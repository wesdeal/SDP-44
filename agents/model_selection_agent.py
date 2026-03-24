"""agents/model_selection_agent.py — Model Selection Agent (Phase 6.1)

Reads task_spec.json, eval_protocol.json, and catalogs/model_catalog.json,
applies deterministic tier-gated selection, and writes selected_models.json.

Architecture authority: AGENT_ARCHITECTURE.md §4 Agent 5.
No legacy files are imported. No LLM calls are made.
"""

import json
import os
import tempfile

from core.manifest import read_manifest, update_stage
from models.model_registry import ModelRegistry

# ---------------------------------------------------------------------------
# Tier preference orderings — AGENT_ARCHITECTURE.md §4 Agent 5
# Key is registry_key (matches model_catalog.json registry_key field).
# ---------------------------------------------------------------------------

_TIER_PREFERENCES = {
    "tabular_regression": {
        "baseline": ["DummyRegressor", "LinearModel"],
        "classical": ["XGBoost", "LightGBM", "RandomForest", "SVR"],
        "specialized": ["SVR", "RandomForest"],
    },
    "tabular_classification": {
        "baseline": ["DummyClassifier", "LinearModel"],
        "classical": ["XGBoost", "LightGBM", "RandomForest"],
        "specialized": ["RandomForest", "LightGBM"],
    },
    "time_series_forecasting": {
        "baseline": ["ARIMA", "DummyRegressor"],
        "classical": ["XGBoost", "RandomForest"],
        "specialized": ["Chronos", "LSTM", "ARIMA"],
    },
}

_TIERS = ("baseline", "classical", "specialized")


class ModelSelectionAgent:
    """Agent 5: Model Selection Agent."""

    def run(self, manifest_path: str) -> None:
        """Select exactly 3 models and write selected_models.json.

        Sets stage status: running → completed | failed.

        Args:
            manifest_path: Path to the run's job_manifest.json.
        """
        update_stage(manifest_path, "model_selection", "running")

        try:
            manifest = read_manifest(manifest_path)
            task_spec = _load_task_spec(manifest)
            eval_protocol = _load_eval_protocol(manifest)
            catalog = _load_catalog(manifest_path)

            selected_models_doc = _select_models(
                manifest, task_spec, eval_protocol, catalog
            )

            artifact_path = _artifact_path(manifest_path)
            _write_json_atomic(artifact_path, selected_models_doc)

            update_stage(
                manifest_path,
                "model_selection",
                "completed",
                artifacts={"selected_models": artifact_path},
            )
        except Exception as exc:
            update_stage(
                manifest_path,
                "model_selection",
                "failed",
                error=str(exc),
            )
            raise


# ---------------------------------------------------------------------------
# Selection logic (pure functions)
# ---------------------------------------------------------------------------

def _select_models(
    manifest: dict,
    task_spec: dict,
    eval_protocol: dict,
    catalog: dict,
) -> dict:
    """Build the selected_models.json document.

    Raises:
        ValueError: With MODEL_POOL_INSUFFICIENT prefix if a required tier cannot
            be filled for the given task_type.
    """
    run_id = manifest["run_id"]
    task_type = task_spec["task_type"]

    if task_type not in _TIER_PREFERENCES:
        raise ValueError(
            f"MODEL_POOL_INSUFFICIENT: task_type={task_type!r} has no tier preference "
            "configuration in ModelSelectionAgent. No compatible model pool defined."
        )

    registry_keys_available = set(ModelRegistry.list_available_models())

    # Build per-tier sets of available registry keys and rejection log
    available_by_tier: dict[str, set] = {t: set() for t in _TIERS}
    models_considered: list[str] = []
    models_rejected: list[dict] = []

    for m in catalog["models"]:
        name = m["name"]
        registry_key = m["registry_key"]
        tier = m["tier"]

        if task_type not in m.get("compatible_tasks", []):
            models_rejected.append({
                "name": name,
                "rejection_reason": (
                    f"task_type {task_type!r} not in compatible_tasks "
                    f"{m.get('compatible_tasks', [])}"
                ),
            })
            continue

        if not m.get("available", False):
            models_rejected.append({
                "name": name,
                "rejection_reason": "catalog available=false",
            })
            continue

        if registry_key not in registry_keys_available:
            models_rejected.append({
                "name": name,
                "rejection_reason": (
                    f"registry_key {registry_key!r} not importable "
                    "(not present in ModelRegistry)"
                ),
            })
            continue

        models_considered.append(name)
        if tier in available_by_tier:
            available_by_tier[tier].add(registry_key)

    # Tier-gated selection
    preferences = _TIER_PREFERENCES[task_type]
    selected: list[dict] = []
    selected_names: set[str] = set()

    for tier in _TIERS:
        pref_list = preferences[tier]
        preferred = pref_list[0]

        chosen = None
        for key in pref_list:
            if key in available_by_tier[tier] and key not in selected_names:
                chosen = key
                break

        if chosen is None:
            raise ValueError(
                f"MODEL_POOL_INSUFFICIENT: No {tier} model available for "
                f"task_type={task_type!r}. "
                f"Available {tier} candidates: {sorted(available_by_tier[tier])}. "
                f"Preference order: {pref_list}. "
                "Ensure at least one model per tier is implemented and registered."
            )

        if chosen != preferred:
            if preferred not in available_by_tier[tier]:
                substitution_reason = (
                    f"Preferred model {preferred!r} is not available in the registry "
                    f"or catalog for task_type={task_type!r}; substituted with {chosen!r}."
                )
            else:
                substitution_reason = (
                    f"Preferred model {preferred!r} was already assigned to another tier; "
                    f"substituted with {chosen!r}."
                )
            substituted_from = preferred
        else:
            substituted_from = None
            substitution_reason = None

        catalog_entry = _find_catalog_entry(catalog, chosen)
        selected.append({
            "name": chosen,
            "tier": tier,
            "rationale": _build_rationale(chosen, tier, task_type, catalog_entry),
            "substituted_from": substituted_from,
            "substitution_reason": substitution_reason,
        })
        selected_names.add(chosen)

    # Hard assertion — architecture requires exactly 3 unique models
    assert len(selected_names) == 3, (
        f"Duplicate model in selection output: {[s['name'] for s in selected]}"
    )

    return {
        "run_id": run_id,
        "task_type": task_type,
        "selected_models": selected,
        "selection_strategy": "tier_gated_deterministic",
        "models_considered": models_considered,
        "models_rejected": models_rejected,
    }


def _build_rationale(
    registry_key: str,
    tier: str,
    task_type: str,
    catalog_entry: dict,
) -> str:
    desc = catalog_entry.get("description", registry_key)
    tier_label = tier.capitalize()
    return f"{tier_label} slot for {task_type}: {desc}"


def _find_catalog_entry(catalog: dict, registry_key: str) -> dict:
    for m in catalog["models"]:
        if m["registry_key"] == registry_key:
            return m
    return {}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _artifact_path(manifest_path: str) -> str:
    """Return absolute path for selected_models.json derived from manifest location."""
    run_dir = os.path.dirname(os.path.abspath(manifest_path))
    return os.path.join(run_dir, "artifacts", "selected_models.json")


def _catalog_path(manifest_path: str) -> str:
    """Return the path to model_catalog.json.

    The catalog is a static repository asset, not a run artifact.  Its
    location is derived from *this file's* path so that the lookup is
    independent of where the run workspace was created.

    This file lives at {repo_root}/agents/model_selection_agent.py, so the
    catalog is one directory up under catalogs/.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "catalogs", "model_catalog.json")


def _load_catalog(manifest_path: str) -> dict:
    path = _catalog_path(manifest_path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"MISSING_CATALOG: model_catalog.json not found at {path!r}."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_task_spec(manifest: dict) -> dict:
    task_spec_path = manifest["stages"]["problem_classification"]["artifacts"]["task_spec"]
    if not os.path.exists(task_spec_path):
        raise FileNotFoundError(
            f"MISSING_TASK_SPEC: task_spec.json not found at {task_spec_path!r}. "
            "Ensure ProblemClassificationAgent completed successfully."
        )
    with open(task_spec_path, encoding="utf-8") as f:
        return json.load(f)


def _load_eval_protocol(manifest: dict) -> dict:
    eval_path = manifest["stages"]["evaluation_protocol"]["artifacts"]["eval_protocol"]
    if not os.path.exists(eval_path):
        raise FileNotFoundError(
            f"MISSING_EVAL_PROTOCOL: eval_protocol.json not found at {eval_path!r}. "
            "Ensure EvaluationProtocolAgent completed successfully."
        )
    with open(eval_path, encoding="utf-8") as f:
        return json.load(f)


def _write_json_atomic(path: str, data: dict) -> None:
    """Write *data* as JSON to *path* atomically (temp-file + rename).

    Creates parent directories as needed.
    """
    dir_name = os.path.dirname(path)
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
