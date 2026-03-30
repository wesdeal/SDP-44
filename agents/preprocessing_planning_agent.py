"""agents/preprocessing_planning_agent.py — Preprocessing Planning Agent (Phase 5.3)

Planning sub-phase for Agent 3.  Reads dataset_profile.json and task_spec.json,
produces a preprocessing_plan.json, then delegates all execution to
preprocessing_execution.execute() (the Phase 5.2 execution layer).

Architecture authority: AGENT_ARCHITECTURE.md §4 Agent 3.
No legacy files are imported.
"""

import json
import os

from core.manifest import read_manifest, update_stage
from agents.preprocessing_execution import execute as _execute

# ---------------------------------------------------------------------------
# Catalog path — derived from this file's location; never a runtime literal.
# ---------------------------------------------------------------------------

_CATALOG_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "catalogs",
        "transformer_catalog.json",
    )
)


# ---------------------------------------------------------------------------
# Public agent class
# ---------------------------------------------------------------------------


class PreprocessingPlanningAgent:
    """Agent 3 planning sub-phase: builds a preprocessing plan and runs it."""

    def run(self, manifest_path: str) -> None:
        """Build a preprocessing plan and delegate to the execution layer.

        Stage lifecycle:
        - Sets ``preprocessing_planning`` → ``"running"`` immediately.
        - If plan construction fails: sets ``"failed"`` and re-raises.
        - If plan construction succeeds: calls ``execute()``, which handles
          the final ``"running"`` refresh + ``"completed"``/``"failed"``
          transition and artifact path writes.

        Args:
            manifest_path: Path to the run's job_manifest.json.
        """
        update_stage(manifest_path, "preprocessing_planning", "running")

        try:
            manifest = read_manifest(manifest_path)
            dataset_profile = _load_dataset_profile(manifest)
            task_spec = _load_task_spec(manifest)
            plan = _build_plan(dataset_profile, task_spec, manifest["run_id"])
        except Exception as exc:
            try:
                update_stage(
                    manifest_path,
                    "preprocessing_planning",
                    "failed",
                    error=str(exc),
                )
            except Exception:
                pass
            raise

        # Execution layer handles its own running → completed/failed transitions.
        _execute(plan, manifest_path)


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------


def _build_plan(dataset_profile: dict, task_spec: dict, run_id: str) -> dict:
    """Return a validated preprocessing_plan dict.

    Attempts LLM-assisted planning first.  Falls back to the minimal safe
    heuristic plan on any failure or invalid LLM output.
    """
    catalog_map = _load_transformer_catalog()
    modality = task_spec["modality"]

    plan = None

    try:
        llm_raw = _call_llm_for_plan(dataset_profile, task_spec)
        if llm_raw is not None:
            validated = _validate_llm_plan(llm_raw, catalog_map, modality, task_spec)
            if validated is not None:
                validated["run_id"] = run_id
                validated["plan_source"] = "llm"
                plan = validated
    except Exception:
        pass  # Any failure in LLM path → fall through to heuristic.

    if plan is None:
        plan = _heuristic_fallback_plan(run_id, task_spec, dataset_profile)

    return plan


def _call_llm_for_plan(dataset_profile: dict, task_spec: dict):
    """Attempt LLM-assisted plan generation.

    Returns a raw plan dict parsed from the LLM response, or ``None`` when
    the LLM is unavailable (no API key, missing package, or call failure).

    No LLM integration is currently active in this repository.  The function
    returns ``None`` immediately when no API key is configured, which triggers
    the heuristic fallback path (AGENT_ARCHITECTURE.md §4 Agent 3 Failure
    Handling).
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        import openai  # noqa: PLC0415 — intentional late import
    except ImportError:
        return None

    prompt = _build_llm_prompt(dataset_profile, task_spec)

    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a preprocessing planning assistant. "
                        "Return ONLY a JSON object conforming to the "
                        "preprocessing_plan schema. Do not include any text "
                        "outside the JSON object."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=2048,
        )
        raw_text = response.choices[0].message.content.strip()

        # Strip markdown code fences if the model wrapped the JSON.
        if raw_text.startswith("```"):
            lines = raw_text.splitlines()
            lines = [ln for ln in lines if not ln.startswith("```")]
            raw_text = "\n".join(lines).strip()

        return json.loads(raw_text)
    except Exception:
        return None


def _build_llm_prompt(dataset_profile: dict, task_spec: dict) -> str:
    """Return a compact JSON prompt for LLM preprocessing planning."""
    modality = task_spec["modality"]
    target_col = task_spec["target_col"]
    time_col = task_spec.get("time_col")
    group_col = task_spec.get("group_col")

    col_summary = [
        {
            "name": c["name"],
            "type": c["inferred_type"],
            "missing_fraction": c.get("missing_fraction", 0.0),
        }
        for c in dataset_profile.get("columns", [])
    ]

    exclude = [c for c in [time_col, group_col] if c]

    return json.dumps(
        {
            "task": "preprocessing_plan",
            "modality": modality,
            "target_col": target_col,
            "exclude_columns_from_features": exclude,
            "columns": col_summary,
            "instruction": (
                "Return a preprocessing_plan JSON object with these top-level "
                "fields: steps (list), preserve_temporal_order (bool), "
                "exclude_columns_from_features (list). "
                "Each step must have: order (int), method (string from the "
                "transformer catalog), parameters (dict), "
                "applies_to (all_numeric|all_categorical|features_only|target_only|[list]), "
                "reason (string), skip_columns (list). "
                f"Valid methods for modality {modality!r}: imputation, z_norm, "
                "min_max (all modalities); detrend, differencing, smoothing "
                "(time_series only); label_encode, onehot_encode, log_transform, "
                "remove_outliers (tabular_iid only). "
                "Do NOT transform the target column or excluded columns."
            ),
        },
        indent=2,
    )


def _validate_llm_plan(
    raw: object,
    catalog_map: dict,
    modality: str,
    task_spec: dict,
) -> dict | None:
    """Validate and sanitize an LLM-produced plan dict.

    Filters out steps with unknown methods, modality violations, missing
    required fields, or invalid order values.  Returns ``None`` if no valid
    steps remain after filtering.
    """
    if not isinstance(raw, dict):
        return None

    steps = raw.get("steps")
    if not isinstance(steps, list) or not steps:
        return None

    target_col = task_spec["target_col"]
    time_col = task_spec.get("time_col")
    group_col = task_spec.get("group_col")
    protected = {c for c in [target_col, time_col, group_col] if c}

    valid_steps = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        method = step.get("method", "")
        if method not in catalog_map:
            continue
        if modality not in catalog_map[method]["allowed_modalities"]:
            continue
        order = step.get("order")
        if not isinstance(order, int):
            continue
        applies_to = step.get("applies_to", "features_only")
        # If applies_to is a column list, strip protected columns.
        if isinstance(applies_to, list):
            applies_to = [c for c in applies_to if c not in protected]
            if not applies_to:
                continue
        skip_cols = step.get("skip_columns") or []
        if not isinstance(skip_cols, list):
            skip_cols = []
        parameters = step.get("parameters") or {}
        if not isinstance(parameters, dict):
            parameters = {}
        valid_steps.append(
            {
                "order": order,
                "method": method,
                "parameters": parameters,
                "applies_to": applies_to,
                "reason": str(step.get("reason", "")),
                "skip_columns": skip_cols,
            }
        )

    if not valid_steps:
        return None

    # Re-sequence order to be contiguous starting at 1.
    valid_steps.sort(key=lambda s: s["order"])
    for i, s in enumerate(valid_steps, 1):
        s["order"] = i

    exclude = raw.get("exclude_columns_from_features") or []
    if not isinstance(exclude, list):
        exclude = []
    # Always include time_col and group_col in exclusions.
    for col in [time_col, group_col]:
        if col and col not in exclude:
            exclude.append(col)

    preserve_temporal = raw.get("preserve_temporal_order")
    if not isinstance(preserve_temporal, bool):
        preserve_temporal = modality == "time_series"

    return {
        "steps": valid_steps,
        "preserve_temporal_order": preserve_temporal,
        "exclude_columns_from_features": exclude,
    }


def _heuristic_fallback_plan(
    run_id: str, task_spec: dict, dataset_profile: dict = None
) -> dict:
    """Return the minimal safe heuristic preprocessing plan.

    For ``time_series`` modality, builds a richer plan that includes lag
    features and rolling statistics so that tree-based models (XGBoost,
    RandomForest) receive meaningful temporal features.  Rolling/lag steps
    are applied to the known original feature columns (derived from
    dataset_profile) to avoid feature explosion from chained transforms.

    For other modalities: imputation → z_norm (unchanged from prior behaviour).

    ``plan_source`` is set to ``"heuristic_fallback"``.
    """
    modality = task_spec["modality"]
    time_col = task_spec.get("time_col")
    group_col = task_spec.get("group_col")
    target_col = task_spec.get("target_col")

    exclude = [c for c in [time_col, group_col] if c]

    if modality != "time_series":
        return {
            "run_id": run_id,
            "steps": [
                {
                    "order": 1,
                    "method": "imputation",
                    "parameters": {"strategy": "mean"},
                    "applies_to": "features_only",
                    "reason": "Fill missing values before normalization.",
                    "skip_columns": [],
                },
                {
                    "order": 2,
                    "method": "z_norm",
                    "parameters": {},
                    "applies_to": "all_numeric",
                    "reason": "Standardize numeric features to zero mean and unit variance.",
                    "skip_columns": [],
                },
            ],
            "preserve_temporal_order": False,
            "exclude_columns_from_features": exclude,
            "plan_source": "heuristic_fallback",
        }

    # --- Time-series specific plan ---
    # Derive the original numeric feature column names from the dataset profile
    # so that rolling_stats and lag_features apply only to the original columns,
    # avoiding the quadratic feature explosion that would occur if rolling/lag
    # transforms were applied to each other's output columns.
    always_excluded = {c for c in [time_col, group_col, target_col] if c}
    original_feature_cols: list = []
    if dataset_profile is not None:
        for col_info in dataset_profile.get("columns", []):
            name = col_info.get("name", "")
            if (
                col_info.get("inferred_type") == "numeric"
                and name not in always_excluded
            ):
                original_feature_cols.append(name)

    # Fall back to "features_only" keyword when profile is unavailable.
    applies_to_features = original_feature_cols if original_feature_cols else "features_only"

    steps = [
        {
            "order": 1,
            "method": "imputation",
            "parameters": {"strategy": "mean"},
            "applies_to": "features_only",
            "reason": "Fill missing values before feature engineering.",
            "skip_columns": [],
        },
        {
            "order": 2,
            "method": "rolling_stats",
            "parameters": {
                "window": 24,
                "include_target_lags": False,
                "target_col": target_col,
            },
            "applies_to": applies_to_features,
            "reason": (
                "Rolling mean/std (24-step window) on original features to capture "
                "local trends without leaking from the target."
            ),
            "skip_columns": [],
        },
        {
            "order": 3,
            "method": "lag_features",
            "parameters": {
                "lags": [1, 2, 3, 6, 24],
                "include_target_lags": True,
                "target_col": target_col,
            },
            "applies_to": applies_to_features,
            "reason": (
                "Lag features (1, 2, 3, 6, 24 steps) on original features plus "
                "target lags so tree-based models can learn temporal dynamics."
            ),
            "skip_columns": [],
        },
        {
            "order": 4,
            "method": "z_norm",
            "parameters": {},
            "applies_to": "all_numeric",
            "reason": (
                "Standardize all numeric columns (original + rolling + lag) "
                "to zero mean and unit variance."
            ),
            "skip_columns": [],
        },
    ]

    return {
        "run_id": run_id,
        "steps": steps,
        "preserve_temporal_order": True,
        "exclude_columns_from_features": exclude,
        "plan_source": "heuristic_fallback",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_dataset_profile(manifest: dict) -> dict:
    """Read dataset_profile.json via the artifact path in the manifest."""
    path = manifest["stages"]["ingestion"]["artifacts"]["dataset_profile"]
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_task_spec(manifest: dict) -> dict:
    """Read task_spec.json via the artifact path in the manifest."""
    path = manifest["stages"]["problem_classification"]["artifacts"]["task_spec"]
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_transformer_catalog() -> dict:
    """Return transformer catalog as {name: entry} dict."""
    with open(_CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)
    return {entry["name"]: entry for entry in catalog["transformers"]}
