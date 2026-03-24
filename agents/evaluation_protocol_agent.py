"""agents/evaluation_protocol_agent.py — Evaluation Protocol Agent (Phase 5.1)

Reads task_spec.json and dataset_profile.json, applies a deterministic routing
table to define the evaluation protocol, and writes eval_protocol.json.

Architecture authority: AGENT_ARCHITECTURE.md §4 Agent 4.
No legacy files are imported. No LLM calls are made.
"""

import json
import os
import tempfile

from core.manifest import read_manifest, update_stage

# ---------------------------------------------------------------------------
# Constants (must match AGENT_ARCHITECTURE.md §4 Agent 4)
# ---------------------------------------------------------------------------

_MINIMUM_TEST_SAMPLES = 30
_DEFAULT_TRAIN_FRACTION = 0.7
_DEFAULT_VAL_FRACTION = 0.1
_DEFAULT_TEST_FRACTION = 0.2

# Full global metric catalog — order and fields match AGENT_ARCHITECTURE.md §4 Agent 4.
# grouped_prediction is not listed in the architecture catalog but the routing table
# assigns it primary_metric=rmse; metrics_for_this_run for that task_type is handled
# explicitly via _GROUPED_PREDICTION_METRICS below.
_METRIC_CATALOG = [
    {
        "name": "rmse",
        "display_name": "RMSE",
        "higher_is_better": False,
        "applicable_tasks": ["tabular_regression", "time_series_forecasting"],
    },
    {
        "name": "mae",
        "display_name": "MAE",
        "higher_is_better": False,
        "applicable_tasks": ["tabular_regression", "time_series_forecasting"],
    },
    {
        "name": "mape",
        "display_name": "MAPE (%)",
        "higher_is_better": False,
        "applicable_tasks": ["tabular_regression", "time_series_forecasting"],
    },
    {
        "name": "smape",
        "display_name": "sMAPE (%)",
        "higher_is_better": False,
        "applicable_tasks": ["time_series_forecasting"],
    },
    {
        "name": "pinball_loss",
        "display_name": "Pinball Loss",
        "higher_is_better": False,
        "applicable_tasks": ["time_series_forecasting"],
        "requires": "quantile",
    },
    {
        "name": "coverage_80",
        "display_name": "Coverage @ 80%",
        "higher_is_better": True,
        "applicable_tasks": ["time_series_forecasting"],
        "requires": "quantile",
    },
    {
        "name": "interval_width_80",
        "display_name": "Interval Width @ 80%",
        "higher_is_better": False,
        "applicable_tasks": ["time_series_forecasting"],
        "requires": "quantile",
    },
    {
        "name": "r2",
        "display_name": "R²",
        "higher_is_better": True,
        "applicable_tasks": ["tabular_regression"],
    },
    {
        "name": "accuracy",
        "display_name": "Accuracy",
        "higher_is_better": True,
        "applicable_tasks": ["tabular_classification"],
    },
    {
        "name": "f1_weighted",
        "display_name": "F1 (weighted)",
        "higher_is_better": True,
        "applicable_tasks": ["tabular_classification"],
    },
    {
        "name": "roc_auc",
        "display_name": "ROC-AUC",
        "higher_is_better": True,
        "applicable_tasks": ["tabular_classification"],
    },
]

# grouped_prediction metrics: treated as regression-like (primary_metric=rmse per routing table)
_GROUPED_PREDICTION_METRICS = ["rmse", "mae", "mape", "r2"]

# Disabled CV block used for all non-time_series_cv strategies
_CV_DISABLED = {
    "enabled": False,
    "method": None,
    "n_folds": None,
    "initial_train_size": None,
    "step_size": None,
    "val_window_size": None,
    "gap": None,
    "aggregate": None,
    "use_for": None,
}


class EvaluationProtocolAgent:
    """Agent 4: Evaluation Protocol Agent."""

    def run(self, manifest_path: str) -> None:
        """Determine the evaluation protocol and write eval_protocol.json.

        Sets stage status: running → completed | failed.

        Args:
            manifest_path: Path to the run's job_manifest.json.
        """
        update_stage(manifest_path, "evaluation_protocol", "running")

        try:
            manifest = read_manifest(manifest_path)
            task_spec = _load_task_spec(manifest)
            profile = _load_dataset_profile(manifest)
            protocol = _build_protocol(manifest, task_spec, profile)
            artifact_path = _artifact_path(manifest_path)
            _write_json_atomic(artifact_path, protocol)
            update_stage(
                manifest_path,
                "evaluation_protocol",
                "completed",
                artifacts={"eval_protocol": artifact_path},
            )
        except Exception as exc:
            update_stage(
                manifest_path,
                "evaluation_protocol",
                "failed",
                error=str(exc),
            )
            raise


# ---------------------------------------------------------------------------
# Protocol construction (pure functions)
# ---------------------------------------------------------------------------

def _build_protocol(manifest: dict, task_spec: dict, profile: dict) -> dict:
    """Build the complete eval_protocol dict from task_spec and profile."""
    run_id = manifest["run_id"]
    task_type = task_spec["task_type"]
    num_rows = profile["num_rows"]

    warnings: list[str] = []

    # Step 1: Route to initial protocol via routing table
    split_strategy, shuffle, primary_metric, rationale = _route(task_spec, warnings)

    # Step 2: Resolve split fractions
    train_fraction = _DEFAULT_TRAIN_FRACTION
    val_fraction = _DEFAULT_VAL_FRACTION
    test_fraction = _DEFAULT_TEST_FRACTION

    # Guardrail: minimum test samples — reduce val fraction if needed
    n_test = int(num_rows * test_fraction)
    if n_test < _MINIMUM_TEST_SAMPLES:
        required_test_fraction = _MINIMUM_TEST_SAMPLES / num_rows
        if required_test_fraction <= 0.3:
            excess = required_test_fraction - test_fraction
            test_fraction = required_test_fraction
            val_fraction = max(0.0, val_fraction - excess)
            warnings.append(
                f"MINIMUM_TEST_SAMPLES: test set would have {n_test} rows (< {_MINIMUM_TEST_SAMPLES}). "
                f"Adjusted test_fraction to {test_fraction:.4f}, val_fraction to {val_fraction:.4f}."
            )
        else:
            warnings.append(
                f"MINIMUM_TEST_SAMPLES_UNRESOLVABLE: dataset has only {num_rows} rows. "
                f"Test set will have {n_test} rows (< {_MINIMUM_TEST_SAMPLES}). "
                "Proceeding with default fractions."
            )

    # Step 3: Resolve optional columns
    stratify_on = None
    group_col = task_spec.get("group_col")
    time_col = task_spec.get("time_col")
    target_col = task_spec.get("target_col")

    if split_strategy == "stratified":
        stratify_on = target_col

    if split_strategy == "group_kfold":
        if not group_col:
            warnings.append(
                "GROUP_COL_MISSING: group_kfold requires group_col but none was set in task_spec. "
                "Downgrading split_strategy to random."
            )
            split_strategy = "random"
            shuffle = True
            rationale += " [downgraded from group_kfold: group_col missing]"

    # Step 4: prediction_type (point unless config specifies quantile)
    prediction_type = "point"
    quantiles: list = []

    # Step 5: metrics_for_this_run
    metrics_for_this_run = _select_metrics(task_type, prediction_type)

    return {
        "run_id": run_id,
        "task_type": task_type,
        "split_strategy": split_strategy,
        "split_rationale": rationale,
        "train_fraction": round(train_fraction, 6),
        "val_fraction": round(val_fraction, 6),
        "test_fraction": round(test_fraction, 6),
        "shuffle": shuffle,
        "stratify_on": stratify_on,
        "group_col": group_col if split_strategy == "group_kfold" else None,
        "time_col": time_col if split_strategy in ("chronological", "time_series_cv") else None,
        "cv": _CV_DISABLED,
        "prediction_type": prediction_type,
        "quantiles": quantiles,
        "primary_metric": primary_metric,
        "metrics": _METRIC_CATALOG,
        "metrics_for_this_run": metrics_for_this_run,
        "minimum_test_samples": _MINIMUM_TEST_SAMPLES,
        "warnings": warnings,
    }


def _route(task_spec: dict, warnings: list) -> tuple[str, bool, str, str]:
    """Apply the protocol routing table.

    Returns (split_strategy, shuffle, primary_metric, rationale).
    Routing table: AGENT_ARCHITECTURE.md §4 Agent 4.
    """
    task_type = task_spec["task_type"]
    classification_subtype = task_spec.get("classification_subtype")
    target_cardinality = task_spec.get("target_cardinality", 0)
    num_rows = None  # Not available here; minimum sample check uses profile

    if task_type == "tabular_regression":
        return (
            "random",
            True,
            "rmse",
            "tabular_regression → random split 70/10/20, primary metric RMSE.",
        )

    if task_type == "tabular_classification":
        if classification_subtype == "binary":
            strategy, primary, note = "stratified", "roc_auc", "binary classification → stratified split, primary metric ROC-AUC."
        else:
            strategy, primary, note = "stratified", "f1_weighted", "multiclass classification → stratified split, primary metric F1 (weighted)."

        # Guardrail: stratified requires ≥2 classes; rough proxy using cardinality
        if target_cardinality < 2:
            warnings.append(
                "STRATIFY_INVALID: target_cardinality < 2; cannot stratify. "
                "Downgrading split_strategy to random."
            )
            return "random", True, primary, note + " [downgraded from stratified: target_cardinality < 2]"

        return strategy, True, primary, note

    if task_type == "time_series_forecasting":
        return (
            "chronological",
            False,
            "mae",
            "time_series_forecasting → chronological split 70/10/20, shuffle=false, primary metric MAE.",
        )

    if task_type == "grouped_prediction":
        return (
            "group_kfold",
            False,
            "rmse",
            "grouped_prediction → group_kfold split, shuffle=false, primary metric RMSE.",
        )

    raise ValueError(
        f"UNSUPPORTED_TASK_TYPE: task_type={task_type!r} is not supported by EvaluationProtocolAgent."
    )


def _select_metrics(task_type: str, prediction_type: str) -> list[str]:
    """Return the list of metric names applicable to this task_type and prediction_type."""
    if task_type == "grouped_prediction":
        return list(_GROUPED_PREDICTION_METRICS)

    result = []
    for m in _METRIC_CATALOG:
        if task_type not in m["applicable_tasks"]:
            continue
        requires = m.get("requires")
        if requires and requires != prediction_type:
            continue
        result.append(m["name"])
    return result


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _artifact_path(manifest_path: str) -> str:
    """Return absolute path for eval_protocol.json derived from manifest location."""
    run_dir = os.path.dirname(os.path.abspath(manifest_path))
    return os.path.join(run_dir, "artifacts", "eval_protocol.json")


def _load_task_spec(manifest: dict) -> dict:
    """Read task_spec.json via the path stored in the manifest."""
    task_spec_path = manifest["stages"]["problem_classification"]["artifacts"]["task_spec"]
    if not os.path.exists(task_spec_path):
        raise FileNotFoundError(
            f"MISSING_TASK_SPEC: task_spec.json not found at {task_spec_path!r}. "
            "Ensure ProblemClassificationAgent completed successfully."
        )
    with open(task_spec_path, encoding="utf-8") as f:
        return json.load(f)


def _load_dataset_profile(manifest: dict) -> dict:
    """Read dataset_profile.json via the path stored in the manifest."""
    profile_path = manifest["stages"]["ingestion"]["artifacts"]["dataset_profile"]
    with open(profile_path, encoding="utf-8") as f:
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
