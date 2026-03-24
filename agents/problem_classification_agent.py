"""agents/problem_classification_agent.py — Problem Classification Agent (Phase 4.2)

Reads dataset_profile.json, classifies the ML task type and modality,
confirms the target variable, and writes task_spec.json.

Architecture authority: AGENT_ARCHITECTURE.md §4 Agent 2.
No legacy files are imported.
"""

import json
import os
import tempfile

from core.manifest import read_manifest, update_stage

# ---------------------------------------------------------------------------
# Enum constants (must match AGENT_ARCHITECTURE.md §4 Agent 2)
# ---------------------------------------------------------------------------

_VALID_TASK_TYPES = frozenset({
    "tabular_classification",
    "tabular_regression",
    "time_series_forecasting",
    "grouped_prediction",
})

_TASK_TO_MODALITY = {
    "time_series_forecasting": "time_series",
    "grouped_prediction": "grouped_tabular",
    "tabular_classification": "tabular_iid",
    "tabular_regression": "tabular_iid",
}

# Canonical column names accepted as unambiguous target indicators
_CANONICAL_TARGET_NAMES = frozenset({"target", "label", "y", "output"})

# Integer target cardinality threshold for classification heuristic
_MAX_INT_CLASSIFICATION_CARDINALITY = 20

# Max unique-value fraction for a categorical column to qualify as a group key
_GROUP_MAX_UNIQUE_FRACTION = 0.5


class ProblemClassificationAgent:
    """Agent 2: Problem Classification Agent."""

    def run(self, manifest_path: str) -> None:
        """Classify the ML task and write task_spec.json.

        Sets stage status: running → completed | failed.

        Args:
            manifest_path: Path to the run's job_manifest.json.
        """
        update_stage(manifest_path, "problem_classification", "running")

        try:
            manifest = read_manifest(manifest_path)
            profile = _load_profile(manifest)
            task_spec = self._build_task_spec(manifest, profile)
            artifact_path = _artifact_path(manifest_path)
            _write_json_atomic(artifact_path, task_spec)
            update_stage(
                manifest_path,
                "problem_classification",
                "completed",
                artifacts={"task_spec": artifact_path},
            )
        except Exception as exc:
            update_stage(
                manifest_path,
                "problem_classification",
                "failed",
                error=str(exc),
            )
            raise

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def _build_task_spec(self, manifest: dict, profile: dict) -> dict:
        run_id = manifest["run_id"]
        columns = profile["columns"]
        col_map = {c["name"]: c for c in columns}
        detected_datetime_columns = profile.get("detected_datetime_columns", [])
        llm_analysis = profile.get("llm_analysis", {})
        num_rows = profile["num_rows"]

        warnings: list[str] = []

        # Step 1: Confirm target column
        target_col, target_reasoning, task_confidence = _confirm_target(
            col_map, llm_analysis, warnings
        )

        # Guardrail: target_col must exist in column list
        if target_col not in col_map:
            raise ValueError(
                f"TARGET_NOT_FOUND: target_col {target_col!r} not found in "
                "dataset_profile columns."
            )

        target_profile = col_map[target_col]

        # Guardrail: target too sparse
        if target_profile["missing_fraction"] > 0.5:
            raise ValueError(
                f"TARGET_TOO_SPARSE: target column {target_col!r} has "
                f"{target_profile['missing_fraction']:.1%} missing values "
                "(threshold: 50%)."
            )

        # Guardrail: target constant
        if target_profile["unique_count"] <= 1:
            raise ValueError(
                f"TARGET_CONSTANT: target column {target_col!r} has only "
                f"{target_profile['unique_count']} unique value(s); "
                "cannot train a model."
            )

        # Step 2: Detect time_col from datetime columns (exclude target)
        time_col = _detect_time_col(col_map, detected_datetime_columns, target_col)

        # Step 3: Detect group_col
        group_col = _detect_group_col(
            col_map, target_col, detected_datetime_columns, num_rows
        )

        # Step 4: Classify task type (priority order per architecture)
        task_type, modality, task_reasoning, task_confidence, time_col, group_col = (
            _classify_task(
                target_profile,
                detected_datetime_columns,
                col_map,
                time_col,
                group_col,
                task_confidence,
                target_reasoning,
                warnings,
            )
        )

        # Guardrail: TS selected but no time_col
        if task_type == "time_series_forecasting" and time_col is None:
            raise ValueError(
                "TS_NO_TIME_COLUMN: task_type=time_series_forecasting was selected "
                "but no time column could be confirmed."
            )

        # Step 5: Target metadata
        target_dtype = (
            "numeric"
            if target_profile["inferred_type"] == "numeric"
            else "categorical"
        )
        target_cardinality = target_profile["unique_count"]

        # Step 6: Subtypes
        classification_subtype = None
        regression_subtype = None

        if task_type == "tabular_classification":
            classification_subtype = (
                "binary" if target_cardinality <= 2 else "multiclass"
            )

        if task_type in ("tabular_regression", "time_series_forecasting"):
            regression_subtype = _infer_regression_subtype(target_profile)

        # Step 7: Forecast horizon and is_multivariate_ts (time-series only)
        forecast_horizon = None
        is_multivariate_ts = None
        if task_type == "time_series_forecasting":
            cfg = manifest.get("config", {})
            forecast_horizon = int(cfg.get("forecast_horizon", 10))
            is_multivariate_ts = bool(llm_analysis.get("is_multivariate", False))

        return {
            "run_id": run_id,
            "task_type": task_type,
            "modality": modality,
            "target_col": target_col,
            "target_dtype": target_dtype,
            "target_cardinality": target_cardinality,
            "time_col": time_col,
            "group_col": group_col,
            "forecast_horizon": forecast_horizon,
            "is_multivariate_ts": is_multivariate_ts,
            "classification_subtype": classification_subtype,
            "regression_subtype": regression_subtype,
            "task_confidence": task_confidence,
            "task_reasoning": task_reasoning,
            "warnings": warnings,
        }


# ---------------------------------------------------------------------------
# Module-level helpers (pure functions, no side effects)
# ---------------------------------------------------------------------------

def _artifact_path(manifest_path: str) -> str:
    """Return absolute path for task_spec.json derived from manifest location."""
    run_dir = os.path.dirname(os.path.abspath(manifest_path))
    return os.path.join(run_dir, "artifacts", "task_spec.json")


def _load_profile(manifest: dict) -> dict:
    """Read dataset_profile.json via the path stored in the manifest."""
    profile_path = manifest["stages"]["ingestion"]["artifacts"]["dataset_profile"]
    with open(profile_path, encoding="utf-8") as f:
        return json.load(f)


def _confirm_target(
    col_map: dict, llm_analysis: dict, warnings: list
) -> tuple[str, str, str]:
    """Return (target_col, reasoning, confidence) per architecture §4 Agent 2.

    Priority order:
    1. LLM high-confidence suggestion.
    2. Exactly one canonical-named column ("target", "label", "y", "output").
    3. Exactly one float column with no missing values.
    4. LLM medium/low suggestion with warning.
    5. Last numeric column heuristic with warning.
    Raises ValueError if no candidate can be found.
    """
    suggested = llm_analysis.get("suggested_target_variable")
    llm_confidence = llm_analysis.get("target_confidence", "low")
    llm_reasoning = llm_analysis.get("target_reasoning", "")

    # 1. High-confidence LLM suggestion
    if suggested and llm_confidence == "high" and suggested in col_map:
        return suggested, f"LLM high-confidence suggestion: {llm_reasoning}", "high"

    # 2. Canonical column name (exactly one match)
    canonical_matches = [n for n in _CANONICAL_TARGET_NAMES if n in col_map]
    if len(canonical_matches) == 1:
        col = canonical_matches[0]
        return col, f"Canonical target name {col!r} found in dataset.", "medium"

    # 3. Exactly one float column with no missing values
    float_no_missing = [
        name
        for name, prof in col_map.items()
        if prof["inferred_type"] == "numeric"
        and prof["missing_fraction"] == 0.0
        and any(t in prof["dtype_pandas"] for t in ("float64", "float32", "float16"))
    ]
    if len(float_no_missing) == 1:
        col = float_no_missing[0]
        return (
            col,
            f"Single float column with no missing values: {col!r}.",
            "medium",
        )

    # 4. LLM medium/low suggestion with warning
    if suggested and suggested in col_map:
        warnings.append(
            f"target_col {suggested!r} accepted from LLM suggestion at "
            f"confidence={llm_confidence!r}. Verify manually."
        )
        return (
            suggested,
            f"LLM suggestion (low confidence): {llm_reasoning}",
            "low",
        )

    # 5. Heuristic fallback: last numeric column
    warnings.append(
        "LLM target suggestion unavailable or invalid. "
        "Falling back to last numeric column heuristic."
    )
    numeric_cols = [
        name for name, prof in col_map.items() if prof["inferred_type"] == "numeric"
    ]
    if numeric_cols:
        col = numeric_cols[-1]
        return (
            col,
            "Heuristic fallback: last numeric column selected as target.",
            "low",
        )

    raise ValueError(
        "TARGET_NOT_FOUND: No suitable target column could be identified. "
        "Provide a dataset with at least one numeric column or configure target_col explicitly."
    )


def _detect_time_col(
    col_map: dict, detected_datetime_columns: list, target_col: str
) -> str | None:
    """Return the best time column, preferring monotonically increasing ones."""
    candidates = [c for c in detected_datetime_columns if c != target_col]
    if not candidates:
        return None
    # Prefer monotonically increasing
    for c in candidates:
        if col_map.get(c, {}).get("is_monotonically_increasing"):
            return c
    return candidates[0]


def _detect_group_col(
    col_map: dict,
    target_col: str,
    detected_datetime_columns: list,
    num_rows: int,
) -> str | None:
    """Detect a group key: categorical, low cardinality, with a time dimension.

    Returns None if no datetime dimension exists (no grouped_prediction possible).
    """
    if not detected_datetime_columns:
        return None

    max_unique = max(2, int(num_rows * _GROUP_MAX_UNIQUE_FRACTION))

    for name, prof in col_map.items():
        if name == target_col:
            continue
        if name in detected_datetime_columns:
            continue
        if prof["inferred_type"] != "categorical":
            continue
        unique_count = prof["unique_count"]
        if 2 <= unique_count <= max_unique:
            return name

    return None


def _classify_task(
    target_profile: dict,
    detected_datetime_columns: list,
    col_map: dict,
    time_col: str | None,
    group_col: str | None,
    task_confidence: str,
    target_reasoning: str,
    warnings: list,
) -> tuple[str, str, str, str, str | None, str | None]:
    """Return (task_type, modality, reasoning, confidence, time_col, group_col).

    Priority order is strictly per AGENT_ARCHITECTURE.md §4 Agent 2.
    """
    target_name = target_profile["name"]
    target_inferred = target_profile["inferred_type"]
    target_unique = target_profile["unique_count"]
    dtype_pandas = target_profile["dtype_pandas"]

    # Priority 1: time_series_forecasting
    # Condition: datetime cols present, numeric target, time_col is monotonically increasing
    if detected_datetime_columns and target_inferred == "numeric" and time_col is not None:
        time_prof = col_map.get(time_col, {})
        if time_prof.get("is_monotonically_increasing"):
            return (
                "time_series_forecasting",
                "time_series",
                (
                    f"Datetime column {time_col!r} is monotonically increasing and "
                    f"target {target_name!r} is numeric → time_series_forecasting."
                ),
                task_confidence,
                time_col,
                group_col,
            )

    # Priority 2: grouped_prediction
    if group_col is not None:
        return (
            "grouped_prediction",
            "grouped_tabular",
            (
                f"Group column {group_col!r} detected with datetime dimension "
                f"→ grouped_prediction."
            ),
            task_confidence,
            time_col,
            group_col,
        )

    # Priority 3: tabular_classification
    is_int_low_card = (
        "int" in dtype_pandas
        and target_unique <= _MAX_INT_CLASSIFICATION_CARDINALITY
    )
    if target_inferred == "categorical" or is_int_low_card:
        return (
            "tabular_classification",
            "tabular_iid",
            (
                f"Target {target_name!r} is categorical or integer with "
                f"cardinality {target_unique} ≤ {_MAX_INT_CLASSIFICATION_CARDINALITY} "
                f"→ tabular_classification."
            ),
            task_confidence,
            None,
            None,
        )

    # Priority 4: tabular_regression
    if target_inferred == "numeric":
        return (
            "tabular_regression",
            "tabular_iid",
            f"Target {target_name!r} is numeric → tabular_regression.",
            task_confidence,
            None,
            None,
        )

    # Default (ambiguous heuristics, LLM unavailable)
    warnings.append(
        "CLASSIFICATION_AMBIGUOUS: All heuristics were inconclusive. "
        "Defaulting to tabular_regression. Verify task type manually."
    )
    return (
        "tabular_regression",
        "tabular_iid",
        "All heuristics ambiguous; defaulting to tabular_regression (safe default).",
        "low",
        None,
        None,
    )


def _infer_regression_subtype(target_profile: dict) -> str:
    """Heuristic regression subtype: count, bounded, or standard."""
    dtype = target_profile["dtype_pandas"]
    min_val = target_profile.get("min")
    max_val = target_profile.get("max")
    has_negative = target_profile.get("has_negative_values", True)

    # Count: integer dtype, non-negative values
    if "int" in dtype and not has_negative:
        return "count"

    # Bounded: float values entirely within [0, 1]
    if (
        min_val is not None
        and max_val is not None
        and min_val >= 0.0
        and max_val <= 1.0
    ):
        return "bounded"

    return "standard"


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
