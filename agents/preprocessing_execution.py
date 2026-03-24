"""agents/preprocessing_execution.py — Preprocessing Execution sub-phase (Phase 5.2)

Accepts a preprocessing_plan.json-style dict and applies the plan deterministically
to the input dataset. Produces processed_data.csv and preprocessing_manifest.json.

Architecture authority: AGENT_ARCHITECTURE.md §4 Agent 3 (execution sub-phase).
No LLM calls are made. No legacy files are imported.
"""

import json
import os
import tempfile

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from core.manifest import read_manifest, update_stage

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_STEP_FAILURES = 2  # abort when strictly more than this many steps fail

# Derived at import time from this file's location; never a runtime hardcoded string.
_CATALOG_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "catalogs",
        "transformer_catalog.json",
    )
)

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def execute(plan: dict, manifest_path: str) -> None:
    """Apply *plan* to the input dataset and write preprocessing artifacts.

    Reads the original input file and task_spec artifact from the manifest.
    Writes:
      - processed_data.csv
      - preprocessing_manifest.json
      - preprocessing_plan.json (verbatim copy of *plan*)

    Sets the ``preprocessing_planning`` stage: running → completed | failed.

    Args:
        plan:           Preprocessing plan dict (preprocessing_plan.json schema).
        manifest_path:  Path to the run's job_manifest.json.
    """
    update_stage(manifest_path, "preprocessing_planning", "running")

    try:
        manifest = read_manifest(manifest_path)
        task_spec = _load_task_spec(manifest)
        df = _load_input(manifest)

        processed_df, steps_applied = _apply_plan(plan, df, task_spec)

        run_dir = os.path.dirname(os.path.abspath(manifest_path))
        artifacts_dir = os.path.join(run_dir, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        processed_data_path = os.path.join(artifacts_dir, "processed_data.csv")
        preproc_manifest_path = os.path.join(artifacts_dir, "preprocessing_manifest.json")
        plan_path = os.path.join(artifacts_dir, "preprocessing_plan.json")

        processed_df.to_csv(processed_data_path, index=False)

        excluded = _excluded_columns(plan, task_spec)
        target_col = task_spec["target_col"]
        feature_columns = [
            c for c in processed_df.columns
            if c != target_col and c not in excluded
        ]

        preproc_manifest = {
            "run_id": manifest["run_id"],
            "steps_applied": steps_applied,
            "final_shape": {
                "rows": int(len(processed_df)),
                "cols": int(len(processed_df.columns)),
            },
            "feature_columns": feature_columns,
            "target_column": target_col,
            "time_column": task_spec.get("time_col"),
        }

        _write_json_atomic(preproc_manifest_path, preproc_manifest)
        _write_json_atomic(plan_path, plan)

        update_stage(
            manifest_path,
            "preprocessing_planning",
            "completed",
            artifacts={
                "preprocessing_plan": plan_path,
                "processed_data": processed_data_path,
                "preprocessing_manifest": preproc_manifest_path,
            },
        )

    except Exception as exc:
        update_stage(
            manifest_path,
            "preprocessing_planning",
            "failed",
            error=str(exc),
        )
        raise


# ---------------------------------------------------------------------------
# Plan application
# ---------------------------------------------------------------------------


def _apply_plan(
    plan: dict, df: pd.DataFrame, task_spec: dict
) -> tuple[pd.DataFrame, list]:
    """Execute all steps in *plan* on *df* and return (processed_df, steps_applied).

    Steps are sorted by their ``order`` field before execution.
    Up to _MAX_STEP_FAILURES individual step failures are tolerated; the step is
    skipped and logged. Exceeding this limit raises RuntimeError.
    """
    catalog_map = _load_transformer_catalog()
    modality = task_spec["modality"]
    target_col = task_spec["target_col"]
    excluded = _excluded_columns(plan, task_spec)

    steps = sorted(plan.get("steps", []), key=lambda s: s["order"])

    # Snapshot target column to verify it is unchanged after non-target steps.
    target_snapshot = (
        df[target_col].reset_index(drop=True).copy()
        if target_col in df.columns
        else None
    )

    steps_applied: list[dict] = []
    failure_count = 0
    current_df = df.copy()

    for step in steps:
        method = step.get("method", "")
        applies_to = step.get("applies_to", "features_only")
        skip_columns = step.get("skip_columns", [])
        parameters = step.get("parameters", {})
        order = step.get("order")

        # Guardrail 1: method must exist in catalog.
        if method not in catalog_map:
            failure_count += 1
            steps_applied.append(
                _skipped_step_record(order, method, parameters, "UNKNOWN_TRANSFORM")
            )
            if failure_count > _MAX_STEP_FAILURES:
                raise RuntimeError(
                    f"UNKNOWN_TRANSFORM: method {method!r} not in transformer catalog. "
                    f"Failures exceeded limit ({_MAX_STEP_FAILURES})."
                )
            continue

        # Guardrail 2: method must be allowed for current modality.
        allowed = catalog_map[method]["allowed_modalities"]
        if modality not in allowed:
            failure_count += 1
            steps_applied.append(
                _skipped_step_record(
                    order,
                    method,
                    parameters,
                    f"MODALITY_VIOLATION: {method!r} not allowed for modality {modality!r}",
                )
            )
            if failure_count > _MAX_STEP_FAILURES:
                raise RuntimeError(
                    f"MODALITY_VIOLATION: method {method!r} not allowed for "
                    f"modality {modality!r}. "
                    f"Failures exceeded limit ({_MAX_STEP_FAILURES})."
                )
            continue

        cols = _resolve_columns(
            applies_to, skip_columns, current_df, target_col, excluded
        )
        rows_before = len(current_df)

        try:
            current_df, fitted_params, columns_affected = _apply_transform(
                method, parameters, current_df, cols, excluded
            )
        except Exception as exc:
            failure_count += 1
            steps_applied.append(
                _skipped_step_record(order, method, parameters, str(exc))
            )
            if failure_count > _MAX_STEP_FAILURES:
                raise RuntimeError(
                    f"Step {order} ({method!r}) failed: {exc}. "
                    f"Failures exceeded limit ({_MAX_STEP_FAILURES})."
                ) from exc
            continue

        steps_applied.append(
            {
                "order": order,
                "method": method,
                "parameters_used": parameters,
                "fitted_params": fitted_params,
                "rows_before": rows_before,
                "rows_after": int(len(current_df)),
                "columns_affected": columns_affected,
            }
        )

    # Post-execution guardrail A: no NaNs may remain in feature columns.
    feature_cols = [
        c for c in current_df.columns
        if c != target_col and c not in excluded
    ]
    for col in feature_cols:
        if current_df[col].isna().any():
            if pd.api.types.is_numeric_dtype(current_df[col]):
                fill = current_df[col].mean()
                current_df[col] = current_df[col].fillna(
                    fill if pd.notna(fill) else 0.0
                )
            else:
                mode = current_df[col].mode()
                fill = mode.iloc[0] if len(mode) > 0 else "missing"
                current_df[col] = current_df[col].fillna(fill)

    # Post-execution guardrail B: target column must be unchanged unless a
    # target_only step was explicitly present in the plan.
    target_was_targeted = any(
        s.get("applies_to") == "target_only" for s in steps
    )
    if (
        not target_was_targeted
        and target_snapshot is not None
        and target_col in current_df.columns
        and len(current_df) == len(target_snapshot)
    ):
        current_target = current_df[target_col].reset_index(drop=True)
        if not current_target.equals(target_snapshot):
            raise RuntimeError(
                "TARGET_MODIFIED: target column values changed during preprocessing "
                "without an explicit target_only step."
            )

    return current_df, steps_applied


# ---------------------------------------------------------------------------
# Transform dispatcher
# ---------------------------------------------------------------------------


def _apply_transform(
    method: str,
    parameters: dict,
    df: pd.DataFrame,
    cols: list,
    excluded: set,
) -> tuple[pd.DataFrame, dict, list]:
    """Dispatch *method* and return (df, fitted_params, columns_affected)."""
    if method == "imputation":
        return _apply_imputation(df, cols, parameters)
    if method == "z_norm":
        return _apply_z_norm(df, cols)
    if method == "min_max":
        return _apply_min_max(df, cols)
    if method == "log_transform":
        return _apply_log_transform(df, cols)
    if method == "remove_outliers":
        return _apply_remove_outliers(df, cols, parameters)
    if method == "detrend":
        return _apply_detrend(df, cols, parameters)
    if method == "differencing":
        return _apply_differencing(df, cols)
    if method == "smoothing":
        return _apply_smoothing(df, cols, parameters)
    if method == "label_encode":
        return _apply_label_encode(df, cols)
    if method == "onehot_encode":
        return _apply_onehot_encode(df, cols, parameters)
    # Should never reach here after catalog validation, but guard defensively.
    raise ValueError(f"UNKNOWN_TRANSFORM: {method!r}")


# ---------------------------------------------------------------------------
# Per-method transform implementations
# ---------------------------------------------------------------------------


def _apply_imputation(
    df: pd.DataFrame, cols: list, parameters: dict
) -> tuple[pd.DataFrame, dict, list]:
    strategy = parameters.get("strategy", "mean")
    df = df.copy()
    fitted: dict = {}
    affected: list = []

    for col in cols:
        if not df[col].isna().any():
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            if strategy == "median":
                fill = df[col].median()
            elif strategy == "constant":
                fill = float(parameters.get("fill_value", 0))
            else:  # mean or most_frequent for numeric
                fill = df[col].mean()
            fill = float(fill) if pd.notna(fill) else 0.0
            df[col] = df[col].fillna(fill)
            fitted[col] = {"fill_value": fill}
        else:
            # Categorical: always use most_frequent regardless of strategy param.
            mode = df[col].mode()
            fill = str(mode.iloc[0]) if len(mode) > 0 else ""
            df[col] = df[col].fillna(fill)
            fitted[col] = {"fill_value": fill}

        affected.append(col)

    return df, fitted, affected


def _apply_z_norm(
    df: pd.DataFrame, cols: list
) -> tuple[pd.DataFrame, dict, list]:
    df = df.copy()
    fitted: dict = {}
    affected: list = []

    for col in cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        mean = float(df[col].mean())
        # ddof=0: population std, deterministic regardless of sample count.
        std = float(df[col].std(ddof=0))
        if std == 0 or np.isnan(std):
            fitted[col] = {"mean": mean, "std": 0.0, "skipped_reason": "constant column"}
            continue
        df[col] = (df[col] - mean) / std
        fitted[col] = {"mean": mean, "std": std}
        affected.append(col)

    return df, fitted, affected


def _apply_min_max(
    df: pd.DataFrame, cols: list
) -> tuple[pd.DataFrame, dict, list]:
    df = df.copy()
    fitted: dict = {}
    affected: list = []

    for col in cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        rng = max_val - min_val
        if rng == 0:
            fitted[col] = {"min": min_val, "max": max_val, "skipped_reason": "zero range"}
            continue
        df[col] = (df[col] - min_val) / rng
        fitted[col] = {"min": min_val, "max": max_val}
        affected.append(col)

    return df, fitted, affected


def _apply_log_transform(
    df: pd.DataFrame, cols: list
) -> tuple[pd.DataFrame, dict, list]:
    df = df.copy()
    affected: list = []

    for col in cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        valid = df[col].dropna()
        if len(valid) > 0 and (valid < 0).any():
            # log is undefined for negative values; skip this column silently.
            continue
        df[col] = np.log1p(df[col])
        affected.append(col)

    return df, {}, affected


def _apply_remove_outliers(
    df: pd.DataFrame, cols: list, parameters: dict
) -> tuple[pd.DataFrame, dict, list]:
    method = parameters.get("method", "iqr")
    threshold = float(parameters.get("threshold", 1.5))
    df = df.copy()

    mask = pd.Series(True, index=df.index)
    for col in cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if method == "iqr":
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            mask &= (df[col] >= lower) & (df[col] <= upper)
        elif method == "zscore":
            std = df[col].std(ddof=0)
            if std == 0:
                continue
            z = (df[col] - df[col].mean()) / std
            mask &= z.abs() <= threshold

    df = df[mask].reset_index(drop=True)
    fitted = {"method": method, "threshold": threshold}
    return df, fitted, cols


def _apply_detrend(
    df: pd.DataFrame, cols: list, parameters: dict
) -> tuple[pd.DataFrame, dict, list]:
    from scipy import signal  # optional dependency; import locally

    trend_type = parameters.get("type", "linear")
    df = df.copy()
    affected: list = []

    for col in cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        fill = df[col].mean() if pd.notna(df[col].mean()) else 0.0
        arr = df[col].fillna(fill).to_numpy(dtype=float)
        if len(arr) < 2:
            continue
        df[col] = signal.detrend(arr, type=trend_type)
        affected.append(col)

    return df, {"type": trend_type}, affected


def _apply_differencing(
    df: pd.DataFrame, cols: list
) -> tuple[pd.DataFrame, dict, list]:
    df = df.copy()
    affected: list = []

    for col in cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        df[col] = df[col].diff()
        affected.append(col)

    if affected:
        # First row becomes NaN after diff; drop it and re-index.
        df = df.iloc[1:].reset_index(drop=True)

    return df, {}, affected


def _apply_smoothing(
    df: pd.DataFrame, cols: list, parameters: dict
) -> tuple[pd.DataFrame, dict, list]:
    window = int(parameters.get("window", 3))
    df = df.copy()
    affected: list = []

    for col in cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        df[col] = df[col].rolling(window=window, min_periods=1).mean()
        affected.append(col)

    return df, {"window": window}, affected


def _apply_label_encode(
    df: pd.DataFrame, cols: list
) -> tuple[pd.DataFrame, dict, list]:
    df = df.copy()
    fitted: dict = {}
    affected: list = []

    for col in cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        fitted[col] = {
            "encoding": {
                str(cls): int(idx)
                for idx, cls in enumerate(le.classes_)
            }
        }
        affected.append(col)

    return df, fitted, affected


def _apply_onehot_encode(
    df: pd.DataFrame, cols: list, parameters: dict
) -> tuple[pd.DataFrame, dict, list]:
    max_cardinality = int(parameters.get("max_cardinality", 20))
    df = df.copy()
    fitted: dict = {}
    affected: list = []
    drop_cols: list = []

    for col in cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        if df[col].nunique() > max_cardinality:
            continue
        dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
        fitted[col] = {"categories": list(dummies.columns)}
        df = pd.concat([df, dummies], axis=1)
        drop_cols.append(col)
        affected.extend(list(dummies.columns))

    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    return df, fitted, affected


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _load_transformer_catalog() -> dict:
    """Return transformer catalog as {name: entry} dict."""
    with open(_CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)
    return {entry["name"]: entry for entry in catalog["transformers"]}


def _load_task_spec(manifest: dict) -> dict:
    """Read task_spec.json via the path stored in the manifest."""
    path = manifest["stages"]["problem_classification"]["artifacts"]["task_spec"]
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_input(manifest: dict) -> pd.DataFrame:
    """Load the original input file as declared in the manifest."""
    file_path = manifest["input"]["file_path"]
    file_format = manifest["input"]["file_format"]
    if file_format == "csv":
        return pd.read_csv(file_path)
    if file_format == "parquet":
        return pd.read_parquet(file_path)
    if file_format == "json":
        return pd.read_json(file_path)
    raise ValueError(f"Unsupported file_format: {file_format!r}")


def _excluded_columns(plan: dict, task_spec: dict) -> set:
    """Return the set of columns that must never be transformed as features.

    Always includes time_col and group_col from task_spec. Also includes any
    columns listed in ``plan["exclude_columns_from_features"]``.
    """
    excluded = set(plan.get("exclude_columns_from_features", []))
    time_col = task_spec.get("time_col")
    group_col = task_spec.get("group_col")
    if time_col:
        excluded.add(time_col)
    if group_col:
        excluded.add(group_col)
    return excluded


def _resolve_columns(
    applies_to,
    skip_columns: list,
    df: pd.DataFrame,
    target_col: str,
    excluded: set,
) -> list:
    """Return the list of columns a step should be applied to.

    Target column is always excluded unless applies_to == "target_only".
    Time and group columns are in *excluded* and are always excluded.
    """
    skip = set(skip_columns or [])
    # Target is excluded by default (not from normalization; never modified
    # unless the step explicitly targets it).
    always_exclude = excluded | {target_col}

    if applies_to == "target_only":
        return [target_col] if target_col in df.columns else []

    if applies_to == "all_numeric":
        cols = [
            c for c in df.columns
            if c not in always_exclude
            and pd.api.types.is_numeric_dtype(df[c])
        ]
    elif applies_to == "all_categorical":
        cols = [
            c for c in df.columns
            if c not in always_exclude
            and not pd.api.types.is_numeric_dtype(df[c])
        ]
    elif applies_to == "features_only":
        cols = [c for c in df.columns if c not in always_exclude]
    elif isinstance(applies_to, list):
        cols = [
            c for c in applies_to
            if c not in always_exclude and c in df.columns
        ]
    else:
        # Unrecognised value: treat as features_only (safe default).
        cols = [c for c in df.columns if c not in always_exclude]

    return [c for c in cols if c not in skip]


def _skipped_step_record(
    order, method: str, parameters: dict, reason: str
) -> dict:
    """Return a step record representing a skipped/failed step."""
    return {
        "order": order,
        "method": method,
        "parameters_used": parameters,
        "fitted_params": {},
        "rows_before": None,
        "rows_after": None,
        "columns_affected": [],
        "skipped_reason": reason,
    }


def _write_json_atomic(path: str, data: dict) -> None:
    """Write *data* as JSON to *path* atomically via temp-file + rename."""
    dir_name = os.path.dirname(path)
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
