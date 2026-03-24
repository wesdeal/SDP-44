"""agents/evaluation_agent.py — Evaluation & Comparison Agent (Phase 7.2)

Loads each successfully-trained model, re-derives the test split deterministically
from eval_protocol.json + processed_data.csv + preprocessing_manifest.json,
computes protocol-defined metrics via core/metric_engine, generates diagnostic
plots, and writes evaluation_report.json, comparison_table.json, and plots/.

Architecture authority: AGENT_ARCHITECTURE.md §4 Agent 7.
No legacy files imported. No other agent called directly.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from core.manifest import read_manifest, update_stage
from core.splitter import split_data
from core.metric_engine import compute_all
from models.model_registry import ModelRegistry


class EvaluationAgent:
    """Agent 7: Evaluation & Comparison Agent."""

    def run(self, manifest_path: str) -> None:
        """Evaluate all successfully-trained models and write evaluation artifacts.

        Sets stage status: running → completed | failed.

        Args:
            manifest_path: Path to the run's job_manifest.json.
        """
        update_stage(manifest_path, "evaluation", "running")

        _stage_finalized = False
        try:
            manifest = read_manifest(manifest_path)
            config = manifest.get("config", {})
            run_id = manifest["run_id"]

            # --- Load prerequisite artifacts ---
            processed_data, preproc_manifest, eval_protocol, training_results = (
                _load_inputs(manifest)
            )

            feature_columns = preproc_manifest["feature_columns"]
            target_column = preproc_manifest["target_column"]
            metrics_for_this_run = eval_protocol["metrics_for_this_run"]
            primary_metric = eval_protocol["primary_metric"]
            task_type = eval_protocol["task_type"]

            random_seed = int(config.get("random_seed", 42))

            # --- Re-derive test split deterministically ---
            X_test, y_test = _derive_test_split(
                processed_data=processed_data,
                eval_protocol=eval_protocol,
                feature_columns=feature_columns,
                target_column=target_column,
                random_seed=random_seed,
            )

            actual_n_test = len(y_test)
            y_true_arr = np.asarray(y_test)

            # Validate row count matches training stage's recorded n_test
            expected_n_test = training_results.get("split_info", {}).get("n_test")
            if expected_n_test is not None and actual_n_test != expected_n_test:
                raise ValueError(
                    f"TEST_SET_MISMATCH: reconstructed test set has {actual_n_test} rows "
                    f"but training stage recorded n_test={expected_n_test}. "
                    "Ensure processed_data.csv and eval_protocol.json are unchanged."
                )

            # Validate no target leakage
            if target_column in X_test.columns:
                raise RuntimeError(
                    f"TARGET_LEAKAGE: target column {target_column!r} found in X_test."
                )

            # --- Artifact directories ---
            run_dir = os.path.dirname(os.path.abspath(manifest_path))
            artifacts_dir = os.path.join(run_dir, "artifacts")
            plots_dir = os.path.join(run_dir, "plots")
            os.makedirs(artifacts_dir, exist_ok=True)
            os.makedirs(plots_dir, exist_ok=True)

            # --- Evaluate each model ---
            model_records: list[dict] = []

            for model_entry in training_results["models"]:
                model_name = model_entry["name"]
                tier = model_entry["tier"]

                # Skip models that failed training — do not invent results
                if model_entry.get("status") != "success":
                    model_records.append({
                        "name": model_name,
                        "tier": tier,
                        "status": "failed",
                        "metrics": {m: None for m in metrics_for_this_run},
                        "n_test_samples": actual_n_test,
                        "inference_time_ms": None,
                        "plot_path": None,
                        "error": (
                            f"Skipped: training status was "
                            f"{model_entry.get('status', 'unknown')!r}"
                        ),
                    })
                    continue

                model_path = model_entry.get("model_path")
                record: dict = {
                    "name": model_name,
                    "tier": tier,
                    "status": "failed",
                    "metrics": {m: None for m in metrics_for_this_run},
                    "n_test_samples": actual_n_test,
                    "inference_time_ms": None,
                    "plot_path": None,
                    "error": None,
                }

                try:
                    # Load trained model artifact
                    model = ModelRegistry.create_model(model_name, {})
                    model.load(Path(model_path))

                    if not model.is_trained:
                        raise RuntimeError(
                            f"LOAD_FAILED: model.is_trained is False after load() "
                            f"for {model_name!r}."
                        )

                    # Inference on the shared test set
                    t_infer = time.time()
                    y_pred = model.predict(X_test)
                    inference_ms = (time.time() - t_infer) * 1000.0

                    y_pred_arr = np.asarray(y_pred)

                    # Compute metrics via metric_engine (never reimplemented here)
                    raw_metrics = compute_all(y_true_arr, y_pred_arr, metrics_for_this_run)

                    # Sanitize: replace inf/nan with None
                    sanitized: dict = {}
                    for m_name, val in raw_metrics.items():
                        if val is None:
                            sanitized[m_name] = None
                        elif isinstance(val, float) and not math.isfinite(val):
                            sanitized[m_name] = None
                        else:
                            sanitized[m_name] = val

                    # Generate diagnostic plot (non-fatal on failure)
                    plot_path = _generate_plot(
                        model_name=model_name,
                        y_true=y_true_arr,
                        y_pred=y_pred_arr,
                        task_type=task_type,
                        plots_dir=plots_dir,
                    )

                    record.update({
                        "status": "evaluated",
                        "metrics": sanitized,
                        "inference_time_ms": round(inference_ms, 3),
                        "plot_path": plot_path,
                        "error": None,
                    })

                except Exception as model_exc:  # noqa: BLE001
                    record["error"] = str(model_exc)

                model_records.append(record)

            # --- Stage-level status ---
            n_evaluated = sum(1 for r in model_records if r["status"] == "evaluated")

            if n_evaluated < 2:
                stage_status = "failed"
                stage_error = (
                    f"Only {n_evaluated} model(s) evaluated successfully; "
                    "at least 2 required (AGENT_ARCHITECTURE.md §4 Agent 7)."
                )
            else:
                stage_status = "completed"
                stage_error = None

            # --- Write artifacts (always, so assembly can use partial results) ---
            eval_report = {
                "run_id": run_id,
                "primary_metric": primary_metric,
                "test_split_size": actual_n_test,
                "models": model_records,
                "evaluated_at": _now_iso(),
            }

            comparison_table = _build_comparison_table(
                run_id=run_id,
                model_records=model_records,
                primary_metric=primary_metric,
                eval_protocol=eval_protocol,
            )

            eval_report_path = os.path.join(artifacts_dir, "evaluation_report.json")
            comparison_table_path = os.path.join(artifacts_dir, "comparison_table.json")
            _write_json_atomic(eval_report_path, eval_report)
            _write_json_atomic(comparison_table_path, comparison_table)

            artifacts = {
                "evaluation_report": eval_report_path,
                "comparison_table": comparison_table_path,
                "plots_dir": plots_dir,
            }
            update_stage(
                manifest_path,
                "evaluation",
                stage_status,
                artifacts=artifacts,
                error=stage_error,
            )
            _stage_finalized = True

            if stage_status == "failed":
                raise RuntimeError(stage_error)

        except Exception as exc:
            if not _stage_finalized:
                update_stage(manifest_path, "evaluation", "failed", error=str(exc))
            raise


# ---------------------------------------------------------------------------
# Test-split reconstruction
# ---------------------------------------------------------------------------


def _derive_test_split(
    processed_data: pd.DataFrame,
    eval_protocol: dict,
    feature_columns: list,
    target_column: str,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Re-derive X_test, y_test deterministically using the same logic as TrainingAgent.

    Mirrors training_agent.py split logic exactly so the test set is identical.
    """
    strategy = eval_protocol["split_strategy"]
    time_col = eval_protocol.get("time_col")
    group_col = eval_protocol.get("group_col")
    stratify_on = eval_protocol.get("stratify_on")
    train_fraction = eval_protocol["train_fraction"]
    val_fraction = eval_protocol["val_fraction"]
    test_fraction = eval_protocol["test_fraction"]

    # Build the same df as TrainingAgent (same column selection)
    cols_needed = list(feature_columns) + [target_column]
    for extra in (time_col, group_col):
        if extra and extra in processed_data.columns and extra not in cols_needed:
            cols_needed.append(extra)
    df = processed_data[cols_needed].copy()

    # Guardrail: verify and sort for chronological split
    if strategy == "chronological":
        if time_col is None:
            raise ValueError(
                "MISSING_TIME_COL: chronological split_strategy requires time_col "
                "but eval_protocol.time_col is null."
            )
        if not _is_sorted(df, time_col):
            df = df.sort_values(time_col, kind="stable").reset_index(drop=True)

    split_result = split_data(
        df=df,
        target_col=target_column,
        strategy=strategy,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        random_seed=random_seed,
        stratify_on=stratify_on,
        group_col=group_col,
        time_col=time_col,
    )

    # Shape A (standard) or shape B (time_series_cv) — test set is the same key
    X_test_raw = split_result["X_test"]
    y_test = split_result["y_test"]

    # Restrict X to feature_columns only (exclude time_col, group_col, etc.)
    fc_present = [c for c in feature_columns if c in X_test_raw.columns]
    X_test = X_test_raw[fc_present]

    return X_test, y_test


# ---------------------------------------------------------------------------
# Comparison table construction
# ---------------------------------------------------------------------------


def _build_comparison_table(
    run_id: str,
    model_records: list[dict],
    primary_metric: str,
    eval_protocol: dict,
) -> dict:
    """Build comparison_table.json from evaluated model records."""
    higher_is_better = _metric_higher_is_better(eval_protocol, primary_metric)

    # Only rank models that were evaluated and have a valid primary metric value
    rankable = [
        r for r in model_records
        if r["status"] == "evaluated"
        and r["metrics"].get(primary_metric) is not None
    ]

    # Sort: ascending for lower-is-better metrics, descending for higher-is-better
    rankable_sorted = sorted(
        rankable,
        key=lambda r: r["metrics"][primary_metric],
        reverse=higher_is_better,
    )

    ranking = []
    for rank_idx, rec in enumerate(rankable_sorted, start=1):
        ranking.append({
            "rank": rank_idx,
            "model_name": rec["name"],
            "tier": rec["tier"],
            "primary_metric_value": rec["metrics"][primary_metric],
            "is_best": rank_idx == 1,
            "all_metrics": rec["metrics"],
        })

    return {
        "run_id": run_id,
        "ranked_by": primary_metric,
        "ranking": ranking,
    }


def _metric_higher_is_better(eval_protocol: dict, metric_name: str) -> bool:
    """Look up higher_is_better for metric_name from eval_protocol.metrics list."""
    for m in eval_protocol.get("metrics", []):
        if m["name"] == metric_name:
            return bool(m.get("higher_is_better", False))
    return False


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------


def _generate_plot(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str,
    plots_dir: str,
) -> Optional[str]:
    """Generate a diagnostic plot PNG for one model.

    Returns the absolute plot path on success, None on any failure.
    Plot generation failures are non-fatal per the architecture.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_path = os.path.join(plots_dir, f"{model_name}_eval.png")

        if task_type in ("tabular_regression", "time_series_forecasting", "grouped_prediction"):
            _plot_regression(plt, model_name, y_true, y_pred, task_type, plot_path)
        elif task_type == "tabular_classification":
            _plot_classification(plt, model_name, y_true, y_pred, plot_path)
        else:
            # Unknown task type — generate a basic scatter as fallback
            _plot_regression(plt, model_name, y_true, y_pred, task_type, plot_path)

        return plot_path

    except Exception:  # noqa: BLE001
        return None


def _plot_regression(
    plt,
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str,
    plot_path: str,
) -> None:
    """Actual vs Predicted scatter + residual distribution [+ prediction trace for TS]."""
    is_ts = task_type == "time_series_forecasting"
    n_cols = 3 if is_ts else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))

    # Panel 1: Actual vs Predicted scatter
    axes[0].scatter(y_true, y_pred, alpha=0.4, s=10)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    axes[0].plot([lo, hi], [lo, hi], "r--", linewidth=1)
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predicted")
    axes[0].set_title(f"{model_name} — Actual vs Predicted")

    # Panel 2: Residual distribution histogram
    residuals = y_true - y_pred
    axes[1].hist(residuals, bins=min(30, max(5, len(residuals) // 10)))
    axes[1].axvline(0, color="r", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")

    # Panel 3 (TS only): Prediction trace — first 100 time steps
    if is_ts:
        n_trace = min(100, len(y_true))
        axes[2].plot(range(n_trace), y_true[:n_trace], label="Actual", linewidth=1)
        axes[2].plot(range(n_trace), y_pred[:n_trace], label="Predicted",
                     linewidth=1, linestyle="--")
        axes[2].set_xlabel("Step")
        axes[2].set_ylabel("Value")
        axes[2].set_title(f"Prediction Trace (first {n_trace} steps)")
        axes[2].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(plot_path, dpi=80)
    plt.close(fig)


def _plot_classification(
    plt,
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    plot_path: str,
) -> None:
    """Confusion matrix [+ ROC curve for binary classification]."""
    from sklearn.metrics import confusion_matrix

    classes = np.unique(y_true)
    is_binary = len(classes) == 2
    n_cols = 2 if is_binary else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    # Panel 1: Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    im = axes[0].imshow(cm, interpolation="nearest", aspect="auto")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    axes[0].set_title(f"{model_name} — Confusion Matrix")
    fig.colorbar(im, ax=axes[0])

    # Panel 2 (binary only): ROC curve
    if is_binary:
        try:
            from sklearn.metrics import roc_curve, roc_auc_score
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            auc_val = roc_auc_score(y_true, y_pred)
            axes[1].plot(fpr, tpr, label=f"AUC={auc_val:.3f}")
            axes[1].plot([0, 1], [0, 1], "k--", linewidth=1)
            axes[1].set_xlabel("False Positive Rate")
            axes[1].set_ylabel("True Positive Rate")
            axes[1].set_title("ROC Curve")
            axes[1].legend(fontsize=8)
        except Exception:  # noqa: BLE001
            axes[1].set_visible(False)

    plt.tight_layout()
    fig.savefig(plot_path, dpi=80)
    plt.close(fig)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _load_inputs(manifest: dict) -> tuple:
    """Load all prerequisite artifacts for the evaluation stage.

    Returns:
        (processed_data_df, preproc_manifest, eval_protocol, training_results)
    """
    stages = manifest["stages"]

    processed_data_path = stages["preprocessing_planning"]["artifacts"]["processed_data"]
    if not os.path.exists(processed_data_path):
        raise FileNotFoundError(
            f"MISSING_PROCESSED_DATA: {processed_data_path!r}. "
            "Ensure PreprocessingPlanningAgent completed successfully."
        )
    processed_data = pd.read_csv(processed_data_path)

    preproc_manifest_path = stages["preprocessing_planning"]["artifacts"][
        "preprocessing_manifest"
    ]
    if not os.path.exists(preproc_manifest_path):
        raise FileNotFoundError(
            f"MISSING_PREPROC_MANIFEST: {preproc_manifest_path!r}."
        )
    with open(preproc_manifest_path, encoding="utf-8") as f:
        preproc_manifest = json.load(f)

    eval_protocol_path = stages["evaluation_protocol"]["artifacts"]["eval_protocol"]
    if not os.path.exists(eval_protocol_path):
        raise FileNotFoundError(
            f"MISSING_EVAL_PROTOCOL: {eval_protocol_path!r}. "
            "Ensure EvaluationProtocolAgent completed successfully."
        )
    with open(eval_protocol_path, encoding="utf-8") as f:
        eval_protocol = json.load(f)

    training_results_path = stages["training"]["artifacts"]["training_results"]
    if not os.path.exists(training_results_path):
        raise FileNotFoundError(
            f"MISSING_TRAINING_RESULTS: {training_results_path!r}. "
            "Ensure TrainingAgent completed successfully."
        )
    with open(training_results_path, encoding="utf-8") as f:
        training_results = json.load(f)

    return processed_data, preproc_manifest, eval_protocol, training_results


def _is_sorted(df: pd.DataFrame, col: str) -> bool:
    """Return True if df[col] is non-decreasing."""
    if len(df) <= 1:
        return True
    try:
        values = df[col].to_numpy()
        return bool((values[:-1] <= values[1:]).all())
    except Exception:  # noqa: BLE001
        return False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: str, data: dict) -> None:
    """Write *data* as JSON to *path* atomically (temp-file + rename)."""
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
