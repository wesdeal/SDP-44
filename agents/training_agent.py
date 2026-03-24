"""agents/training_agent.py — Training Agent (Phase 7.1)

Reads selected_models.json, computes the data split once via core/splitter.py,
trains each selected model (optionally with Optuna tuning), and writes
training_results.json plus per-model artifacts under trained_models/.

Architecture authority: AGENT_ARCHITECTURE.md §4 Agent 6.
No legacy files imported. No other agent called directly.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from pathlib import Path

import pandas as pd

from core.manifest import read_manifest, update_stage
from core.splitter import split_data
from core.metric_engine import compute_all
from models.model_registry import ModelRegistry
from models.hyperparameter_tuner import HyperparameterTuner


class TrainingAgent:
    """Agent 6: Training Agent."""

    def run(self, manifest_path: str) -> None:
        """Train all selected models and write training_results.json.

        Sets stage status: running → completed | partial_failure | failed.

        Args:
            manifest_path: Path to the run's job_manifest.json.
        """
        update_stage(manifest_path, "training", "running")

        _stage_finalized = False
        try:
            manifest = read_manifest(manifest_path)
            config = manifest.get("config", {})
            run_id = manifest["run_id"]

            # --- Load prerequisite artifacts ---
            processed_data, preproc_manifest, eval_protocol, selected_models_doc = (
                _load_inputs(manifest)
            )

            feature_columns = preproc_manifest["feature_columns"]
            target_column = preproc_manifest["target_column"]

            # eval_protocol is authoritative for split parameters
            strategy = eval_protocol["split_strategy"]
            time_col = eval_protocol.get("time_col")
            group_col = eval_protocol.get("group_col")
            stratify_on = eval_protocol.get("stratify_on")
            train_fraction = eval_protocol["train_fraction"]
            val_fraction = eval_protocol["val_fraction"]
            test_fraction = eval_protocol["test_fraction"]
            minimum_test_samples = eval_protocol.get("minimum_test_samples", 30)
            primary_metric = eval_protocol.get("primary_metric", "rmse")

            random_seed = int(config.get("random_seed", 42))

            # Build DataFrame with all columns the splitter needs
            cols_needed = list(feature_columns) + [target_column]
            for extra in (time_col, group_col):
                if extra and extra in processed_data.columns and extra not in cols_needed:
                    cols_needed.append(extra)
            df = processed_data[cols_needed].copy()

            # Guardrail: verify and ensure data sorted by time_col for chronological split
            if strategy == "chronological":
                if time_col is None:
                    raise ValueError(
                        "MISSING_TIME_COL: chronological split_strategy requires time_col "
                        "but eval_protocol.time_col is null."
                    )
                if not _is_sorted(df, time_col):
                    df = df.sort_values(time_col, kind="stable").reset_index(drop=True)

            # --- Compute the split ONCE before any model training ---
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

            # Unpack shape A (standard) or shape B (time_series_cv)
            if "folds" in split_result:
                # Shape B: use last fold for train/val; frozen test tail
                last_fold = split_result["folds"][-1]
                X_train_raw = last_fold["X_train"]
                y_train = last_fold["y_train"]
                X_val_raw = last_fold["X_val"]
                y_val = last_fold["y_val"]
                X_test_raw = split_result["X_test"]
                y_test = split_result["y_test"]
            else:
                X_train_raw = split_result["X_train"]
                y_train = split_result["y_train"]
                X_val_raw = split_result["X_val"]
                y_val = split_result["y_val"]
                X_test_raw = split_result["X_test"]
                y_test = split_result["y_test"]

            # Restrict X splits to feature_columns only (exclude time_col, group_col, etc.)
            fc_present = [c for c in feature_columns if c in X_train_raw.columns]
            X_train = X_train_raw[fc_present]
            X_val = X_val_raw[fc_present]
            X_test = X_test_raw[fc_present]

            # Enforce minimum test samples — fail the stage, not per-model
            n_test = int(len(y_test))
            if n_test < minimum_test_samples:
                raise ValueError(
                    f"INSUFFICIENT_TEST_SAMPLES: test set has {n_test} rows, "
                    f"minimum_test_samples={minimum_test_samples}. Aborting training stage."
                )

            split_info = {
                "strategy": strategy,
                "n_train": int(len(y_train)),
                "n_val": int(len(y_val)),
                "n_test": n_test,
            }

            # Frozen data_splits for HyperparameterTuner (shape A contract)
            data_splits = {
                "X_train": X_train,
                "y_train": y_train,
                "X_val": X_val,
                "y_val": y_val,
                "X_test": X_test,
                "y_test": y_test,
            }

            # --- Artifact directories ---
            run_dir = os.path.dirname(os.path.abspath(manifest_path))
            artifacts_dir = os.path.join(run_dir, "artifacts")
            trained_models_dir = os.path.join(run_dir, "trained_models")
            os.makedirs(artifacts_dir, exist_ok=True)
            os.makedirs(trained_models_dir, exist_ok=True)

            tune = bool(config.get("tune_hyperparameters", False))
            n_trials = int(config.get("n_optuna_trials", 20))
            task_type = selected_models_doc.get("task_type", "")

            # --- Train each selected model; catch per-model failures ---
            model_records: list[dict] = []
            trained_model_paths: dict[str, str] = {}

            for model_info in selected_models_doc["selected_models"]:
                model_name = model_info["name"]
                tier = model_info["tier"]
                model_dir = os.path.join(trained_models_dir, model_name)
                os.makedirs(model_dir, exist_ok=True)

                record: dict = {
                    "name": model_name,
                    "tier": tier,
                    "status": "failed",
                    "hyperparameters": {},
                    "hyperparameter_source": "default",
                    "best_val_score": None,
                    "training_duration_seconds": 0.0,
                    "training_history": {},
                    "model_path": model_dir,
                    "error": None,
                }

                t_start = time.time()
                try:
                    if tune:
                        tuner = HyperparameterTuner(
                            model_name=model_name,
                            data_splits=data_splits,
                            n_trials=n_trials,
                            primary_metric=primary_metric,
                            task_type=task_type,
                        )
                        best_params = tuner.tune()
                        raw_score = tuner.best_score
                        best_val_score = (
                            float(raw_score)
                            if raw_score is not None and math.isfinite(float(raw_score))
                            else None
                        )
                        hyperparameter_source = "optuna_tuned"
                    else:
                        # Seed default params with task_type so models that
                        # require it (e.g. LinearModel) can build without tuning.
                        best_params = {"task_type": task_type} if task_type else {}
                        best_val_score = None
                        hyperparameter_source = "default"

                    model = ModelRegistry.create_model(model_name, best_params)
                    model.build()
                    training_history = model.train(X_train, y_train, X_val, y_val)

                    if not model.is_trained:
                        raise RuntimeError(
                            f"UNTRAINED_MODEL: model.is_trained is False after train() "
                            f"for {model_name!r}."
                        )

                    model.save(Path(model_dir))

                    # Compute val score when tuning was skipped
                    if best_val_score is None:
                        y_val_pred = model.predict(X_val)
                        val_metrics = compute_all(y_val, y_val_pred, [primary_metric])
                        raw = val_metrics.get(primary_metric)
                        best_val_score = (
                            float(raw)
                            if raw is not None and math.isfinite(float(raw))
                            else None
                        )

                    duration = round(time.time() - t_start, 4)
                    record.update({
                        "status": "success",
                        "hyperparameters": best_params if best_params else {},
                        "hyperparameter_source": hyperparameter_source,
                        "best_val_score": best_val_score,
                        "training_duration_seconds": duration,
                        "training_history": training_history if training_history else {},
                        "model_path": model_dir,
                        "error": None,
                    })
                    trained_model_paths[model_name] = model_dir

                except Exception as model_exc:  # noqa: BLE001
                    record["training_duration_seconds"] = round(time.time() - t_start, 4)
                    record["error"] = str(model_exc)

                model_records.append(record)

            # --- Determine stage-level status ---
            n_success = sum(1 for r in model_records if r["status"] == "success")
            total = len(model_records)
            if n_success == 0:
                stage_status = "failed"
                stage_error = f"All {total} selected models failed to train."
            elif n_success < total:
                stage_status = "partial_failure"
                stage_error = None
            else:
                stage_status = "completed"
                stage_error = None

            # --- Write training_results.json ---
            training_results = {
                "run_id": run_id,
                "models": model_records,
                "split_info": split_info,
                "feature_columns": list(fc_present),
                "target_column": target_column,
            }
            training_results_path = os.path.join(artifacts_dir, "training_results.json")
            _write_json_atomic(training_results_path, training_results)

            artifacts = {
                "training_results": training_results_path,
            }
            update_stage(
                manifest_path,
                "training",
                stage_status,
                artifacts=artifacts,
                error=stage_error,
            )
            _stage_finalized = True

            if stage_status == "failed":
                raise RuntimeError(stage_error)

        except Exception as exc:
            if not _stage_finalized:
                update_stage(manifest_path, "training", "failed", error=str(exc))
            raise


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _load_inputs(manifest: dict) -> tuple:
    """Load all prerequisite artifacts for the training stage.

    Returns:
        (processed_data_df, preproc_manifest, eval_protocol, selected_models)
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

    selected_models_path = stages["model_selection"]["artifacts"]["selected_models"]
    if not os.path.exists(selected_models_path):
        raise FileNotFoundError(
            f"MISSING_SELECTED_MODELS: {selected_models_path!r}. "
            "Ensure ModelSelectionAgent completed successfully."
        )
    with open(selected_models_path, encoding="utf-8") as f:
        selected_models = json.load(f)

    return processed_data, preproc_manifest, eval_protocol, selected_models


def _is_sorted(df: pd.DataFrame, col: str) -> bool:
    """Return True if df[col] is non-decreasing."""
    if len(df) <= 1:
        return True
    try:
        values = df[col].to_numpy()
        return bool((values[:-1] <= values[1:]).all())
    except Exception:  # noqa: BLE001
        return False


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
