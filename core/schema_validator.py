import json

_REQUIRED_FIELDS = {
    "dataset_profile": {
        "run_id", "num_rows", "num_columns", "columns",
    },
    "task_spec": {
        "run_id", "task_type", "target_col",
    },
    "eval_protocol": {
        "run_id", "task_type", "split_strategy", "train_fraction", "val_fraction",
        "test_fraction", "shuffle", "stratify_on", "group_col", "time_col",
        "cv", "prediction_type", "quantiles", "primary_metric", "metrics",
        "metrics_for_this_run", "minimum_test_samples",
    },
    "preprocessing_manifest": {
        "run_id", "steps_applied", "final_shape", "feature_columns",
        "target_column", "time_column",
    },
    "preprocessing_plan": {
        "run_id", "steps", "preserve_temporal_order",
        "exclude_columns_from_features", "plan_source",
    },
    "selected_models": {
        "run_id", "task_type", "selected_models",
        "selection_strategy", "models_considered", "models_rejected",
    },
    "training_results": {
        "run_id", "models", "split_info", "feature_columns", "target_column",
    },
    "evaluation_report": {
        "run_id", "primary_metric", "test_split_size", "models", "evaluated_at",
    },
    "comparison_table": {
        "run_id", "ranked_by", "ranking",
    },
    "leaderboard": {
        "run_id", "dataset_name", "task_type", "primary_metric",
        "higher_is_better", "models", "generated_at",
    },
    "run_summary": {
        "run_id", "assembled_at", "stages",
    },
}


def validate(artifact_path: str, schema_name: str) -> bool:
    if schema_name not in _REQUIRED_FIELDS:
        raise ValueError(f"Unknown schema_name: {schema_name!r}")

    with open(artifact_path, "r") as f:
        data = json.load(f)

    return _REQUIRED_FIELDS[schema_name].issubset(data.keys())
