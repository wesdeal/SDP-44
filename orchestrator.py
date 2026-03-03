"""orchestrator.py — Skeleton pipeline orchestrator.

Public API
----------
run_pipeline(input_file: str, config: dict) -> str

This is the sole entry point for pipeline execution (IMPLEMENTATION_RULES.md §3).
All stage coordination, manifest management, and artifact creation happen here.

Phase 1.2 — stub stages only. No real agent imports or logic.
Real agent calls replace stubs in Phase 7.4.
"""

import json
import os

from core.manifest import initialize_manifest, read_manifest, update_stage


# ---------------------------------------------------------------------------
# Internal path helpers (all paths derived from manifest_path, never hardcoded)
# ---------------------------------------------------------------------------

def _run_dir(manifest_path: str) -> str:
    """Return the run directory (parent of job_manifest.json)."""
    return os.path.dirname(os.path.abspath(manifest_path))


def _artifacts_dir(manifest_path: str) -> str:
    """Return the artifacts/ subdirectory for this run."""
    return os.path.join(_run_dir(manifest_path), "artifacts")


def _write_empty_json(path: str) -> None:
    """Write an empty JSON object to *path*, creating parent dirs as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({}, f)


def _write_empty_file(path: str) -> None:
    """Write an empty file at *path*, creating parent dirs as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").close()


# ---------------------------------------------------------------------------
# Stub stages — inline placeholders, no real agent imports
# Each stub writes placeholder artifacts and returns the artifacts dict.
# ---------------------------------------------------------------------------

def _stub_ingestion(manifest_path: str) -> dict:
    arts = _artifacts_dir(manifest_path)
    dataset_profile = os.path.join(arts, "dataset_profile.json")
    _write_empty_json(dataset_profile)
    return {"dataset_profile": dataset_profile}


def _stub_problem_classification(manifest_path: str) -> dict:
    arts = _artifacts_dir(manifest_path)
    task_spec = os.path.join(arts, "task_spec.json")
    _write_empty_json(task_spec)
    return {"task_spec": task_spec}


def _stub_preprocessing_planning(manifest_path: str) -> dict:
    arts = _artifacts_dir(manifest_path)
    preprocessing_plan = os.path.join(arts, "preprocessing_plan.json")
    processed_data = os.path.join(arts, "processed_data.csv")
    preprocessing_manifest = os.path.join(arts, "preprocessing_manifest.json")
    _write_empty_json(preprocessing_plan)
    _write_empty_file(processed_data)
    _write_empty_json(preprocessing_manifest)
    return {
        "preprocessing_plan": preprocessing_plan,
        "processed_data": processed_data,
        "preprocessing_manifest": preprocessing_manifest,
    }


def _stub_evaluation_protocol(manifest_path: str) -> dict:
    arts = _artifacts_dir(manifest_path)
    eval_protocol = os.path.join(arts, "eval_protocol.json")
    _write_empty_json(eval_protocol)
    return {"eval_protocol": eval_protocol}


def _stub_model_selection(manifest_path: str) -> dict:
    arts = _artifacts_dir(manifest_path)
    selected_models = os.path.join(arts, "selected_models.json")
    _write_empty_json(selected_models)
    return {"selected_models": selected_models}


def _stub_training(manifest_path: str) -> dict:
    arts = _artifacts_dir(manifest_path)
    training_results = os.path.join(arts, "training_results.json")
    trained_models = os.path.join(_run_dir(manifest_path), "trained_models")
    _write_empty_json(training_results)
    os.makedirs(trained_models, exist_ok=True)
    return {
        "training_results": training_results,
        "trained_model_paths": trained_models,
    }


def _stub_evaluation(manifest_path: str) -> dict:
    arts = _artifacts_dir(manifest_path)
    evaluation_report = os.path.join(arts, "evaluation_report.json")
    comparison_table = os.path.join(arts, "comparison_table.json")
    plots_dir = os.path.join(_run_dir(manifest_path), "plots")
    _write_empty_json(evaluation_report)
    _write_empty_json(comparison_table)
    os.makedirs(plots_dir, exist_ok=True)
    return {
        "evaluation_report": evaluation_report,
        "comparison_table": comparison_table,
        "plots_dir": plots_dir,
    }


def _stub_artifact_assembly(manifest_path: str) -> dict:
    dashboard_dir = os.path.join(_run_dir(manifest_path), "dashboard")
    os.makedirs(dashboard_dir, exist_ok=True)
    return {"dashboard_bundle": dashboard_dir}


# ---------------------------------------------------------------------------
# Stage registry — canonical order (AGENT_ARCHITECTURE.md §5)
# ---------------------------------------------------------------------------

_STAGES = [
    ("ingestion",               _stub_ingestion),
    ("problem_classification",  _stub_problem_classification),
    ("preprocessing_planning",  _stub_preprocessing_planning),
    ("evaluation_protocol",     _stub_evaluation_protocol),
    ("model_selection",         _stub_model_selection),
    ("training",                _stub_training),
    ("evaluation",              _stub_evaluation),
    ("artifact_assembly",       _stub_artifact_assembly),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pipeline(input_file: str, config: dict) -> str:
    """Execute the full pipeline and return the path to the dashboard bundle.

    Args:
        input_file: Path to the raw input dataset (CSV / Parquet / JSON).
        config: Run-level configuration dict.  ``"runs_dir"`` controls where
            run workspaces are created; defaults to ``"runs"``.

    Returns:
        Path to the completed ``dashboard/`` directory.
    """
    # 1. Initialize manifest — generates run_id, writes job_manifest.json.
    manifest = initialize_manifest(input_file, config)
    run_id = manifest["run_id"]

    runs_dir = config.get("runs_dir", "runs")
    manifest_path = os.path.join(runs_dir, run_id, "job_manifest.json")

    # 2. Execute each stub stage in canonical order: pending → running → completed.
    for stage_name, stub_fn in _STAGES:
        update_stage(manifest_path, stage_name, "running")
        artifacts = stub_fn(manifest_path)
        update_stage(manifest_path, stage_name, "completed", artifacts=artifacts)

    # 3. Return the dashboard bundle path recorded by artifact_assembly.
    final_manifest = read_manifest(manifest_path)
    return final_manifest["stages"]["artifact_assembly"]["artifacts"]["dashboard_bundle"]
