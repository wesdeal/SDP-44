"""orchestrator.py — Pipeline orchestrator.

Public API
----------
run_pipeline(input_file: str, config: dict) -> str
run_pipeline_async(input_file: str, config: dict) -> str

This is the sole entry point for pipeline execution (IMPLEMENTATION_RULES.md §3).
All stage coordination happens here; agents are never called from each other.
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor

from core.manifest import initialize_manifest, read_manifest

from agents.ingestion_agent import IngestionAgent
from agents.problem_classification_agent import ProblemClassificationAgent
from agents.preprocessing_planning_agent import PreprocessingPlanningAgent
from agents.evaluation_protocol_agent import EvaluationProtocolAgent
from agents.model_selection_agent import ModelSelectionAgent
from agents.training_agent import TrainingAgent
from agents.evaluation_agent import EvaluationAgent
from agents.artifact_assembly_agent import ArtifactAssemblyAgent


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

    # 2. Stage 1: Ingestion
    IngestionAgent().run(manifest_path)
    _require_completed(manifest_path, "ingestion")

    # 3. Stage 2: Problem Classification
    ProblemClassificationAgent().run(manifest_path)
    _require_completed(manifest_path, "problem_classification")

    # 4. Stages 3 & 4: Preprocessing Planning + Evaluation Protocol (parallel).
    #    Both agents must reach "completed" before model_selection may start.
    _run_parallel_pair(manifest_path)
    _require_completed(manifest_path, "preprocessing_planning")
    _require_completed(manifest_path, "evaluation_protocol")

    # 5. Stage 5: Model Selection
    ModelSelectionAgent().run(manifest_path)
    _require_completed(manifest_path, "model_selection")

    # 6. Stage 6: Training.
    #    partial_failure (some models failed) does not raise — evaluation still runs.
    #    A hard failure (all models failed) raises — skip evaluation, run assembly.
    training_failed = False
    try:
        TrainingAgent().run(manifest_path)
    except Exception:
        training_failed = True

    # 7. Stage 7: Evaluation.
    #    Skipped only when training completely failed.
    #    Any evaluation failure is non-fatal: artifact_assembly assembles whatever exists.
    if not training_failed:
        try:
            EvaluationAgent().run(manifest_path)
        except Exception:
            pass

    # 8. Stage 8: Artifact Assembly — always runs regardless of upstream failures.
    ArtifactAssemblyAgent().run(manifest_path)

    final_manifest = read_manifest(manifest_path)
    return final_manifest["stages"]["artifact_assembly"]["artifacts"]["dashboard_bundle"]


# ---------------------------------------------------------------------------
# Async public API
# ---------------------------------------------------------------------------

def run_pipeline_async(input_file: str, config: dict) -> str:
    """Initialize the manifest and start the pipeline in a background thread.

    Returns the run_id immediately. All pipeline stages run asynchronously in a
    daemon thread. Callers can poll the manifest via /api/runs/{run_id}/status
    to track progress.

    Args:
        input_file: Path to the raw input dataset (CSV / Parquet / JSON).
        config: Run-level configuration dict (same as run_pipeline).

    Returns:
        The run_id UUID string.
    """
    manifest = initialize_manifest(input_file, config)
    run_id = manifest["run_id"]
    runs_dir = config.get("runs_dir", "runs")
    manifest_path = os.path.join(runs_dir, run_id, "job_manifest.json")

    t = threading.Thread(
        target=_run_from_manifest,
        args=(manifest_path,),
        daemon=True,
        name=f"pipeline-{run_id[:8]}",
    )
    t.start()
    return run_id


def _run_from_manifest(manifest_path: str) -> None:
    """Execute all pipeline stages for an already-initialized manifest.

    Mirrors the body of run_pipeline() but skips initialize_manifest since
    the manifest already exists. Used by run_pipeline_async().
    """
    try:
        IngestionAgent().run(manifest_path)
        _require_completed(manifest_path, "ingestion")

        ProblemClassificationAgent().run(manifest_path)
        _require_completed(manifest_path, "problem_classification")

        _run_parallel_pair(manifest_path)
        _require_completed(manifest_path, "preprocessing_planning")
        _require_completed(manifest_path, "evaluation_protocol")

        ModelSelectionAgent().run(manifest_path)
        _require_completed(manifest_path, "model_selection")

        training_failed = False
        try:
            TrainingAgent().run(manifest_path)
        except Exception:
            training_failed = True

        if not training_failed:
            try:
                EvaluationAgent().run(manifest_path)
            except Exception:
                pass

        ArtifactAssemblyAgent().run(manifest_path)
    except Exception:
        import traceback
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_completed(manifest_path: str, stage_name: str) -> None:
    """Raise RuntimeError if the named stage did not reach 'completed' status.

    Enforces the dependency gate: a downstream stage must not start until all
    required upstream stages are confirmed completed in the manifest.
    """
    manifest = read_manifest(manifest_path)
    status = manifest["stages"][stage_name]["status"]
    if status != "completed":
        error = manifest["stages"][stage_name].get("error") or ""
        raise RuntimeError(
            f"Stage '{stage_name}' did not complete (status={status!r}). {error}".strip()
        )


def _run_parallel_pair(manifest_path: str) -> None:
    """Run preprocessing_planning and evaluation_protocol concurrently.

    Both agents manage their own stage status transitions. Agent-level exceptions
    are captured so both stages are allowed to finish before failure is surfaced.
    If either agent fails its exception is re-raised after both futures settle.
    """
    exc_results: dict = {
        "preprocessing_planning": None,
        "evaluation_protocol": None,
    }

    def _run(agent, key: str) -> None:
        try:
            agent.run(manifest_path)
        except Exception as exc:
            exc_results[key] = exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        f3 = executor.submit(_run, PreprocessingPlanningAgent(), "preprocessing_planning")
        f4 = executor.submit(_run, EvaluationProtocolAgent(), "evaluation_protocol")
        f3.result()  # propagate scheduling-level errors from the executor itself
        f4.result()

    # Surface agent-level failures after both stages have settled.
    for exc in exc_results.values():
        if exc is not None:
            raise exc
