"""tests/integration/test_skeleton_pipeline.py

Phase 1.3 — Smoke test for the skeleton orchestrator.

Verifies that run_pipeline():
  - creates job_manifest.json
  - advances every stage to "completed"
  - writes all stage artifacts to disk
  - creates the dashboard/ directory

No mocking.  An isolated temporary directory is used as runs_dir so the
test is self-contained and run_id is never hardcoded.
"""

import json
import os
import tempfile

import pytest

import orchestrator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Canonical stage order from AGENT_ARCHITECTURE.md §5
_STAGE_NAMES = [
    "ingestion",
    "problem_classification",
    "preprocessing_planning",
    "evaluation_protocol",
    "model_selection",
    "training",
    "evaluation",
    "artifact_assembly",
]

# Path to the sample dataset required by this integration test
_INPUT_FILE = os.path.join(
    os.path.dirname(__file__), "..", "..", "inputs", "ETTh1.csv"
)


# ---------------------------------------------------------------------------
# Fixture — isolated run workspace
# ---------------------------------------------------------------------------

@pytest.fixture()
def pipeline_result():
    """Run the skeleton pipeline once and yield a result bundle.

    Yields:
        dict with keys:
            dashboard_bundle  – return value of run_pipeline()
            run_dir           – runs/{run_id}/ directory
            manifest_path     – path to job_manifest.json
            manifest          – parsed manifest dict
    """
    with tempfile.TemporaryDirectory() as tmp_runs_dir:
        config = {"runs_dir": tmp_runs_dir, "random_seed": 42}
        dashboard_bundle = orchestrator.run_pipeline(_INPUT_FILE, config)

        # run_dir is the parent of dashboard/
        run_dir = os.path.dirname(os.path.abspath(dashboard_bundle))
        manifest_path = os.path.join(run_dir, "job_manifest.json")

        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)

        yield {
            "dashboard_bundle": dashboard_bundle,
            "run_dir": run_dir,
            "manifest_path": manifest_path,
            "manifest": manifest,
        }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSkeletonPipeline:
    """Smoke tests for the Phase-1.2 skeleton orchestrator."""

    def test_manifest_file_exists(self, pipeline_result):
        """job_manifest.json must be present on disk."""
        assert os.path.isfile(pipeline_result["manifest_path"]), (
            f"job_manifest.json not found at {pipeline_result['manifest_path']}"
        )

    def test_manifest_has_run_id(self, pipeline_result):
        """Manifest must carry a non-empty run_id."""
        run_id = pipeline_result["manifest"]["run_id"]
        assert isinstance(run_id, str) and run_id, "run_id is missing or empty"

    def test_all_stages_completed(self, pipeline_result):
        """Every canonical stage must reach status 'completed'."""
        stages = pipeline_result["manifest"]["stages"]
        for stage_name in _STAGE_NAMES:
            assert stage_name in stages, f"Stage '{stage_name}' missing from manifest"
            actual = stages[stage_name]["status"]
            assert actual == "completed", (
                f"Stage '{stage_name}' has status '{actual}', expected 'completed'"
            )

    def test_stage_artifacts_exist_on_disk(self, pipeline_result):
        """Every artifact path recorded in the manifest must exist on disk."""
        stages = pipeline_result["manifest"]["stages"]
        for stage_name in _STAGE_NAMES:
            artifacts = stages[stage_name].get("artifacts", {})
            for key, path in artifacts.items():
                assert os.path.exists(path), (
                    f"Stage '{stage_name}' artifact '{key}' not found at '{path}'"
                )

    def test_dashboard_directory_exists(self, pipeline_result):
        """dashboard/ directory must exist and be a directory."""
        db = pipeline_result["dashboard_bundle"]
        assert os.path.isdir(db), f"dashboard/ not a directory: {db}"

    def test_return_value_matches_manifest(self, pipeline_result):
        """run_pipeline() return value must match the artifact recorded in the manifest."""
        recorded = pipeline_result["manifest"]["stages"]["artifact_assembly"][
            "artifacts"
        ]["dashboard_bundle"]
        assert pipeline_result["dashboard_bundle"] == recorded, (
            "Return value of run_pipeline() does not match manifest artifact"
        )
