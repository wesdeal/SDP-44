"""tests/integration/test_full_pipeline_timeseries.py — Phase 7.5 full-pipeline
integration test for time-series forecasting.

Uses ETTh1.csv (inputs/ETTh1.csv) as the input dataset.

Verifies:
1. Run completes to the expected end state.
2. All canonical artifacts exist on disk.
3. Manifest stage statuses are architecture-consistent.
4. Outputs are coherent with the time_series_forecasting task type.
5. Two runs with the same seed produce identical metric values.

No mocking. No legacy imports. No direct agent invocation.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

import orchestrator

# ---------------------------------------------------------------------------
# Canonical stage names (AGENT_ARCHITECTURE.md §5)
# ---------------------------------------------------------------------------

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

_SEED = 42

# Path to the ETTh1 dataset (TEAM_IMPLEMENTATION_PLAN.md §7.5 requirement)
_ETTH1_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "inputs", "ETTh1.csv"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_dir_from_dashboard(dashboard_path: str) -> str:
    return os.path.dirname(os.path.abspath(dashboard_path))


def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def timeseries_run():
    """Run the full pipeline once on ETTh1.csv.

    Yields a dict with:
        dashboard       – return value of run_pipeline()
        run_dir         – runs/{run_id}/ directory
        manifest_path   – path to job_manifest.json
        manifest        – parsed manifest dict
        artifacts_dir   – runs/{run_id}/artifacts/ directory
    """
    if not os.path.isfile(_ETTH1_PATH):
        pytest.skip(f"ETTh1.csv not found at {_ETTH1_PATH!r}; skipping time-series test.")

    with tempfile.TemporaryDirectory() as workspace:
        runs_dir = os.path.join(workspace, "runs")
        config = {"runs_dir": runs_dir, "random_seed": _SEED}

        dashboard = orchestrator.run_pipeline(_ETTH1_PATH, config)
        run_dir = _run_dir_from_dashboard(dashboard)
        manifest_path = os.path.join(run_dir, "job_manifest.json")
        manifest = _load_json(manifest_path)

        yield {
            "dashboard": dashboard,
            "run_dir": run_dir,
            "manifest_path": manifest_path,
            "manifest": manifest,
            "artifacts_dir": os.path.join(run_dir, "artifacts"),
        }


@pytest.fixture(scope="module")
def timeseries_determinism_pair():
    """Run the pipeline twice on ETTh1.csv with the same seed.

    Yields (run1_info, run2_info) where each is a dict with artifacts_dir.
    Uses separate runs_dir subdirectories so the runs are fully independent.
    """
    if not os.path.isfile(_ETTH1_PATH):
        pytest.skip(f"ETTh1.csv not found at {_ETTH1_PATH!r}; skipping determinism test.")

    with tempfile.TemporaryDirectory() as workspace:
        results = []
        for i in range(2):
            runs_dir = os.path.join(workspace, f"runs_{i}")
            config = {"runs_dir": runs_dir, "random_seed": _SEED}
            dashboard = orchestrator.run_pipeline(_ETTH1_PATH, config)
            run_dir = _run_dir_from_dashboard(dashboard)
            results.append({
                "artifacts_dir": os.path.join(run_dir, "artifacts"),
            })

        yield results[0], results[1]


# ---------------------------------------------------------------------------
# Tests — pipeline completion
# ---------------------------------------------------------------------------


class TestTimeseriesPipelineCompletion:
    """Verify the pipeline runs to completion on the ETTh1 time-series dataset."""

    def test_manifest_exists(self, timeseries_run):
        assert os.path.isfile(timeseries_run["manifest_path"]), (
            f"job_manifest.json not found at {timeseries_run['manifest_path']}"
        )

    def test_manifest_has_run_id(self, timeseries_run):
        run_id = timeseries_run["manifest"]["run_id"]
        assert isinstance(run_id, str) and run_id, "run_id is missing or empty"

    def test_all_stages_completed(self, timeseries_run):
        stages = timeseries_run["manifest"]["stages"]
        for stage in _STAGE_NAMES:
            assert stage in stages, f"Stage '{stage}' missing from manifest"
            status = stages[stage]["status"]
            assert status == "completed", (
                f"Stage '{stage}' has status '{status}', expected 'completed'"
            )

    def test_stage_artifacts_exist_on_disk(self, timeseries_run):
        stages = timeseries_run["manifest"]["stages"]
        for stage in _STAGE_NAMES:
            for key, path in stages[stage].get("artifacts", {}).items():
                assert os.path.exists(path), (
                    f"Stage '{stage}' artifact '{key}' not found at '{path}'"
                )

    def test_dashboard_directory_exists(self, timeseries_run):
        assert os.path.isdir(timeseries_run["dashboard"]), (
            f"dashboard/ not a directory: {timeseries_run['dashboard']}"
        )

    def test_return_value_matches_manifest(self, timeseries_run):
        recorded = timeseries_run["manifest"]["stages"]["artifact_assembly"][
            "artifacts"
        ]["dashboard_bundle"]
        assert timeseries_run["dashboard"] == recorded


# ---------------------------------------------------------------------------
# Tests — canonical artifact existence
# ---------------------------------------------------------------------------


class TestTimeseriesArtifactsExist:
    """Verify every canonical artifact is present in the artifacts/ directory."""

    _REQUIRED_FILES = [
        "dataset_profile.json",
        "task_spec.json",
        "preprocessing_plan.json",
        "processed_data.csv",
        "preprocessing_manifest.json",
        "eval_protocol.json",
        "selected_models.json",
        "training_results.json",
        "evaluation_report.json",
        "comparison_table.json",
    ]

    def test_required_artifacts_exist(self, timeseries_run):
        arts_dir = timeseries_run["artifacts_dir"]
        for fname in self._REQUIRED_FILES:
            path = os.path.join(arts_dir, fname)
            assert os.path.isfile(path), f"Required artifact missing: {fname}"


# ---------------------------------------------------------------------------
# Tests — task-type coherence
# ---------------------------------------------------------------------------


class TestTimeseriesTaskTypeCoherence:
    """Verify task classification and downstream outputs are time-series-consistent."""

    def test_task_spec_is_time_series_forecasting(self, timeseries_run):
        task_spec = _load_json(
            os.path.join(timeseries_run["artifacts_dir"], "task_spec.json")
        )
        assert task_spec["task_type"] == "time_series_forecasting", (
            f"Expected task_type='time_series_forecasting', got {task_spec['task_type']!r}"
        )

    def test_task_spec_has_time_col(self, timeseries_run):
        task_spec = _load_json(
            os.path.join(timeseries_run["artifacts_dir"], "task_spec.json")
        )
        assert task_spec.get("time_col") is not None, (
            "task_spec.time_col must be set for time_series_forecasting"
        )

    def test_eval_protocol_split_strategy_is_time_series_appropriate(self, timeseries_run):
        proto = _load_json(
            os.path.join(timeseries_run["artifacts_dir"], "eval_protocol.json")
        )
        ts_strategies = {"chronological", "time_series_cv"}
        assert proto["split_strategy"] in ts_strategies, (
            f"eval_protocol split_strategy for time series must be one of "
            f"{ts_strategies}, got {proto['split_strategy']!r}"
        )

    def test_eval_protocol_has_time_col(self, timeseries_run):
        proto = _load_json(
            os.path.join(timeseries_run["artifacts_dir"], "eval_protocol.json")
        )
        assert proto.get("time_col") is not None, (
            "eval_protocol.time_col must be set for time_series_forecasting"
        )

    def test_selected_models_count_is_three(self, timeseries_run):
        selected = _load_json(
            os.path.join(timeseries_run["artifacts_dir"], "selected_models.json")
        )
        assert len(selected["selected_models"]) == 3, (
            "ModelSelectionAgent must select exactly 3 models (AGENT_ARCHITECTURE.md §6.2)"
        )

    def test_selected_models_tier_coverage(self, timeseries_run):
        selected = _load_json(
            os.path.join(timeseries_run["artifacts_dir"], "selected_models.json")
        )
        tiers = {m["tier"] for m in selected["selected_models"]}
        for required_tier in ("baseline", "classical", "specialized"):
            assert required_tier in tiers, (
                f"Required tier '{required_tier}' not represented in selected models "
                "(AGENT_ARCHITECTURE.md §6.3)"
            )

    def test_comparison_table_has_ranked_models(self, timeseries_run):
        ct = _load_json(
            os.path.join(timeseries_run["artifacts_dir"], "comparison_table.json")
        )
        assert len(ct.get("ranking", [])) >= 1, (
            "comparison_table.json ranking list must contain at least one evaluated model"
        )

    def test_comparison_table_models_have_metrics(self, timeseries_run):
        ct = _load_json(
            os.path.join(timeseries_run["artifacts_dir"], "comparison_table.json")
        )
        for entry in ct.get("ranking", []):
            assert "all_metrics" in entry, (
                f"Ranking entry for {entry.get('model_name')!r} missing 'all_metrics'"
            )
            assert len(entry["all_metrics"]) > 0, (
                f"Ranking entry for {entry.get('model_name')!r} has empty metrics"
            )


# ---------------------------------------------------------------------------
# Tests — determinism
# ---------------------------------------------------------------------------


class TestTimeseriesDeterminism:
    """Two pipeline runs with the same seed must produce identical metric values."""

    def test_comparison_table_metrics_are_identical(self, timeseries_determinism_pair):
        run1, run2 = timeseries_determinism_pair

        ct1 = _load_json(os.path.join(run1["artifacts_dir"], "comparison_table.json"))
        ct2 = _load_json(os.path.join(run2["artifacts_dir"], "comparison_table.json"))

        # Index by model name for stable comparison (order may not be guaranteed)
        metrics1 = {e["model_name"]: e["all_metrics"] for e in ct1.get("ranking", [])}
        metrics2 = {e["model_name"]: e["all_metrics"] for e in ct2.get("ranking", [])}

        assert set(metrics1.keys()) == set(metrics2.keys()), (
            f"Evaluated model sets differ between runs: "
            f"{set(metrics1)} vs {set(metrics2)}"
        )

        for model_name in metrics1:
            m1 = metrics1[model_name]
            m2 = metrics2[model_name]
            for metric_name, val1 in m1.items():
                val2 = m2.get(metric_name)
                assert val1 == val2, (
                    f"Metric '{metric_name}' for model '{model_name}' differs between "
                    f"runs with seed={_SEED}: {val1} (run 1) vs {val2} (run 2)"
                )

    def test_primary_metric_values_are_identical(self, timeseries_determinism_pair):
        run1, run2 = timeseries_determinism_pair

        ct1 = _load_json(os.path.join(run1["artifacts_dir"], "comparison_table.json"))
        ct2 = _load_json(os.path.join(run2["artifacts_dir"], "comparison_table.json"))

        pv1 = {e["model_name"]: e["primary_metric_value"] for e in ct1.get("ranking", [])}
        pv2 = {e["model_name"]: e["primary_metric_value"] for e in ct2.get("ranking", [])}

        assert pv1 == pv2, (
            f"primary_metric_value map differs between runs with seed={_SEED}:\n"
            f"  run 1: {pv1}\n  run 2: {pv2}"
        )
