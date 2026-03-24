"""tests/integration/test_full_pipeline_regression.py — Phase 7.5 full-pipeline
integration test for tabular regression.

Verifies:
1. Run completes to the expected end state.
2. All canonical artifacts exist on disk.
3. Manifest stage statuses are architecture-consistent.
4. Outputs are coherent with the tabular_regression task type.
5. Two runs with the same seed produce identical metric values.

No mocking. No legacy imports. No direct agent invocation.
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pandas as pd
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

# ---------------------------------------------------------------------------
# Synthetic regression dataset (deterministic)
# ---------------------------------------------------------------------------


def _write_synthetic_regression_csv(path: str, n: int = 500, seed: int = _SEED) -> None:
    """Write a deterministic tabular regression CSV to *path*.

    The dataset has no datetime columns, five numeric features, and a
    continuous numeric target — forcing problem_classification to resolve
    to tabular_regression.
    """
    rng = np.random.default_rng(seed)
    feat1 = rng.normal(0.0, 1.0, n)
    feat2 = rng.uniform(-5.0, 5.0, n)
    feat3 = rng.exponential(1.0, n)
    feat4 = rng.normal(2.0, 0.5, n)
    feat5 = rng.uniform(0.0, 10.0, n)
    target = (
        3.0 * feat1
        - 2.0 * feat2
        + 0.5 * feat3
        + feat4
        + rng.normal(0.0, 0.1, n)
    )
    df = pd.DataFrame({
        "feat1": feat1,
        "feat2": feat2,
        "feat3": feat3,
        "feat4": feat4,
        "feat5": feat5,
        "target": target,
    })
    df.to_csv(path, index=False)


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
def regression_run():
    """Run the full pipeline once on the synthetic regression dataset.

    Yields a dict with:
        dashboard       – return value of run_pipeline()
        run_dir         – runs/{run_id}/ directory
        manifest_path   – path to job_manifest.json
        manifest        – parsed manifest dict
        artifacts_dir   – runs/{run_id}/artifacts/ directory
    """
    with tempfile.TemporaryDirectory() as workspace:
        input_path = os.path.join(workspace, "synthetic_regression.csv")
        _write_synthetic_regression_csv(input_path)

        runs_dir = os.path.join(workspace, "runs")
        config = {"runs_dir": runs_dir, "random_seed": _SEED}

        dashboard = orchestrator.run_pipeline(input_path, config)
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
def regression_determinism_pair():
    """Run the pipeline twice with the same seed.

    Yields (run1_info, run2_info) where each is a dict with artifacts_dir.
    Uses separate runs_dir subdirectories so the runs are fully independent.
    """
    with tempfile.TemporaryDirectory() as workspace:
        input_path = os.path.join(workspace, "synthetic_regression.csv")
        _write_synthetic_regression_csv(input_path)

        results = []
        for i in range(2):
            runs_dir = os.path.join(workspace, f"runs_{i}")
            config = {"runs_dir": runs_dir, "random_seed": _SEED}
            dashboard = orchestrator.run_pipeline(input_path, config)
            run_dir = _run_dir_from_dashboard(dashboard)
            results.append({
                "artifacts_dir": os.path.join(run_dir, "artifacts"),
            })

        yield results[0], results[1]


# ---------------------------------------------------------------------------
# Tests — pipeline completion
# ---------------------------------------------------------------------------


class TestRegressionPipelineCompletion:
    """Verify the pipeline runs to completion on a tabular regression dataset."""

    def test_manifest_exists(self, regression_run):
        assert os.path.isfile(regression_run["manifest_path"]), (
            f"job_manifest.json not found at {regression_run['manifest_path']}"
        )

    def test_manifest_has_run_id(self, regression_run):
        run_id = regression_run["manifest"]["run_id"]
        assert isinstance(run_id, str) and run_id, "run_id is missing or empty"

    def test_all_stages_completed(self, regression_run):
        stages = regression_run["manifest"]["stages"]
        for stage in _STAGE_NAMES:
            assert stage in stages, f"Stage '{stage}' missing from manifest"
            status = stages[stage]["status"]
            assert status == "completed", (
                f"Stage '{stage}' has status '{status}', expected 'completed'"
            )

    def test_stage_artifacts_exist_on_disk(self, regression_run):
        stages = regression_run["manifest"]["stages"]
        for stage in _STAGE_NAMES:
            for key, path in stages[stage].get("artifacts", {}).items():
                assert os.path.exists(path), (
                    f"Stage '{stage}' artifact '{key}' not found at '{path}'"
                )

    def test_dashboard_directory_exists(self, regression_run):
        assert os.path.isdir(regression_run["dashboard"]), (
            f"dashboard/ not a directory: {regression_run['dashboard']}"
        )

    def test_return_value_matches_manifest(self, regression_run):
        recorded = regression_run["manifest"]["stages"]["artifact_assembly"][
            "artifacts"
        ]["dashboard_bundle"]
        assert regression_run["dashboard"] == recorded


# ---------------------------------------------------------------------------
# Tests — canonical artifact existence
# ---------------------------------------------------------------------------


class TestRegressionArtifactsExist:
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

    def test_required_artifacts_exist(self, regression_run):
        arts_dir = regression_run["artifacts_dir"]
        for fname in self._REQUIRED_FILES:
            path = os.path.join(arts_dir, fname)
            assert os.path.isfile(path), f"Required artifact missing: {fname}"


# ---------------------------------------------------------------------------
# Tests — task-type coherence
# ---------------------------------------------------------------------------


class TestRegressionTaskTypeCoherence:
    """Verify task classification and downstream outputs are regression-consistent."""

    def test_task_spec_is_tabular_regression(self, regression_run):
        task_spec = _load_json(
            os.path.join(regression_run["artifacts_dir"], "task_spec.json")
        )
        assert task_spec["task_type"] == "tabular_regression", (
            f"Expected task_type='tabular_regression', got {task_spec['task_type']!r}"
        )

    def test_selected_models_count_is_three(self, regression_run):
        selected = _load_json(
            os.path.join(regression_run["artifacts_dir"], "selected_models.json")
        )
        assert len(selected["selected_models"]) == 3, (
            "ModelSelectionAgent must select exactly 3 models (AGENT_ARCHITECTURE.md §6.2)"
        )

    def test_selected_models_tier_coverage(self, regression_run):
        selected = _load_json(
            os.path.join(regression_run["artifacts_dir"], "selected_models.json")
        )
        tiers = {m["tier"] for m in selected["selected_models"]}
        for required_tier in ("baseline", "classical", "specialized"):
            assert required_tier in tiers, (
                f"Required tier '{required_tier}' not represented in selected models "
                "(AGENT_ARCHITECTURE.md §6.3)"
            )

    def test_comparison_table_has_ranked_models(self, regression_run):
        ct = _load_json(
            os.path.join(regression_run["artifacts_dir"], "comparison_table.json")
        )
        assert len(ct.get("ranking", [])) >= 1, (
            "comparison_table.json ranking list must contain at least one evaluated model"
        )

    def test_comparison_table_models_have_metrics(self, regression_run):
        ct = _load_json(
            os.path.join(regression_run["artifacts_dir"], "comparison_table.json")
        )
        for entry in ct.get("ranking", []):
            assert "all_metrics" in entry, (
                f"Ranking entry for {entry.get('model_name')!r} missing 'all_metrics'"
            )
            assert len(entry["all_metrics"]) > 0, (
                f"Ranking entry for {entry.get('model_name')!r} has empty metrics"
            )

    def test_eval_protocol_split_strategy_is_regression_appropriate(self, regression_run):
        proto = _load_json(
            os.path.join(regression_run["artifacts_dir"], "eval_protocol.json")
        )
        # Tabular regression must not use a time-series-only split strategy
        assert proto["split_strategy"] != "time_series_cv", (
            "eval_protocol split_strategy should not be 'time_series_cv' for tabular regression"
        )


# ---------------------------------------------------------------------------
# Tests — determinism
# ---------------------------------------------------------------------------


class TestRegressionDeterminism:
    """Two pipeline runs with the same seed must produce identical metric values."""

    def test_comparison_table_metrics_are_identical(self, regression_determinism_pair):
        run1, run2 = regression_determinism_pair

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

    def test_primary_metric_values_are_identical(self, regression_determinism_pair):
        run1, run2 = regression_determinism_pair

        ct1 = _load_json(os.path.join(run1["artifacts_dir"], "comparison_table.json"))
        ct2 = _load_json(os.path.join(run2["artifacts_dir"], "comparison_table.json"))

        pv1 = {e["model_name"]: e["primary_metric_value"] for e in ct1.get("ranking", [])}
        pv2 = {e["model_name"]: e["primary_metric_value"] for e in ct2.get("ranking", [])}

        assert pv1 == pv2, (
            f"primary_metric_value map differs between runs with seed={_SEED}:\n"
            f"  run 1: {pv1}\n  run 2: {pv2}"
        )
