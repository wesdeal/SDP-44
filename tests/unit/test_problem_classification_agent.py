"""tests/unit/test_problem_classification_agent.py

Unit tests for ProblemClassificationAgent (Agent 2).

Coverage:
- tabular_classification  (categorical target, int low-cardinality target)
- tabular_regression       (float target, no datetime)
- time_series_forecasting  (monotonically-increasing datetime col, numeric target)
- grouped_prediction       (datetime col + categorical group, time col not monotonic)
- classification_subtype: binary / multiclass
- Failure paths: TARGET_CONSTANT, TARGET_TOO_SPARSE, TARGET_NOT_FOUND (no numeric col)
- run() method: writes task_spec.json, updates manifest status (happy path + failure)

Architecture authority: AGENT_ARCHITECTURE.md §4 Agent 2.
"""

import json
import os

import pytest

from Pipeline.agents.problem_classification_agent import ProblemClassificationAgent

# ---------------------------------------------------------------------------
# Shared agent instance
# ---------------------------------------------------------------------------

AGENT = ProblemClassificationAgent()

# ---------------------------------------------------------------------------
# Profile / manifest builders
# ---------------------------------------------------------------------------


def _col(
    name: str,
    inferred_type: str,
    dtype_pandas: str,
    unique_count: int,
    missing_fraction: float = 0.0,
    is_monotonically_increasing: bool = False,
    min_val=None,
    max_val=None,
    has_negative_values: bool = False,
) -> dict:
    """Build a minimal column profile entry matching dataset_profile.json structure."""
    col: dict = {
        "name": name,
        "inferred_type": inferred_type,
        "dtype_pandas": dtype_pandas,
        "unique_count": unique_count,
        "missing_fraction": missing_fraction,
        "is_monotonically_increasing": is_monotonically_increasing,
        "has_negative_values": has_negative_values,
    }
    if min_val is not None:
        col["min"] = min_val
    if max_val is not None:
        col["max"] = max_val
    return col


def _profile(
    columns: list,
    detected_datetime_columns: list | None = None,
    llm_analysis: dict | None = None,
    num_rows: int = 100,
) -> dict:
    """Build a minimal dataset_profile dict."""
    return {
        "num_rows": num_rows,
        "columns": columns,
        "detected_datetime_columns": detected_datetime_columns or [],
        "llm_analysis": llm_analysis or {},
    }


def _manifest(run_id: str = "test-run", config: dict | None = None) -> dict:
    """Build a minimal manifest dict for _build_task_spec (no file I/O needed)."""
    return {
        "run_id": run_id,
        "config": config or {},
        "stages": {},
    }


# ---------------------------------------------------------------------------
# Helpers for run() integration tests
# ---------------------------------------------------------------------------

_ALL_STAGES = [
    "ingestion",
    "problem_classification",
    "preprocessing_planning",
    "evaluation_protocol",
    "model_selection",
    "training",
    "evaluation",
    "artifact_assembly",
]


def _stage_entry(status: str = "pending") -> dict:
    return {
        "status": status,
        "started_at": None,
        "completed_at": None,
        "artifacts": {},
        "error": None,
    }


def _write_run_dir(tmp_path, profile: dict, config: dict | None = None) -> str:
    """Create a minimal run workspace and return the path to job_manifest.json."""
    run_dir = tmp_path / "runs" / "test-run"
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True)

    profile_path = str(artifacts_dir / "dataset_profile.json")
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f)

    stages = {name: _stage_entry("pending") for name in _ALL_STAGES}
    stages["ingestion"]["status"] = "completed"
    stages["ingestion"]["artifacts"]["dataset_profile"] = profile_path

    manifest = {
        "run_id": "test-run",
        "config": config or {},
        "stages": stages,
    }
    manifest_path = str(run_dir / "job_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    return manifest_path


# ===========================================================================
# Task type: tabular_classification
# ===========================================================================


class TestTabularClassification:
    def test_binary_categorical_target(self):
        """Categorical target with 2 unique values → binary classification."""
        cols = [
            _col("feature1", "numeric", "float64", unique_count=80),
            _col("label", "categorical", "object", unique_count=2),
        ]
        spec = AGENT._build_task_spec(_manifest(), _profile(cols))

        assert spec["task_type"] == "tabular_classification"
        assert spec["modality"] == "tabular_iid"
        assert spec["target_col"] == "label"
        assert spec["classification_subtype"] == "binary"
        assert spec["time_col"] is None
        assert spec["group_col"] is None

    def test_multiclass_categorical_target(self):
        """Categorical target with >2 unique values → multiclass classification."""
        cols = [
            _col("feature1", "numeric", "float64", unique_count=80),
            _col("label", "categorical", "object", unique_count=5),
        ]
        spec = AGENT._build_task_spec(_manifest(), _profile(cols))

        assert spec["task_type"] == "tabular_classification"
        assert spec["classification_subtype"] == "multiclass"

    def test_int_low_cardinality_target(self):
        """Integer target with cardinality ≤ 20 → classification (arch §4 Agent 2)."""
        cols = [
            _col("feature1", "numeric", "float64", unique_count=80),
            _col("target", "numeric", "int64", unique_count=4),
        ]
        spec = AGENT._build_task_spec(_manifest(), _profile(cols))

        assert spec["task_type"] == "tabular_classification"
        assert spec["modality"] == "tabular_iid"
        assert spec["classification_subtype"] == "multiclass"


# ===========================================================================
# Task type: tabular_regression
# ===========================================================================


class TestTabularRegression:
    def test_float_target_no_datetime(self):
        """Numeric float target, no datetime columns → tabular_regression."""
        cols = [
            _col("feature1", "numeric", "float64", unique_count=80),
            _col("target", "numeric", "float64", unique_count=90,
                 min_val=-5.0, max_val=50.0, has_negative_values=True),
        ]
        spec = AGENT._build_task_spec(_manifest(), _profile(cols))

        assert spec["task_type"] == "tabular_regression"
        assert spec["modality"] == "tabular_iid"
        assert spec["target_col"] == "target"
        assert spec["time_col"] is None
        assert spec["group_col"] is None
        assert spec["classification_subtype"] is None

    def test_regression_subtype_standard(self):
        """Float target with negative values → regression_subtype='standard'."""
        cols = [
            _col("feature1", "numeric", "float64", unique_count=80),
            _col("target", "numeric", "float64", unique_count=90,
                 min_val=-5.0, max_val=50.0, has_negative_values=True),
        ]
        spec = AGENT._build_task_spec(_manifest(), _profile(cols))

        assert spec["regression_subtype"] == "standard"

    def test_regression_subtype_bounded(self):
        """Float target entirely in [0, 1] → regression_subtype='bounded'."""
        cols = [
            _col("feature1", "numeric", "float64", unique_count=80),
            _col("target", "numeric", "float64", unique_count=90,
                 min_val=0.0, max_val=1.0, has_negative_values=False),
        ]
        spec = AGENT._build_task_spec(_manifest(), _profile(cols))

        assert spec["task_type"] == "tabular_regression"
        assert spec["regression_subtype"] == "bounded"


# ===========================================================================
# Task type: time_series_forecasting
# ===========================================================================


class TestTimeSeriesForecasting:
    def _ts_profile(self) -> dict:
        """Minimal profile that triggers time_series_forecasting."""
        cols = [
            _col("date", "numeric", "datetime64[ns]", unique_count=100,
                 is_monotonically_increasing=True),
            _col("value", "numeric", "float64", unique_count=95,
                 min_val=0.0, max_val=100.0),
        ]
        return _profile(
            cols,
            detected_datetime_columns=["date"],
            llm_analysis={
                "suggested_target_variable": "value",
                "target_confidence": "high",
                "target_reasoning": "numeric value column",
            },
        )

    def test_basic_routing(self):
        """Monotonically-increasing datetime + numeric target → time_series_forecasting."""
        spec = AGENT._build_task_spec(_manifest(), self._ts_profile())

        assert spec["task_type"] == "time_series_forecasting"
        assert spec["modality"] == "time_series"
        assert spec["time_col"] == "date"
        assert spec["target_col"] == "value"

    def test_default_forecast_horizon(self):
        """forecast_horizon defaults to 10 when not set in config."""
        spec = AGENT._build_task_spec(_manifest(), self._ts_profile())

        assert spec["forecast_horizon"] == 10

    def test_custom_forecast_horizon(self):
        """forecast_horizon is read from manifest config when present."""
        spec = AGENT._build_task_spec(
            _manifest(config={"forecast_horizon": 24}),
            self._ts_profile(),
        )

        assert spec["forecast_horizon"] == 24

    def test_time_col_and_group_col_types(self):
        """time_col is a string; group_col is None for a plain time-series."""
        spec = AGENT._build_task_spec(_manifest(), self._ts_profile())

        assert isinstance(spec["time_col"], str)
        assert spec["group_col"] is None


# ===========================================================================
# Task type: grouped_prediction
# ===========================================================================


class TestGroupedPrediction:
    def test_grouped_prediction_non_monotonic_time(self):
        """
        Datetime col present but NOT monotonically increasing → TS priority 1 skipped.
        Low-cardinality categorical col present → grouped_prediction (priority 2).
        """
        cols = [
            _col("date", "numeric", "datetime64[ns]", unique_count=100,
                 is_monotonically_increasing=False),
            _col("store_id", "categorical", "object", unique_count=5),
            _col("target", "numeric", "float64", unique_count=80),
        ]
        profile = _profile(
            cols,
            detected_datetime_columns=["date"],
            llm_analysis={
                "suggested_target_variable": "target",
                "target_confidence": "high",
                "target_reasoning": "sales value",
            },
            num_rows=100,
        )
        spec = AGENT._build_task_spec(_manifest(), profile)

        assert spec["task_type"] == "grouped_prediction"
        assert spec["modality"] == "grouped_tabular"
        assert spec["group_col"] == "store_id"
        assert spec["target_col"] == "target"
        assert spec["time_col"] is not None  # date col is still captured


# ===========================================================================
# Failure paths
# ===========================================================================


class TestFailurePaths:
    def test_target_constant_raises(self):
        """TARGET_CONSTANT: target has only 1 unique value → ValueError."""
        cols = [
            _col("feature1", "numeric", "float64", unique_count=80),
            _col("target", "numeric", "float64", unique_count=1),
        ]
        with pytest.raises(ValueError, match="TARGET_CONSTANT"):
            AGENT._build_task_spec(_manifest(), _profile(cols))

    def test_target_too_sparse_raises(self):
        """TARGET_TOO_SPARSE: target missing_fraction > 0.5 → ValueError."""
        cols = [
            _col("feature1", "numeric", "float64", unique_count=80),
            _col("target", "numeric", "float64", unique_count=40,
                 missing_fraction=0.6),
        ]
        with pytest.raises(ValueError, match="TARGET_TOO_SPARSE"):
            AGENT._build_task_spec(_manifest(), _profile(cols))

    def test_no_numeric_column_raises(self):
        """TARGET_NOT_FOUND when dataset has only categorical columns."""
        cols = [
            _col("cat1", "categorical", "object", unique_count=5),
            _col("cat2", "categorical", "object", unique_count=3),
        ]
        with pytest.raises(ValueError, match="TARGET_NOT_FOUND"):
            AGENT._build_task_spec(_manifest(), _profile(cols))


# ===========================================================================
# Integration: run() writes task_spec.json and updates manifest on disk
# ===========================================================================


class TestRunMethod:
    def test_run_writes_task_spec_and_sets_completed(self, tmp_path):
        """Happy path: run() produces task_spec.json and marks stage completed."""
        cols = [
            _col("feature1", "numeric", "float64", unique_count=80),
            _col("target", "numeric", "float64", unique_count=90,
                 min_val=0.0, max_val=100.0),
        ]
        manifest_path = _write_run_dir(tmp_path, _profile(cols))

        AGENT.run(manifest_path)

        task_spec_path = os.path.join(
            os.path.dirname(manifest_path), "artifacts", "task_spec.json"
        )
        assert os.path.exists(task_spec_path), "task_spec.json must be written"

        with open(task_spec_path, encoding="utf-8") as f:
            spec = json.load(f)

        assert spec["task_type"] == "tabular_regression"
        assert spec["target_col"] == "target"
        assert spec["run_id"] == "test-run"

        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

        assert manifest["stages"]["problem_classification"]["status"] == "completed"
        assert "task_spec" in manifest["stages"]["problem_classification"]["artifacts"]

    def test_run_sets_failed_status_on_error(self, tmp_path):
        """Failure path: run() marks stage failed and re-raises the exception."""
        cols = [
            _col("feature1", "numeric", "float64", unique_count=80),
            _col("target", "numeric", "float64", unique_count=1),  # constant
        ]
        manifest_path = _write_run_dir(tmp_path, _profile(cols))

        with pytest.raises(ValueError, match="TARGET_CONSTANT"):
            AGENT.run(manifest_path)

        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

        assert manifest["stages"]["problem_classification"]["status"] == "failed"
        assert manifest["stages"]["problem_classification"]["error"] is not None
