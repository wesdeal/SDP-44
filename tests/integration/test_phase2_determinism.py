"""tests/integration/test_phase2_determinism.py — Phase 2.4 determinism verification.

Verifies:
1. split_data produces byte-identical splits across two calls with the same seed.
2. compute_all produces identical metric dicts across two calls on the same arrays.
3. schema_validator.validate behaves correctly on real JSON files.

No mocking.  All randomness is controlled via a fixed seed.
"""

from __future__ import annotations

import json
import tempfile
import os

import numpy as np
import pandas as pd
import pytest

from core.splitter import split_data
from core.metric_engine import compute_all
from core.schema_validator import validate

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SEED = 42
N_ROWS = 200
REGRESSION_METRICS = ["rmse", "mae", "mape", "r2", "smape"]


def _make_regression_df(seed: int = SEED, n: int = N_ROWS) -> pd.DataFrame:
    """Return a fully deterministic regression DataFrame."""
    rng = np.random.default_rng(seed)
    feat1 = rng.normal(0.0, 1.0, n)
    feat2 = rng.uniform(-5.0, 5.0, n)
    target = 3.0 * feat1 - 2.0 * feat2 + rng.normal(0.0, 0.1, n)
    return pd.DataFrame({"feat1": feat1, "feat2": feat2, "target": target})


# ---------------------------------------------------------------------------
# 1. split_data determinism
# ---------------------------------------------------------------------------


class TestSplitDataDeterminism:
    """Two calls with the same seed must produce identical output."""

    def _assert_splits_equal(self, s1: dict, s2: dict) -> None:
        """Assert every DataFrame and Series in two split dicts are identical."""
        assert set(s1.keys()) == set(s2.keys()), "Split dicts have different keys"
        for key in s1:
            obj1, obj2 = s1[key], s2[key]
            if isinstance(obj1, pd.DataFrame):
                pd.testing.assert_frame_equal(
                    obj1.reset_index(drop=True),
                    obj2.reset_index(drop=True),
                    check_exact=True,
                    obj=f"split_data key={key!r}",
                )
            elif isinstance(obj1, pd.Series):
                pd.testing.assert_series_equal(
                    obj1.reset_index(drop=True),
                    obj2.reset_index(drop=True),
                    check_exact=True,
                    obj=f"split_data key={key!r}",
                )
            else:
                assert obj1 == obj2, f"Mismatch at key={key!r}: {obj1} != {obj2}"

    def test_random_strategy_determinism(self):
        df = _make_regression_df()
        kwargs = dict(
            target_col="target",
            strategy="random",
            train_fraction=0.7,
            val_fraction=0.15,
            test_fraction=0.15,
            random_seed=SEED,
        )
        s1 = split_data(df.copy(), **kwargs)
        s2 = split_data(df.copy(), **kwargs)
        self._assert_splits_equal(s1, s2)

    def test_chronological_strategy_determinism(self):
        df = _make_regression_df()
        df["time"] = range(N_ROWS)
        kwargs = dict(
            target_col="target",
            strategy="chronological",
            train_fraction=0.7,
            val_fraction=0.15,
            test_fraction=0.15,
            random_seed=SEED,
            time_col="time",
        )
        s1 = split_data(df.copy(), **kwargs)
        s2 = split_data(df.copy(), **kwargs)
        self._assert_splits_equal(s1, s2)

    def test_time_series_cv_determinism(self):
        df = _make_regression_df()
        df["time"] = range(N_ROWS)
        kwargs = dict(
            target_col="target",
            strategy="time_series_cv",
            train_fraction=0.6,
            val_fraction=0.2,
            test_fraction=0.2,
            random_seed=SEED,
            time_col="time",
            cv_method="expanding",
        )
        s1 = split_data(df.copy(), **kwargs)
        s2 = split_data(df.copy(), **kwargs)

        # Compare folds list
        assert len(s1["folds"]) == len(s2["folds"]), "Number of CV folds differs"
        for i, (f1, f2) in enumerate(zip(s1["folds"], s2["folds"])):
            for key in f1:
                obj1, obj2 = f1[key], f2[key]
                if isinstance(obj1, pd.DataFrame):
                    pd.testing.assert_frame_equal(
                        obj1.reset_index(drop=True),
                        obj2.reset_index(drop=True),
                        check_exact=True,
                        obj=f"fold[{i}][{key!r}]",
                    )
                else:
                    pd.testing.assert_series_equal(
                        obj1.reset_index(drop=True),
                        obj2.reset_index(drop=True),
                        check_exact=True,
                        obj=f"fold[{i}][{key!r}]",
                    )
        # Compare frozen test set
        pd.testing.assert_frame_equal(
            s1["X_test"].reset_index(drop=True),
            s2["X_test"].reset_index(drop=True),
            check_exact=True,
        )
        pd.testing.assert_series_equal(
            s1["y_test"].reset_index(drop=True),
            s2["y_test"].reset_index(drop=True),
            check_exact=True,
        )
        assert s1["meta"] == s2["meta"]

    def test_split_sizes_are_consistent(self):
        """Total rows across train/val/test must equal input row count."""
        df = _make_regression_df()
        result = split_data(
            df,
            target_col="target",
            strategy="random",
            train_fraction=0.7,
            val_fraction=0.15,
            test_fraction=0.15,
            random_seed=SEED,
        )
        total = len(result["X_train"]) + len(result["X_val"]) + len(result["X_test"])
        assert total == N_ROWS

    def test_target_column_is_not_in_feature_splits(self):
        df = _make_regression_df()
        result = split_data(
            df,
            target_col="target",
            strategy="random",
            train_fraction=0.7,
        val_fraction=0.15,
        test_fraction=0.15,
        random_seed=SEED,
    )

        assert "target" not in result["X_train"].columns
        assert "target" not in result["X_val"].columns
        assert "target" not in result["X_test"].columns

        assert result["y_train"].name == "target"
        assert result["y_val"].name == "target"
        assert result["y_test"].name == "target"


# ---------------------------------------------------------------------------
# 2. compute_all determinism
# ---------------------------------------------------------------------------


class TestComputeAllDeterminism:
    """Two calls with the same arrays must produce identical metric dicts."""

    def _make_fixed_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(SEED)
        y_true = rng.normal(0.0, 1.0, 100)
        noise = rng.normal(0.0, 0.2, 100)
        y_pred = y_true + noise
        return y_true, y_pred

    def test_regression_metrics_determinism(self):
        y_true, y_pred = self._make_fixed_arrays()
        r1 = compute_all(y_true, y_pred, REGRESSION_METRICS)
        r2 = compute_all(y_true, y_pred, REGRESSION_METRICS)
        assert r1 == r2, f"compute_all returned different results:\n{r1}\nvs\n{r2}"

    def test_all_regression_metrics_non_null(self):
        """Every standard regression metric must return a float, not None."""
        y_true, y_pred = self._make_fixed_arrays()
        result = compute_all(y_true, y_pred, REGRESSION_METRICS)
        for metric, value in result.items():
            assert value is not None, f"Metric {metric!r} returned None unexpectedly"
            assert isinstance(value, float), f"Metric {metric!r} is not a float: {value!r}"

    def test_unknown_metric_returns_none(self):
        y_true, y_pred = self._make_fixed_arrays()
        result = compute_all(y_true, y_pred, ["rmse", "nonexistent_metric"])
        assert result["nonexistent_metric"] is None
        assert result["rmse"] is not None

    def test_metric_values_are_finite(self):
        y_true, y_pred = self._make_fixed_arrays()
        result = compute_all(y_true, y_pred, REGRESSION_METRICS)
        for metric, value in result.items():
            if value is not None:
                assert np.isfinite(value), f"Metric {metric!r} is not finite: {value}"

    def test_two_calls_same_seed_produce_identical_floats(self):
        """Verify bit-for-bit equality (not just approximate) of metric values."""
        rng = np.random.default_rng(SEED)
        y_true = rng.normal(0.0, 5.0, 500)
        y_pred = y_true * 0.95 + rng.normal(0.0, 0.5, 500)

        r1 = compute_all(y_true.copy(), y_pred.copy(), REGRESSION_METRICS)
        r2 = compute_all(y_true.copy(), y_pred.copy(), REGRESSION_METRICS)

        for metric in REGRESSION_METRICS:
            assert r1[metric] == r2[metric], (
                f"Metric {metric!r} differs: {r1[metric]} vs {r2[metric]}"
            )


# ---------------------------------------------------------------------------
# 3. schema_validator behavior
# ---------------------------------------------------------------------------


class TestSchemaValidator:
    """Validate schema_validator.validate against real JSON files on disk."""

    def _write_json(self, tmp_dir: str, filename: str, payload: dict) -> str:
        path = os.path.join(tmp_dir, filename)
        with open(path, "w") as f:
            json.dump(payload, f)
        return path

    def test_valid_dataset_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_json(tmp, "dataset_profile.json", {
                "run_id": "run_001",
                "num_rows": 200,
                "num_columns": 3,
                "columns": ["feat1", "feat2", "target"],
                "extra_field": "ignored",
            })
            assert validate(path, "dataset_profile") is True

    def test_invalid_dataset_profile_missing_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_json(tmp, "bad_profile.json", {
                "run_id": "run_001",
                "num_rows": 200,
                # missing num_columns and columns
            })
            assert validate(path, "dataset_profile") is False

    def test_valid_task_spec(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_json(tmp, "task_spec.json", {
                "run_id": "run_001",
                "task_type": "regression",
                "target_col": "target",
            })
            assert validate(path, "task_spec") is True

    def test_invalid_task_spec_missing_task_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_json(tmp, "bad_task_spec.json", {
                "run_id": "run_001",
                "target_col": "target",
                # missing task_type
            })
            assert validate(path, "task_spec") is False

    def test_unknown_schema_name_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_json(tmp, "dummy.json", {"foo": "bar"})
            with pytest.raises(ValueError, match="Unknown schema_name"):
                validate(path, "nonexistent_schema")

    def test_empty_json_object_fails_all_schemas(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_json(tmp, "empty.json", {})
            assert validate(path, "dataset_profile") is False
            assert validate(path, "task_spec") is False

    def test_validate_is_deterministic_across_calls(self):
        """Same file → same result on repeated calls."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_json(tmp, "profile.json", {
                "run_id": "run_002",
                "num_rows": 100,
                "num_columns": 2,
                "columns": ["a", "b"],
            })
            results = [validate(path, "dataset_profile") for _ in range(5)]
            assert all(r == results[0] for r in results), (
                f"validate returned inconsistent results: {results}"
            )
