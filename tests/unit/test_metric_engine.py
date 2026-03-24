"""tests/unit/test_metric_engine.py — Unit tests for core/metric_engine.py.

Covers:
- normal regression case (rmse, mae, mape, r2, smape)
- normal classification case (accuracy, f1_weighted, roc_auc)
- incompatible metric returns None (e.g. regression metric on class labels)
- mape zero division safety (all-zero y_true)
- roc_auc multiclass returns None
- unknown metric name returns None
- smape zero/zero pair contributes 0
- perfect predictions yield expected values
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from Pipeline.core.metric_engine import compute_all


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def reg_arrays():
    """Simple regression arrays with known errors."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
    return y_true, y_pred


@pytest.fixture
def clf_arrays():
    """Binary classification arrays (hard labels)."""
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])
    return y_true, y_pred


@pytest.fixture
def clf_proba_arrays():
    """Binary classification arrays with predicted probabilities for roc_auc."""
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_pred_proba = np.array([0.1, 0.4, 0.8, 0.9, 0.2, 0.3, 0.05, 0.95])
    return y_true, y_pred_proba


# ---------------------------------------------------------------------------
# Normal regression case
# ---------------------------------------------------------------------------


class TestRegressionMetrics:
    def test_rmse_is_float(self, reg_arrays):
        y_true, y_pred = reg_arrays
        result = compute_all(y_true, y_pred, ["rmse"])
        assert isinstance(result["rmse"], float)
        assert result["rmse"] > 0

    def test_mae_is_float(self, reg_arrays):
        y_true, y_pred = reg_arrays
        result = compute_all(y_true, y_pred, ["mae"])
        assert isinstance(result["mae"], float)
        assert result["mae"] > 0

    def test_mape_is_float(self, reg_arrays):
        y_true, y_pred = reg_arrays
        result = compute_all(y_true, y_pred, ["mape"])
        assert isinstance(result["mape"], float)
        assert result["mape"] > 0

    def test_r2_is_float(self, reg_arrays):
        y_true, y_pred = reg_arrays
        result = compute_all(y_true, y_pred, ["r2"])
        assert isinstance(result["r2"], float)
        assert result["r2"] <= 1.0

    def test_smape_is_float(self, reg_arrays):
        y_true, y_pred = reg_arrays
        result = compute_all(y_true, y_pred, ["smape"])
        assert isinstance(result["smape"], float)
        assert 0.0 <= result["smape"] <= 2.0

    def test_all_regression_metrics_together(self, reg_arrays):
        y_true, y_pred = reg_arrays
        metrics = ["rmse", "mae", "mape", "r2", "smape"]
        result = compute_all(y_true, y_pred, metrics)
        assert set(result.keys()) == set(metrics)
        for name, val in result.items():
            assert val is not None, f"{name} should not be None for valid regression inputs"

    def test_perfect_predictions_rmse_zero(self):
        y = np.array([1.0, 2.0, 3.0])
        result = compute_all(y, y, ["rmse"])
        assert result["rmse"] == pytest.approx(0.0)

    def test_perfect_predictions_r2_one(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        result = compute_all(y, y, ["r2"])
        assert result["r2"] == pytest.approx(1.0)

    def test_rmse_equals_sqrt_mse(self, reg_arrays):
        y_true, y_pred = reg_arrays
        result = compute_all(y_true, y_pred, ["rmse"])
        from sklearn.metrics import mean_squared_error
        expected = math.sqrt(mean_squared_error(y_true, y_pred))
        assert result["rmse"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Normal classification case
# ---------------------------------------------------------------------------


class TestClassificationMetrics:
    def test_accuracy_is_float(self, clf_arrays):
        y_true, y_pred = clf_arrays
        result = compute_all(y_true, y_pred, ["accuracy"])
        assert isinstance(result["accuracy"], float)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_f1_weighted_is_float(self, clf_arrays):
        y_true, y_pred = clf_arrays
        result = compute_all(y_true, y_pred, ["f1_weighted"])
        assert isinstance(result["f1_weighted"], float)
        assert 0.0 <= result["f1_weighted"] <= 1.0

    def test_roc_auc_binary(self, clf_proba_arrays):
        y_true, y_pred_proba = clf_proba_arrays
        result = compute_all(y_true, y_pred_proba, ["roc_auc"])
        assert isinstance(result["roc_auc"], float)
        assert 0.0 <= result["roc_auc"] <= 1.0

    def test_perfect_accuracy(self):
        y = np.array([0, 1, 0, 1])
        result = compute_all(y, y, ["accuracy"])
        assert result["accuracy"] == pytest.approx(1.0)

    def test_all_classification_metrics_together(self, clf_proba_arrays):
        y_true, y_pred_proba = clf_proba_arrays
        metrics = ["accuracy", "f1_weighted", "roc_auc"]
        result = compute_all(y_true, y_pred_proba, metrics)
        assert set(result.keys()) == set(metrics)


# ---------------------------------------------------------------------------
# Incompatible metric returns None
# ---------------------------------------------------------------------------


class TestIncompatibleMetrics:
    def test_roc_auc_on_regression_values_returns_none(self):
        # Continuous float targets — not binary classes → None
        y_true = np.array([1.5, 2.3, 3.1, 4.7])
        y_pred = np.array([1.6, 2.2, 3.0, 4.8])
        result = compute_all(y_true, y_pred, ["roc_auc"])
        assert result["roc_auc"] is None

    def test_unknown_metric_returns_none(self, reg_arrays):
        y_true, y_pred = reg_arrays
        result = compute_all(y_true, y_pred, ["nonexistent_metric"])
        assert result["nonexistent_metric"] is None

    def test_mixed_known_and_unknown(self, reg_arrays):
        y_true, y_pred = reg_arrays
        result = compute_all(y_true, y_pred, ["rmse", "unknown_metric"])
        assert isinstance(result["rmse"], float)
        assert result["unknown_metric"] is None


# ---------------------------------------------------------------------------
# MAPE zero division safety
# ---------------------------------------------------------------------------


class TestMapeZeroDivision:
    def test_all_zero_y_true_returns_none(self):
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        result = compute_all(y_true, y_pred, ["mape"])
        assert result["mape"] is None

    def test_partial_zero_y_true_skips_zeros(self):
        # Only non-zero elements are used; should not raise
        y_true = np.array([0.0, 2.0, 4.0])
        y_pred = np.array([1.0, 2.0, 4.0])
        result = compute_all(y_true, y_pred, ["mape"])
        assert result["mape"] is not None
        assert isinstance(result["mape"], float)
        # mape over non-zero elements: |2-2|/2 + |4-4|/4 = 0.0
        assert result["mape"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ROC-AUC multiclass returns None
# ---------------------------------------------------------------------------


class TestRocAucMulticlass:
    def test_three_class_returns_none(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0.3, 0.4, 0.9, 0.2, 0.5, 0.8])
        result = compute_all(y_true, y_pred, ["roc_auc"])
        assert result["roc_auc"] is None

    def test_single_class_returns_none(self):
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([0.9, 0.8, 0.7, 0.95])
        result = compute_all(y_true, y_pred, ["roc_auc"])
        assert result["roc_auc"] is None


# ---------------------------------------------------------------------------
# SMAPE edge cases
# ---------------------------------------------------------------------------


class TestSmape:
    def test_smape_zero_zero_pair_contributes_zero(self):
        # When both y_true and y_pred are 0, that pair contributes 0 (not NaN)
        y_true = np.array([0.0, 1.0])
        y_pred = np.array([0.0, 1.0])
        result = compute_all(y_true, y_pred, ["smape"])
        assert result["smape"] == pytest.approx(0.0)

    def test_smape_bounded_between_zero_and_two(self):
        rng = np.random.default_rng(42)
        y_true = rng.uniform(0.1, 10.0, size=100)
        y_pred = rng.uniform(0.1, 10.0, size=100)
        result = compute_all(y_true, y_pred, ["smape"])
        assert 0.0 <= result["smape"] <= 2.0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_inputs_same_outputs(self, reg_arrays):
        y_true, y_pred = reg_arrays
        metrics = ["rmse", "mae", "mape", "r2", "smape"]
        r1 = compute_all(y_true, y_pred, metrics)
        r2 = compute_all(y_true, y_pred, metrics)
        for name in metrics:
            assert r1[name] == r2[name], f"{name} is not deterministic"

    def test_output_keys_match_input_list(self, reg_arrays):
        y_true, y_pred = reg_arrays
        metrics = ["rmse", "r2", "unknown"]
        result = compute_all(y_true, y_pred, metrics)
        assert set(result.keys()) == set(metrics)
