"""core/metric_engine.py — Deterministic metric computation.

Public API:
    compute_all(y_true, y_pred, metrics_list) -> dict[str, float | None]

Supported metric names:
    Regression : rmse, mae, mape, r2, smape
    Classification: accuracy, f1_weighted, roc_auc

Authority: AGENT_ARCHITECTURE.md §3 and TEAM_IMPLEMENTATION_PLAN.md §2.2.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

_SUPPORTED_METRICS = frozenset(
    {"rmse", "mae", "mape", "r2", "smape", "accuracy", "f1_weighted", "roc_auc"}
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_all(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics_list: list[str],
) -> dict[str, Any]:
    """Compute each metric in *metrics_list* and return a result dict.

    Returns ``{metric_name: value}`` where *value* is a ``float`` on success
    or ``None`` if the metric is incompatible with the data or fails for any
    reason.  This function never raises.

    Args:
        y_true: Ground-truth target values.
        y_pred: Model predictions.  For classification metrics that require
            probabilities (roc_auc), pass predicted probabilities; for all
            other classification metrics pass hard class labels.
        metrics_list: Names of metrics to compute.  Unknown names are silently
            mapped to ``None``.

    Returns:
        Dictionary mapping each requested metric name to its value or ``None``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    result: dict[str, Any] = {}
    for name in metrics_list:
        result[name] = _compute_one(name, y_true, y_pred)
    return result


# ---------------------------------------------------------------------------
# Internal dispatch
# ---------------------------------------------------------------------------


def _compute_one(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """Return the scalar metric value, or None on any failure."""
    try:
        if name == "rmse":
            return math.sqrt(mean_squared_error(y_true, y_pred))
        if name == "mae":
            return float(mean_absolute_error(y_true, y_pred))
        if name == "mape":
            return _mape(y_true, y_pred)
        if name == "r2":
            return float(r2_score(y_true, y_pred))
        if name == "smape":
            return _smape(y_true, y_pred)
        if name == "accuracy":
            return float(accuracy_score(y_true, y_pred))
        if name == "f1_weighted":
            return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        if name == "roc_auc":
            return _roc_auc(y_true, y_pred)
        # Unknown metric name
        return None
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Metric implementations
# ---------------------------------------------------------------------------


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """Mean Absolute Percentage Error; returns None if all true values are zero."""
    mask = y_true != 0
    if not mask.any():
        return None
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """Symmetric Mean Absolute Percentage Error (0–2 scale).

    SMAPE = mean(2 * |y_true - y_pred| / (|y_true| + |y_pred|))

    Pairs where both y_true and y_pred are zero contribute 0 to the sum.
    """
    numerator = 2.0 * np.abs(y_true - y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    # Avoid division by zero: 0/0 → 0 (both values identical and zero)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_sample = np.where(denominator == 0, 0.0, numerator / denominator)
    return float(np.mean(per_sample))


def _roc_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """ROC-AUC for binary classification only; returns None for multiclass."""
    classes = np.unique(y_true)
    if len(classes) != 2:
        return None
    return float(roc_auc_score(y_true, y_pred))
