"""tests/unit/test_chronos_model.py — Smoke tests for ChronosModel.

Covers:
- model builds and loads pretrained weights without error
- train() stores context and sets is_trained=True
- predict() returns a 1-D array of the correct length
- predict_quantiles() returns correct shape per quantile and monotone ordering
- update_context() appends new observations
- predict raises ValueError before train() is called

NOTE: Tests that require a live model (build + train + predict) are grouped in
a single test to avoid pytest fixture teardown issues with PyTorch's thread-pool
allocator on macOS, which causes a segfault when a loaded transformer model is
kept alive across test collection boundaries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

try:
    import torch
    from Pipeline.models.time_series.chronos_model import ChronosModel, CHRONOS_AVAILABLE
except ImportError:
    CHRONOS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not CHRONOS_AVAILABLE,
    reason="chronos-forecasting not installed",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HISTORY_LEN = 128
HORIZON = 12
_QUANTILES = [0.1, 0.5, 0.9]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(seed: int = 42):
    rng = np.random.default_rng(seed)
    y = pd.Series(rng.standard_normal(HISTORY_LEN))
    X = pd.DataFrame({"lag_1": y.shift(1).fillna(0).values})
    return X, y


def _horizon_X():
    return pd.DataFrame({"lag_1": np.zeros(HORIZON)})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_predict_raises_before_train():
    """A fresh un-built, un-trained instance must raise before predict."""
    m = ChronosModel("Chronos", {"model_size": "tiny"})
    m.is_trained = False  # ensure guard is active
    with pytest.raises(ValueError, match="train\\(\\)"):
        m.predict(_horizon_X())


def test_full_inference_pipeline():
    """
    End-to-end smoke test: build → train → predict → predict_quantiles.

    Kept as one test so the loaded ChronosPipeline object is not shared
    across pytest fixture teardown boundaries (avoids macOS SIGSEGV in
    PyTorch's allocator when the model lives past a test boundary).
    """
    # PyTorch's OpenMP thread pool conflicts with pytest's signal handling on
    # macOS, causing a segfault in torch.nn.functional.embedding. Single-
    # threaded inference avoids this; Chronos-tiny is fast enough on CPU.
    torch.set_num_threads(1)

    m = ChronosModel("Chronos", {"model_size": "tiny", "num_samples": 10})

    # --- build ---
    m.build()
    assert m.model is not None, "build() must set self.model"

    # --- train ---
    X, y = _make_data()
    m.train(X, y)
    assert m.is_trained is True, "train() must set is_trained=True"
    assert m.context_data is not None
    assert len(m.context_data) == HISTORY_LEN

    # --- predict ---
    X_h = _horizon_X()
    preds = m.predict(X_h)

    assert preds.shape == (HORIZON,), (
        f"Expected shape ({HORIZON},), got {preds.shape}"
    )
    assert np.all(np.isfinite(preds)), "Point forecast contains non-finite values"

    # --- predict_quantiles ---
    result = m.predict_quantiles(X_h, quantiles=_QUANTILES)
    for q in _QUANTILES:
        assert q in result, f"Missing quantile {q}"
        assert result[q].shape == (HORIZON,), (
            f"Quantile {q}: expected ({HORIZON},), got {result[q].shape}"
        )

    # Monotone ordering: q10 <= q50 <= q90 for every timestep
    assert np.all(result[0.1] <= result[0.5] + 1e-6), "q10 > q50 at some step"
    assert np.all(result[0.5] <= result[0.9] + 1e-6), "q50 > q90 at some step"

    # --- update_context ---
    original_len = len(m.context_data)
    new_obs = np.array([1.0, 2.0, 3.0])
    m.update_context(new_obs)
    assert len(m.context_data) == original_len + 3
