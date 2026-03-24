"""models/time_series/arima_model.py — ARIMA model for time series forecasting.

Uses statsmodels ARIMA with a fixed order (1,1,1) so outputs are deterministic
given the same y_train values. No neural-network or transformer dependency.

Architecture authority: AGENT_ARCHITECTURE.md §4 / model_catalog.json.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# Allow sibling-package import when run in isolation (mirrors pattern in chronos_model.py)
sys.path.append(str(Path(__file__).parent.parent))
from base_model import BaseModel  # noqa: E402 (path manipulation must precede)


class ARIMAModel(BaseModel):
    """ARIMA(p,d,q) wrapper for univariate time-series forecasting.

    Defaults to order (1,1,1), which is stable and CPU-fast.
    Deterministic: statsmodels ARIMA uses exact MLE — same data → same coefficients.
    """

    def __init__(self, model_name: str = "ARIMA", hyperparameters: Dict[str, Any] = None):
        if hyperparameters is None:
            hyperparameters = {}
        super().__init__(model_name, hyperparameters)
        self._order = (
            int(hyperparameters.get("p", 1)),
            int(hyperparameters.get("d", 1)),
            int(hyperparameters.get("q", 1)),
        )
        self._fitted = None

    def build(self) -> None:
        """No-op: statsmodels ARIMA is built inside train()."""

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> Dict:
        from statsmodels.tsa.arima.model import ARIMA

        values = y_train.values.astype(float)
        model = ARIMA(values, order=self._order)
        self._fitted = model.fit(method_kwargs={"warn_convergence": False})
        self.model = self._fitted
        self.is_trained = True
        return {"order": list(self._order), "aic": float(self._fitted.aic)}

    def load(self, load_path: Path) -> None:
        """Override to restore self._fitted after base-class joblib load."""
        super().load(load_path)
        self._fitted = self.model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained or self._fitted is None:
            raise RuntimeError("ARIMAModel must be trained before predict().")
        n_steps = len(X)
        forecast = self._fitted.forecast(steps=n_steps)
        return np.asarray(forecast, dtype=float)
