"""
SVR Model — Support Vector Regression wrapper.

Specialized tabular regression model (AGENT_ARCHITECTURE.md §6.3, specialized tier).
Compatible tasks: tabular_regression only.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import time

from sklearn.svm import SVR

from base_model import BaseModel


class SVRModel(BaseModel):
    """Support Vector Regression for tabular regression tasks."""

    def __init__(self, model_name: str = "SVR", hyperparameters: Dict[str, Any] = None):
        if hyperparameters is None:
            hyperparameters = {}
        super().__init__(model_name, hyperparameters)

    def build(self):
        """Instantiate sklearn SVR with current hyperparameters."""
        self.model = SVR(
            C=self.hyperparameters.get("C", 1.0),
            kernel=self.hyperparameters.get("kernel", "rbf"),
            epsilon=self.hyperparameters.get("epsilon", 0.1),
            gamma=self.hyperparameters.get("gamma", "scale"),
        )

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> Dict:
        if self.model is None:
            raise ValueError("Model must be built before training. Call build() first.")
        start_time = time.time()
        self.model.fit(X_train, y_train)
        training_time = time.time() - start_time
        self.is_trained = True
        self.metadata["training_time_seconds"] = training_time
        self.metadata["n_training_samples"] = len(X_train)
        return {"training_time": training_time}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions.")
        return self.model.predict(X)
