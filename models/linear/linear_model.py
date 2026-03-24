"""
Linear Model - Unified interface for Ridge (regression) and LogisticRegression (classification).
Task type is determined at build() time from hyperparameters["task_type"].
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import time

from sklearn.linear_model import Ridge, LogisticRegression

from base_model import BaseModel

SUPPORTED_TASKS = {"tabular_regression", "tabular_classification"}


class LinearModel(BaseModel):
    """
    Unified linear model wrapper.

    Selects the underlying estimator based on task_type in hyperparameters:
        - tabular_regression     → sklearn.linear_model.Ridge
        - tabular_classification → sklearn.linear_model.LogisticRegression

    Hyperparameters (all optional):
        task_type   : str   - "tabular_regression" or "tabular_classification"
        alpha       : float - regularization strength for Ridge (default: 1.0)
        C           : float - inverse regularization strength for LogisticRegression (default: 1.0)
        max_iter    : int   - max iterations for LogisticRegression (default: 1000)
        random_state: int   - random seed (default: 42)
    """

    def __init__(self, model_name: str = "LinearModel", hyperparameters: Dict[str, Any] = None):
        if hyperparameters is None:
            hyperparameters = {}
        super().__init__(model_name, hyperparameters)

    def build(self):
        """
        Instantiate the underlying sklearn estimator based on task_type.

        Raises:
            ValueError: If task_type is missing or not in SUPPORTED_TASKS.
        """
        task_type = self.hyperparameters.get("task_type")
        if task_type not in SUPPORTED_TASKS:
            raise ValueError(
                f"LinearModel requires hyperparameters['task_type'] to be one of "
                f"{sorted(SUPPORTED_TASKS)}, got: {task_type!r}"
            )

        random_state = self.hyperparameters.get("random_state", 42)

        if task_type == "tabular_regression":
            self.model = Ridge(
                alpha=self.hyperparameters.get("alpha", 1.0),
                random_state=random_state,
            )
        else:  # tabular_classification
            self.model = LogisticRegression(
                C=self.hyperparameters.get("C", 1.0),
                max_iter=self.hyperparameters.get("max_iter", 1000),
                random_state=random_state,
            )

        self.metadata["task_type"] = task_type

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
