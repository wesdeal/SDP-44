"""
Dummy Model Implementations - Baseline wrappers for sklearn dummy estimators.
DummyRegressor: predicts training-set mean.
DummyClassifier: predicts most frequent class.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import time

from sklearn.dummy import DummyRegressor as _SklearnDummyRegressor
from sklearn.dummy import DummyClassifier as _SklearnDummyClassifier

from base_model import BaseModel


class DummyRegressorModel(BaseModel):
    """
    Baseline regressor that predicts the training-set mean (or median/constant).
    Wraps sklearn.dummy.DummyRegressor.
    """

    def __init__(self, model_name: str = "DummyRegressor", hyperparameters: Dict[str, Any] = None):
        if hyperparameters is None:
            hyperparameters = {}
        super().__init__(model_name, hyperparameters)

    def build(self):
        """Build DummyRegressor with strategy from hyperparameters (default: mean)."""
        self.model = _SklearnDummyRegressor(
            strategy=self.hyperparameters.get("strategy", "mean"),
        )

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> Dict:
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


class DummyClassifierModel(BaseModel):
    """
    Baseline classifier that predicts the most frequent class (or uniform random).
    Wraps sklearn.dummy.DummyClassifier.
    """

    def __init__(self, model_name: str = "DummyClassifier", hyperparameters: Dict[str, Any] = None):
        if hyperparameters is None:
            hyperparameters = {}
        super().__init__(model_name, hyperparameters)

    def build(self):
        """Build DummyClassifier with strategy from hyperparameters (default: most_frequent)."""
        self.model = _SklearnDummyClassifier(
            strategy=self.hyperparameters.get("strategy", "most_frequent"),
            random_state=self.hyperparameters.get("random_state", 42),
        )

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> Dict:
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
