"""
Random Forest Model Implementation
Ensemble of decision trees for robust predictions
"""

from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import time

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from base_model import BaseModel


class RandomForestModel(BaseModel):
    """
    Random Forest implementation for regression tasks.
    Ensemble method using multiple decision trees.
    """

    def __init__(self, model_name: str = "RandomForest", hyperparameters: Dict[str, Any] = None):
        if hyperparameters is None:
            hyperparameters = {}
        super().__init__(model_name, hyperparameters)

    def build(self):
        """Build Random Forest model with specified hyperparameters."""
        self.model = RandomForestRegressor(
            n_estimators=self.hyperparameters.get('n_estimators', 100),
            max_depth=self.hyperparameters.get('max_depth', 10),
            min_samples_split=self.hyperparameters.get('min_samples_split', 5),
            min_samples_leaf=self.hyperparameters.get('min_samples_leaf', 2),
            max_features=self.hyperparameters.get('max_features', 'sqrt'),
            bootstrap=self.hyperparameters.get('bootstrap', True),
            random_state=self.hyperparameters.get('random_state', 42),
            n_jobs=-1,  # Use all CPU cores
            verbose=0
        )

        print(f"✓ Random Forest model built with {len(self.hyperparameters)} hyperparameters")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> Dict:
        """Train Random Forest model."""
        print(f"Training Random Forest on {len(X_train)} samples...")

        start_time = time.time()

        # Train the model
        self.model.fit(X_train, y_train)

        training_time = time.time() - start_time
        self.is_trained = True

        # Store metadata
        self.metadata['training_time_seconds'] = training_time
        self.metadata['n_features'] = X_train.shape[1]
        self.metadata['n_training_samples'] = len(X_train)
        self.metadata['feature_names'] = list(X_train.columns)

        print(f"✓ Training completed in {training_time:.2f} seconds")

        return {
            'training_time': training_time,
            'n_estimators': self.model.n_estimators
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions!")

        predictions = self.model.predict(X)
        return predictions

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        if not self.is_trained:
            raise ValueError("Model must be trained first!")

        importance = self.model.feature_importances_
        feature_names = self.model.feature_names_in_

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return importance_df
