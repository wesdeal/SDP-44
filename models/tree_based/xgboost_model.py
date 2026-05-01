"""
XGBoost Model Implementation
Gradient boosting model for time series forecasting
"""
from xgboost import XGBClassifier, XGBRegressor
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Any, Optional
import time

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from base_model import BaseModel


class XGBoostModel(BaseModel):
    """
    XGBoost implementation for regression tasks.
    Uses gradient boosting with tree-based learners.
    """

    def __init__(self, model_name: str = "XGBoost", hyperparameters: Dict[str, Any] = None):
        """
        Initialize XGBoost model

        Args:
            model_name: Name of the model
            hyperparameters: Dict of XGBoost hyperparameters
        """
        if hyperparameters is None:
            hyperparameters = {}
        super().__init__(model_name, hyperparameters)
        self._label_encoder = None

    def build(self):
        """
        Build XGBoost model with specified hyperparameters.

        Key hyperparameters:
            - n_estimators: Number of boosting rounds (trees)
            - max_depth: Maximum tree depth
            - learning_rate: Step size shrinkage
            - subsample: Fraction of samples per tree
            - colsample_bytree: Fraction of features per tree
            - reg_alpha: L1 regularization
            - reg_lambda: L2 regularization
        """
        task_type = self.hyperparameters.get('task_type', 'tabular_regression')
        ModelClass = XGBClassifier if 'classification' in task_type else XGBRegressor
        self.model = ModelClass(
            n_estimators=self.hyperparameters.get('n_estimators', 100),
            max_depth=self.hyperparameters.get('max_depth', 6),
            learning_rate=self.hyperparameters.get('learning_rate', 0.1),
            subsample=self.hyperparameters.get('subsample', 0.8),
            colsample_bytree=self.hyperparameters.get('colsample_bytree', 0.8),
            reg_alpha=self.hyperparameters.get('reg_alpha', 0.0),
            reg_lambda=self.hyperparameters.get('reg_lambda', 1.0),
            random_state=self.hyperparameters.get('random_state', 42),
            tree_method='hist',
            eval_metric='mlogloss' if 'classification' in task_type else 'rmse',
            verbosity=0
        )

        print(f"✓ XGBoost model built with {len(self.hyperparameters)} hyperparameters")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> Dict:
        """
        Train XGBoost model.

        Training process:
        1. Build trees sequentially
        2. Each tree corrects errors from previous trees
        3. Combine predictions from all trees

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)

        Returns:
            training_history: Dict with training metrics
        """
        print(f"Training XGBoost on {len(X_train)} samples...")

        start_time = time.time()

        # Prepare evaluation set for monitoring
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        if isinstance(self.model, XGBClassifier):
            self._label_encoder = LabelEncoder()
            y_train = self._label_encoder.fit_transform(y_train)
            if y_val is not None:
                y_val = self._label_encoder.transform(y_val)
            eval_set = [(X_train, y_train)]
            if X_val is not None and y_val is not None:
                eval_set.append((X_val, y_val))

        # Train the model
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False
        )

        training_time = time.time() - start_time
        self.is_trained = True

        # Store metadata
        self.metadata['training_time_seconds'] = training_time
        self.metadata['n_features'] = X_train.shape[1]
        self.metadata['n_training_samples'] = len(X_train)
        self.metadata['feature_names'] = list(X_train.columns)

        # Get training results
        results = self.model.evals_result() if hasattr(self.model, 'evals_result') else {}

        print(f"✓ Training completed in {training_time:.2f} seconds")

        return {
            'training_time': training_time,
            'evals_result': results,
            'n_estimators_used': self.model.n_estimators
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.

        Prediction process:
        1. Pass features through each tree
        2. Sum all tree predictions (weighted by learning_rate)
        3. Return final prediction

        Args:
            X: Features to predict on

        Returns:
            Predictions as numpy array
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions!")

        predictions = self.model.predict(X)

        if self._label_encoder is not None:
            predictions = self._label_encoder.inverse_transform(
                predictions.astype(int)
            )

        return predictions

    def save(self, save_path):
        super().save(save_path)
        if self._label_encoder is not None:
            import joblib
            from pathlib import Path
            joblib.dump(self._label_encoder, Path(save_path) / "XGBoost_label_encoder.pkl")

    def load(self, load_path):
        super().load(load_path)
        from pathlib import Path
        import joblib
        encoder_path = Path(load_path) / "XGBoost_label_encoder.pkl"
        if encoder_path.exists():
            self._label_encoder = joblib.load(encoder_path)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.

        Returns:
            DataFrame with features ranked by importance
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first!")

        importance = self.model.feature_importances_
        feature_names = self.model.feature_names_in_

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return importance_df
