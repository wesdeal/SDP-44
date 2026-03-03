"""
Base Model Class - Abstract interface for all models
All model implementations (XGBoost, LSTM, etc.) inherit from this class
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import joblib
import json
from pathlib import Path
import time


class BaseModel(ABC):
    """
    Abstract base class for all models.
    Ensures consistent interface across different model types.

    Every model must implement:
        - build(): Initialize model with hyperparameters
        - train(): Train the model on data
        - predict(): Make predictions on new data
    """

    def __init__(self, model_name: str, hyperparameters: Dict[str, Any]):
        """
        Initialize base model

        Args:
            model_name: Name of the model (e.g., "XGBoost", "LSTM")
            hyperparameters: Dictionary of hyperparameters
        """
        self.model_name = model_name
        self.hyperparameters = hyperparameters
        self.model = None  # The actual model object (sklearn, keras, etc.)
        self.is_trained = False
        self.metadata = {
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': self.__class__.__name__
        }

    @abstractmethod
    def build(self):
        """
        Build/initialize the model with hyperparameters.

        Example for XGBoost:
            self.model = XGBRegressor(n_estimators=100, max_depth=6)

        Example for LSTM:
            self.model = Sequential()
            self.model.add(LSTM(units=64, ...))
        """
        pass

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> Dict:
        """
        Train the model on training data.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)

        Returns:
            training_history: Dict containing training metrics/history
                Example: {"train_loss": [0.5, 0.3, 0.2], "val_loss": [0.6, 0.4, 0.3]}
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Features to predict on

        Returns:
            Predictions as numpy array
        """
        pass

    def save(self, save_path: Path):
        """
        Save the trained model and metadata to disk.

        Saves two files:
            1. {model_name}.pkl - the trained model object
            2. {model_name}_metadata.json - hyperparameters and training info

        Args:
            save_path: Directory to save model files
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        if not self.is_trained:
            print(f"Warning: Saving untrained model '{self.model_name}'")

        # Save model using joblib (efficient for sklearn/xgboost models)
        model_file = save_path / f"{self.model_name}.pkl"
        joblib.dump(self.model, model_file)

        # Save metadata as JSON
        metadata_file = save_path / f"{self.model_name}_metadata.json"
        metadata_to_save = {
            'model_name': self.model_name,
            'hyperparameters': self.hyperparameters,
            'is_trained': self.is_trained,
            **self.metadata
        }

        with open(metadata_file, 'w') as f:
            json.dump(metadata_to_save, f, indent=2, default=str)

        print(f"✓ Model saved to {save_path}")

    def load(self, load_path: Path):
        """
        Load a previously trained model from disk.

        Args:
            load_path: Directory containing saved model files
        """
        load_path = Path(load_path)
        model_file = load_path / f"{self.model_name}.pkl"

        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        # Load model
        self.model = joblib.load(model_file)
        self.is_trained = True

        # Load metadata if available
        metadata_file = load_path / f"{self.model_name}_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                loaded_metadata = json.load(f)
                self.hyperparameters = loaded_metadata.get('hyperparameters', {})
                self.metadata.update(loaded_metadata)

        print(f"✓ Model loaded from {load_path}")

    def get_params(self) -> Dict:
        """Get model hyperparameters"""
        return self.hyperparameters.copy()

    def __repr__(self) -> str:
        """String representation of model"""
        status = "trained" if self.is_trained else "untrained"
        return f"{self.model_name}({status}, params={len(self.hyperparameters)})"
