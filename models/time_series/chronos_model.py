"""
Chronos Model Implementation
Pretrained foundation model for zero-shot time series forecasting
Uses Amazon's Chronos transformer-based model
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import time
import torch

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from base_model import BaseModel

try:
    from chronos import ChronosPipeline
    CHRONOS_AVAILABLE = True
except ImportError:
    CHRONOS_AVAILABLE = False
    print("Warning: chronos-forecasting not installed. Install with: pip install git+https://github.com/amazon-science/chronos-forecasting.git")


class ChronosModel(BaseModel):
    """
    Chronos implementation for zero-shot time series forecasting.

    Chronos is a pretrained transformer-based model that can forecast
    time series without task-specific training (zero-shot).

    Key Features:
    - Zero-shot forecasting (no training required)
    - Handles multiple time series
    - Supports different model sizes (tiny, mini, small, base, large)
    - Probabilistic forecasting with quantiles
    """

    def __init__(self, model_name: str = "Chronos", hyperparameters: Dict[str, Any] = None):
        """
        Initialize Chronos model

        Args:
            model_name: Name of the model
            hyperparameters: Dict of Chronos hyperparameters
                - model_size: 'tiny', 'mini', 'small', 'base', 'large' (default: 'small')
                - prediction_length: Number of steps to forecast (default: 12)
                - context_length: Historical window size (default: 512)
                - num_samples: Number of sample paths for probabilistic forecasting (default: 20)
                - temperature: Sampling temperature for generation (default: 1.0)
                - top_k: Top-k sampling parameter (default: 50)
                - top_p: Nucleus sampling parameter (default: 1.0)
        """
        if not CHRONOS_AVAILABLE:
            raise ImportError(
                "Chronos not installed. Install with:\n"
                "pip install git+https://github.com/amazon-science/chronos-forecasting.git"
            )

        if hyperparameters is None:
            hyperparameters = {}
        super().__init__(model_name, hyperparameters)

        # Store context for zero-shot forecasting
        self.context_data = None
        self.feature_names = None

    def build(self):
        """
        Build Chronos model pipeline.

        Note: Chronos is a pretrained model, so 'build' loads the pretrained weights.
        No architecture construction is needed.
        """
        model_size = self.hyperparameters.get('model_size', 'small')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Map model size to Hugging Face model ID
        model_id_map = {
            'tiny': 'amazon/chronos-t5-tiny',
            'mini': 'amazon/chronos-t5-mini',
            'small': 'amazon/chronos-t5-small',
            'base': 'amazon/chronos-t5-base',
            'large': 'amazon/chronos-t5-large'
        }

        model_id = model_id_map.get(model_size, 'amazon/chronos-t5-small')

        print(f"Loading Chronos model: {model_id} on {device}...")
        self.model = ChronosPipeline.from_pretrained(
            model_id,
            device_map=device,
            torch_dtype=torch.bfloat16 if device == 'cuda' else torch.float32
        )

        self.metadata['model_id'] = model_id
        self.metadata['device'] = device

        print(f"✓ Chronos model loaded: {model_size} ({model_id})")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> Dict:
        """
        Prepare Chronos for forecasting.

        Note: Chronos is a zero-shot model and doesn't require traditional training.
        This method stores the training data as context for forecasting.

        Args:
            X_train: Training features (used for feature names only)
            y_train: Training target (historical time series for context)
            X_val: Validation features (optional, not used)
            y_val: Validation target (optional, not used)

        Returns:
            training_history: Dict with metadata
        """
        print(f"Preparing Chronos for forecasting on {len(y_train)} historical samples...")

        start_time = time.time()

        # Store the historical data as context
        # Chronos uses the historical target values for forecasting
        self.context_data = y_train.values
        self.feature_names = list(X_train.columns)
        self.is_trained = True

        # Get context length from hyperparameters
        context_length = self.hyperparameters.get('context_length', 512)

        # Store metadata
        self.metadata['training_time_seconds'] = time.time() - start_time
        self.metadata['n_features'] = X_train.shape[1]
        self.metadata['n_training_samples'] = len(X_train)
        self.metadata['context_length'] = min(context_length, len(y_train))
        self.metadata['feature_names'] = self.feature_names
        self.metadata['model_type'] = 'zero-shot'

        print(f"✓ Chronos prepared in {self.metadata['training_time_seconds']:.2f} seconds")
        print(f"  Context length: {self.metadata['context_length']} samples")
        print(f"  Note: Chronos is zero-shot - no gradient-based training performed")

        return {
            'training_time': self.metadata['training_time_seconds'],
            'context_length': self.metadata['context_length'],
            'model_type': 'zero-shot'
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using Chronos.

        Chronos uses the historical context to forecast future values.
        The prediction length is determined by the number of rows in X.

        Args:
            X: Features to predict on (length determines forecast horizon)

        Returns:
            Predictions as numpy array (median of sample paths)
        """
        if not self.is_trained:
            raise ValueError("Model must be prepared (call train()) before making predictions!")

        prediction_length = len(X)
        context_length = self.hyperparameters.get('context_length', 512)
        num_samples = self.hyperparameters.get('num_samples', 20)
        temperature = self.hyperparameters.get('temperature', 1.0)
        top_k = self.hyperparameters.get('top_k', 50)
        top_p = self.hyperparameters.get('top_p', 1.0)

        # Use the most recent context_length samples
        context = self.context_data[-context_length:] if len(self.context_data) > context_length else self.context_data

        # Convert to torch tensor
        context_tensor = torch.tensor(context, dtype=torch.float32)

        # Generate forecast
        forecast = self.model.predict(
            context=context_tensor,
            prediction_length=prediction_length,
            num_samples=num_samples,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

        # forecast shape: [num_samples, prediction_length]
        # Use median across samples as point forecast
        predictions = np.median(forecast.numpy(), axis=0)

        return predictions

    def predict_quantiles(self, X: pd.DataFrame, quantiles: list = [0.1, 0.5, 0.9]) -> Dict[str, np.ndarray]:
        """
        Make probabilistic predictions with quantiles.

        Args:
            X: Features to predict on
            quantiles: List of quantiles to compute (e.g., [0.1, 0.5, 0.9])

        Returns:
            Dict mapping quantile to predictions
            Example: {0.1: array([...]), 0.5: array([...]), 0.9: array([...])}
        """
        if not self.is_trained:
            raise ValueError("Model must be prepared before making predictions!")

        prediction_length = len(X)
        context_length = self.hyperparameters.get('context_length', 512)
        num_samples = self.hyperparameters.get('num_samples', 100)  # More samples for better quantile estimation

        context = self.context_data[-context_length:] if len(self.context_data) > context_length else self.context_data
        context_tensor = torch.tensor(context, dtype=torch.float32)

        # Generate forecast
        forecast = self.model.predict(
            context=context_tensor,
            prediction_length=prediction_length,
            num_samples=num_samples
        )

        # Compute quantiles across samples
        forecast_np = forecast.numpy()  # Shape: [num_samples, prediction_length]

        quantile_forecasts = {}
        for q in quantiles:
            quantile_forecasts[q] = np.quantile(forecast_np, q, axis=0)

        return quantile_forecasts

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.

        Note: Chronos is a foundation model that uses only the historical
        target values for forecasting. It doesn't use external features,
        so traditional feature importance is not applicable.

        Returns:
            DataFrame indicating Chronos doesn't compute feature importance
        """
        return pd.DataFrame({
            'feature': ['N/A'],
            'importance': [0.0],
            'note': ['Chronos uses historical target values only (zero-shot forecasting)']
        })

    def update_context(self, new_data: np.ndarray):
        """
        Update the historical context with new observations.
        Useful for rolling forecasts.

        Args:
            new_data: New observations to append to context
        """
        if self.context_data is None:
            self.context_data = new_data
        else:
            self.context_data = np.concatenate([self.context_data, new_data])

        print(f"✓ Context updated: {len(self.context_data)} total samples")
