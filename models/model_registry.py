"""
Model Registry - Central registry for all available models
Maps model names (from metadata) to Python classes
"""

from typing import Dict, Type
from .base_model import BaseModel
from .tree_based.xgboost_model import XGBoostModel
from .tree_based.random_forest_model import RandomForestModel

# Try to import Chronos (optional dependency)
try:
    from .time_series.chronos_model import ChronosModel
    CHRONOS_AVAILABLE = True
except Exception as e:
    CHRONOS_AVAILABLE = False
    print(f"Note: Chronos model not available - {e}")


class ModelRegistry:
    """
    Central registry mapping model names to implementations.

    Usage:
        model = ModelRegistry.create_model("XGBoost", hyperparameters)
        model.build()
        model.train(X_train, y_train)
    """

    # Dictionary mapping string names to Python classes
    _registry: Dict[str, Type[BaseModel]] = {
        'XGBoost': XGBoostModel,
        'RandomForest': RandomForestModel,
        # Add LSTM and ARIMA when implemented
        # 'LSTM': LSTMModel,
        # 'ARIMA': ARIMAModel,
    }

    # Add Chronos if available
    if CHRONOS_AVAILABLE:
        _registry['Chronos'] = ChronosModel

    @classmethod
    def get_model_class(cls, model_name: str) -> Type[BaseModel]:
        """
        Look up a model class by name.

        Args:
            model_name: String like "XGBoost" (from metadata)

        Returns:
            The Python class (not an instance)

        Raises:
            ValueError: If model not found in registry
        """
        # Normalize name (handle case variations)
        normalized_name = model_name.strip()

        if normalized_name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Model '{model_name}' not found in registry.\n"
                f"Available models: {available}"
            )

        return cls._registry[normalized_name]

    @classmethod
    def create_model(cls, model_name: str, hyperparameters: Dict) -> BaseModel:
        """
        Factory method: Create and return a model instance.

        Args:
            model_name: Name of model (e.g., "XGBoost")
            hyperparameters: Dict of hyperparameters

        Returns:
            Initialized model instance (ready to build and train)
        """
        model_class = cls.get_model_class(model_name)
        model_instance = model_class(model_name, hyperparameters)
        return model_instance

    @classmethod
    def list_available_models(cls) -> list:
        """
        Get list of all registered models.

        Returns:
            List of model names
        """
        return list(cls._registry.keys())

    @classmethod
    def register_model(cls, model_name: str, model_class: Type[BaseModel]):
        """
        Dynamically add a new model to the registry.
        Useful for plugins or custom models.

        Args:
            model_name: Name for the model
            model_class: Class that inherits from BaseModel

        Raises:
            TypeError: If model_class doesn't inherit from BaseModel
        """
        if not issubclass(model_class, BaseModel):
            raise TypeError(f"{model_class} must inherit from BaseModel")

        cls._registry[model_name] = model_class
        print(f"✓ Registered model: {model_name}")
