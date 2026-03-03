"""
Hyperparameter Tuner - Automatic hyperparameter optimization using Optuna
"""

from typing import Dict, Any
import optuna
from sklearn.metrics import mean_squared_error
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from .model_registry import ModelRegistry


class HyperparameterTuner:
    """
    Automatic hyperparameter optimization using Bayesian optimization.

    Uses Optuna's TPE (Tree-structured Parzen Estimator) sampler to
    intelligently search the hyperparameter space.
    """

    def __init__(self, model_name: str, data_splits: Dict, n_trials: int = 50):
        """
        Args:
            model_name: Which model to tune (e.g., "XGBoost")
            data_splits: Dict with train/val/test data
            n_trials: Number of trials to run
        """
        self.model_name = model_name
        self.data_splits = data_splits
        self.n_trials = n_trials
        self.best_params = None
        self.best_score = float('inf')

    def tune(self) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.

        Returns:
            Best hyperparameters found
        """
        print(f"  🔍 Tuning hyperparameters ({self.n_trials} trials)...")

        # Create optimization study
        study = optuna.create_study(
            direction='minimize',  # Minimize validation RMSE
            sampler=optuna.samplers.TPESampler(seed=42)
        )

        # Run optimization
        study.optimize(
            self._objective,
            n_trials=self.n_trials,
            show_progress_bar=False,  # Set to True for progress bar
            n_jobs=1
        )

        # Get best result
        self.best_params = study.best_params
        self.best_score = study.best_value

        print(f"  ✓ Best validation RMSE: {self.best_score:.4f}")
        print(f"  ✓ Best params: {self.best_params}")

        return self.best_params

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for optimization.

        Args:
            trial: Optuna trial object

        Returns:
            Validation RMSE (lower is better)
        """
        # Get hyperparameters for this trial
        hyperparameters = self._get_search_space(trial)

        try:
            # Create and train model
            model = ModelRegistry.create_model(self.model_name, hyperparameters)
            model.build()

            model.train(
                self.data_splits['X_train'],
                self.data_splits['y_train'],
                self.data_splits['X_val'],
                self.data_splits['y_val']
            )

            # Evaluate on validation set
            y_val_pred = model.predict(self.data_splits['X_val'])
            val_rmse = np.sqrt(mean_squared_error(
                self.data_splits['y_val'],
                y_val_pred
            ))

            return val_rmse

        except Exception as e:
            # If training fails, return large penalty
            print(f"    Trial failed: {e}")
            return float('inf')

    def _get_search_space(self, trial: optuna.Trial) -> Dict:
        """
        Define hyperparameter search space for each model.

        Args:
            trial: Optuna trial

        Returns:
            Dict of hyperparameters for this trial
        """

        if self.model_name == 'XGBoost':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': 42
            }

        elif self.model_name == 'RandomForest':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.8]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'random_state': 42
            }

        elif self.model_name == 'Chronos':
            return {
                'model_size': trial.suggest_categorical('model_size', ['tiny', 'mini', 'small', 'base']),
                'prediction_length': trial.suggest_categorical('prediction_length', [12, 24, 48, 96]),
                'context_length': trial.suggest_categorical('context_length', [256, 512, 1024]),
                'num_samples': trial.suggest_int('num_samples', 10, 50),
                'temperature': trial.suggest_float('temperature', 0.5, 1.5),
                'top_k': trial.suggest_int('top_k', 20, 100),
                'top_p': trial.suggest_float('top_p', 0.8, 1.0)
            }

        else:
            return {}
