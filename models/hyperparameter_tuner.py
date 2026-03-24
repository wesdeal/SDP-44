"""
Hyperparameter Tuner - Automatic hyperparameter optimization using Optuna
"""

from typing import Dict, Any, Optional
import optuna
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from .model_registry import ModelRegistry
from core.metric_engine import compute_all

# Optimization direction per metric, derived from evaluation_protocol_agent._METRIC_CATALOG.
# True = higher is better (maximize); False = lower is better (minimize).
_METRIC_HIGHER_IS_BETTER: Dict[str, bool] = {
    "rmse": False,
    "mae": False,
    "mape": False,
    "smape": False,
    "pinball_loss": False,
    "interval_width_80": False,
    "r2": True,
    "accuracy": True,
    "f1_weighted": True,
    "roc_auc": True,
    "coverage_80": True,
}


class HyperparameterTuner:
    """
    Automatic hyperparameter optimization using Bayesian optimization.

    Uses Optuna's TPE (Tree-structured Parzen Estimator) sampler to
    intelligently search the hyperparameter space.
    """

    def __init__(
        self,
        model_name: str,
        data_splits: Dict,
        n_trials: int = 50,
        primary_metric: str = "rmse",
        task_type: Optional[str] = None,
    ):
        """
        Args:
            model_name: Which model to tune (e.g., "XGBoost")
            data_splits: Dict with train/val/test data
            n_trials: Number of trials to run
            primary_metric: Metric name from eval_protocol.json (e.g. "rmse", "roc_auc")
            task_type: Pipeline task type; required when tuning LinearModel
        """
        self.model_name = model_name
        self.data_splits = data_splits
        self.n_trials = n_trials
        self.primary_metric = primary_metric
        self.task_type = task_type
        self.best_params = None
        self.best_score = None
        higher_is_better = _METRIC_HIGHER_IS_BETTER.get(primary_metric, False)
        self._direction = "maximize" if higher_is_better else "minimize"

    def tune(self) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.

        Returns:
            Best hyperparameters found
        """
        print(f"  🔍 Tuning hyperparameters ({self.n_trials} trials)...")

        # Create optimization study
        study = optuna.create_study(
            direction=self._direction,
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

        print(f"  ✓ Best validation {self.primary_metric}: {self.best_score:.4f}")
        print(f"  ✓ Best params: {self.best_params}")

        return self.best_params

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for optimization.

        Args:
            trial: Optuna trial object

        Returns:
            Validation metric value (direction determined by primary_metric)
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

            # Evaluate on validation set using primary_metric
            y_val_pred = model.predict(self.data_splits['X_val'])
            metrics = compute_all(
                self.data_splits['y_val'],
                y_val_pred,
                [self.primary_metric],
            )
            score = metrics.get(self.primary_metric)
            if score is None:
                return float('inf') if self._direction == "minimize" else float('-inf')
            return score

        except Exception as e:
            # If training fails, return sentinel penalty in the worst direction
            print(f"    Trial failed: {e}")
            return float('inf') if self._direction == "minimize" else float('-inf')

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

        elif self.model_name == 'LinearModel':
            # task_type is a required fixed parameter (not tuned); default to regression
            task_type = self.task_type or "tabular_regression"
            if task_type == "tabular_regression":
                return {
                    'task_type': task_type,
                    'alpha': trial.suggest_float('alpha', 1e-4, 100.0, log=True),
                    'random_state': 42,
                }
            else:  # tabular_classification
                return {
                    'task_type': task_type,
                    'C': trial.suggest_float('C', 1e-4, 100.0, log=True),
                    'max_iter': trial.suggest_int('max_iter', 100, 2000),
                    'random_state': 42,
                }

        elif self.model_name == 'SVR':
            return {
                'C': trial.suggest_float('C', 1e-2, 1e3, log=True),
                'epsilon': trial.suggest_float('epsilon', 1e-3, 1.0, log=True),
                'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear']),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            }

        elif self.model_name == 'DummyRegressor':
            # strategy is the only meaningful hyperparameter for DummyRegressor
            return {
                'strategy': trial.suggest_categorical('strategy', ['mean', 'median']),
                'random_state': 42,
            }

        else:
            return {}
