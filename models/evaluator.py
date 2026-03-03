"""
Model Evaluator - Calculate metrics and generate diagnostic plots
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict
import json


class ModelEvaluator:
    """
    Comprehensive model evaluation with metrics and visualizations.
    """

    def __init__(self):
        self.metrics_history = []

    def evaluate(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                 save_path: Path = None) -> Dict:
        """
        Full evaluation pipeline.

        Args:
            model: Trained model with .predict() method
            X_test: Test features
            y_test: Test target (ground truth)
            save_path: Where to save plots (optional)

        Returns:
            Dictionary of metrics
        """
        print(f"\nEvaluating {model.model_name}...")

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred)
        metrics['model_name'] = model.model_name
        metrics['n_test_samples'] = len(y_test)

        # Print summary
        self._print_metrics(metrics)

        # Generate plots
        if save_path:
            self.generate_plots(y_test, y_pred, model.model_name, save_path)

        # Store in history
        self.metrics_history.append(metrics)

        return metrics

    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict:
        """Calculate comprehensive metrics."""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        try:
            mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        except:
            mape = None

        r2 = r2_score(y_true, y_pred)
        residuals = y_true - y_pred

        metrics = {
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape) if mape is not None else None,
            'r2': float(r2),
            'mean_residual': float(residuals.mean()),
            'std_residual': float(residuals.std()),
            'max_error': float(np.abs(residuals).max()),
        }

        return metrics

    def _print_metrics(self, metrics: Dict):
        """Pretty print metrics."""
        print(f"\n{'='*50}")
        print(f"  Model: {metrics['model_name']}")
        print(f"{'='*50}")
        print(f"  RMSE:  {metrics['rmse']:.4f}")
        print(f"  MAE:   {metrics['mae']:.4f}")
        if metrics['mape']:
            print(f"  MAPE:  {metrics['mape']:.2f}%")
        print(f"  R²:    {metrics['r2']:.4f}")
        print(f"{'='*50}\n")

    def generate_plots(self, y_true, y_pred, model_name: str, save_path: Path):
        """Generate diagnostic plots."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{model_name} - Evaluation', fontsize=16, fontweight='bold')

        # Plot 1: Predictions vs Actual
        axes[0, 0].scatter(y_true, y_pred, alpha=0.5, s=20)
        min_val, max_val = y_true.min(), y_true.max()
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual')
        axes[0, 0].set_ylabel('Predicted')
        axes[0, 0].set_title('Predictions vs Actual')
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Residuals
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5, s=20)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Residuals Distribution
        axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1, 0].set_xlabel('Residual')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Residuals Distribution')

        # Plot 4: Time Series
        n_points = min(200, len(y_true))
        axes[1, 1].plot(y_true.values[:n_points], label='Actual', lw=2, alpha=0.7)
        axes[1, 1].plot(y_pred[:n_points], label='Predicted', lw=2, alpha=0.7)
        axes[1, 1].set_xlabel('Time Steps')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_title(f'Predictions Over Time (first {n_points} points)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = save_path / f'{model_name}_evaluation.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Plots saved to {plot_path}")

    def compare_models(self, save_path: Path = None) -> pd.DataFrame:
        """Compare all evaluated models."""
        if not self.metrics_history:
            return pd.DataFrame()

        comparison = pd.DataFrame(self.metrics_history)
        comparison = comparison.sort_values('rmse')

        print("\n" + "="*80)
        print("MODEL COMPARISON (sorted by RMSE)")
        print("="*80)
        print(comparison[['model_name', 'rmse', 'mae', 'r2']].to_string(index=False))
        print("="*80 + "\n")

        if save_path:
            comparison_file = Path(save_path) / 'model_comparison.json'
            comparison.to_json(comparison_file, indent=2, orient='records')
            print(f"✓ Comparison saved to {comparison_file}")

        return comparison
