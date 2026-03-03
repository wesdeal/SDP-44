"""
Model Trainer - Main orchestrator for training pipeline
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
import json
from datetime import datetime
from sklearn.model_selection import train_test_split

import sys
sys.path.append(str(Path(__file__).parent))

from .model_registry import ModelRegistry
from .evaluator import ModelEvaluator
from .hyperparameter_tuner import HyperparameterTuner


class ModelTrainer:
    """
    Main training orchestrator.

    Workflow:
    1. Load preprocessed data + metadata
    2. Split into train/val/test
    3. For each recommended model:
       a. Tune hyperparameters (optional)
       b. Train model
       c. Evaluate on test set
       d. Save model + results
    4. Generate comparison report
    """

    def __init__(self, pipeline_outputs_dir: Path):
        """
        Args:
            pipeline_outputs_dir: Path to Pipeline/outputs/
        """
        self.outputs_dir = Path(pipeline_outputs_dir)
        self.models_dir = self.outputs_dir.parent / "models"
        self.results_dir = self.models_dir / "results"
        self.evaluator = ModelEvaluator()

        # Create results directories
        (self.results_dir / "trained_models").mkdir(parents=True, exist_ok=True)
        (self.results_dir / "evaluation_reports").mkdir(parents=True, exist_ok=True)
        (self.results_dir / "experiments").mkdir(parents=True, exist_ok=True)

    def load_preprocessed_data(self, dataset_name: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Load preprocessed CSV and metadata JSON.

        Args:
            dataset_name: e.g., "ETTh1"

        Returns:
            (dataframe, metadata_dict)
        """
        # Load cleaned data
        data_path = self.outputs_dir / f"{dataset_name}_processed.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"Processed data not found: {data_path}")

        df = pd.read_csv(data_path)
        print(f"✓ Loaded data: {df.shape}")

        # Load metadata
        metadata_path = self.outputs_dir / f"{dataset_name}_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        print(f"✓ Loaded metadata")

        return df, metadata

    def prepare_data_splits(self, df: pd.DataFrame, target_col: str,
                           test_size=0.2, val_size=0.1,
                           is_time_series=True) -> Dict:
        """
        Split data into train/validation/test sets.

        Args:
            df: Full dataset
            target_col: Name of target column
            test_size: Fraction for test set (default 20%)
            val_size: Fraction of remaining for validation (default 10%)
            is_time_series: If True, use chronological split

        Returns:
            Dict with X_train, y_train, X_val, y_val, X_test, y_test
        """
        print(f"\nPreparing data splits (test={test_size}, val={val_size})...")

        # Remove non-feature columns
        feature_cols = [col for col in df.columns if col not in [target_col, 'date']]
        X = df[feature_cols]
        y = df[target_col]

        print(f"  Features: {list(X.columns)}")
        print(f"  Target: {target_col}")

        if is_time_series:
            # Chronological split
            n = len(df)
            test_idx = int(n * (1 - test_size))
            val_idx = int(test_idx * (1 - val_size))

            X_train, X_val, X_test = X[:val_idx], X[val_idx:test_idx], X[test_idx:]
            y_train, y_val, y_test = y[:val_idx], y[val_idx:test_idx], y[test_idx:]

            print(f"  Time series split (chronological):")
        else:
            # Random split
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size, random_state=42
            )

            print(f"  Random split (shuffled):")

        print(f"    Train: {len(X_train)} samples")
        print(f"    Val:   {len(X_val)} samples")
        print(f"    Test:  {len(X_test)} samples")

        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }

    def train_all_recommended_models(self, dataset_name: str,
                                     tune_hyperparameters: bool = True,
                                     n_trials: int = 20) -> Dict:
        """
        MAIN TRAINING FUNCTION

        Trains all models recommended by LLM metadata extractor.

        Args:
            dataset_name: e.g., "ETTh1"
            tune_hyperparameters: If True, run auto hyperparameter optimization
            n_trials: Number of trials for tuning

        Returns:
            Dict of results for each model
        """
        print(f"\n{'='*70}")
        print(f"  TRAINING MODELS FOR: {dataset_name}")
        print(f"{'='*70}\n")

        # Step 1: Load data and metadata
        df, metadata = self.load_preprocessed_data(dataset_name)

        target_col = metadata['metadata']['target_variable']
        recommended_models = metadata['metadata']['recommended_models']

        # Filter to only models we have implemented
        available_models = ModelRegistry.list_available_models()
        recommended_models = [m for m in recommended_models if m in available_models]

        print(f"\nTarget Variable: {target_col}")
        print(f"Recommended Models: {recommended_models}")
        print(f"Available Models: {available_models}")

        if not recommended_models:
            print("\n⚠ No recommended models are implemented yet!")
            return {}

        # Step 2: Prepare data splits
        data_splits = self.prepare_data_splits(df, target_col)

        # Step 3: Train each recommended model
        results = {}

        for i, model_name in enumerate(recommended_models, 1):
            print(f"\n{'─'*70}")
            print(f"[{i}/{len(recommended_models)}] Training: {model_name}")
            print(f"{'─'*70}")

            try:
                # Step 3a: Get/tune hyperparameters
                if tune_hyperparameters:
                    print(f"\n🔧 Tuning hyperparameters...")
                    tuner = HyperparameterTuner(
                        model_name,
                        data_splits,
                        n_trials=n_trials
                    )
                    hyperparameters = tuner.tune()
                else:
                    hyperparameters = self._get_default_hyperparameters(model_name)
                    print(f"\n📋 Using default hyperparameters: {hyperparameters}")

                # Step 3b: Create and train model
                model = ModelRegistry.create_model(model_name, hyperparameters)
                model.build()

                training_history = model.train(
                    data_splits['X_train'],
                    data_splits['y_train'],
                    data_splits['X_val'],
                    data_splits['y_val']
                )

                # Step 3c: Evaluate on test set
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                eval_save_path = self.results_dir / "evaluation_reports" / f"{dataset_name}_{model_name}_{timestamp}"

                evaluation_results = self.evaluator.evaluate(
                    model,
                    data_splits['X_test'],
                    data_splits['y_test'],
                    save_path=eval_save_path
                )

                # Step 3d: Save trained model
                model_save_path = self.results_dir / "trained_models" / f"{dataset_name}_{model_name}_{timestamp}"
                model.save(model_save_path)

                # Store results
                results[model_name] = {
                    'status': 'success',
                    'hyperparameters': hyperparameters,
                    'training_history': training_history,
                    'evaluation': evaluation_results,
                    'model_path': str(model_save_path),
                    'eval_path': str(eval_save_path),
                    'timestamp': timestamp
                }

                print(f"\n✓ {model_name} completed successfully!")
                print(f"  RMSE: {evaluation_results['rmse']:.4f}")
                print(f"  R²: {evaluation_results['r2']:.4f}")

            except Exception as e:
                import traceback
                print(f"\n✗ {model_name} failed: {str(e)}")
                traceback.print_exc()
                results[model_name] = {
                    'status': 'failed',
                    'error': str(e)
                }

        # Step 4: Generate comparison report
        print(f"\n{'='*70}")
        print("GENERATING COMPARISON REPORT")
        print(f"{'='*70}")

        comparison_path = self.results_dir / "experiments"
        self.evaluator.compare_models(save_path=comparison_path)

        # Save experiment summary
        experiment_file = comparison_path / f"{dataset_name}_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(experiment_file, 'w') as f:
            json.dump({
                'dataset': dataset_name,
                'timestamp': datetime.now().isoformat(),
                'results': results
            }, f, indent=2, default=str)

        print(f"\n✓ All training complete!")
        print(f"✓ Results saved to: {self.results_dir}")

        return results

    def _get_default_hyperparameters(self, model_name: str) -> Dict:
        """Get sensible default hyperparameters for each model."""
        defaults = {
            'XGBoost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            },
            'RandomForest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            }
        }

        return defaults.get(model_name, {})
