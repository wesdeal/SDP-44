"""
Test script to verify Chronos model integration with the pipeline
"""

import sys
from pathlib import Path

# Add Pipeline directory to path
pipeline_dir = Path(__file__).parent
sys.path.insert(0, str(pipeline_dir))
sys.path.insert(0, str(pipeline_dir / "models"))

import pandas as pd
import numpy as np
from models.model_registry import ModelRegistry

def test_chronos_basic():
    """Test basic Chronos functionality"""
    print("=" * 60)
    print("TEST 1: Basic Chronos Model Creation and Build")
    print("=" * 60)

    try:
        # Test model creation
        hyperparameters = {
            'model_size': 'tiny',  # Use tiny for faster testing
            'prediction_length': 12,
            'context_length': 256,
            'num_samples': 20
        }

        model = ModelRegistry.create_model("Chronos", hyperparameters)
        print(f"✓ Model created: {model}")

        # Test model build (loads pretrained weights)
        print("\nBuilding Chronos model (loading pretrained weights)...")
        model.build()
        print(f"✓ Model built successfully")
        print(f"  Metadata: {model.metadata}")

        return True

    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("\nTo use Chronos, install it with:")
        print("  pip install git+https://github.com/amazon-science/chronos-forecasting.git")
        return False
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chronos_train_predict():
    """Test Chronos training (context preparation) and prediction"""
    print("\n" + "=" * 60)
    print("TEST 2: Chronos Train and Predict")
    print("=" * 60)

    try:
        # Create synthetic time series data
        np.random.seed(42)
        n_samples = 500

        # Generate time series with trend and seasonality
        t = np.arange(n_samples)
        trend = 0.05 * t
        seasonality = 10 * np.sin(2 * np.pi * t / 50)
        noise = np.random.normal(0, 2, n_samples)
        y = trend + seasonality + noise + 50

        # Create DataFrame (Chronos doesn't use features, but we need X for interface consistency)
        X = pd.DataFrame({
            'lag_1': np.roll(y, 1),
            'lag_2': np.roll(y, 2),
            'trend': t
        })
        X.iloc[:2] = 0  # Handle initial lags
        y_series = pd.Series(y, name='target')

        # Split data
        train_size = int(0.7 * n_samples)
        val_size = int(0.1 * n_samples)

        X_train = X[:train_size]
        y_train = y_series[:train_size]
        X_val = X[train_size:train_size+val_size]
        y_val = y_series[train_size:train_size+val_size]
        X_test = X[train_size+val_size:]
        y_test = y_series[train_size+val_size:]

        print(f"\nData splits:")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Val:   {len(X_val)} samples")
        print(f"  Test:  {len(X_test)} samples")

        # Create and build model
        hyperparameters = {
            'model_size': 'tiny',
            'prediction_length': len(X_test),
            'context_length': 256,
            'num_samples': 20
        }

        model = ModelRegistry.create_model("Chronos", hyperparameters)
        model.build()

        # Train (prepare context)
        print("\nPreparing Chronos for forecasting...")
        train_result = model.train(X_train, y_train, X_val, y_val)
        print(f"✓ Training result: {train_result}")

        # Predict
        print("\nGenerating predictions...")
        y_pred = model.predict(X_test)

        print(f"✓ Predictions generated: {len(y_pred)} values")
        print(f"  Prediction range: [{y_pred.min():.2f}, {y_pred.max():.2f}]")
        print(f"  Actual range:     [{y_test.min():.2f}, {y_test.max():.2f}]")

        # Calculate simple metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        print(f"\nMetrics:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")

        return True

    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chronos_registry():
    """Test that Chronos is properly registered"""
    print("\n" + "=" * 60)
    print("TEST 3: Model Registry Integration")
    print("=" * 60)

    try:
        available_models = ModelRegistry.list_available_models()
        print(f"Available models: {available_models}")

        if 'Chronos' in available_models:
            print("✓ Chronos is registered in ModelRegistry")
        else:
            print("✗ Chronos is NOT registered in ModelRegistry")
            return False

        # Test getting model class
        chronos_class = ModelRegistry.get_model_class('Chronos')
        print(f"✓ Chronos class retrieved: {chronos_class}")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chronos_quantile_forecasting():
    """Test Chronos probabilistic forecasting"""
    print("\n" + "=" * 60)
    print("TEST 4: Chronos Quantile Forecasting")
    print("=" * 60)

    try:
        # Create synthetic data
        np.random.seed(42)
        n_samples = 300
        t = np.arange(n_samples)
        y = 10 * np.sin(2 * np.pi * t / 50) + 50 + np.random.normal(0, 2, n_samples)

        X = pd.DataFrame({'time': t})
        y_series = pd.Series(y, name='target')

        # Split
        train_size = 250
        X_train = X[:train_size]
        y_train = y_series[:train_size]
        X_test = X[train_size:]

        # Create model
        hyperparameters = {
            'model_size': 'tiny',
            'prediction_length': len(X_test),
            'context_length': 200,
            'num_samples': 50  # More samples for better quantile estimates
        }

        model = ModelRegistry.create_model("Chronos", hyperparameters)
        model.build()
        model.train(X_train, y_train)

        # Get quantile forecasts
        print("\nGenerating quantile forecasts (10th, 50th, 90th percentiles)...")
        quantile_forecasts = model.predict_quantiles(X_test, quantiles=[0.1, 0.5, 0.9])

        print(f"✓ Quantile forecasts generated:")
        for q, forecast in quantile_forecasts.items():
            print(f"  {int(q*100)}th percentile: {len(forecast)} values, range [{forecast.min():.2f}, {forecast.max():.2f}]")

        # Verify median matches standard prediction
        median_pred = quantile_forecasts[0.5]
        standard_pred = model.predict(X_test)

        diff = np.abs(median_pred - standard_pred).mean()
        print(f"\n  Difference between median and standard prediction: {diff:.4f}")

        if diff < 0.1:
            print("✓ Median prediction matches standard prediction (as expected)")

        return True

    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("CHRONOS MODEL INTEGRATION TESTS")
    print("=" * 60)

    results = []

    # Test 1: Registry integration
    results.append(("Registry Integration", test_chronos_registry()))

    # Test 2: Basic creation and build
    results.append(("Basic Creation & Build", test_chronos_basic()))

    # Test 3: Train and predict (only if basic test passed)
    if results[-1][1]:
        results.append(("Train & Predict", test_chronos_train_predict()))

        # Test 4: Quantile forecasting (only if train/predict passed)
        if results[-1][1]:
            results.append(("Quantile Forecasting", test_chronos_quantile_forecasting()))

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:30s} {status}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        print("\nChronos is fully integrated and ready to use.")
        print("\nNext steps:")
        print("  1. Run preprocessing_pipeline.py on your dataset")
        print("  2. The LLM may now recommend Chronos as a model")
        print("  3. Run trainer.py to train all recommended models")
        print("  4. Compare Chronos with other models in results/")
    else:
        print("\n" + "=" * 60)
        print("SOME TESTS FAILED")
        print("=" * 60)
        print("\nIf you see import errors, install Chronos:")
        print("  pip install git+https://github.com/amazon-science/chronos-forecasting.git")
        print("\nIf you see other errors, check the traceback above.")


if __name__ == "__main__":
    main()
