# Model Training Pipeline

Automated model training and hyperparameter optimization for time series forecasting.

## Overview

This model training stage automatically:
1. Loads preprocessed data from the preprocessing pipeline
2. Reads LLM-recommended models from metadata
3. Performs Bayesian hyperparameter optimization
4. Trains and evaluates each model
5. Generates comparison reports and visualizations

## Directory Structure

```
models/
├── __init__.py                    # Package initialization
├── base_model.py                  # Abstract base class for all models
├── model_registry.py              # Central model registry
├── trainer.py                     # Main training orchestrator
├── evaluator.py                   # Model evaluation and metrics
├── hyperparameter_tuner.py        # Automatic hyperparameter optimization
│
├── tree_based/                    # Tree-based models
│   ├── __init__.py
│   ├── xgboost_model.py          # XGBoost implementation
│   └── random_forest_model.py    # Random Forest implementation
│
├── time_series/                   # Time series specific models
│   ├── __init__.py
│   ├── lstm_model.py             # LSTM (to be implemented)
│   └── arima_model.py            # ARIMA (to be implemented)
│
└── results/                       # Training outputs
    ├── trained_models/            # Saved .pkl model files
    ├── evaluation_reports/        # Metrics and plots
    └── experiments/               # Experiment tracking
```

## Quick Start

### Basic Usage

```python
from models.trainer import ModelTrainer

# Initialize trainer
trainer = ModelTrainer("Pipeline/outputs")

# Train all recommended models
results = trainer.train_all_recommended_models(
    dataset_name="ETTh1",
    tune_hyperparameters=True,
    n_trials=50
)
```

### Run Test Script

```bash
cd Pipeline
python test_model_training.py
```

## How It Works

### 1. Data Loading
- Reads `{dataset}_processed.csv` from outputs/
- Reads `{dataset}_metadata.json` for recommended models

### 2. Data Splitting
- **Time Series**: Chronological split (70% train, 10% val, 20% test)
- **Tabular**: Random split with same proportions

### 3. Hyperparameter Tuning (Optional)
- Uses **Optuna** for Bayesian optimization
- TPE (Tree-structured Parzen Estimator) sampler
- Minimizes validation RMSE
- Configurable number of trials (default: 20)

### 4. Model Training
- Instantiates model with best hyperparameters
- Trains on training set
- Monitors performance on validation set
- Saves trained model (.pkl file)

### 5. Evaluation
- Calculates metrics on test set:
  - **RMSE**: Root Mean Squared Error
  - **MAE**: Mean Absolute Error
  - **MAPE**: Mean Absolute Percentage Error
  - **R²**: Coefficient of Determination
- Generates diagnostic plots:
  - Predictions vs Actual scatter
  - Residuals plot
  - Residuals distribution
  - Time series comparison

### 6. Results Saving
- Trained models → `results/trained_models/`
- Evaluation plots → `results/evaluation_reports/`
- Experiment logs → `results/experiments/`
- Model comparison JSON

## Hyperparameter Search Spaces

### XGBoost
```python
{
    'n_estimators': [50, 500],        # Number of trees
    'max_depth': [3, 15],             # Tree depth
    'learning_rate': [0.01, 0.3],     # Step size (log scale)
    'subsample': [0.6, 1.0],          # Sample fraction per tree
    'colsample_bytree': [0.6, 1.0],   # Feature fraction per tree
    'reg_alpha': [1e-8, 10],          # L1 regularization
    'reg_lambda': [1e-8, 10],         # L2 regularization
}
```

### Random Forest
```python
{
    'n_estimators': [50, 300],
    'max_depth': [5, 30],
    'min_samples_split': [2, 20],
    'min_samples_leaf': [1, 10],
    'max_features': ['sqrt', 'log2', 0.5, 0.8],
    'bootstrap': [True, False]
}
```

## Adding New Models

### Step 1: Implement Model Class

```python
# models/tree_based/my_model.py

from ..base_model import BaseModel
import pandas as pd
import numpy as np

class MyModel(BaseModel):
    def build(self):
        # Initialize your model
        self.model = YourModelClass(**self.hyperparameters)
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        # Training logic
        self.model.fit(X_train, y_train)
        self.is_trained = True
        return {'training_time': ...}
    
    def predict(self, X):
        return self.model.predict(X)
```

### Step 2: Register Model

```python
# models/model_registry.py

from .tree_based.my_model import MyModel

class ModelRegistry:
    _registry = {
        'XGBoost': XGBoostModel,
        'RandomForest': RandomForestModel,
        'MyModel': MyModel,  # Add here
    }
```

### Step 3: Add Hyperparameter Space

```python
# models/hyperparameter_tuner.py

def _get_search_space(self, trial):
    if self.model_name == 'MyModel':
        return {
            'param1': trial.suggest_int('param1', 10, 100),
            'param2': trial.suggest_float('param2', 0.01, 0.1),
        }
```

## Evaluation Metrics Explained

- **RMSE**: Average prediction error (same units as target). Lower is better.
- **MAE**: Average absolute error. More robust to outliers than RMSE.
- **MAPE**: Error as percentage of actual value. Good for relative comparison.
- **R²**: Variance explained by model (1 = perfect, 0 = baseline, <0 = worse than mean).

## Output Files

### Trained Models
```
results/trained_models/
└── ETTh1_XGBoost_20251202_151230/
    ├── XGBoost.pkl              # Serialized model
    └── XGBoost_metadata.json    # Hyperparameters + training info
```

### Evaluation Reports
```
results/evaluation_reports/
└── ETTh1_XGBoost_20251202_151230/
    └── XGBoost_evaluation.png   # 4-panel diagnostic plot
```

### Experiment Logs
```
results/experiments/
├── model_comparison.json         # Side-by-side metrics
└── ETTh1_experiment_20251202_151230.json  # Full experiment details
```

## Best Practices

1. **Always tune hyperparameters** for production models
2. **Use more trials** (50-100) for better results (but slower)
3. **Check evaluation plots** to diagnose issues:
   - Scattered residuals = good
   - Patterns in residuals = model missing something
   - Predictions track actual in time series plot = good temporal modeling
4. **Compare multiple models** before selecting best
5. **Save experiment metadata** for reproducibility

## Troubleshooting

### Issue: Model training fails
- Check that preprocessed data exists in `outputs/`
- Verify metadata JSON has `target_variable` and `recommended_models`
- Ensure required packages installed (`xgboost`, `sklearn`, `optuna`)

### Issue: Hyperparameter tuning is slow
- Reduce `n_trials` (but may hurt performance)
- Use simpler search spaces
- Enable parallel optimization: `n_jobs=-1` in Optuna

### Issue: Poor model performance
- Try more tuning trials
- Check preprocessing quality
- Verify data splits are appropriate
- Try different models

## Dependencies

```bash
pip install xgboost scikit-learn optuna pandas numpy matplotlib
```

## Future Enhancements

- [ ] Implement LSTM model
- [ ] Implement ARIMA model  
- [ ] Add early stopping for tree models
- [ ] Add cross-validation support
- [ ] Add ensemble methods
- [ ] Add feature importance analysis
- [ ] Add SHAP explanations
- [ ] Add MLflow integration for experiment tracking
