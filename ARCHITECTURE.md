# Pipeline Architecture - How Everything Connects

## Connection Between Stages

### The Bridge: `outputs/{dataset}_metadata.json`

This JSON file is the **contract** between preprocessing and model training stages.

```json
{
  "metadata": {
    "dataset_name": "ETTh1",
    "target_variable": "OT",  ← Models use this as y target
    "recommended_models": [    ← Models trainer loops through this
      "XGBoost",
      "RandomForest"
    ],
    "preprocessing_steps": [   ← Shows what was done to data
      {
        "method": "remove_outliers",
        "parameters": {"method": "iqr", "threshold": 1.5},
        "order": 1
      },
      ...
    ],
    "has_seasonality": true,   ← Helps models understand data
    "has_trend": true,
    "is_multivariate": true,
    "num_rows": 17420,
    "num_columns": 8
  }
}
```

---

## Code-Level Connection

### preprocessing_pipeline.py (Line 400-422)

```python
def main(input_file_path: str = "./inputs/ETTh1.csv"):
    # ... Extract metadata and preprocess ...
    
    # CRITICAL: Save to outputs directory
    outputs_dir = script_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    input_filename = Path(input_file_path).stem  # "ETTh1"
    
    # Save metadata
    metadata_output_path = outputs_dir / f"{input_filename}_metadata.json"
    with open(metadata_output_path, 'w') as f:
        f.write(json.dumps({
            'metadata': asdict(metadata),  # Contains recommended_models, target_variable
            'preprocessing_applied': preprocessing_methods
        }, indent=2, default=str))
    
    # Save cleaned data
    processed_data_path = outputs_dir / f"{input_filename}_processed.csv"
    df_processed.to_csv(processed_data_path, index=False)
    
    return metadata, df_processed
```

### models/trainer.py (Lines 45-75)

```python
def load_preprocessed_data(self, dataset_name: str):
    """
    Load preprocessed CSV and metadata JSON.
    
    This is where the connection happens!
    """
    # Load cleaned data
    data_path = self.outputs_dir / f"{dataset_name}_processed.csv"
    df = pd.read_csv(data_path)  # ← Reads preprocessed data
    
    # Load metadata
    metadata_path = self.outputs_dir / f"{dataset_name}_metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)  # ← Reads recommendations
    
    return df, metadata

def train_all_recommended_models(self, dataset_name: str, ...):
    # Load data and metadata
    df, metadata = self.load_preprocessed_data(dataset_name)
    
    # Extract key information from metadata
    target_col = metadata['metadata']['target_variable']  # ← Gets "OT"
    recommended_models = metadata['metadata']['recommended_models']  # ← Gets ["XGBoost", "RandomForest"]
    
    # Split data
    data_splits = self.prepare_data_splits(df, target_col)
    
    # Train each recommended model
    for model_name in recommended_models:  # ← Loops through LLM recommendations
        model = ModelRegistry.create_model(model_name, hyperparameters)
        model.train(data_splits['X_train'], data_splits['y_train'])
        # ... evaluate and save ...
```

---

## File System Connection

```
Pipeline/
│
├── preprocessing_pipeline.py
│   │
│   └─► WRITES TO ─────────────┐
│                               │
├── outputs/                    │ ← Bridge directory
│   ├── ETTh1_processed.csv    │ ← Cleaned data
│   └── ETTh1_metadata.json ◄──┘ ← Recommendations
│       │
│       └─► CONTAINS:
│           • target_variable: "OT"
│           • recommended_models: ["XGBoost", "RandomForest"]
│           • preprocessing_steps: [...]
│           • data characteristics
│
└── models/
    └── trainer.py
        │
        └─► READS FROM ────────┘
            • Loads ETTh1_processed.csv
            • Parses ETTh1_metadata.json
            • Trains recommended models
```

---

## Data Flow Example

### Step-by-Step Execution

1. **User runs:** `python run_full_pipeline.py --input inputs/ETTh1.csv`

2. **preprocessing_pipeline.py executes:**
   ```python
   # Line 382: Extract metadata
   metadata = extractor.extract_from_file("./inputs/ETTh1.csv")
   
   # LLM responds:
   # {
   #   "target_variable": "OT",
   #   "recommended_models": ["XGBoost", "RandomForest"],
   #   ...
   # }
   
   # Line 396: Apply preprocessing
   df_processed = pipeline.apply_from_metadata(df, metadata)
   
   # Line 417: Save outputs
   # outputs/ETTh1_processed.csv ← cleaned data
   # outputs/ETTh1_metadata.json ← recommendations
   ```

3. **models/trainer.py executes:**
   ```python
   # Line 55: Load from outputs/
   df, metadata = self.load_preprocessed_data("ETTh1")
   
   # Line 67: Extract recommendations
   target_col = metadata['metadata']['target_variable']  # "OT"
   recommended_models = ['XGBoost', 'RandomForest']
   
   # Line 90: For each model
   for model_name in recommended_models:
       # Line 96: Tune hyperparameters
       tuner = HyperparameterTuner(model_name, data_splits)
       best_params = tuner.tune()  # 20 trials
       
       # Line 105: Train
       model = ModelRegistry.create_model(model_name, best_params)
       model.train(X_train, y_train)
       
       # Line 115: Evaluate
       metrics = evaluator.evaluate(model, X_test, y_test)
       
       # Line 124: Save
       model.save(save_path)
   ```

4. **Results saved to:**
   ```
   models/results/
   ├── trained_models/
   │   ├── ETTh1_XGBoost_20251202/
   │   └── ETTh1_RandomForest_20251202/
   ├── evaluation_reports/
   │   └── ETTh1_XGBoost_20251202/
   │       └── XGBoost_evaluation.png
   └── experiments/
       └── model_comparison.json
   ```

---

## Variable Tracing

### Target Variable Path

```
inputs/ETTh1.csv
    │
    │ Column: "OT" (temperature)
    │
    ▼
[LLM Analysis]
    │
    │ Identifies: target_variable = "OT"
    │
    ▼
outputs/ETTh1_metadata.json
    │
    │ {
    │   "metadata": {
    │     "target_variable": "OT"  ← Stored here
    │   }
    │ }
    │
    ▼
models/trainer.py
    │
    │ Line 67: target_col = metadata['metadata']['target_variable']
    │ Line 90: y = df[target_col]  # Extract "OT" column
    │
    ▼
Train/Test Split
    │
    │ y_train = [30.5, 27.8, ...]  # OT values
    │ y_test = [28.3, 29.1, ...]
    │
    ▼
Model Training
    │
    │ model.train(X_train, y_train)
    │            ^^^^^^^^  ^^^^^^^^
    │            features  target "OT"
    │
    ▼
Predictions
    │
    │ y_pred = model.predict(X_test)
    │ # Predicted "OT" values
```

### Model Recommendations Path

```
inputs/ETTh1.csv
    │
    │ Time series, 7 features, 1 target
    │ Has seasonality, trend, multiple features
    │
    ▼
[LLM Analysis]
    │
    │ Determines suitable models:
    │ • XGBoost (handles non-linear patterns)
    │ • RandomForest (robust, parallel trees)
    │ • LSTM (for sequential patterns) - not implemented yet
    │ • ARIMA (for time series) - not implemented yet
    │
    ▼
outputs/ETTh1_metadata.json
    │
    │ {
    │   "metadata": {
    │     "recommended_models": [
    │       "XGBoost",
    │       "RandomForest",
    │       "LSTM",
    │       "ARIMA"
    │     ]
    │   }
    │ }
    │
    ▼
models/trainer.py
    │
    │ Line 68: recommended_models = metadata['metadata']['recommended_models']
    │ Line 72: available_models = ModelRegistry.list_available_models()
    │           # ["XGBoost", "RandomForest"] - only implemented ones
    │ Line 73: recommended_models = [m for m in recommended_models if m in available_models]
    │           # Filters to ["XGBoost", "RandomForest"]
    │
    ▼
Training Loop
    │
    │ for model_name in ["XGBoost", "RandomForest"]:
    │     model = ModelRegistry.create_model(model_name, params)
    │     model.train(...)
    │     model.evaluate(...)
    │     model.save(...)
    │
    ▼
Results
    │
    │ • models/results/trained_models/ETTh1_XGBoost_20251202/
    │ • models/results/trained_models/ETTh1_RandomForest_20251202/
```

---

## Summary

**The Connection Point:** `outputs/{dataset}_metadata.json`

**What flows through:**
1. **Target variable** (which column to predict)
2. **Recommended models** (which models to train)
3. **Data characteristics** (seasonality, trend, etc.)
4. **Preprocessing applied** (what was done to clean data)

**Why it works:**
- Preprocessing stage saves what models need
- Model training stage reads what preprocessing created
- JSON format ensures structured, parseable data
- No hardcoded assumptions - everything is data-driven

**To run end-to-end:**
```bash
python run_full_pipeline.py --input inputs/ETTh1.csv
```

This single command:
1. Extracts metadata with LLM
2. Preprocesses data
3. Saves to `outputs/`
4. Trains recommended models
5. Evaluates and compares
6. Saves trained models to `models/results/`
