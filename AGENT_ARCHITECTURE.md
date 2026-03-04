# Pipeline Agent Architecture — Full Design Document

**Version:** v1 Design
**Date:** 2026-02-17
**Scope:** `/Pipeline` directory only. No frontend, no UI, CPU-only execution.

---

## 0. Goal Confirmation

From a raw dataset upload (CSV / Parquet / JSON), the pipeline automatically:

1. Profiles the dataset and extracts semantic metadata (LLM-assisted)
2. Classifies the ML task and modality
3. Plans and executes appropriate preprocessing
4. Defines an evaluation protocol (splits + metrics) that matches the task
5. Selects exactly **3** models from a fixed internal pool of ~10
6. Trains each model with Optuna-tuned hyperparameters on a protocol-appropriate split
7. Evaluates all 3 on an **identical** held-out test set using **identical** metrics
8. Assembles dashboard-ready artifacts (metrics, plots, model cards, leaderboard)

Every pipeline stage is handled by a **specialized agent**. Agents communicate **only via well-defined artifacts** written to a shared `runs/{run_id}/` workspace. No agent calls another agent's code directly.

---

## 1. Mapping to `high_level.md`

| `high_level.md` Concept | Design Agent |
|---|---|
| LLM metadata extraction, dataset profile | **Ingestion / Metadata Agent** |
| Planning Agent — preprocessing, transformers, models, splits | **Problem Classification Agent** + **Preprocessing Planning Agent** + **Evaluation Protocol Agent** + **Model Selection Agent** |
| Training/Testing Agent | **Training Agent** |
| Eval Agent — metrics and comparison for dashboard | **Evaluation & Comparison Agent** + **Artifact Assembly Agent** |
| Catalogs for agents to browse | **Model Catalog** (`catalogs/model_catalog.json`) + **Transformer Catalog** (`catalogs/transformer_catalog.json`) |

> Note: `high_level.md` lists one "Planning Agent". This design splits it into three distinct agents (Problem Classification, Preprocessing Planning, Evaluation Protocol) to enforce single-responsibility and prevent the planning agent from making decisions it cannot validate.

---

## 2. The JobRun Manifest

Every agent **reads** the manifest at startup and **writes** exactly its own stage fields before exit. The manifest is the single source of truth for a pipeline run.

**File path:** `runs/{run_id}/job_manifest.json`

```json
{
  "run_id": "string (UUID4)",
  "created_at": "ISO-8601 timestamp",
  "updated_at": "ISO-8601 timestamp",
  "status": "pending | running | completed | failed | partial_failure",
  "input": {
    "file_path": "string — absolute path to raw input file",
    "file_format": "csv | parquet | json",
    "original_filename": "string"
  },
  "stages": {
    "ingestion": {
      "status": "pending | running | completed | failed",
      "started_at": "ISO-8601 | null",
      "completed_at": "ISO-8601 | null",
      "artifacts": {
        "dataset_profile": "runs/{run_id}/artifacts/dataset_profile.json"
      },
      "error": "string | null"
    },
    "problem_classification": {
      "status": "pending | running | completed | failed",
      "started_at": "ISO-8601 | null",
      "completed_at": "ISO-8601 | null",
      "artifacts": {
        "task_spec": "runs/{run_id}/artifacts/task_spec.json"
      },
      "error": "string | null"
    },
    "preprocessing_planning": {
      "status": "pending | running | completed | failed",
      "started_at": "ISO-8601 | null",
      "completed_at": "ISO-8601 | null",
      "artifacts": {
        "preprocessing_plan": "runs/{run_id}/artifacts/preprocessing_plan.json",
        "processed_data": "runs/{run_id}/artifacts/processed_data.csv",
        "preprocessing_manifest": "runs/{run_id}/artifacts/preprocessing_manifest.json"
      },
      "error": "string | null"
    },
    "evaluation_protocol": {
      "status": "pending | running | completed | failed",
      "started_at": "ISO-8601 | null",
      "completed_at": "ISO-8601 | null",
      "artifacts": {
        "eval_protocol": "runs/{run_id}/artifacts/eval_protocol.json"
      },
      "error": "string | null"
    },
    "model_selection": {
      "status": "pending | running | completed | failed",
      "started_at": "ISO-8601 | null",
      "completed_at": "ISO-8601 | null",
      "artifacts": {
        "selected_models": "runs/{run_id}/artifacts/selected_models.json"
      },
      "error": "string | null"
    },
    "training": {
      "status": "pending | running | completed | failed | partial_failure",
      "started_at": "ISO-8601 | null",
      "completed_at": "ISO-8601 | null",
      "artifacts": {
        "training_results": "runs/{run_id}/artifacts/training_results.json",
        "trained_model_paths": {
          "ModelA": "runs/{run_id}/trained_models/ModelA/",
          "ModelB": "runs/{run_id}/trained_models/ModelB/",
          "ModelC": "runs/{run_id}/trained_models/ModelC/"
        }
      },
      "error": "string | null"
    },
    "evaluation": {
      "status": "pending | running | completed | failed",
      "started_at": "ISO-8601 | null",
      "completed_at": "ISO-8601 | null",
      "artifacts": {
        "evaluation_report": "runs/{run_id}/artifacts/evaluation_report.json",
        "comparison_table": "runs/{run_id}/artifacts/comparison_table.json",
        "plots_dir": "runs/{run_id}/plots/"
      },
      "error": "string | null"
    },
    "ensembling": {
      "status": "pending | running | completed | failed",
      "started_at": "ISO-8601 | null",
      "completed_at": "ISO-8601 | null",
      "artifacts": {
        "ensemble_spec": "runs/{run_id}/artifacts/ensemble_spec.json",
        "ensemble_predictions": "runs/{run_id}/artifacts/ensemble_predictions.csv",
        "ensemble_report": "runs/{run_id}/artifacts/ensemble_report.json"
      },
      "error": "string | null"
    },
    "artifact_assembly": {
      "status": "pending | running | completed | failed",
      "started_at": "ISO-8601 | null",
      "completed_at": "ISO-8601 | null",
      "artifacts": {
        "dashboard_bundle": "runs/{run_id}/dashboard/"
      },
      "error": "string | null"
    }
  },
  "config": {
    "tune_hyperparameters": true,
    "n_optuna_trials": 20,
    "random_seed": 42,
    "cpu_only": true
  }
}
```

**Write rules:**
- Each agent sets its stage `status` to `"running"` on start, updates `artifacts` paths, then sets `completed` or `failed`.
- Only the agent that owns a stage may write to that stage's key.
- The `updated_at` field is refreshed on every write.
- A stage may not start unless all its declared dependencies are `"completed"`.

---

## 3. Model Pool — Fixed Catalog

The internal model pool is defined in `catalogs/model_catalog.json`. Agents do not import model code directly — they reference models by name through this catalog. The Model Selection Agent uses the catalog to gate and choose. The Training Agent uses the registry to instantiate.

**File:** `catalogs/model_catalog.json`

```json
{
  "models": [
    {
      "name": "DummyRegressor",
      "tier": "baseline",
      "compatible_tasks": ["tabular_regression", "time_series_forecasting"],
      "requires_features": false,
      "cpu_feasible": true,
      "registry_key": "DummyRegressor",
      "description": "Predicts the training set mean. Required baseline for regression tasks."
    },
    {
      "name": "DummyClassifier",
      "tier": "baseline",
      "compatible_tasks": ["tabular_classification"],
      "requires_features": false,
      "cpu_feasible": true,
      "registry_key": "DummyClassifier",
      "description": "Predicts majority class or uniform random. Required baseline for classification."
    },
    {
      "name": "LinearModel",
      "tier": "baseline",
      "compatible_tasks": ["tabular_regression", "tabular_classification"],
      "requires_features": true,
      "cpu_feasible": true,
      "registry_key": "LinearModel",
      "description": "Ridge regression (regression) or LogisticRegression (classification). Interpretable baseline."
    },
    {
      "name": "RandomForest",
      "tier": "classical",
      "compatible_tasks": ["tabular_regression", "tabular_classification", "time_series_forecasting"],
      "requires_features": true,
      "cpu_feasible": true,
      "registry_key": "RandomForest",
      "description": "Ensemble of decision trees. Robust, parallelizable, handles mixed types well."
    },
    {
      "name": "XGBoost",
      "tier": "classical",
      "compatible_tasks": ["tabular_regression", "tabular_classification", "time_series_forecasting"],
      "requires_features": true,
      "cpu_feasible": true,
      "registry_key": "XGBoost",
      "description": "Gradient boosted trees. High-capacity, widely used benchmark for tabular data."
    },
    {
      "name": "LightGBM",
      "tier": "classical",
      "compatible_tasks": ["tabular_regression", "tabular_classification", "time_series_forecasting"],
      "requires_features": true,
      "cpu_feasible": true,
      "registry_key": "LightGBM",
      "description": "Histogram-based gradient boosting. Fast on larger datasets."
    },
    {
      "name": "ARIMA",
      "tier": "specialized",
      "compatible_tasks": ["time_series_forecasting"],
      "requires_features": false,
      "cpu_feasible": true,
      "registry_key": "ARIMA",
      "description": "Classical univariate time-series model. Strong baseline for single-series forecasting."
    },
    {
      "name": "LSTM",
      "tier": "specialized",
      "compatible_tasks": ["time_series_forecasting"],
      "requires_features": true,
      "cpu_feasible": true,
      "description": "Recurrent neural network with long-range memory. CPU-feasible at small scale.",
      "registry_key": "LSTM"
    },
    {
      "name": "Chronos",
      "tier": "specialized",
      "compatible_tasks": ["time_series_forecasting"],
      "requires_features": false,
      "cpu_feasible": true,
      "registry_key": "Chronos",
      "description": "Pretrained zero-shot forecasting transformer (Amazon). Works without feature engineering."
    },
    {
      "name": "SVR",
      "tier": "classical",
      "compatible_tasks": ["tabular_regression"],
      "requires_features": true,
      "cpu_feasible": true,
      "registry_key": "SVR",
      "description": "Support Vector Regression. Good for small-to-medium datasets with non-linear boundaries."
    }
  ]
}
```

**Model selection constraint rules (deterministic, enforced by Model Selection Agent):**

| Task Type | Baseline Slot | Classical Slot | Specialized Slot |
|---|---|---|---|
| `tabular_regression` | DummyRegressor or LinearModel | XGBoost or RandomForest or LightGBM | SVR or RandomForest (second choice) |
| `tabular_classification` | DummyClassifier or LinearModel | XGBoost or RandomForest or LightGBM | RandomForest or LightGBM (second choice) |
| `time_series_forecasting` | ARIMA | RandomForest or XGBoost (windowed) | Chronos or LSTM |

The agent must always fill all three tier slots. If a specialized model is unavailable (import error), fall back to the next eligible model in that tier, log the substitution, and mark `substitution_reason` in `selected_models.json`.

---

## 4. Agent Specifications

---

### Agent 1 — Ingestion / Metadata Agent

**Purpose:**
Parse the raw uploaded file, compute statistical column profiles, detect temporal structure, and call the LLM to extract semantic metadata (dataset intent, target variable candidate, data quality issues). This agent is the entry point and does **not** make any ML decisions.

**Decisions made by this agent:**
- Column dtype inference (numeric vs categorical vs datetime)
- Sample row selection for LLM context
- LLM prompt construction and response parsing

**Deterministic operations:**
- File reading, shape computation, missing fraction, cardinality, value range, sample values

#### Inputs

| Artifact | Schema |
|---|---|
| Raw file (CSV/Parquet/JSON) | Any tabular format |
| `job_manifest.json` (`input` block) | File path, format, original filename |

#### Outputs

**`runs/{run_id}/artifacts/dataset_profile.json`**

```json
{
  "run_id": "string",
  "dataset_name": "string",
  "file_format": "csv | parquet | json",
  "num_rows": "int",
  "num_columns": "int",
  "columns": [
    {
      "name": "string",
      "dtype_pandas": "string",
      "inferred_type": "numeric | categorical | datetime | boolean | unknown",
      "missing_count": "int",
      "missing_fraction": "float [0, 1]",
      "unique_count": "int",
      "sample_values": ["list of up to 5 representative values"],
      "min": "float | null",
      "max": "float | null",
      "mean": "float | null",
      "std": "float | null",
      "has_negative_values": "bool",
      "is_temporal": "bool",
      "is_monotonically_increasing": "bool | null"
    }
  ],
  "has_header": "bool",
  "detected_datetime_columns": ["list of column names"],
  "llm_analysis": {
    "dataset_description": "string — LLM-generated semantic description",
    "suggested_target_variable": "string — column name",
    "target_confidence": "high | medium | low",
    "target_reasoning": "string",
    "known_quality_issues": ["list of strings"],
    "has_trend": "bool",
    "has_seasonality": "bool",
    "is_multivariate": "bool",
    "data_source_hint": "string",
    "ingestion_date": "YYYY-MM-DD"
  },
  "profiling_completed_at": "ISO-8601"
}
```

#### Tools Used
- `pandas` / `pyarrow` for file ingestion and profiling
- `numpy` for statistics
- OpenAI API (`gpt-4o-mini`) via `code_interpreter` tool — LLM receives a sample of the file (first 50 rows + column stats) and returns the `llm_analysis` block
- JSON schema validator (validates LLM response before accepting)

#### Validation Checks & Guardrails
- File is readable and has at least 2 rows and 1 column → abort with `INGESTION_EMPTY_FILE` error
- At least 1 non-constant numeric column → warn if not present
- LLM response must parse as valid JSON matching the `llm_analysis` schema → retry if not
- `suggested_target_variable` must exist as an actual column name → if not, set `target_confidence: low` and mark column as `null`
- If `missing_fraction` > 0.9 for any column → flag it as `likely_unusable`

#### Failure Handling
- LLM call failure / timeout: retry up to 2 times with exponential backoff (2s, 6s)
- LLM JSON parse failure: retry with a stricter prompt asking for JSON only; if still fails, set `llm_analysis` to defaults with all `null` fields — pipeline continues with heuristic-only mode
- File read failure: abort immediately, set stage status to `failed`, write error to manifest

#### Output Flow to Next Agent
Problem Classification Agent reads `dataset_profile.json` directly. It does not call the Ingestion Agent — it reads the artifact file path from the manifest.

---

### Agent 2 — Problem Classification Agent

**Purpose:**
Given the dataset profile, determine: (a) the ML task type, (b) the data modality, (c) the confirmed target variable, (d) any grouping column, (e) the time column if present, (f) forecast horizon if time-series. This agent drives all downstream routing — every other agent's behavior is conditioned on this output.

**Decisions made by this agent:**
- Task type classification (4-way enum)
- Confirmation or rejection of LLM-suggested target variable
- Identification of group key for grouped splits

**Deterministic operations:**
- Heuristic checks on column stats and dtypes

#### Inputs

| Artifact | Schema |
|---|---|
| `dataset_profile.json` | Full profile from Agent 1 |

#### Outputs

**`runs/{run_id}/artifacts/task_spec.json`**

```json
{
  "run_id": "string",
  "task_type": "tabular_classification | tabular_regression | time_series_forecasting | grouped_prediction",
  "modality": "tabular_iid | time_series | grouped_tabular",
  "target_col": "string — confirmed column name",
  "target_dtype": "numeric | categorical",
  "target_cardinality": "int — number of unique values in target",
  "time_col": "string | null — column used as time index",
  "group_col": "string | null — column used for group-based splits",
  "forecast_horizon": "int | null — steps ahead to forecast (time-series only)",
  "is_multivariate_ts": "bool | null — time-series only",
  "classification_subtype": "binary | multiclass | null",
  "regression_subtype": "standard | count | bounded | null",
  "task_confidence": "high | medium | low",
  "task_reasoning": "string — explanation of how task was inferred",
  "warnings": ["list of strings — non-blocking issues"]
}
```

**Task type decision logic (heuristic, in priority order):**

```
IF detected_datetime_columns is not empty
   AND target_col is numeric
   AND rows are ordered by time (is_monotonically_increasing = true on time col)
   → task_type = "time_series_forecasting"
   → modality = "time_series"

ELSE IF a group_col is detected (column with repeated IDs and a time dimension)
   → task_type = "grouped_prediction"
   → modality = "grouped_tabular"

ELSE IF target_col.inferred_type == "categorical"
   OR target_col.unique_count <= 20 AND target_col.dtype is integer
   → task_type = "tabular_classification"
   → modality = "tabular_iid"

ELSE IF target_col.inferred_type == "numeric"
   → task_type = "tabular_regression"
   → modality = "tabular_iid"

ELSE
   → task_type = "tabular_regression" (safe default)
   → task_confidence = "low"
   → emit warning
```

**Target variable confirmation logic:**

```
1. If LLM suggested a target and target_confidence == "high":
   → accept LLM suggestion
2. If LLM confidence is "medium" or "low":
   → check: is there exactly one column named "target", "label", "y", "output"?
      → if yes, use that column
   → else: is there exactly one column with dtype float that has no missing values?
      → if yes, use that column
   → else: use LLM suggestion with a low-confidence warning
3. If LLM analysis was unavailable:
   → run heuristics only, emit low-confidence warning
```

#### Tools Used
- Pure Python heuristics (no external calls)
- `pandas` dtype inspection
- Optional LLM call only for ambiguous cases (`task_confidence` remains `low` without it)

#### Validation Checks & Guardrails
- `target_col` must exist in `dataset_profile.columns[*].name` → abort if not found
- `task_type` must be one of the 4 known enum values → abort otherwise
- If `time_series_forecasting` is selected but no `time_col` is found → abort with `TS_NO_TIME_COLUMN`
- If target has > 50% missing values → abort with `TARGET_TOO_SPARSE`
- If target has only 1 unique value (constant) → abort with `TARGET_CONSTANT`

#### Failure Handling
- If all heuristics are ambiguous and LLM is unavailable: default to `tabular_regression`, `task_confidence: low`, emit strong warning to manifest; pipeline continues
- Abort conditions (unrecoverable): target not found, target constant, no time column for TS task

#### Output Flow to Next Agent
`task_spec.json` is read independently by:
- Preprocessing Planning Agent (determines which transforms are safe)
- Evaluation Protocol Agent (determines split strategy + metrics)
- Model Selection Agent (gates candidate models by task type)

---

### Agent 3 — Preprocessing Planning Agent

**Purpose:**
Given the dataset profile and task spec, produce an ordered preprocessing plan, then execute it deterministically to produce the cleaned dataset. This agent plans AND executes in two sub-phases:
- **Planning sub-phase** (LLM-assisted): produces `preprocessing_plan.json`
- **Execution sub-phase** (deterministic): produces `processed_data.csv` + `preprocessing_manifest.json`

**Decisions made by this agent:**
- Which preprocessing steps to apply, in what order, with what parameters (LLM-assisted)
- Whether to preserve temporal ordering (enforced by task_spec)

**Deterministic operations:**
- Execution of preprocessing steps from the plan using the transformer catalog
- Saving the scaler/imputer fit parameters for potential inference-time reuse

#### Inputs

| Artifact | Schema |
|---|---|
| `dataset_profile.json` | Full profile |
| `task_spec.json` | Task type, target col, time col |

#### Outputs

**`runs/{run_id}/artifacts/preprocessing_plan.json`**

```json
{
  "run_id": "string",
  "steps": [
    {
      "order": 1,
      "method": "string — key from transformer catalog",
      "parameters": { "key": "value" },
      "applies_to": "all_numeric | all_categorical | [column_list] | target_only | features_only",
      "reason": "string",
      "skip_columns": ["list of column names to exclude from this step"]
    }
  ],
  "preserve_temporal_order": "bool",
  "exclude_columns_from_features": ["time_col, group_col, any ID columns"],
  "plan_source": "llm | heuristic_fallback"
}
```

**`runs/{run_id}/artifacts/processed_data.csv`**
The cleaned, transformed dataset with the same column structure (minus excluded columns).

**`runs/{run_id}/artifacts/preprocessing_manifest.json`**
Records exactly what was applied (including fitted parameters like scaler mean/std, imputer fill values) for reproducibility and potential inversion at inference time.

```json
{
  "run_id": "string",
  "steps_applied": [
    {
      "order": 1,
      "method": "string",
      "parameters_used": {},
      "fitted_params": { "col_name": { "mean": 0.0, "std": 1.0 } },
      "rows_before": "int",
      "rows_after": "int",
      "columns_affected": ["list"]
    }
  ],
  "final_shape": { "rows": "int", "cols": "int" },
  "feature_columns": ["list of column names after exclusions"],
  "target_column": "string",
  "time_column": "string | null"
}
```

#### Transformer Catalog
**File:** `catalogs/transformer_catalog.json`

```json
{
  "transformers": [
    { "name": "imputation", "allowed_modalities": ["tabular_iid", "time_series", "grouped_tabular"], "params": ["strategy"] },
    { "name": "z_norm", "allowed_modalities": ["tabular_iid", "time_series", "grouped_tabular"], "params": [] },
    { "name": "min_max", "allowed_modalities": ["tabular_iid", "time_series", "grouped_tabular"], "params": [] },
    { "name": "log_transform", "allowed_modalities": ["tabular_iid", "tabular_regression"], "params": [] },
    { "name": "remove_outliers", "allowed_modalities": ["tabular_iid"], "params": ["method", "threshold"] },
    { "name": "detrend", "allowed_modalities": ["time_series"], "params": ["type"] },
    { "name": "differencing", "allowed_modalities": ["time_series"], "params": [] },
    { "name": "smoothing", "allowed_modalities": ["time_series"], "params": ["window"] },
    { "name": "label_encode", "allowed_modalities": ["tabular_iid", "grouped_tabular"], "params": [] },
    { "name": "onehot_encode", "allowed_modalities": ["tabular_iid", "grouped_tabular"], "params": ["max_cardinality"] }
  ]
}
```

**Key modality-gating rules enforced during plan validation:**
- `remove_outliers` is **never** applied to time-series data (would break temporal structure)
- `differencing` and `detrend` are **only** applied to time-series data
- The time column and group column are **never** transformed — they are excluded from all steps
- The target column is **excluded** from normalization unless explicitly needed (prevents data leakage)

#### Tools Used
- LLM (`gpt-4o-mini`) for planning sub-phase — receives column stats + task_spec, returns `steps[]`
- `pandas`, `scipy`, `sklearn.preprocessing` for execution sub-phase
- Existing `PreprocessingPipeline` class in `preprocessing_pipeline.py` (refactored to accept a plan dict)
- Transformer catalog for plan validation

#### Validation Checks & Guardrails
- Each step's `method` must exist in transformer catalog → reject unknown methods
- Each step's `method` must be allowed for current modality → reject invalid methods
- After execution: `processed_data.csv` must have same number of rows as input (unless `remove_outliers` was applied, in which case log the row delta)
- After execution: no NaN values in feature columns → if any remain, apply fallback mean imputation
- Target column must survive unchanged (verify by checking its values didn't change)

#### Failure Handling
- LLM plan fails or returns invalid JSON: fall back to a minimal safe plan (imputation → z_norm for numeric features)
- Individual step execution fails: skip the step, log it in `preprocessing_manifest.json`, continue
- If more than 2 steps fail: abort stage, mark `failed`, do not write processed_data

#### Output Flow to Next Agent
`processed_data.csv` and `preprocessing_manifest.json` are read by the Training Agent. The `preprocessing_plan.json` is included in the dashboard bundle.

---

### Agent 4 — Evaluation Protocol Agent

**Purpose:**
Define the evaluation protocol deterministically from the task spec: which split strategy to use, what the primary ranking metric is, and the full set of metrics to compute. This agent makes **no LLM calls** — all decisions are rule-based.

**Decisions made by this agent:**
- Split strategy (random stratified / chronological / group-based)
- Split ratios
- Primary metric (used for model ranking)
- Full metric set (used for evaluation and comparison table)

**Deterministic operations:**
- All operations are deterministic lookups from task_spec

#### Inputs

| Artifact | Schema |
|---|---|
| `task_spec.json` | Task type, modality, target dtype, group col, num rows |
| `dataset_profile.json` | `num_rows` for minimum size checks |

#### Outputs

**`runs/{run_id}/artifacts/eval_protocol.json`**

```json
{
  "run_id": "string",
  "task_type": "string",

  "split_strategy": "random | stratified | chronological | group_kfold | time_series_cv",
  "train_fraction": 0.7,
  "val_fraction": 0.1,
  "test_fraction": 0.2,

  "shuffle": "bool — always false for time_series",
  "stratify_on": "string | null — target column name for stratified",
  "group_col": "string | null — for group_kfold",
  "time_col": "string | null — for chronological/time_series_cv",

  "cv": {
    "enabled": "bool",
    "method": "rolling | expanding",
    "n_folds": "int",
    "initial_train_size": "int — rows or timesteps",
    "step_size": "int — rows or timesteps",
    "val_window_size": "int — rows or timesteps",
    "gap": "int — optional embargo between train and val to reduce leakage",
    "aggregate": "mean | median",
    "use_for": "tuning_only | tuning_and_model_ranking"
  },

  "prediction_type": "point | quantile",
  "quantiles": ["list[float] — required when prediction_type == quantile (e.g., [0.1,0.5,0.9])"],

  "primary_metric": "string — single metric name used to rank models",
  "metrics": [
    { "name": "rmse", "display_name": "RMSE", "higher_is_better": false, "applicable_tasks": ["tabular_regression", "time_series_forecasting"] },
    { "name": "mae", "display_name": "MAE", "higher_is_better": false, "applicable_tasks": ["tabular_regression", "time_series_forecasting"] },
    { "name": "mape", "display_name": "MAPE (%)", "higher_is_better": false, "applicable_tasks": ["tabular_regression", "time_series_forecasting"] },
    { "name": "smape", "display_name": "sMAPE (%)", "higher_is_better": false, "applicable_tasks": ["time_series_forecasting"] },

    { "name": "pinball_loss", "display_name": "Pinball Loss", "higher_is_better": false, "applicable_tasks": ["time_series_forecasting"], "requires": "quantile" },
    { "name": "coverage_80", "display_name": "Coverage @ 80%", "higher_is_better": true, "applicable_tasks": ["time_series_forecasting"], "requires": "quantile" },
    { "name": "interval_width_80", "display_name": "Interval Width @ 80%", "higher_is_better": false, "applicable_tasks": ["time_series_forecasting"], "requires": "quantile" },

    { "name": "r2", "display_name": "R²", "higher_is_better": true, "applicable_tasks": ["tabular_regression"] },
    { "name": "accuracy", "display_name": "Accuracy", "higher_is_better": true, "applicable_tasks": ["tabular_classification"] },
    { "name": "f1_weighted", "display_name": "F1 (weighted)", "higher_is_better": true, "applicable_tasks": ["tabular_classification"] },
    { "name": "roc_auc", "display_name": "ROC-AUC", "higher_is_better": true, "applicable_tasks": ["tabular_classification"] }
  ],

  "metrics_for_this_run": ["list of metric names applicable to this task_type and prediction_type"],
  "minimum_test_samples": 30
}
```

**Protocol routing table:**

| `task_type` | `split_strategy` | `shuffle` | `primary_metric` | Notes |
|---|---|---|---|---|
| `tabular_regression` | `random` | true | `rmse` | Split 70/10/20 |
| `tabular_classification` (binary) | `stratified` | true | `roc_auc` | Stratify on target |
| `tabular_classification` (multiclass) | `stratified` | true | `f1_weighted` | Stratify on target |
| `time_series_forecasting` | `chronological` | false | `mae` | Split preserves order |
| `grouped_prediction` | `group_kfold` | false | `rmse` | No cross-group leakage |

**Why MAE is primary for time-series instead of RMSE:**
RMSE heavily penalizes large spikes common in time-series data and is sensitive to a single outlier point. MAE is more robust for initial model comparison. RMSE is still computed and reported.

#### Tools Used
- Pure Python — no external dependencies
- Reads from `task_spec.json` and `dataset_profile.json`

#### Validation Checks & Guardrails
- Test fraction must yield at least `minimum_test_samples` rows → if not, emit warning and reduce val fraction
- `stratified` split requires target with at least 2 classes and no class with fewer than 2 samples → if violated, downgrade to `random`
- `group_kfold` requires `group_col` to be set in task_spec → if not found, downgrade to `random`

#### Failure Handling
- All decisions are deterministic — no retries needed
- The only failure mode is a missing or invalid `task_spec.json` → abort with `MISSING_TASK_SPEC`

#### Output Flow to Next Agent
`eval_protocol.json` is read by:
- Model Selection Agent (to check metric compatibility)
- Training Agent (for split execution)
- Evaluation & Comparison Agent (for metric computation)

---

### Agent 5 — Model Selection Agent

**Purpose:**
Select exactly 3 models from the fixed model pool. Selection is deterministic and rule-based (no LLM). The agent enforces the 3-tier constraint (1 baseline, 1 classical, 1 specialized) and writes a rationale for each selection.

**Decisions made by this agent:**
- Which 3 models from the catalog to use for this run
- Substitutions when a preferred model is unavailable

**Deterministic operations:**
- All selection logic is rule-based from the catalog and task_spec

#### Inputs

| Artifact | Schema |
|---|---|
| `task_spec.json` | `task_type` |
| `eval_protocol.json` | `primary_metric` for tie-breaking |
| `catalogs/model_catalog.json` | Full model pool |

#### Outputs

**`runs/{run_id}/artifacts/selected_models.json`**

```json
{
  "run_id": "string",
  "task_type": "string",
  "selected_models": [
    {
      "name": "string — registry key",
      "tier": "baseline | classical | specialized",
      "rationale": "string — why this model was selected for this task",
      "substituted_from": "string | null — original preferred model if substituted",
      "substitution_reason": "string | null"
    },
    { "..." },
    { "..." }
  ],
  "selection_strategy": "tier_gated_deterministic",
  "models_considered": ["list of all catalog models that passed task gate"],
  "models_rejected": [
    { "name": "string", "rejection_reason": "string" }
  ]
}
```

**Selection algorithm (pseudocode):**

```
candidates = [m for m in catalog if task_type in m.compatible_tasks]

# Gate: check registry availability (try import)
available_candidates = [m for m in candidates if ModelRegistry.has(m.registry_key)]

# Tier 1: Baseline slot
baseline_candidates = [m for m in available_candidates if m.tier == "baseline"]
selected_baseline = baseline_candidates[0]   # prefer Dummy* over Linear for true baseline

# Tier 2: Classical slot
classical_candidates = [m for m in available_candidates if m.tier == "classical"
                        and m != selected_baseline]
# Preference order for classical: XGBoost > LightGBM > RandomForest > SVR
selected_classical = first match in preference order

# Tier 3: Specialized slot
specialized_candidates = [m for m in available_candidates if m.tier == "specialized"
                          and m not in [selected_baseline, selected_classical]]
selected_specialized = first match in preference order

# If < 3 models available after filtering:
# Fill remaining slots with the next best available from any tier (no duplicates)
# Log substitution in selected_models.json

final_selection = [selected_baseline, selected_classical, selected_specialized]
assert len(set(final_selection)) == 3  # no duplicates
```

**Time-series preference ordering:**
- Baseline slot: ARIMA (preferred) → DummyRegressor
- Classical slot: XGBoost (windowed features) → RandomForest
- Specialized slot: Chronos (preferred, zero-shot) → LSTM

**Why ARIMA is baseline for time-series (not DummyRegressor):**
ARIMA is a well-understood classical forecasting model with known statistical properties, making it the appropriate baseline against which learned models are measured. DummyRegressor (predict-the-mean) is still computed as a sanity check but is not a tier-1 entry in time-series runs.

#### Tools Used
- `catalogs/model_catalog.json` (static JSON)
- `ModelRegistry.list_available_models()` to check which models can be imported on this machine
- No LLM calls

#### Validation Checks & Guardrails
- Exactly 3 models must be selected → hard requirement; abort if impossible
- No duplicate model names → assert before writing output
- Each model must be present in `ModelRegistry` → if not, substitute and log
- All 3 models must be compatible with `task_type` → enforced by catalog gate

#### Failure Handling
- If fewer than 3 catalog models are compatible with the task type: this represents a catalog gap → abort with `MODEL_POOL_INSUFFICIENT` and log which task/tier combination is missing
- If a preferred model fails import check: substitute to next preference, log substitution

#### Output Flow to Next Agent
Training Agent reads `selected_models[*].name` and instantiates each via `ModelRegistry.create_model()`.

---

### Agent 6 — Training Agent

**Purpose:**
For each of the 3 selected models: apply the protocol-defined split, optionally tune hyperparameters with Optuna, train on the training portion, and save the trained model artifact. The Training Agent is the only agent that calls `ModelRegistry` and writes `.pkl` files.

**Decisions made by this agent:**
- Whether to use default or tuned hyperparameters (configured in `job_manifest.json`)
- Optuna search space per model (from `HyperparameterTuner`)

**Deterministic operations:**
- Data split execution from `eval_protocol.json`
- Model save/load via `BaseModel.save()`

#### Inputs

| Artifact | Schema |
|---|---|
| `runs/{run_id}/artifacts/processed_data.csv` | Cleaned tabular data |
| `preprocessing_manifest.json` | Feature columns, target column |
| `eval_protocol.json` | Split strategy, fractions, time/group col |
| `selected_models.json` | 3 model names to train |
| `job_manifest.json` | `config.tune_hyperparameters`, `config.n_optuna_trials`, `config.random_seed` |

#### Outputs

**`runs/{run_id}/trained_models/{model_name}/`** (one directory per model)
- `{model_name}.pkl` — serialized model object
- `{model_name}_metadata.json` — hyperparameters, training duration, training history

**`runs/{run_id}/artifacts/training_results.json`**

```json
{
  "run_id": "string",
  "models": [
    {
      "name": "string",
      "tier": "string",
      "status": "success | failed",
      "hyperparameters": { "key": "value" },
      "hyperparameter_source": "optuna_tuned | default",
      "best_val_score": "float | null — validation metric from Optuna",
      "training_duration_seconds": "float",
      "training_history": {},
      "model_path": "string",
      "error": "string | null"
    }
  ],
  "split_info": {
    "strategy": "string",
    "n_train": "int",
    "n_val": "int",
    "n_test": "int"
  },
  "feature_columns": ["list"],
  "target_column": "string"
}
```

**Split execution rules (from `eval_protocol.json`):**

| `split_strategy` | Implementation |
|---|---|
| `random` | `sklearn.model_selection.train_test_split(shuffle=True, random_state=seed)` |
| `stratified` | `train_test_split(stratify=y, shuffle=True, random_state=seed)` |
| `chronological` | Slice by row index: first 70% train, next 10% val, last 20% test. No shuffling at any step. |
| `group_kfold` | `sklearn.model_selection.GroupShuffleSplit` with group column |

**Critical invariant:** The test split is computed once at the start of the training stage and stored. All 3 models see **the identical `X_test`, `y_test`** — this is enforced by splitting once and passing the same numpy arrays to each model's training sub-loop.

#### Tools Used
- `ModelRegistry.create_model()` for instantiation
- `HyperparameterTuner` (Optuna, TPE sampler, minimize val loss)
- `sklearn.model_selection` for splits
- `BaseModel.save()` for persistence
- `pandas`, `numpy`

#### Validation Checks & Guardrails
- Split must be computed before any model training begins
- `n_test` must be >= `eval_protocol.minimum_test_samples` → abort if not
- After training: check `model.is_trained == True` before saving → reject untrained models
- Optuna objective must never return `NaN` → catch and return `float('inf')` as penalty
- For time-series splits: verify the data is sorted by `time_col` before slicing

#### Failure Handling
- Per-model exception: catch, set model status to `failed`, write error to `training_results.json`, continue to next model
- If all 3 models fail: set stage status to `failed`; pipeline does not continue to evaluation
- If 2 out of 3 models succeed: set stage status to `partial_failure`; evaluation continues with the 2 successful models
- Optuna trial failure: return `float('inf')` penalty, Optuna continues with next trial

#### Output Flow to Next Agent
Evaluation & Comparison Agent reads `training_results.json` to discover model paths, then loads each `.pkl` via `BaseModel.load()`.

---

### Agent 7 — Evaluation & Comparison Agent

**Purpose:**
Load each trained model, run inference on the **shared test set**, compute the protocol-defined metrics, generate diagnostic plots, and produce a ranked comparison table. This agent enforces metric comparability by ensuring all three models are evaluated identically.

**Decisions made by this agent:**
- None — all evaluation is deterministic given the test set and eval_protocol

**Deterministic operations:**
- All metric computations
- Plot generation

#### Inputs

| Artifact | Schema |
|---|---|
| `training_results.json` | Model paths, feature/target columns |
| `eval_protocol.json` | Metrics list, primary metric |
| `runs/{run_id}/artifacts/processed_data.csv` | Source for re-extracting test split |
| `runs/{run_id}/trained_models/{model_name}/` | Saved model files |

**How the shared test set is reconstructed:**
The agent re-applies the **exact same split logic** from `eval_protocol.json` (same fractions, same seed, same strategy) to the `processed_data.csv`. Because the split is deterministic (same seed, same file, same strategy), this reproduces the identical `X_test`, `y_test` used during training.

#### Outputs

**`runs/{run_id}/artifacts/evaluation_report.json`**

```json
{
  "run_id": "string",
  "primary_metric": "string",
  "test_split_size": "int",
  "models": [
    {
      "name": "string",
      "tier": "string",
      "status": "evaluated | failed",
      "metrics": {
        "rmse": "float | null",
        "mae": "float | null",
        "mape": "float | null",
        "r2": "float | null",
        "accuracy": "float | null",
        "f1_weighted": "float | null",
        "roc_auc": "float | null",
        "smape": "float | null"
      },
      "n_test_samples": "int",
      "inference_time_ms": "float",
      "plot_path": "string | null"
    }
  ],
  "evaluated_at": "ISO-8601"
}
```

**`runs/{run_id}/artifacts/comparison_table.json`**

```json
{
  "run_id": "string",
  "ranked_by": "string — primary_metric name",
  "ranking": [
    {
      "rank": 1,
      "model_name": "string",
      "tier": "string",
      "primary_metric_value": "float",
      "is_best": true,
      "all_metrics": { "..." }
    }
  ]
}
```

**Plots generated per model (`runs/{run_id}/plots/{model_name}_eval.png`):**
- Predicted vs Actual scatter (regression/forecasting)
- Residuals over time / residuals vs fitted
- Residual distribution histogram
- Prediction trace (first N time steps, for TS tasks)
- Confusion matrix (classification tasks only)
- ROC curve (binary classification only)

#### How Metric Comparability is Enforced

1. **Single test set:** The test set is never modified between models. All 3 models predict on the same `X_test` numpy array (same rows, same column order).
2. **Identical metric functions:** All metrics are computed by a single `MetricEngine.compute_all(y_true, y_pred, task_type)` call that only returns metrics from `eval_protocol.metrics_for_this_run`.
3. **No task-switching:** The eval_protocol specifies a fixed metric set for the run. A model cannot be evaluated on different metrics than its peers.
4. **Scale consistency:** Because all models receive the same `processed_data.csv` (normalized by the same scaler), predictions are in the same scale. Raw-scale RMSE and MAE are directly comparable.
5. **NaN handling:** If a metric computation returns NaN (e.g., MAPE when `y_true` contains zeros), it is recorded as `null` for all models, not just the failing model.

#### Tools Used
- `ModelRegistry` / `BaseModel.load()` for loading models
- `sklearn.metrics` for standard metrics
- Custom `smape` implementation (symmetric MAPE)
- `matplotlib` for diagnostic plots
- Existing `ModelEvaluator` class (refactored to accept `eval_protocol` instead of hardcoding metrics)

#### Validation Checks & Guardrails
- Verify reconstructed test set has same row count as recorded in `training_results.json`
- Verify all metric values are finite floats or null → no `inf` in output
- Each model's `n_test_samples` must equal the same value across all 3 models → hard check
- Verify no target leakage: confirm `target_col` is not in `X_test.columns`

#### Failure Handling
- Per-model load failure: mark status `failed`, skip to next model, continue comparison
- Plot generation failure: log warning, set `plot_path: null`, do not abort
- If fewer than 2 models can be evaluated: abort Evaluation Agent, set stage to `failed`

#### Output Flow to Next Agent
Artifact Assembly Agent reads `evaluation_report.json` and `comparison_table.json` to build the dashboard bundle.

---

### Agent 8 — Ensemble Agent

**Purpose:**
Combine multiple trained model predictions into a single ensemble prediction artifact and evaluate it using the same protocol.

**Inputs**
| Artifact | Notes |
|---|---|
| `evaluation_report.json` | Must include each model’s predictions path or in-memory reference |
| `comparison_table.json` | Used to derive weights (if weight_strategy == "metric_weighted") |
| `eval_protocol.json` | Defines prediction_type, quantiles, and which metrics to compute |

**Outputs**
`runs/{run_id}/artifacts/ensemble_spec.json`
```json
{
  "run_id": "string",
  "method": "mean | median | weighted_mean",
  "weight_strategy": "uniform | metric_weighted",
  "weight_metric": "string | null",
  "weights": { "ModelA": 0.2, "ModelB": 0.5, "ModelC": 0.3 },
  "notes": "string"
}
```

`runs/{run_id}/artifacts/ensemble_predictions.csv`
- For point forecasts: a single `y_pred` column aligned to test timestamps/rows.
- For quantiles: columns like `q0.1`, `q0.5`, `q0.9`.

`runs/{run_id}/artifacts/ensemble_report.json`
- Same schema shape as a single model entry in `evaluation_report.json`, plus the `ensemble_spec` reference.

**Guardrails**
- Ensemble uses only predictions produced on the identical frozen test set (no leakage).
- If any required quantile column is missing from any model, ensemble downgrades to point-only (median/mean) and logs a warning.


---

### Agent 9 — Artifact Assembly Agent

**Purpose:**
Gather all run artifacts and assemble them into a clean, frontend-consumable dashboard bundle under `runs/{run_id}/dashboard/`. This agent writes no new data — it curates, renames, and restructures existing artifacts.

**Decisions made by this agent:**
- None — pure assembly

**Deterministic operations:**
- File copy/rename, JSON merge, leaderboard generation, model card generation

#### Inputs

All artifacts from all prior stages (read from `job_manifest.json` artifact paths).

#### Outputs

**`runs/{run_id}/dashboard/`**

```
dashboard/
├── run_summary.json          # Final JobRun manifest state
├── leaderboard.json          # Ranked model comparison
├── pipeline_log.json         # Stage timing and status summary
├── model_cards/
│   ├── {ModelA}_card.json    # One card per model
│   ├── {ModelB}_card.json
│   └── {ModelC}_card.json
├── plots/
│   ├── {ModelA}_eval.png
│   ├── {ModelB}_eval.png
│   ├── {ModelC}_eval.png
│   └── comparison_chart.png  # Bar chart of primary metric across 3 models
├── artifacts/
│   ├── dataset_profile.json  # Passthrough from ingestion
│   ├── task_spec.json        # Passthrough from classification
│   ├── preprocessing_plan.json
│   └── eval_protocol.json
└── README.txt                # Human-readable guide to what's in this bundle
```

**Model card schema (`model_cards/{ModelName}_card.json`):**

```json
{
  "model_name": "string",
  "tier": "baseline | classical | specialized",
  "task_type": "string",
  "rationale_for_selection": "string — from selected_models.json",
  "hyperparameters": {},
  "hyperparameter_source": "optuna_tuned | default",
  "training_duration_seconds": "float",
  "metrics": {
    "primary_metric_name": "float",
    "all_metrics": {}
  },
  "rank": "int — 1 is best",
  "is_recommended": "bool — true only for rank=1",
  "plot_path": "string — relative path from dashboard/",
  "trained_model_path": "string — relative path to .pkl file"
}
```

**`leaderboard.json`** (consumed directly by frontend):

```json
{
  "run_id": "string",
  "dataset_name": "string",
  "task_type": "string",
  "primary_metric": "string",
  "higher_is_better": "bool",
  "models": [
    {
      "rank": 1,
      "name": "string",
      "tier": "string",
      "primary_metric_value": "float",
      "delta_from_baseline": "float — difference from rank-3 (baseline) model",
      "delta_pct": "float — percent improvement over baseline",
      "recommendation": "string — 'Best Overall' | 'Strong Performer' | 'Baseline'"
    }
  ],
  "generated_at": "ISO-8601"
}
```

#### Tools Used
- `shutil.copy2` for file copying
- `json` for manifest merging
- `matplotlib` for `comparison_chart.png` (simple grouped bar chart)
- No LLM calls, no new computation

#### Validation Checks & Guardrails
- All required artifact paths from the manifest must exist → if missing, log error for that artifact and continue
- `leaderboard.json` must have exactly as many entries as successfully evaluated models
- `comparison_chart.png` must be generated even if only 2 of 3 models succeeded

#### Failure Handling
- Missing artifact: log error, set `"missing": true` in the bundle's `pipeline_log.json`, continue
- Chart generation failure: log warning, continue
- This stage never hard-aborts — it assembles whatever is available

---

## 5. Proposed Folder Structure

```
Pipeline/
│
├── agents/                          # One module per agent
│   ├── __init__.py
│   ├── ingestion_agent.py           # Agent 1
│   ├── problem_classification_agent.py  # Agent 2
│   ├── preprocessing_planning_agent.py  # Agent 3 (planning sub-phase)
│   ├── preprocessing_execution.py   # Agent 3 (execution sub-phase)
│   ├── evaluation_protocol_agent.py # Agent 4
│   ├── model_selection_agent.py     # Agent 5
│   ├── training_agent.py            # Agent 6
│   ├── evaluation_agent.py          # Agent 7
│   └── artifact_assembly_agent.py   # Agent 8
│
├── catalogs/                        # Static configuration (agents read, never write)
│   ├── model_catalog.json           # ~10 model definitions + metadata
│   └── transformer_catalog.json     # Allowed preprocessing methods per modality
│
├── core/                            # Shared utilities used by multiple agents
│   ├── __init__.py
│   ├── manifest.py                  # JobRun manifest read/write helpers
│   ├── splitter.py                  # All split strategy implementations
│   ├── metric_engine.py             # Unified metric computation (replaces evaluator.py)
│   └── schema_validator.py          # JSON schema validation for all artifacts
│
├── models/                          # Model implementations (unchanged from current)
│   ├── __init__.py
│   ├── base_model.py
│   ├── model_registry.py
│   ├── hyperparameter_tuner.py      # Updated to accept eval_protocol for tuning metric
│   ├── tree_based/
│   │   ├── xgboost_model.py
│   │   ├── random_forest_model.py
│   │   └── lightgbm_model.py        # New
│   ├── linear/
│   │   └── linear_model.py          # New — Ridge/Logistic unified model
│   ├── time_series/
│   │   ├── arima_model.py
│   │   ├── lstm_model.py
│   │   └── chronos_model.py
│   └── dummy/
│       └── dummy_model.py           # New — wraps sklearn DummyRegressor/Classifier
│
├── runs/                            # All pipeline run outputs live here
│   └── {run_id}/                    # One directory per run
│       ├── job_manifest.json        # THE single source of truth
│       ├── artifacts/               # Intermediate artifacts (agent outputs)
│       │   ├── dataset_profile.json
│       │   ├── task_spec.json
│       │   ├── preprocessing_plan.json
│       │   ├── preprocessing_manifest.json
│       │   ├── processed_data.csv
│       │   ├── eval_protocol.json
│       │   ├── selected_models.json
│       │   ├── training_results.json
│       │   ├── evaluation_report.json
│       │   └── comparison_table.json
│       ├── trained_models/          # One subdirectory per model
│       │   ├── XGBoost/
│       │   │   ├── XGBoost.pkl
│       │   │   └── XGBoost_metadata.json
│       │   ├── RandomForest/
│       │   └── Chronos/
│       ├── plots/                   # All evaluation plots
│       └── dashboard/               # Dashboard-ready bundle (Agent 8 output)
│           ├── run_summary.json
│           ├── leaderboard.json
│           ├── pipeline_log.json
│           ├── model_cards/
│           ├── plots/
│           └── artifacts/
│
├── inputs/                          # Raw user-uploaded datasets (read-only)
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   └── AirPassengers.csv
│
├── orchestrator.py                  # Pipeline entry point; spawns agents in sequence
├── high_level.md                    # Original guiding document (unchanged)
├── AGENT_ARCHITECTURE.md            # This document
│
└── [Legacy files — to be refactored]
    ├── preprocessing_pipeline.py    # → split into agents/ingestion_agent.py + preprocessing_*
    ├── preprocessor.py              # → absorbed into agents/preprocessing_execution.py
    ├── models/trainer.py            # → refactored into agents/training_agent.py
    ├── models/evaluator.py          # → refactored into core/metric_engine.py
    └── run_full_pipeline.py         # → replaced by orchestrator.py
```

---

## 6. Pipeline Sequence Diagram

```
User / API
    │
    │  upload(file_path)
    ▼
orchestrator.py
    │
    │  create run_id, initialize job_manifest.json
    │
    ├─► [Agent 1] Ingestion / Metadata Agent
    │       │  reads:  raw file
    │       │  writes: dataset_profile.json
    │       │
    │       ▼ (on success)
    ├─► [Agent 2] Problem Classification Agent
    │       │  reads:  dataset_profile.json
    │       │  writes: task_spec.json
    │       │
    │       ├─────────────────────────────────┐
    │       │                                 │
    │       ▼ (parallel start)                ▼ (parallel start)
    ├─► [Agent 3] Preprocessing Agent    [Agent 4] Eval Protocol Agent
    │       │  reads:  profile, task_spec      │  reads:  task_spec, profile
    │       │  writes: processed_data.csv      │  writes: eval_protocol.json
    │       │          preprocessing_plan.json  │          (deterministic, no LLM)
    │       │          preprocessing_manifest   │
    │       │                                  │
    │       └──────────────┬───────────────────┘
    │                      │ (both must complete)
    │                      ▼
    ├─► [Agent 5] Model Selection Agent
    │       │  reads:  task_spec.json
    │       │          eval_protocol.json
    │       │          model_catalog.json
    │       │  writes: selected_models.json
    │       │
    │       ▼ (on success)
    ├─► [Agent 6] Training Agent
    │       │  reads:  processed_data.csv
    │       │          preprocessing_manifest.json
    │       │          eval_protocol.json
    │       │          selected_models.json
    │       │
    │       │  FOR EACH of 3 selected models:
    │       │    ├── apply protocol split (once, shared across models)
    │       │    ├── tune hyperparameters (Optuna, val set)
    │       │    ├── train on X_train, y_train
    │       │    └── save .pkl + metadata
    │       │
    │       │  writes: training_results.json
    │       │          trained_models/{model_name}/
    │       │
    │       ▼ (on success or partial_failure with ≥2 models)
    ├─► [Agent 7] Evaluation & Comparison Agent
    │       │  reads:  training_results.json
    │       │          eval_protocol.json
    │       │          processed_data.csv
    │       │          trained_models/*/
    │       │
    │       │  FOR EACH trained model:
    │       │    ├── reconstruct identical test set (deterministic)
    │       │    ├── run predict(X_test)
    │       │    ├── compute all metrics in eval_protocol
    │       │    └── generate diagnostic plots
    │       │
    │       │  rank models by primary_metric
    │       │  writes: evaluation_report.json
    │       │          comparison_table.json
    │       │          plots/
    │       │
    │       ▼ (always runs if ≥1 model evaluated)
    └─► [Agent 8] Artifact Assembly Agent
            │  reads:  all prior artifacts (via job_manifest.json)
            │  writes: dashboard/ bundle
            │
            ▼
        dashboard/
            ├── leaderboard.json
            ├── model_cards/
            ├── plots/
            └── run_summary.json
                        │
                        ▼
                 API response to frontend
```

**Branching conditions:**

```
After Agent 2 (Problem Classification):
  IF task_type == "time_series_forecasting":
      → Eval Protocol uses chronological split, MAE primary
      → Model Selection gates to TS-compatible models
  IF task_type == "tabular_classification":
      → Eval Protocol uses stratified split, ROC-AUC or F1 primary
      → Model Selection gates to classification-compatible models
  IF task_type == "tabular_regression":
      → Eval Protocol uses random split, RMSE primary
      → Model Selection gates to regression-compatible models

After Agent 6 (Training):
  IF all 3 models succeed:  → continue to Evaluation (normal path)
  IF 2 of 3 succeed:        → continue to Evaluation (partial_failure logged)
  IF 1 of 3 succeeds:       → continue to Evaluation (strong warning logged)
  IF 0 models succeed:      → abort; skip Evaluation and Assembly
```

---

## 7. Metric Comparability — Detailed Explanation

The core problem: two models (e.g., ARIMA and XGBoost) may produce predictions in different formats or scales. This section defines how the pipeline ensures their metrics are directly comparable.

### Rule 1 — Identical Test Set

The Training Agent computes the split **once**, at the very beginning of its execution, and stores:
- `X_test`: a fixed numpy array (frozen before any model trains)
- `y_test`: a fixed numpy array (frozen before any model trains)

The Evaluation Agent **re-derives** the same test set using the identical split parameters. Because `processed_data.csv` is immutable after the Preprocessing Agent completes, and the split is seeded and deterministic, `X_test` and `y_test` are byte-for-byte identical across all model evaluations.

### Rule 2 — Identical Metric Functions

All metrics are computed through a single `MetricEngine.compute_all(y_true, y_pred, task_type, metrics_list)` function. There are no model-specific metric branches. Each model's output is passed through the same function with the same parameters.

### Rule 3 — Output Shape Normalization

Before metrics are computed, all model predictions are normalized to shape `(n_test,)` — a flat 1D numpy array. Models that produce 2D outputs (e.g., an LSTM that outputs `(n_test, horizon)`) must reduce to the primary forecast step at index `[0]` before evaluation.

This normalization is enforced in the Evaluation Agent before calling `MetricEngine`.

### Rule 4 — Scale Consistency via Shared Preprocessing

Because all models receive `processed_data.csv` (normalized by the Preprocessing Agent), all models predict in the **same normalized scale**. RMSE and MAE values are directly comparable across models.

If the dashboard needs to display predictions in the original scale, the `preprocessing_manifest.json` contains the inverse-transform parameters (scaler mean/std). The Artifact Assembly Agent applies this inversion when generating plots for display — but the comparison metrics are always in normalized scale for consistency.

### Rule 5 — No Metric Exclusions Per Model

Every model is evaluated on **every metric** in `eval_protocol.metrics_for_this_run`. If a model's prediction causes a metric computation to fail (e.g., predicted all-zeros for MAPE), the metric is recorded as `null` for that model, and `null` is also recorded for **all other models** for that metric in the `comparison_table.json`. This prevents any model from appearing better simply because a broken metric was excluded from its column.

---

## 8. Existing Code Mapping — What Gets Refactored

| Current File | Agent(s) It Becomes |
|---|---|
| `preprocessing_pipeline.py` — `MetadataExtractor` | Agent 1 (Ingestion Agent) |
| `preprocessing_pipeline.py` — `PreprocessingPipeline` | Agent 3 (Preprocessing Execution sub-phase) |
| `models/trainer.py` — `prepare_data_splits()` | `core/splitter.py` |
| `models/trainer.py` — `train_all_recommended_models()` | Agent 6 (Training Agent) |
| `models/evaluator.py` — `ModelEvaluator` | `core/metric_engine.py` + Agent 7 |
| `models/hyperparameter_tuner.py` | Stays in `models/`, refactored to accept `eval_protocol` |
| `models/model_registry.py` | Stays in `models/`, unchanged |
| `models/base_model.py` | Stays in `models/`, unchanged |
| `run_full_pipeline.py` | `orchestrator.py` |

**What changes in existing code:**
- `trainer.py`: The model recommendation list (currently `metadata['recommended_models']`) is replaced by reading from `selected_models.json`. The split logic is moved to `core/splitter.py`.
- `evaluator.py`: The hardcoded regression metrics (RMSE, MAE, R²) are replaced by a dynamic metric set from `eval_protocol.json`.
- `preprocessing_pipeline.py`: The LLM model recommendation logic is removed (that is now Agent 5's job). The metadata extractor focuses only on dataset profiling.

---

## 9. Orchestrator Design (`orchestrator.py`)

The orchestrator is a thin sequential runner. It does not contain any ML logic — it only:
1. Creates `run_id` (UUID4)
2. Initializes `job_manifest.json`
3. Calls each agent in dependency order
4. Checks the manifest after each agent completes
5. Aborts if a stage fails and the failure is non-recoverable

```python
# Pseudocode — not implementation
def run_pipeline(input_file: str, config: dict) -> str:
    run_id = generate_run_id()
    manifest = initialize_manifest(run_id, input_file, config)

    run_agent(IngestionAgent, manifest)
    check_stage(manifest, "ingestion")          # abort if failed

    run_agent(ProblemClassificationAgent, manifest)
    check_stage(manifest, "problem_classification")  # abort if failed

    run_agents_parallel([
        PreprocessingPlanningAgent,
        EvaluationProtocolAgent
    ], manifest)
    check_stages(manifest, ["preprocessing_planning", "evaluation_protocol"])

    run_agent(ModelSelectionAgent, manifest)
    check_stage(manifest, "model_selection")

    run_agent(TrainingAgent, manifest)
    check_stage(manifest, "training", allow_partial=True)

    run_agent(EvaluationAgent, manifest)
    check_stage(manifest, "evaluation")

    run_agent(ArtifactAssemblyAgent, manifest)  # always runs

    return manifest["stages"]["artifact_assembly"]["artifacts"]["dashboard_bundle"]
```

The orchestrator exposes a single public function: `run_pipeline(input_file, config) -> dashboard_path`.

---

## 10. Assumptions and Non-Goals for v1

### Assumptions

- **CPU-only:** No GPU acceleration. Chronos uses the `tiny` model size by default. LSTM keeps hidden size <= 64.
- **Single-target:** All tasks predict a single output column. Multi-output regression is not supported in v1.
- **Supervised learning only:** Clustering, dimensionality reduction, and anomaly detection are out of scope.
- **English-language datasets:** The LLM prompts assume column names and any string data are in English.
- **File size limit:** Input files are assumed to fit in memory (< 500 MB uncompressed).
- **Single machine:** No distributed training or multi-process coordination. Agents run sequentially in the same process (except the parallel planning/protocol sub-step, which uses `concurrent.futures.ThreadPoolExecutor`).
- **Stateless agents:** Each agent reads from disk and writes to disk. No agent holds state across runs.
- **Fixed random seed:** All stochastic operations use `config.random_seed = 42` for reproducibility.
- **OpenAI API availability:** Agents 1 and 3 require a valid API key in the environment (`OPENAI_API_KEY`). If unavailable, both fall back to heuristic-only mode.
- **Model pool is fixed:** The ~10 models in `model_catalog.json` are all that exist. Adding a new model requires registering it in the catalog AND implementing it in `models/`.

### Non-Goals for v1

- No streaming or incremental learning
- No cross-run model comparison (each run is independent)
- No model serving or inference API
- No data versioning or experiment tracking (MLflow, W&B, etc.)
- No automatic retraining on new data
- No feature importance ranking or SHAP explanations in v1
- No support for image, audio, or text modalities
- No multi-GPU or distributed training
- No frontend, UI, or web server
- No authentication or multi-tenant isolation
- No caching of LLM responses across runs (each run calls the LLM fresh)
- No ensemble models (the 3 selected models compete; they are not combined)

---

## 11. Key Design Decisions — Rationale

**Why split "Planning Agent" from high_level.md into 4 agents?**
A single planning agent would need to make decisions about preprocessing, evaluation protocol, and model selection simultaneously — all with different inputs and failure modes. Splitting creates clear single responsibilities and allows Agents 3 and 4 to run in parallel (since neither depends on the other's output).

**Why is model selection deterministic (no LLM)?**
Giving the LLM free choice over model selection (current `preprocessing_pipeline.py` behavior) allows it to recommend models that are not implemented or not appropriate for the detected task. The catalog-gated, tier-enforced selection guarantees that exactly 3 compatible, available models are always chosen.

**Why is the test set reconstructed in Agent 7 rather than passed directly?**
Avoiding passing `X_test` as a file between agents (which would require serializing potentially large arrays) in favor of deterministic re-derivation from `eval_protocol.json` is safer and simpler. The split is always deterministic given the same seed and strategy.

**Why is `preprocessing_manifest.json` separate from `preprocessing_plan.json`?**
The plan describes what was *intended*. The manifest describes what was *actually applied* (including fitted scaler parameters, rows before/after, any skipped steps). This separation allows the dashboard to show users both the plan and any deviations.

**Why is ARIMA the time-series baseline instead of DummyRegressor?**
DummyRegressor (predict the mean) is not a meaningful baseline for time-series — a model that outperforms predicting the mean is setting an extremely low bar. ARIMA with auto-selected order is a proper classical baseline that captures temporal autocorrelation and provides a meaningful reference point.

**Why is MAE the primary metric for time-series instead of RMSE?**
Financial and energy time-series (like ETTh1/2) contain occasional large spikes. RMSE squares these errors, making a single spike dominate the comparison. MAE is more robust and leads to fairer initial ranking. RMSE is still computed and shown.
