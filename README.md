# SDP-44 — Automated ML Pipeline

An end-to-end machine learning pipeline that takes a raw tabular dataset and automatically handles everything: profiling, task classification, preprocessing, model selection, training, evaluation, and dashboard output. No configuration required — drop in a CSV and get back a ranked model comparison.

Built as a Senior Design Project.

---

## What It Does

Given a raw dataset (CSV, Parquet, or JSON), the pipeline:

1. **Profiles** every column — dtypes, missing rates, statistics, temporal structure
2. **Classifies** the ML task — regression, classification, time-series forecasting, or grouped prediction
3. **Plans and executes preprocessing** — imputation, normalization, lag features, rolling stats, encoding, and more
4. **Defines an evaluation protocol** — the right split strategy (random, stratified, chronological, group-based) and the right primary metric for the task
5. **Selects exactly 3 models** — one baseline, one classical, one specialized — from a fixed catalog of 10
6. **Trains all 3** with optional Optuna hyperparameter tuning, on an identical train/val/test split
7. **Evaluates all 3** on the same frozen test set using the same metrics, then ranks them
8. **Assembles a dashboard bundle** — leaderboard, model cards, diagnostic plots, and a comparison chart

LLM assistance (OpenAI GPT-4o-mini) is used for dataset profiling and preprocessing planning when an API key is available. The pipeline degrades gracefully to heuristic-only mode without one.

---

## How It Works

The pipeline is built around **9 specialized agents**. Each agent reads its inputs from a shared run workspace and writes its outputs as JSON artifacts. No agent calls another agent's code directly — all communication happens through files on disk, governed by a central `job_manifest.json`.

```
User
 │
 │  run_pipeline("inputs/ETTh1.csv", config)
 ▼
orchestrator.py
 │
 ├─► [1] IngestionAgent          → dataset_profile.json
 │         LLM-assisted column profiling and semantic metadata
 │
 ├─► [2] ProblemClassificationAgent  → task_spec.json
 │         Identifies task type, target column, time/group columns
 │
 ├─► [3] PreprocessingPlanningAgent  → processed_data.csv
 │   [4] EvaluationProtocolAgent     → eval_protocol.json
 │         Runs in parallel. Plans and executes preprocessing.
 │         Defines split strategy and metric set.
 │
 ├─► [5] ModelSelectionAgent     → selected_models.json
 │         Picks exactly 3 models (baseline + classical + specialized)
 │
 ├─► [6] TrainingAgent           → trained_models/ + training_results.json
 │         Splits data once. Optionally tunes with Optuna. Trains all 3.
 │
 ├─► [7] EvaluationAgent         → evaluation_report.json + plots/
 │         Re-derives the identical test set. Evaluates all 3 models.
 │         Generates diagnostic plots and a ranked comparison table.
 │
 └─► [8] ArtifactAssemblyAgent   → dashboard/
           Assembles leaderboard, model cards, plots, and pipeline log.
```

Every run produces a self-contained workspace under `runs/{run_id}/`:

```
runs/{run_id}/
├── job_manifest.json          # Single source of truth; all stage statuses and artifact paths
├── artifacts/
│   ├── dataset_profile.json
│   ├── task_spec.json
│   ├── preprocessing_plan.json
│   ├── preprocessing_manifest.json
│   ├── processed_data.csv
│   ├── eval_protocol.json
│   ├── selected_models.json
│   ├── training_results.json
│   ├── evaluation_report.json
│   └── comparison_table.json
├── trained_models/
│   ├── XGBoost/
│   ├── RandomForest/
│   └── ARIMA/
├── plots/
│   ├── XGBoost_eval.png
│   ├── RandomForest_eval.png
│   └── ARIMA_eval.png
└── dashboard/
    ├── leaderboard.json
    ├── run_summary.json
    ├── pipeline_log.json
    ├── model_cards/
    ├── plots/
    └── artifacts/
```

---

## Model Catalog

The pipeline always selects **exactly 3 models** — one per tier — from the fixed internal catalog.

| Model | Tier | Compatible Tasks |
|---|---|---|
| DummyRegressor | baseline | regression, time-series |
| DummyClassifier | baseline | classification |
| LinearModel (Ridge / Logistic) | baseline | regression, classification |
| RandomForest | classical | regression, classification, time-series |
| XGBoost | classical | regression, classification, time-series |
| LightGBM | classical | regression, classification, time-series |
| SVR | classical | regression |
| ARIMA | specialized | time-series |
| LSTM | specialized | time-series |
| Chronos (Amazon, zero-shot) | specialized | time-series |

**Selection rules by task:**

| Task | Baseline | Classical | Specialized |
|---|---|---|---|
| `tabular_regression` | DummyRegressor | XGBoost | SVR |
| `tabular_classification` | DummyClassifier | XGBoost | RandomForest |
| `time_series_forecasting` | ARIMA | XGBoost | Chronos |

---

## Prerequisites

- Python 3.10+
- (Optional) An OpenAI API key for LLM-assisted profiling and preprocessing planning

---

## Installation

```bash
git clone https://github.com/wesdeal/SDP-44.git
cd SDP-44
pip install -r requirements.txt
```

To enable LLM-assisted mode, set your OpenAI API key:

```bash
export OPENAI_API_KEY=sk-...
```

Or create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

---

## Running the Pipeline

```python
from orchestrator import run_pipeline

dashboard_path = run_pipeline(
    input_file="inputs/ETTh1.csv",
    config={
        "runs_dir": "runs",
        "tune_hyperparameters": False,
        "n_optuna_trials": 20,
        "random_seed": 42,
    }
)

print(f"Dashboard bundle: {dashboard_path}")
```

With hyperparameter tuning enabled:

```python
dashboard_path = run_pipeline(
    input_file="inputs/AirPassengers.csv",
    config={
        "runs_dir": "runs",
        "tune_hyperparameters": True,
        "n_optuna_trials": 30,
        "random_seed": 42,
    }
)
```

---

## Sample Datasets

Three datasets are included in `inputs/` to try immediately:

| File | Description | Task Type |
|---|---|---|
| `ETTh1.csv` | Electricity transformer temperature (hourly) | Time-series forecasting |
| `ETTh2.csv` | Electricity transformer temperature (hourly, 2nd station) | Time-series forecasting |
| `AirPassengers.csv` | Monthly airline passengers 1949–1960 | Time-series forecasting |

---

## Configuration Options

| Key | Type | Default | Description |
|---|---|---|---|
| `runs_dir` | str | `"runs"` | Base directory for run workspaces |
| `tune_hyperparameters` | bool | `false` | Enable Optuna hyperparameter tuning |
| `n_optuna_trials` | int | `20` | Number of Optuna trials per model |
| `random_seed` | int | `42` | Seed for all random operations |
| `forecast_horizon` | int | `10` | Steps ahead to forecast (time-series only) |

---

## Evaluation Metrics

Metrics are assigned automatically based on the detected task type.

| Task | Primary Metric | Also Computed |
|---|---|---|
| `tabular_regression` | RMSE | MAE, MAPE, R² |
| `tabular_classification` (binary) | ROC-AUC | Accuracy, F1 (weighted) |
| `tabular_classification` (multiclass) | F1 (weighted) | Accuracy, ROC-AUC |
| `time_series_forecasting` | MAE | RMSE, MAPE, sMAPE |

All 3 selected models are always evaluated on the **identical frozen test set** using the **identical metric functions** so results are directly comparable.

---

## Project Structure

```
SDP-44/
├── orchestrator.py                  # Pipeline entry point
├── agents/
│   ├── ingestion_agent.py           # Column profiling + LLM analysis
│   ├── problem_classification_agent.py
│   ├── preprocessing_planning_agent.py
│   ├── preprocessing_execution.py   # Deterministic transform executor
│   ├── evaluation_protocol_agent.py
│   ├── model_selection_agent.py
│   ├── training_agent.py
│   ├── evaluation_agent.py
│   └── artifact_assembly_agent.py
├── core/
│   ├── manifest.py                  # Thread-safe manifest read/write
│   ├── splitter.py                  # All split strategies
│   ├── metric_engine.py             # Unified metric computation
│   └── schema_validator.py
├── models/
│   ├── base_model.py                # Abstract base class
│   ├── model_registry.py            # Factory and registry
│   ├── hyperparameter_tuner.py      # Optuna integration
│   ├── tree_based/
│   ├── linear/
│   ├── tabular/
│   ├── time_series/
│   └── dummy/
├── catalogs/
│   ├── model_catalog.json           # Model pool definition
│   └── transformer_catalog.json     # Allowed preprocessing transforms
├── inputs/                          # Sample datasets
├── runs/                            # Pipeline run outputs (git-ignored)
└── requirements.txt
```

---

## Running Tests

```bash
pytest tests/
```

Unit tests cover the manifest, splitter, metric engine, schema validator, and problem classification agent. Integration tests run full pipeline passes for regression and time-series tasks.

---

## Design Principles

**Artifact-based communication.** Agents never import each other. All data flows through JSON files in the run workspace. This makes every stage independently inspectable and restartable.

**Deterministic evaluation.** The test split is computed once from a fixed seed. The evaluation agent re-derives the identical test set from the same parameters rather than receiving it as a file — guaranteeing all models are compared on byte-identical data.

**Graceful degradation.** LLM calls include retry logic and fall back to heuristic defaults on failure, so the pipeline always completes even without an API key or network access.

**Fixed model catalog.** Model selection is rule-based, not LLM-driven. The catalog gates which models are compatible with each task type, and the tier constraint (baseline + classical + specialized) ensures every run includes a meaningful baseline for comparison.
