# Backend Wiring Guide

How to connect this frontend to real pipeline artifact data.

---

## TL;DR — one-function swap

Open `src/data/api.js` and replace the body of `getRun()`:

```js
// BEFORE (mock)
export async function getRun(/* runId */) {
  await new Promise((r) => setTimeout(r, 80));
  return mockRun;
}

// AFTER (real backend)
export async function getRun(runId = "latest") {
  const res = await fetch(`/api/runs/${runId}`);
  if (!res.ok) throw new Error(`HTTP ${res.status} — failed to load run`);
  return await res.json();
}
```

No component files need to change as long as the response shape matches
the mock (see section below).

---

## Response shape contract

The backend must return a single JSON object that merges these artifacts
from `runs/{run_id}/`:

| Frontend field | Artifact file | Notes |
|---|---|---|
| `run_id` | run-level | UUID string |
| `status` | run-level | `"completed"` \| `"running"` \| `"failed"` |
| `created_at` | run-level | ISO 8601 |
| `completed_at` | run-level | ISO 8601 or null |
| `task_spec` | `artifacts/task_spec.json` | full object |
| `dataset_profile` | `artifacts/dataset_profile.json` | full object |
| `eval_protocol` | `artifacts/eval_protocol.json` | must include `split_counts: { train, validation, test }` |
| `selected_models` | `artifacts/selected_models.json` | full object |
| `training_results` | `artifacts/training_results.json` | full object |
| `evaluation_report` | `artifacts/evaluation_report.json` | full object |
| `comparison_table` | `artifacts/comparison_table.json` | full object |
| `preprocessing_plan` | `artifacts/preprocessing_plan.json` | full object |
| `preprocessing_manifest` | `artifacts/preprocessing_manifest.json` | full object |
| `feature_engineering` | derived from preprocessing | see note below |
| `pipeline_metadata` | run-level summary | see note below |

### `feature_engineering` field

This field is not a separate artifact file. It is a derived summary the
backend should compute from `preprocessing_plan` and `preprocessing_manifest`.
Expected shape:

```json
{
  "raw_signals": ["HUFL", "HULL", ...],
  "lag_config": { "columns": [...], "lags": [...], "total_generated": 24 },
  "rolling_config": { "columns": [...], "windows": [...], "statistics": [...], "total_generated": 12 },
  "feature_groups": {
    "raw": [...], "rolling_mean": [...], "rolling_std": [...],
    "lag_features": [...], "target_lags": [...]
  },
  "total_features": 42,
  "rows_dropped_for_lags": 24
}
```

If this field is missing, `FeatureEngineeringPanel` and `PreprocessingDetailsPanel`
will render empty gracefully.

### `pipeline_metadata` field

Run-level summary object. Expected shape:

```json
{
  "run_id": "...",
  "task_type": "time_series_forecasting",
  "target_col": "OT",
  "time_col": "date",
  "primary_metric": "MAE",
  "models_tested": 3,
  "hyperparameter_source": "default",
  "using_mock_data": false,
  "evaluation_timestamp": "...",
  "pipeline_version": "0.1.0",
  "dataset_name": "ETTh1"
}
```

---

## Key metric formatting notes

**MAPE and sMAPE** are stored as fractions (e.g. `0.122`) in artifact JSON.
The frontend multiplies by 100 for display (`12.2%`). The real backend must
store them the same way — as fractions, not percentages — or
`formatMetricByName()` in `src/utils/format.js` will show incorrect values.

---

## Where mock data is consumed

All components access run data through the service layer in
`src/services/runService.js`. The service functions are:

| Function | Used by | Artifact sources |
|---|---|---|
| `deriveRunHeader(run)` | `AppShell`, `RunOverviewHeader` | `run_id`, `status`, `task_spec`, `dataset_profile` |
| `deriveSummaryStats(run)` | `SummaryCards` | `task_spec`, `eval_protocol`, `comparison_table` |
| `deriveLeaderboardRows(run)` | `LeaderboardTable` | `comparison_table`, `training_results`, `evaluation_report` |
| `deriveWinner(run)` | `RecommendationCard` | `comparison_table`, `eval_protocol`, `selected_models` |
| `deriveModelList(run)` | `ModelDetailSection` | `comparison_table`, `training_results`, `evaluation_report`, `selected_models` |
| `deriveMetricChartRows(run)` | `MetricComparisonSection` | `comparison_table`, `training_results`, `evaluation_report` |
| `deriveAdvancedDetails(run)` | `AdvancedDetailsSection` | all preprocessing + metadata fields |

---

## Run-picker / multi-run navigation

`src/data/api.js` also exports `listRuns()` (stub). When the backend is
ready, implement it as:

```js
export async function listRuns() {
  const res = await fetch("/api/runs");
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return await res.json(); // expects: [{ run_id, status, created_at, dataset_name }]
}
```

The `RunPicker` component (not yet built) would call `listRuns()` to populate
a dropdown of available run IDs, then pass the selected ID to `getRun(id)`.

---

## Local development against a real backend

Assuming the Python backend runs on `localhost:8000`:

1. Add a proxy to `frontend/vite.config.js`:

```js
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
});
```

2. Replace `getRun()` as shown at the top of this file.
3. `npm run dev` — the frontend will proxy `/api/runs/...` to the backend.

---

## Model-specific assumptions in the UI

| Component | Assumption | Artifact field |
|---|---|---|
| `TrainingHistoryChart` | Only renders when `training_history` is non-null | `training_results.models[].training_history` |
| `ModelPlotPanel` | Renders a placeholder until `plot_path` image is served | `evaluation_report.models[].plot_path` |
| `RecommendationCard` | Expects `comparison_table.ranking[0].is_best === true` | `comparison_table.ranking[].is_best` |
| `LeaderboardTable` | Expects exactly 3 models | `comparison_table.ranking` (length 3) |
| `MetricBarChart` | MAPE/sMAPE as fractions | `comparison_table.ranking[].all_metrics` |
