# Frontend — Backend Integration Guide

This document explains how to connect the Pipeline dashboard to a live backend,
how the data-fetching and adapter layers work, and how to switch between mock
and live modes.

---

## Architecture overview

```
Browser
  └─ App.jsx                        reads ?runId= from URL
       └─ src/data/api.js            mock/live router
            ├── [mock mode]          returns mockRun.js with a small delay
            └── [live mode]
                 ├─ src/services/api/dashboardFetcher.js
                 │    fetches all artifacts in parallel
                 ├─ src/services/api/artifactClient.js
                 │    per-artifact fetch helpers (GET /api/runs/:id/artifacts/:name)
                 └─ src/services/adapters/runAdapter.js
                      normalizes raw JSON → frontend Run shape
                           ↓
                  src/services/runService.js    (unchanged)
                  React components              (unchanged)
```

No component file changes are needed when switching between mock and live mode.

---

## Environment variables

Copy `.env.example` to `.env.local` and fill in your values.

| Variable | Description | Default |
|---|---|---|
| `VITE_API_BASE` | Backend origin (e.g. `http://localhost:8000`). Empty = mock mode. | `""` |
| `VITE_USE_MOCK` | Force mock mode even when `VITE_API_BASE` is set. | `false` |
| `VITE_DEFAULT_RUN_ID` | Run UUID to load on startup (overridden by `?runId=`). | `299970a3-…` |
| `VITE_PLOT_BASE_URL` | URL prefix for evaluation plot images. | `/plots` |

---

## Running in mock mode (default)

```bash
cd frontend
npm run dev
```

No `.env.local` required. The app uses `src/data/mockRun.js` and shows a live,
fully functional dashboard with the real run's numbers baked in.

---

## Running in live mode

1. Start the Python backend (must expose the artifact API described below).
2. Create `frontend/.env.local`:
   ```
   VITE_API_BASE=http://localhost:8000
   VITE_USE_MOCK=false
   VITE_DEFAULT_RUN_ID=299970a3-0f86-4c91-bc05-4e0f3995ee43
   ```
3. Start the dev server:
   ```bash
   cd frontend
   npm run dev
   ```
   The Vite dev server automatically proxies `/api/*` and `/plots/*` to the
   backend origin, so there are no CORS issues during development.

---

## Required backend endpoints

### Artifact files

```
GET /api/runs/:runId/artifacts/task_spec.json
GET /api/runs/:runId/artifacts/dataset_profile.json
GET /api/runs/:runId/artifacts/eval_protocol.json
GET /api/runs/:runId/artifacts/selected_models.json
GET /api/runs/:runId/artifacts/training_results.json
GET /api/runs/:runId/artifacts/evaluation_report.json
GET /api/runs/:runId/artifacts/comparison_table.json
GET /api/runs/:runId/artifacts/preprocessing_plan.json
GET /api/runs/:runId/artifacts/preprocessing_manifest.json
```

Each endpoint returns the raw artifact JSON file. A `404` response means the
artifact is not yet available; the dashboard degrades gracefully for any missing
artifact.

### Run metadata

```
GET /api/runs/:runId
```

Returns `{ run_id, status, created_at, completed_at }`. Used to populate the
top status bar and run overview header.

### Run list (optional)

```
GET /api/runs
```

Returns `[{ run_id, status, created_at, dataset_name }, ...]`. Used if a
run-picker UI is added. Not required for the single-run dashboard view.

### Plot images (optional)

```
GET /plots/:runId/:filename.png
```

Evaluation plot images are referenced by filename in `evaluation_report.json`.
The adapter converts absolute filesystem paths to `/plots/:runId/:filename`.
If this endpoint is not available, `ModelPlotPanel` shows its styled placeholder.

---

## Adapter layer — field normalization

`src/services/adapters/runAdapter.js` handles these differences between real
artifact JSON and the mock shape the UI was built against:

| Difference | Real artifact | Normalized (UI) |
|---|---|---|
| Metric key casing | `mae`, `rmse`, `mape`, `smape` | `MAE`, `RMSE`, `MAPE`, `sMAPE` |
| XGBoost training history | `training_history.evals_result.validation_1.rmse[]` | `training_history.validation_1[]` + `.metric` |
| Split counts location | `training_results.split_info.n_train/n_val/n_test` | `eval_protocol.split_counts.train/validation/test` |
| primary_metric casing | `"mae"` | `"MAE"` |
| Plot paths | Absolute filesystem path | `/plots/:runId/:filename` |

If the real backend adds fields that differ from the above, update only
`runAdapter.js` — no component changes required.

---

## Run selection

The dashboard loads a run based on (in priority order):

1. `?runId=<uuid>` URL query parameter
2. `VITE_DEFAULT_RUN_ID` environment variable
3. Hard-coded fallback UUID: `299970a3-0f86-4c91-bc05-4e0f3995ee43`

Example: `http://localhost:5173/?runId=abc123-...`

---

## Optional / assumed fields

These backend fields are used when present but gracefully omitted when absent:

| Field | Where | Used by |
|---|---|---|
| `dataset_profile.llm_analysis` | `dataset_profile.json` | Overview header subtitle |
| `comparison_table.ranking[].interpretation` | `comparison_table.json` | RecommendationCard |
| `selected_models.models_rejected` | `selected_models.json` | ModelSelectionPanel |
| `feature_engineering` | Not a real artifact — derived from `preprocessing_plan` | FeatureEngineeringPanel |
| `pipeline_metadata` | Not a real artifact | PipelineMetadataPanel |
| Plot images | `evaluation_report.models[].plot_path` | ModelPlotPanel |

---

## Minimal backend stub (FastAPI example)

If you need a quick local backend to serve existing run artifacts:

```python
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
import json, pathlib

app = FastAPI()
RUNS_DIR = pathlib.Path("../runs")

@app.get("/api/runs/{run_id}")
def get_run_meta(run_id: str):
    manifest = RUNS_DIR / run_id / "job_manifest.json"
    if not manifest.exists():
        raise HTTPException(404)
    data = json.loads(manifest.read_text())
    return {"run_id": run_id, "status": data.get("status", "completed"),
            "created_at": data.get("created_at"), "completed_at": data.get("completed_at")}

@app.get("/api/runs/{run_id}/artifacts/{filename}")
def get_artifact(run_id: str, filename: str):
    path = RUNS_DIR / run_id / "artifacts" / filename
    if not path.exists():
        raise HTTPException(404)
    return json.loads(path.read_text())

app.mount("/plots", StaticFiles(directory=str(RUNS_DIR)), name="plots")
# Note: adjust the StaticFiles directory so /plots/{run_id}/{file} resolves
# to runs/{run_id}/plots/{file} on your filesystem.
```

Start with: `uvicorn server:app --reload --port 8000`
