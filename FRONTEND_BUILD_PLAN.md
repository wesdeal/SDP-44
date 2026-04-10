# Frontend Build Plan

A persistent, phase-by-phase roadmap for the Pipeline project's React + Recharts
dashboard. This file is the single source of truth for the frontend effort and
should be updated as phases complete.

---

## 1. Project Goal

Build a polished, standalone frontend for the AI pipeline project that
visualizes the results of a single pipeline run — specifically the comparison
of the 3 trained models, their metrics, training metadata, and the underlying
strategy choices the pipeline made.

- **Stack:** React 18 + Vite, Recharts for visualization, plain CSS modules +
  a single global theme file.
- **Location:** a separate top-level `frontend/` directory at the project root,
  cleanly isolated from the Python backend.
- **Aesthetic:** dark, technical, data-console — feels like an analytics
  terminal or model arena, not a generic SaaS admin dashboard.
- **Data:** starts against a mock data layer that mirrors the real artifact
  JSON shapes; backend wiring is a one-function swap later.

---

## 2. Design Direction

The UI should feel like a piece of real analytical tooling.

- **Dark analytics console** — near-black charcoal background (`#0a0d12`),
  panel surfaces a touch lighter, thin borders, subtle 1px graph-paper grid
  texture behind the content.
- **Neon-accent palette** — cyan as the primary accent (best model, brand
  mark, key data), amber for secondary rank, purple for the specialized tier,
  green for success/status, red for failures.
- **High-density but readable layout** — dense data with generous internal
  spacing, monospace numbers, sentence-case labels in small caps for
  section headers.
- **Model arena / telemetry / ML evaluation dashboard feel** — borrows from
  trading terminals, observability dashboards, and ML experiment trackers.
- **Inspired by the screenshots the user shared** — high-contrast dark
  surfaces, neon highlights, panel-based grid composition.
- **Not a generic SaaS admin dashboard** — no blue gradients, no pastel
  cards, no rounded everything, no whitespace-as-decoration.

### Color tokens (already defined in `frontend/src/styles/theme.css`)

| Token | Value | Use |
|---|---|---|
| `--bg-0` | `#0a0d12` | page background |
| `--bg-1` | `#10141b` | panel background |
| `--bg-2` / `--bg-3` | `#161b25` / `#1c2230` | inset / hover |
| `--border` / `--border-strong` | `#1f2735` / `#2a3447` | thin panel borders |
| `--text-hi` / `--text-md` / `--text-lo` / `--text-dim` | `#e7ecf3` / `#b3bccd` / `#7c8699` / `#4a5466` | text hierarchy |
| `--accent-cyan` | `#22d3ee` | primary accent / best model |
| `--accent-amber` | `#fbbf24` | rank 2 / warnings |
| `--accent-purple` | `#a78bfa` | specialized tier |
| `--accent-green` / `--accent-red` | `#34d399` / `#f87171` | status |

---

## 3. Core Product Goals

The frontend is a decision-support tool. It should:

- **Compare the 3 tested models clearly** — leaderboard with explicit ranks,
  best model visually emphasized.
- **Show all major metrics**, not just the primary one — MAE, RMSE, MAPE,
  sMAPE, plus training duration and inference time.
- **Surface the winning model clearly** — a "Why this model won"
  recommendation card with the primary metric, the delta versus the other
  models, and a plain-English explanation.
- **Include detailed technical metadata** — hyperparameters, hyperparameter
  source, validation score, training duration, model tier, status.
- **Expose strategy details** — split strategy, split counts, preprocessing
  steps, lag features, rolling windows, model selection rationale.
- **Keep advanced technical information collapsible** so the default view
  stays clean and judges can drill in only when they want to.

---

## 4. Real Backend Context To Reflect

The frontend mock data must reflect the user's actual current run so the demo
feels grounded, not invented.

| Field | Value |
|---|---|
| Task type | `time_series_forecasting` |
| Target column | `OT` |
| Time column | `date` |
| Split strategy | `chronological` |
| Train / Validation / Test counts | `12177 / 1739 / 3480` |
| Primary ranking metric | `MAE` |
| Available metrics | `RMSE`, `MAE`, `MAPE`, `sMAPE` |
| Hyperparameter source | `default` |
| Feature engineering | lag features + rolling statistics |
| Model tiers present | `baseline`, `classical`, `specialized` |

### Current ranking

| Rank | Model | Tier | Notes |
|---|---|---|---|
| 1 | **XGBoost** | classical | Best MAE; training history (validation RMSE per boosting round) is available for the convergence chart. |
| 2 | RandomForest | classical | Robust comparator. |
| 3 | DummyRegressor | baseline | Mean predictor — establishes the floor any real model must beat. |

These exact values are already populated in
`frontend/src/data/mockRun.js` so every component built in later phases will
display realistic numbers.

---

## 5. Proposed Frontend Architecture

A separate `frontend/` directory at the project root, sibling to `agents/`,
`core/`, and `runs/`. Inside, a small, conventional `src/` layout that scales
phase by phase without restructuring.

```
frontend/
├── index.html
├── package.json
├── vite.config.js
├── .gitignore
├── README.md
└── src/
    ├── main.jsx                  # entry — mounts <App/>
    ├── App.jsx                   # loads run, renders shell + dashboard
    ├── layout/
    │   ├── AppShell.jsx          # sticky top bar + content frame
    │   └── AppShell.module.css
    ├── pages/
    │   ├── RunDashboard.jsx      # composes all section components
    │   └── RunDashboard.module.css
    ├── components/               # populated phase by phase
    │   ├── overview/             # Phase 2
    │   ├── leaderboard/          # Phase 2
    │   ├── recommendation/       # Phase 2
    │   ├── charts/               # Phase 3
    │   ├── model-detail/         # Phase 4
    │   └── advanced/             # Phase 5
    ├── data/
    │   ├── mockRun.js            # full mock run; mirrors artifact JSON
    │   └── api.js                # getRun() — currently returns mockRun
    ├── hooks/                    # added when first hook is justified
    ├── utils/
    │   └── format.js             # number / duration / metric formatters
    └── styles/
        ├── theme.css             # design tokens (CSS custom properties)
        └── global.css            # resets + base typography + grid texture
```

**Conventions:**

- One component = one folder when it has its own CSS module; small components
  may colocate in `components/<group>/`.
- All visual tokens come from `theme.css`. No hardcoded colors in component CSS.
- All formatting (numbers, durations, ids) goes through `utils/format.js`.
- All data access goes through `data/api.js`. Components never import `mockRun`
  directly.

---

## 6. Dashboard Information Architecture

The main dashboard page renders sections in this order, top to bottom:

1. **Run overview header** — run id, dataset, task type, target, split
   strategy, primary metric, # models tested, best model.
2. **Recommendation / winner card** — best model, primary metric value,
   delta vs other models, plain-English "why it won".
3. **Leaderboard** — 3-row table of all models, best row visually emphasized.
4. **Metric comparison charts** — grouped bar charts and/or small multiples
   across all metrics; metric selector for focused inspection.
5. **Per-model detail drilldown** — one panel per model with metrics,
   hyperparameters, training metadata, training history (where available),
   plot placeholder.
6. **Advanced collapsible technical details** — split info, preprocessing
   steps, feature engineering, model selection rationale, strategy notes.

The default view (sections 1–5) stays clean. Section 6 is collapsed by default.

---

## 7. Component Plan

Components likely to be built across phases. Names are tentative but the
responsibilities are firm.

### Layout & shell
- `AppShell` — sticky top bar (brand mark, run id, dataset name, task type,
  status pill) + content frame.

### Phase 2 — Overview / Recommendation / Leaderboard
- `RunOverviewHeader` — top metadata strip.
- `SummaryCard` — small reusable stat card (label + value + optional unit).
- `RecommendationCard` — winner block: best model, primary metric, deltas,
  plain-English explanation.
- `Leaderboard` — 3-row table with rank badge, model name, tier badge,
  primary metric, all metrics, training time, inference time, status; best
  row gets cyan border + glow.
- `TierBadge` — small colored pill (`baseline` / `classical` / `specialized`).
- `RankBadge` — `#1` / `#2` / `#3` with rank-specific accent color.

### Phase 3 — Charts
- `ChartContainer` — reusable Recharts wrapper that supplies the dark theme
  axes, grid, tooltip, and legend so individual charts stay short.
- `MetricComparisonPanel` — top-level section with metric selector and a
  grouped bar chart comparing the 3 models on the chosen metric.
- `MetricSelector` — chip-style toggle group (MAE / RMSE / MAPE / sMAPE /
  training_duration / inference_time_ms).
- `MetricSmallMultiples` — optional grid of one mini bar chart per metric
  for an at-a-glance overview.

### Phase 4 — Model drilldown
- `ModelDetailSection` — wrapper with model selector tabs.
- `ModelDetailPanel` — per-model block: metrics table, status, tier badge,
  hyperparameters, hyperparameter source, validation score, training duration,
  inference time.
- `TrainingHistoryChart` — line chart of validation metric over boosting
  rounds / epochs (only rendered when `training_history` is present).
- `PlotPlaceholder` — slot for `plot_path` images; renders a styled empty
  box until backend serves the image.

### Phase 5 — Advanced details
- `AdvancedDetailsAccordion` — collapsible container, all sections collapsed
  by default.
- `SplitInfoPanel` — split strategy, train/val/test counts, time column.
- `PreprocessingPanel` — ordered list of preprocessing steps (plan + manifest)
  with method, parameters, columns affected.
- `FeatureEngineeringPanel` — lag features, rolling windows, window sizes.
- `ModelSelectionPanel` — selection strategy, rationale per selected model,
  list of rejected models with reasons.
- `StrategyNotesPanel` — free-text strategy / decision notes if present.

---

## 8. Data Modeling Plan

The frontend has a single canonical `Run` object that mirrors the union of
backend artifacts. Mock first, real backend later — no component code changes
when swapping.

### Backend artifacts represented (under `runs/{run_id}/artifacts/`)

- `task_spec.json` — task type, target column, time column, forecast horizon,
  reasoning, warnings.
- `dataset_profile.json` — dataset name, row/column counts, per-column stats,
  LLM analysis (description, trend, seasonality).
- `eval_protocol.json` — split strategy, fractions, primary metric, metrics
  list, split counts.
- `selected_models.json` — selection strategy, the 3 chosen models with tier
  + rationale, models considered, models rejected with reasons.
- `training_results.json` — per-model status, hyperparameters, hyperparameter
  source, best validation score, training duration, training history,
  model path.
- `evaluation_report.json` — per-model test metrics, n_test_samples, inference
  time, plot path.
- `comparison_table.json` — ranked comparison: rank, model, tier, primary
  metric value, is_best flag, all metrics.
- `preprocessing_plan.json` / `preprocessing_manifest.json` — ordered
  preprocessing steps (planned + actually executed) with parameters and
  columns affected.

### Frontend `Run` object (already mocked in `src/data/mockRun.js`)

```js
{
  run_id, status, created_at, completed_at,
  task_spec:           { ...mirrors task_spec.json },
  dataset_profile:     { ...mirrors dataset_profile.json },
  eval_protocol:       { ...mirrors eval_protocol.json, plus split_counts },
  selected_models:     { ...mirrors selected_models.json },
  training_results:    { models: [...] },
  evaluation_report:   { models: [...] },
  comparison_table:    { ranked_by, ranking: [...] },
  preprocessing_plan:  { steps: [...] },
  preprocessing_manifest: { steps_applied: [...] },
}
```

### Access pattern

- Components call `getRun()` from `src/data/api.js` (currently returns the
  mock with an 80ms delay).
- Backend wiring later replaces only the body of `getRun()` with a `fetch`
  call. As long as the response shape matches, no component code changes.

---

## 9. Detailed Phase Plan

Each phase is small enough to ship and review in one pass. The user reviews
between phases.

### Phase 1 — Frontend scaffolding ✅ COMPLETE

- [x] Create top-level `frontend/` directory at the project root.
- [x] Initialize React + Vite app (package.json, vite.config.js, index.html).
- [x] Install React 18, React DOM, Recharts, `@vitejs/plugin-react`, Vite.
- [x] Establish folder structure (`src/{layout,pages,components,data,hooks,utils,styles}`).
- [x] Create design tokens in `src/styles/theme.css`.
- [x] Create `src/styles/global.css` (resets, base typography, grid texture).
- [x] Build `AppShell` (sticky top bar + content frame).
- [x] Build placeholder `RunDashboard` page with 6 labelled section cards.
- [x] Create `src/data/mockRun.js` with realistic data for all 9 artifacts.
- [x] Create `src/data/api.js` with `getRun()` returning the mock.
- [x] Create `src/utils/format.js` (number/duration/id formatters).
- [x] Write `frontend/README.md`.
- [x] Verify `npm install` and `npm run build` succeed.

### Phase 2 — Overview and leaderboard

- Build `RunOverviewHeader` displaying run id, dataset, task type, target,
  split strategy, primary metric, # models tested, best model.
- Build `SummaryCard` reusable stat tile.
- Build `RecommendationCard` ("Why XGBoost won") with primary metric value,
  delta versus rank 2 and rank 3, and a short plain-English explanation
  derived from the comparison table.
- Build `Leaderboard` table for all 3 models — rank badge, model name, tier
  badge, primary metric, all secondary metrics, training time, inference time,
  status. Best row visually emphasized (cyan border + subtle glow).
- Build `TierBadge` and `RankBadge` primitives.
- Wire all four section placeholders in `RunDashboard` to the new components.

### Phase 3 — Charts and comparisons

- Add `ChartContainer` Recharts wrapper that locks in dark theme axes, grid,
  tooltip, and legend so chart components stay short.
- Build `MetricSelector` (chip toggles for MAE / RMSE / MAPE / sMAPE /
  training_duration / inference_time_ms).
- Build `MetricComparisonPanel` containing the selector and a grouped Recharts
  bar chart comparing the 3 models on the selected metric.
- Optional: build `MetricSmallMultiples` — a responsive grid of mini bar
  charts, one per metric, for at-a-glance comparison.
- Make charts responsive (width fills container, sensible min height).
- Wire the metric comparison section in `RunDashboard`.

### Phase 4 — Model drilldown

- Build `ModelDetailSection` with tabs (one per model) or a stacked layout.
- Build `ModelDetailPanel` per model: metrics table, status, tier badge,
  hyperparameters, hyperparameter source, validation score, training duration,
  inference time.
- Build `TrainingHistoryChart` rendering the XGBoost validation RMSE curve
  from `training_results.models[].training_history`. Hide gracefully when
  history is null.
- Build `PlotPlaceholder` slot for `plot_path` images (empty styled box with
  label until the backend serves images).
- Wire the per-model detail section in `RunDashboard`.

### Phase 5 — Advanced technical details

- Build `AdvancedDetailsAccordion` (all sections collapsed by default, smooth
  expand/collapse).
- Build `SplitInfoPanel` (strategy, train/val/test counts, time column).
- Build `PreprocessingPanel` rendering the ordered steps from
  `preprocessing_plan` and `preprocessing_manifest` (method, parameters,
  columns affected, rows before/after).
- Build `FeatureEngineeringPanel` summarizing lag lists and rolling window
  configurations.
- Build `ModelSelectionPanel` (selection strategy, per-model rationale,
  rejected models with reasons).
- Build `StrategyNotesPanel` for free-text notes when present.
- Wire the advanced details section in `RunDashboard`.

### Phase 6 — Polish and backend integration prep

- Pass over spacing, typography, label wording, and color usage for
  consistency.
- Add proper loading and error states in `App.jsx` / `getRun()`.
- Improve number formatting consistency across components.
- Verify responsive behavior at common breakpoints (1440 / 1280 / 1024 / 768).
- Document the exact change to swap `getRun()` from mock to a real
  `fetch('/api/runs/:id')` in `frontend/README.md`.
- Optional: add a tiny `RunPicker` that lists available run ids (still mocked
  client-side) so judges can imagine multi-run navigation.

### Phase 7 — Real backend integration, live artifact wiring, and demo hardening

Connects the frontend to actual pipeline artifact JSON rather than relying
solely on mock data. Every existing component remains unchanged; only the
data-access and adapter layers are added or modified.

#### Files added

| File | Purpose |
|---|---|
| `src/services/api/artifactClient.js` | Per-artifact fetch helpers. Each function fetches one artifact file (`GET /api/runs/:runId/artifacts/:name.json`). Returns `null` on 404 so missing artifacts degrade gracefully. |
| `src/services/api/dashboardFetcher.js` | `getRunDashboardData(runId)` — fetches all 9 artifacts in parallel via `Promise.all`. |
| `src/services/adapters/runAdapter.js` | `adaptRawArtifacts(bundle)` — normalizes raw artifact JSON into the frontend Run shape. Handles metric key casing (lowercase → uppercase), XGBoost training history flattening, split count location difference, and plot path resolution. |
| `frontend/.env.example` | Documents all environment variables with descriptions and defaults. |
| `frontend/INTEGRATION.md` | Full integration guide: endpoints, adapter contract, env vars, mock/live toggle, minimal backend stub. |

#### Files modified

| File | Change |
|---|---|
| `src/data/api.js` | Routes `getRun()` to either mock or live fetcher based on `VITE_API_BASE` and `VITE_USE_MOCK`. Exports `DEFAULT_RUN_ID`. |
| `vite.config.js` | Dev-server proxy: when `VITE_API_BASE` is set, forwards `/api/*` and `/plots/*` to the backend origin to avoid CORS issues. |
| `src/App.jsx` | Reads `?runId=<uuid>` from the URL query string; passes it to `getRun()` so judges can navigate between runs without a code change. |

#### Key normalization decisions (runAdapter.js)

- **Metric keys**: Real artifacts use lowercase (`mae`, `rmse`); UI expects uppercase (`MAE`, `RMSE`). The adapter lifts all keys.
- **Training history**: XGBoost real shape is `training_history.evals_result.validation_1.rmse[]`. Component expects flat `training_history.validation_1[]` + `.metric`. Adapter unwraps the nested structure.
- **Split counts**: Real data has `training_results.split_info.{n_train,n_val,n_test}`; UI reads `eval_protocol.split_counts.{train,validation,test}`. Adapter backfills the field.
- **Plot paths**: Real paths are absolute filesystem paths. Adapter strips to `/plots/:runId/:filename`; Vite proxy forwards to backend.

#### Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `VITE_API_BASE` | Backend origin URL. Empty = mock mode. | `""` |
| `VITE_USE_MOCK` | Force mock even when API base is set. | `false` |
| `VITE_DEFAULT_RUN_ID` | Default run UUID. Overridden by `?runId=`. | `299970a3-…` |
| `VITE_PLOT_BASE_URL` | URL prefix for plot images. | `/plots` |

#### How to run

**Mock mode (no backend needed):**
```bash
cd frontend && npm run dev
```

**Live mode:**
```bash
# frontend/.env.local
VITE_API_BASE=http://localhost:8000
VITE_USE_MOCK=false

cd frontend && npm run dev
# Backend must serve /api/runs/:id/artifacts/:name.json and /plots/:id/:file
```

**Load a specific run:**
```
http://localhost:5173/?runId=299970a3-0f86-4c91-bc05-4e0f3995ee43
```

#### Backend gaps / assumptions for a fully live demo

- The backend must expose `GET /api/runs/:runId/artifacts/:filename` for each artifact file.
- Plot images require `GET /plots/:runId/:filename` (or a static file server that resolves the same path).
- `feature_engineering` is not a standalone artifact in the real pipeline — it is derived from `preprocessing_plan`. The `FeatureEngineeringPanel` degrades to an empty state if absent. A future backend enhancement could denormalize this field onto the run response.
- `dataset_profile.json` contents depend on whether the ingestion agent ran successfully; the header subtitle is optional.
- `run_id` must be known before navigation — no automatic run discovery is implemented yet.

---

## 10. Working Agreement For Future Phases

The user wants this built phase by phase, with explicit checkpoints. The
following rules apply to every phase from here on:

1. **One phase at a time.** Do not start the next phase until the user
   approves the previous one.
2. **Explain before changing code.** At the start of each phase, summarize
   what will be built and which files will be created or modified.
3. **Keep changes scoped.** A phase only touches files belonging to that
   phase's section, plus the `RunDashboard` wiring for that section. No
   opportunistic refactoring.
4. **After each phase, summarize.** Report what was created, the file paths,
   how to run and verify, and what to inspect before approving the next
   phase.
5. **Prefer creating new files** over modifying existing ones, except for
   `RunDashboard.jsx`, which is the natural composition point for each new
   section component.
6. **Stop and wait for approval** before moving on. No autopilot.
7. **State assumptions clearly** when making engineering judgment calls.
8. **Update this plan file** as phases complete — flip the phase to `✅
   COMPLETE` and note any deviations from the original plan.

---

## 11. Definition of Done

The frontend is "demo ready" when all of the following are true:

- `cd frontend && npm run dev` boots cleanly with no console errors.
- The dashboard loads against the mock run and renders all six sections:
  overview, recommendation, leaderboard, metric comparison, per-model detail,
  advanced details.
- The winning model (XGBoost in the current run) is unambiguously emphasized
  in both the leaderboard and the recommendation card.
- All four metrics (MAE, RMSE, MAPE, sMAPE) plus training duration and
  inference time can be compared across the 3 models without scrolling
  through raw JSON.
- The XGBoost training history is visualized as a convergence curve.
- Advanced technical details (split, preprocessing, feature engineering,
  selection rationale) are accessible via the collapsible accordion and
  do not clutter the default view.
- The visual aesthetic matches the design direction in section 2 — dark
  charcoal background, neon accents, thin borders, dense but readable
  panels, terminal-leaning typography, subtle grid texture.
- The data layer is structured so swapping mock for a real backend is a
  one-function change in `src/data/api.js`, with the swap documented in
  `frontend/README.md`.
- The build succeeds: `npm run build` produces a clean `dist/` bundle.
- The frontend feels like a real piece of analytical tooling — something a
  judge would believe an ML team actually uses internally.
