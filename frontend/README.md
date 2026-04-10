# Pipeline Frontend

A dark, dense, terminal-inspired React dashboard for visualizing a pipeline run
from the `Pipeline/` backend. Currently runs against mock data; the data layer
is structured so the backend can be wired in by changing one function.

## Stack

- React 18 + Vite
- Recharts (for charts, added in Phase 3)
- Plain CSS modules + a single global theme file (`src/styles/theme.css`)
- No TypeScript, no Tailwind, no UI library

## Getting started

```bash
cd frontend
npm install
npm run dev
```

Then open <http://localhost:5173>.

Other scripts:

- `npm run build` — production build into `dist/`
- `npm run preview` — preview the production build locally

## Folder structure

```
frontend/
├── index.html
├── package.json
├── vite.config.js
└── src/
    ├── main.jsx                # entry — mounts <App/>
    ├── App.jsx                 # loads run data, renders shell + dashboard
    ├── layout/
    │   ├── AppShell.jsx        # top bar + content frame
    │   └── AppShell.module.css
    ├── pages/
    │   ├── RunDashboard.jsx    # main page, composes section components
    │   └── RunDashboard.module.css
    ├── components/             # populated incrementally per phase
    ├── data/
    │   ├── mockRun.js          # full mock run; mirrors real artifact JSON shapes
    │   └── api.js              # getRun() — currently returns mockRun
    ├── hooks/
    ├── utils/
    │   └── format.js           # number / duration formatters
    └── styles/
        ├── theme.css           # design tokens (colors, spacing, type)
        └── global.css          # resets + base typography + grid texture
```

## Mock data → real backend

The mock object in `src/data/mockRun.js` mirrors the union of these backend
artifacts written to `runs/{run_id}/artifacts/`:

- `task_spec.json`
- `dataset_profile.json`
- `eval_protocol.json`
- `selected_models.json`
- `training_results.json`
- `evaluation_report.json`
- `comparison_table.json`
- `preprocessing_plan.json`
- `preprocessing_manifest.json`

To wire in real data, replace the body of `getRun()` in `src/data/api.js` with
a `fetch` against your backend endpoint. As long as the response shape matches
`mockRun`, no component code needs to change.

## Phased build status

- **Phase 1 (current):** scaffolding, theme, app shell, mock data layer,
  placeholder section grid.
- Phase 2: run overview header, recommendation card, leaderboard.
- Phase 3: metric comparison charts (Recharts).
- Phase 4: per-model detail drilldowns.
- Phase 5: collapsible advanced details (split, preprocessing, features).
- Phase 6: polish + backend hookup.
