# Phase 0 & 1 — Implementation Notes

**Status:** Complete
**Tests passing:** 51 / 51
**Authoritative design reference:** `AGENT_ARCHITECTURE.md`

---

## Phase 0 — Scaffolding

No logic. Creates the workspace and static configuration files that every
subsequent phase depends on.

---

### 0.1 — Directory Tree

Created the following directory structure from scratch:

```
Pipeline/
├── agents/               # One module per pipeline stage (populated Phase 4+)
├── catalogs/             # Static JSON config files for models and transformers
├── core/                 # Shared utilities used by all agents
├── runs/                 # Runtime workspaces — one sub-directory per run_id
├── tests/
│   ├── unit/             # Per-module unit tests
│   └── integration/      # End-to-end pipeline tests
└── models/
    ├── dummy/            # DummyRegressor / DummyClassifier (Phase 3)
    └── linear/           # LinearModel / Ridge / LogisticRegression (Phase 3)
```

`__init__.py` files were added to all Python packages so they are importable.

---

### 0.2 — `catalogs/model_catalog.json`

Defines the fixed pool of 10 models. Each entry carries the fields that
`ModelSelectionAgent` reads at runtime:

| Field | Purpose |
|---|---|
| `name` | Human-readable model name |
| `tier` | `baseline`, `classical`, or `specialized` |
| `compatible_tasks` | List of task types the model can handle |
| `requires_features` | Whether the model needs engineered feature columns |
| `cpu_feasible` | All models are CPU-feasible per project scope |
| `available` | Runtime importability flag — `false` means the model is skipped |
| `registry_key` | Key used to look up the implementation in `model_registry.py` |
| `description` | Short prose description |

**Model pool (10 total):**

| Model | Tier | Compatible Tasks | Available |
|---|---|---|---|
| DummyRegressor | baseline | tabular_regression, time_series_forecasting | `true` |
| DummyClassifier | baseline | tabular_classification | `true` |
| LinearModel | baseline | tabular_regression, tabular_classification | `true` |
| RandomForest | classical | tabular_regression, tabular_classification, time_series_forecasting | `true` |
| XGBoost | classical | tabular_regression, tabular_classification, time_series_forecasting | `true` |
| LightGBM | classical | tabular_regression, tabular_classification, time_series_forecasting | **`false`** |
| ARIMA | specialized | time_series_forecasting | `true` |
| LSTM | specialized | time_series_forecasting | `true` |
| Chronos | specialized | time_series_forecasting | **`false`** |
| SVR | classical | tabular_regression | `true` |

**Why LightGBM and Chronos are `false`:**
- LightGBM adds an external `pip` dependency with known platform-specific install
  failures. It is set to `false` until the pipeline passes its first end-to-end
  integration test (Phase 6.3).
- Chronos has optional dependencies and a `try/except` import guard in the
  existing registry. Its catalog entry reflects runtime availability, not
  aspirational availability.

---

### 0.3 — `catalogs/transformer_catalog.json`

Defines the 10 preprocessing transformers that `PreprocessingPlanningAgent` may
select. Each entry specifies:

| Field | Purpose |
|---|---|
| `name` | Transformer identifier (used in `preprocessing_plan.json`) |
| `allowed_modalities` | Which data modalities permit this transformer |
| `params` | Configuration parameters the transformer accepts |

**Transformer list:**

| Transformer | Allowed Modalities | Params |
|---|---|---|
| `imputation` | tabular_iid, time_series, grouped_tabular | `strategy` |
| `z_norm` | tabular_iid, time_series, grouped_tabular | — |
| `min_max` | tabular_iid, time_series, grouped_tabular | — |
| `log_transform` | tabular_iid, tabular_regression | — |
| `remove_outliers` | tabular_iid | `method`, `threshold` |
| `detrend` | time_series | `type` |
| `differencing` | time_series | — |
| `smoothing` | time_series | `window` |
| `label_encode` | tabular_iid, grouped_tabular | — |
| `onehot_encode` | tabular_iid, grouped_tabular | `max_cardinality` |

The catalog is the sole source of truth for which transformers are legal in a
given plan. `PreprocessingPlanningAgent` validates its LLM output against this
catalog before accepting it.

---

## Phase 1 — Manifest + Skeleton Orchestrator

Validates the manifest-driven wiring contract before any real agent is written.
Catches integration failures at the cheapest possible moment.

---

### 1.1 — `core/manifest.py`

Three public functions constitute the only permitted way to read or write
`job_manifest.json`. All agents must use this module; no agent may open the
manifest file directly.

#### `initialize_manifest(input_file, config) -> dict`

Creates a new run workspace and writes the initial manifest to:

```
{config["runs_dir"]}/{run_id}/job_manifest.json
```

- Generates a UUID-4 `run_id`.
- Resolves `input_file` to an absolute path.
- Detects file format from extension (`.csv`, `.parquet`, `.json`); raises
  `ValueError` for unsupported extensions.
- Initialises all 8 stages with `status: "pending"`, `started_at: null`,
  `completed_at: null`, `artifacts: {}`, `error: null`.
- Writes atomically via temp-file + `os.replace` to prevent partial reads.
- Returns the manifest dict (identical to what is on disk).

#### `update_stage(manifest_path, stage_name, status, artifacts=None, error=None)`

Updates a single stage in the on-disk manifest.

- Reads current manifest, modifies only the named stage and root `updated_at`.
- Sets `started_at` when `status == "running"`.
- Sets `completed_at` when `status` is `"completed"`, `"failed"`, or
  `"partial_failure"`.
- Replaces `artifacts` dict only when the kwarg is not `None` (passing `None`
  leaves existing artifacts untouched).
- Validates `status` against the five permitted values; raises `ValueError` for
  unknown status strings.
- Raises `KeyError` for unknown stage names, `FileNotFoundError` if the manifest
  does not exist.
- Writes atomically.

#### `read_manifest(manifest_path) -> dict`

Reads and returns the current manifest from disk. Raises `FileNotFoundError` if
the path does not exist.

---

### 1.2 — `orchestrator.py` (Skeleton)

`run_pipeline(input_file, config) -> str` is the **sole entry point** for
pipeline execution (enforced by `CLAUDE.md §3.4`).

#### Internal structure

Eight private stub functions (`_stub_ingestion`, `_stub_problem_classification`,
…) act as stand-ins for the real agents. Each stub:

1. Derives all paths from `manifest_path` via `_run_dir()` and
   `_artifacts_dir()` — no hardcoded paths.
2. Writes a placeholder artifact (empty JSON object or empty file/directory).
3. Returns an `artifacts` dict that the orchestrator passes to `update_stage`.

Two private path helpers keep path logic in one place:

- `_run_dir(manifest_path)` — returns the run directory (parent of the manifest).
- `_artifacts_dir(manifest_path)` — returns `{run_dir}/artifacts/`.

Two private write helpers:

- `_write_empty_json(path)` — creates parent dirs and writes `{}`.
- `_write_empty_file(path)` — creates parent dirs and writes an empty file.

#### Execution flow

```
initialize_manifest(input_file, config)
    │
    ▼ for each stage in _STAGES (canonical order):
    │   update_stage(... "running")
    │   artifacts = stub_fn(manifest_path)
    │   update_stage(... "completed", artifacts=artifacts)
    │
    ▼
read_manifest(manifest_path)
    │
    ▼
return manifest["stages"]["artifact_assembly"]["artifacts"]["dashboard_bundle"]
```

#### Artifacts written per stub

| Stage | Artifacts |
|---|---|
| ingestion | `artifacts/dataset_profile.json` |
| problem_classification | `artifacts/task_spec.json` |
| preprocessing_planning | `artifacts/preprocessing_plan.json`, `artifacts/processed_data.csv`, `artifacts/preprocessing_manifest.json` |
| evaluation_protocol | `artifacts/eval_protocol.json` |
| model_selection | `artifacts/selected_models.json` |
| training | `artifacts/training_results.json`, `trained_models/` (directory) |
| evaluation | `artifacts/evaluation_report.json`, `artifacts/comparison_table.json`, `plots/` (directory) |
| artifact_assembly | `dashboard/` (directory) |

---

### 1.3 — Tests

#### `tests/unit/test_manifest.py` — 45 tests

Covers `core/manifest.py` in three test classes:

**`TestInitializeManifest` (22 tests)**
- Returns a dict with all required top-level keys.
- `run_id` is a valid UUID-4; successive calls produce unique IDs.
- Initial `status` is `"pending"`.
- `created_at` and `updated_at` are valid ISO-8601 timestamps and are equal on
  initialization.
- `input.file_path` is absolute; format is correctly detected for `.csv`,
  `.parquet`, `.json`; `original_filename` is correct.
- All 8 stages are present and start with `pending`, `null` timestamps, empty
  `artifacts`, and `null` error.
- Config dict is stored verbatim.
- Manifest file is created on disk and its content matches the returned dict.
- Unsupported extension and missing extension both raise `ValueError`.

**`TestUpdateStage` (16 tests)**
- `"running"` sets `started_at`; does not set `completed_at`.
- `"completed"`, `"failed"`, `"partial_failure"` all set `completed_at`.
- `error` message is stored; `artifacts` dict is stored.
- `artifacts=None` leaves existing artifacts untouched.
- Only the named stage is modified; all others are unchanged.
- Root `updated_at` is refreshed; `created_at` is not.
- Full `running → completed` lifecycle works correctly.
- All 8 stages are individually updatable.
- Unknown stage raises `KeyError`; invalid status raises `ValueError`; missing
  manifest file raises `FileNotFoundError`.

**`TestReadManifest` (7 tests)**
- Returns a dict.
- Content matches the initialized manifest.
- Successive reads are deterministic.
- Reflects changes made by `update_stage`.
- Missing file raises `FileNotFoundError`.

#### `tests/integration/test_skeleton_pipeline.py` — 6 tests

End-to-end smoke test for the skeleton orchestrator. Uses
`tempfile.TemporaryDirectory` as `runs_dir` so the test is fully isolated and
`run_id` is never hardcoded. All assertions use the live manifest and filesystem
rather than mocks.

| Test | Assertion |
|---|---|
| `test_manifest_file_exists` | `job_manifest.json` is present on disk |
| `test_manifest_has_run_id` | Manifest carries a non-empty string `run_id` |
| `test_all_stages_completed` | All 8 canonical stages have `status == "completed"` |
| `test_stage_artifacts_exist_on_disk` | Every path in every stage's `artifacts` dict exists on the filesystem |
| `test_dashboard_directory_exists` | The returned path is an existing directory |
| `test_return_value_matches_manifest` | `run_pipeline()` return value equals `stages.artifact_assembly.artifacts.dashboard_bundle` |

---

## Test Run Results

```
$ python -m pytest tests/ -v

51 passed in 0.18s
```

All unit and integration tests pass deterministically.
