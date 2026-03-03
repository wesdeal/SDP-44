# Team Implementation Plan

**Revised TODO list, technical review notes, and work distribution for 5 engineers.**
**Authoritative design:** `AGENT_ARCHITECTURE.md`
**Development rules:** `CLAUDE.md`, `IMPLEMENTATION_RULES.md`

---

## Part A — Revised TODO List

---

### Phase 0 — Scaffolding
*No logic. Creates workspace and static configs.*

**0.1** Create directory tree: `agents/`, `catalogs/`, `core/`, `runs/`, `tests/unit/`, `tests/integration/`, `models/dummy/`, `models/linear/`
- Deps: none

**0.2** Write `catalogs/model_catalog.json` — 10 model entries. Mark `LightGBM` and `Chronos` with `"available": false` until implementations are verified importable. XGBoost, RandomForest, ARIMA, LSTM entries set to `true`.
- Deps: none

**0.3** Write `catalogs/transformer_catalog.json` — preprocessing methods with `allowed_modalities` per entry
- Deps: none

---

### Phase 1 — Manifest + Skeleton Orchestrator *(NEW — moved forward)*
*Validates the manifest-driven wiring before any real agent is written. Catches integration failures at the cheapest possible moment.*

**1.1** `core/manifest.py` — `initialize_manifest()`, `update_stage()`, `read_manifest()`
- Deps: 0.1

**1.2** `orchestrator.py` skeleton — generates `run_id`, creates manifest, calls **stub agents** that write placeholder artifacts and return `completed`, checks manifest after each stage. Full agent wiring comes in Phase 7.
- Deps: 1.1

**1.3** Smoke test: run skeleton orchestrator on `inputs/ETTh1.csv`, assert `job_manifest.json` is created, all stages cycle through `running → completed`, `dashboard/` directory is created.
- Deps: 1.2

---

### Phase 2 — Core Shared Utilities
*Stable before any real agent is written. Includes unit tests.*

**2.1** `core/splitter.py` — implements all 4 strategies (`random`, `stratified`, `chronological`, `group_kfold`). Port from legacy `trainer.py:prepare_data_splits`, do not import it.
- Deps: 0.1

**2.2** `core/metric_engine.py` — `compute_all(y_true, y_pred, metrics_list) → dict`. Handles `null` for failed metrics consistently across all models. Port from legacy `evaluator.py`, do not import it.
- Deps: 0.1

**2.3** `core/schema_validator.py` — stub with `validate(artifact_path, schema_name) → bool`. Ship with schemas for `dataset_profile` and `task_spec` only. Remaining schemas added incrementally as agents are implemented.
- Deps: 0.1

**2.4** Unit tests: `tests/unit/test_splitter.py`, `tests/unit/test_metric_engine.py`, `tests/unit/test_manifest.py`
- Deps: 2.1, 2.2, 1.1

---

### Phase 3 — Model Layer *(consolidated, extracted from original Phase 2)*
*All model registrations in one place. Decoupled from any agent.*

**3.1** `models/dummy/dummy_model.py` — `DummyRegressor` and `DummyClassifier` wrapping sklearn. Register both in `model_registry.py`.
- Deps: 0.1

**3.2** `models/linear/linear_model.py` — unified interface: Ridge (regression) / LogisticRegression (classification). Register in `model_registry.py`.
- Deps: 0.1

**3.3** Verify existing models (XGBoost, RandomForest, ARIMA, LSTM) match their catalog entries — `registry_key`, `compatible_tasks`, and `tier` must be consistent. Fix any mismatches in `model_catalog.json`, not in model code.
- Deps: 0.2, 3.1, 3.2

> **LightGBM deferred to Phase 6.** It introduces an external dependency and installation risk. The pipeline should pass an end-to-end integration test before new external models are added.

---

### Phase 4 — Ingestion Agent + Problem Classification Agent
*Sequential by design — Classification reads Ingestion's output.*

**4.1** `agents/ingestion_agent.py` — port `MetadataExtractor` from legacy `preprocessing_pipeline.py`. Strip all model recommendation logic. Output: `dataset_profile.json` per `AGENT_ARCHITECTURE.md §4 Agent 1`. Add schema to `schema_validator.py`.
- Deps: 1.1, 2.3

**4.2** `agents/problem_classification_agent.py` — heuristic decision tree (dtype, monotonicity, cardinality). No LLM calls. Output: `task_spec.json` per §4 Agent 2. Add schema to `schema_validator.py`.
- Deps: 1.1, 2.3, 4.1

**4.3** Unit test: `tests/unit/test_problem_classification_agent.py` — cover all 4 task types with fixture DataFrames.
- Deps: 4.2

---

### Phase 5 — Preprocessing Agent + Evaluation Protocol Agent *(parallel development)*
*These two agents run in parallel in the pipeline and can be developed in parallel by different engineers.*

**5.1** `agents/evaluation_protocol_agent.py` — deterministic routing table only. No LLM calls. Output: `eval_protocol.json` per §4 Agent 4. Add schema to `schema_validator.py`.
- Deps: 1.1, 4.2

**5.2** `agents/preprocessing_execution.py` — execution sub-phase. Accepts a `preprocessing_plan.json` dict. Records fitted scaler params. Output: `processed_data.csv`, `preprocessing_manifest.json`. Implement the heuristic fallback plan path first — LLM planning layers on top in 5.3.
- Deps: 1.1, 0.3, 4.2

**5.3** `agents/preprocessing_planning_agent.py` — LLM sub-phase. Calls LLM, validates response against transformer catalog, falls back to minimal safe plan on failure. Calls `preprocessing_execution.py` execution path. Output: `preprocessing_plan.json`.
- Deps: 5.2

---

### Phase 6 — Model Selection + Hyperparameter Tuner Update
*Depends on both Phase 5 branches completing.*

**6.1** `agents/model_selection_agent.py` — reads `model_catalog.json`, gates by `task_type`, enforces 3-tier constraint, verifies `ModelRegistry` availability. Output: `selected_models.json` per §4 Agent 5.
- Deps: 1.1, 0.2, 3.1, 3.2, 3.3, 4.2, 5.1

**6.2** Update `models/hyperparameter_tuner.py` — accept `primary_metric` from `eval_protocol.json` instead of hardcoded RMSE. Add search spaces for `LinearModel` and `DummyRegressor`. Read the file in full before modifying.
- Deps: 2.2, 5.1

**6.3** *(Deferred — after integration test passes)* `models/tree_based/lightgbm_model.py` — implement, register in `model_registry.py`, set `"available": true` in `model_catalog.json`, add Optuna search space in tuner.
- Deps: 6.2, integration test from 7.5

---

### Phase 7 — Training + Evaluation + Assembly + Orchestrator Finalization

**7.1** `agents/training_agent.py` — replaces legacy `trainer.py`. Reads `selected_models.json`. Computes split once via `core/splitter.py`, freezes `X_test`/`y_test`. Trains each model, catches per-model failures. Output: `training_results.json`, `trained_models/`.
- Deps: 1.1, 1.2, 2.1, 5.1, 5.2, 6.1, 6.2

**7.2** `agents/evaluation_agent.py` — replaces legacy `evaluator.py`. Re-derives test set deterministically from `eval_protocol.json` + `processed_data.csv`. Calls `metric_engine.compute_all()` identically for all 3 models. Generates plots. Output: `evaluation_report.json`, `comparison_table.json`, `plots/`.
- Deps: 1.1, 2.1, 2.2, 5.1, 7.1

**7.3** `agents/artifact_assembly_agent.py` — assembles `dashboard/` bundle: `leaderboard.json`, `model_cards/`, `comparison_chart.png`, `run_summary.json`.
- Deps: 1.1, 7.2

**7.4** `orchestrator.py` finalization — replace stub agents with real agent calls. Add `ThreadPoolExecutor` for parallel execution of Agents 3+4 (preprocessing + eval protocol). Verify all stage dependency checks.
- Deps: all prior tasks

**7.5** Integration tests: `tests/integration/test_full_pipeline_regression.py`, `tests/integration/test_full_pipeline_timeseries.py` — run on `ETTh1.csv` and one tabular regression dataset. Assert determinism (two runs with same seed produce identical metric values).
- Deps: 7.4

---

### Revised Critical Path

```
0.1 → 1.1 → 1.2 (skeleton) → 1.3 (smoke test)
                    │
                    └─► 2.1 + 2.2 + 2.3 (parallel)
                                │
                    ┌───────────┘
                    │
           4.1 → 4.2 ─┬─► 5.2 → 5.3 ─┐
                       │               ├─► 6.1 → 7.1 → 7.2 → 7.3 → 7.4 → 7.5
                       └─► 5.1 ────────┘
                                          ↑
              3.1 + 3.2 + 3.3 ───────────┘    (6.3 deferred)
```

---

## Part A — What Changed & Why

- **Phase 1 (Skeleton Orchestrator) is new and moved forward.** The original plan builds the orchestrator dead-last (7.3). This means integration failures — broken manifest wiring, missing artifact paths, incorrect stage transitions — are only discovered after all 8 agents are implemented. A skeleton orchestrator with stub agents validates the entire manifest-driven communication contract in Phase 1, at near-zero cost.

- **Model implementations extracted from "Ingestion Agent" phase.** Tasks 2.2–2.4 had nothing to do with ingestion. Grouping them there was misleading and created a false dependency. They are now Phase 3 ("Model Layer"), clearly isolated.

- **LightGBM deferred past first integration test.** LightGBM adds an external `pip` dependency with known platform-specific install failures. Adding it before the pipeline runs end-to-end means a broken import can block all model selection testing. It is implemented in Phase 6.3, gated on a passing integration test.

- **Preprocessing execution (5.2) split from planning (5.3).** The original 4.1 → 4.2 pair required the LLM call to be working before any preprocessing could be tested. Now the execution sub-phase (5.2) is implemented first with the heuristic fallback plan and is immediately testable. The LLM planning layer (5.3) is added on top without blocking.

- **`schema_validator.py` is a stub initially, filled incrementally.** The original plan implied all schemas exist in Phase 1, before any artifact schemas are finalized in implementation. The schema definitions are added as each agent is implemented, keeping the validator useful from day one without requiring a complete schema library upfront.

- **Unit tests are explicit tasks.** The original list had zero testing tasks despite `CLAUDE.md` and `IMPLEMENTATION_RULES.md` both requiring them. Tests for core utilities and classification heuristics are now first-class tasks.

- **`Chronos` left at `"available": false` in catalog until verified.** Chronos has optional dependencies and a try/except import guard in the existing registry. The catalog reflects actual runtime availability rather than aspirational availability.

---

## Part B — Team Task Allocation

---

### Member 1 — Integration Owner

**Responsibility area:** Scaffolding, manifest system, orchestrator (skeleton → final), artifact assembly

**Assigned tasks:** 0.1, 0.2, 0.3, 1.1, 1.2, 1.3, 7.3, 7.4, 7.5

**Deliverables:**
- Working directory structure on day one (unblocks everyone)
- `core/manifest.py` (unblocks all agents)
- Skeleton orchestrator with smoke test passing (validates architecture early)
- Final orchestrator with real agents wired and parallel execution
- `artifact_assembly_agent.py`
- Full integration test suite

**Depends on:** Nobody at start. Depends on Members 2–5 having completed their agents for Phase 7 finalization.

**Who depends on them:** Everyone. Member 1's Phase 0 and 1.1 are the first hard blockers for the entire team.

---

### Member 2 — Core Infrastructure

**Responsibility area:** Shared utilities, testing framework

**Assigned tasks:** 2.1, 2.2, 2.3, 2.4

**Deliverables:**
- `core/splitter.py` with all 4 split strategies and unit tests
- `core/metric_engine.py` with null-safe metric handling and unit tests
- `core/schema_validator.py` stub (schemas filled incrementally as agents ship)
- `tests/unit/test_*.py` for all core modules

**Depends on:** Member 1 (0.1 for directory structure, 1.1 for manifest).

**Who depends on them:** Members 3, 4, and 5 all depend on `splitter.py`, `metric_engine.py`, and `schema_validator.py` before their agents can be completed. Member 2 should finish Phase 2 before Members 3–5 complete their agents.

---

### Member 3 — Data Pipeline (Ingestion + Classification)

**Responsibility area:** Agents 1 and 2 — raw data in, task type out

**Assigned tasks:** 3.1, 3.2, 3.3, 4.1, 4.2, 4.3

**Deliverables:**
- `models/dummy/dummy_model.py` + `models/linear/linear_model.py` registered in registry (3.1–3.3, simple and fast)
- `agents/ingestion_agent.py` outputting valid `dataset_profile.json`
- `agents/problem_classification_agent.py` outputting valid `task_spec.json`
- Unit tests covering all 4 task type branches
- Schema definitions for `dataset_profile` and `task_spec` added to `schema_validator.py`

**Depends on:** Member 1 (manifest, directory tree), Member 2 (schema_validator stub).

**Who depends on them:** Members 4 and 5. Task 4.2 (`task_spec.json`) is required by the preprocessing planning agent (Member 4) and by the model selection agent (Member 5). Member 3 is on the critical path.

---

### Member 4 — Preprocessing + Evaluation Protocol

**Responsibility area:** Agents 3 and 4 — cleaned data out, eval protocol out

**Assigned tasks:** 5.1, 5.2, 5.3

**Deliverables:**
- `agents/evaluation_protocol_agent.py` — deterministic routing table, no LLM, fully testable immediately
- `agents/preprocessing_execution.py` — heuristic fallback first, testable without LLM
- `agents/preprocessing_planning_agent.py` — LLM call layered on top of execution
- Schema definitions for `eval_protocol` and `preprocessing_plan` added to `schema_validator.py`

**Depends on:** Member 1 (manifest), Member 2 (schema_validator), Member 3 (`task_spec.json`).

**Who depends on them:** Member 5 depends on `eval_protocol.json` (for tuner and training agent) and `processed_data.csv` (for training and evaluation). Member 4's work is a hard gate on Phase 6 and 7.

> Member 4 can begin 5.1 (eval protocol, deterministic) immediately after Member 3 ships `task_spec.json`. Task 5.2 (preprocessing execution heuristic path) can be stubbed and tested against fixture data before real ingestion output is available.

---

### Member 5 — Model Layer, Selection, Training, and Evaluation

**Responsibility area:** Model pool integrity, model selection logic, training agent, evaluation agent

**Assigned tasks:** 6.1, 6.2, 6.3 (deferred), 7.1, 7.2

**Deliverables:**
- `agents/model_selection_agent.py` — catalog-gated, tier-enforced selection
- Updated `models/hyperparameter_tuner.py` — reads `primary_metric` from eval protocol
- `agents/training_agent.py` — single frozen test set, per-model failure handling
- `agents/evaluation_agent.py` — deterministic test set re-derivation, identical metrics across all 3 models
- LightGBM implementation (6.3) after first integration test passes

**Depends on:** Member 1 (manifest, orchestrator skeleton), Member 2 (splitter, metric_engine), Member 3 (catalog entries for dummy + linear models), Member 4 (`eval_protocol.json`, `processed_data.csv`).

**Who depends on them:** Member 1 (integration owner) cannot finalize the orchestrator or artifact assembly until training and evaluation agents ship.

---

### Dependency Matrix

| | M1 (Integration) | M2 (Core) | M3 (Data Pipeline) | M4 (Preprocessing) | M5 (Model+Eval) |
|---|---|---|---|---|---|
| **M1** | — | needs 2.1–2.3 for final orchestrator | needs agents 4.1–4.2 | needs agents 5.1–5.3 | needs agents 6.1–7.2 |
| **M2** | needs 0.1, 1.1 | — | — | — | — |
| **M3** | needs 0.1, 1.1 | needs 2.3 | — | — | — |
| **M4** | needs 0.1, 1.1 | needs 2.3 | needs 4.2 | — | — |
| **M5** | needs 0.1, 1.1 | needs 2.1, 2.2 | needs 3.1–3.3 | needs 5.1, 5.2 | — |

### Parallel Work Available After Phase 1

Once Member 1 delivers `core/manifest.py` and the directory skeleton:

- **M2, M3, and M5 (model layer)** can all begin immediately in parallel.
- **M4** can begin `5.1` (eval protocol, deterministic) as soon as M3 ships `task_spec.json`.
- **M5** (model layer tasks 3.1–3.3) has zero dependency on M2, M3, or M4 and can run fully in parallel with Phase 2.
