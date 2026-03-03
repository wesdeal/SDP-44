# CLAUDE.md — Instructions for Claude Code

This file governs all development within the `/Pipeline` directory.
Read it in full before making any change to this codebase.

---

## 1. Single Source of Truth

**`Pipeline/AGENT_ARCHITECTURE.md` is the authoritative design document.**

All architectural decisions, agent contracts, artifact schemas, folder structure,
model pool definitions, and evaluation protocols are specified there.
When any other document conflicts with `AGENT_ARCHITECTURE.md`, `AGENT_ARCHITECTURE.md` wins.

Documents that are **not** authoritative and must not drive new development:

- `PIPELINE_WORKFLOW.md`
- `INTEGRATION_COMPLETE.md`
- `MODEL_TRAINING_REPORT.md`
- `AI_MODEL_PIPELINE_REPORT_SECTION.md`
- `CHRONOS_SETUP.md`
- `ARCHITECTURE.md`
- `README.md`
- `high_level.md`

These files may be read for historical context only. They must never influence
the structure of new code.

---

## 2. Legacy Policy

### 2.1 Legacy Files

The following files are **legacy**. They implement an earlier, non-agent-based
pipeline and must not be imported, called, or extended:

- `preprocessing_pipeline.py`
- `preprocessor.py`
- `run_full_pipeline.py`
- `models/trainer.py`
- `models/evaluator.py`
- `quick_test_pipeline.py`
- Any file placed under `_legacy/`
- Any file listed in `LEGACY_IGNORED.md` (if that file exists)

### 2.2 What You May Do With Legacy Files

- **Read** them to understand existing logic before reimplementing.
- **Extract** and port logic into the new agent modules.
- **Do not** import them from new code under any circumstance.
- **Do not** modify them to make them compatible with the new architecture.

---

## 3. Core Architectural Principles

These principles are non-negotiable. Do not violate them, even if a shortcut
appears simpler.

### 3.1 Manifest-Driven Pipeline

Every pipeline run has a single `runs/{run_id}/job_manifest.json` file.
This file is the only shared mutable state. Every agent reads it at startup,
writes only its own stage block, and reads artifact paths from it rather than
constructing paths independently.

### 3.2 Agent-Based Stages

Each pipeline stage is implemented as a distinct Python module under `agents/`.
An agent is a class or module with a single public `run(manifest_path: str)` method.
No agent may contain logic that belongs to another agent's stage.

### 3.3 Artifact-Only Communication

Agents communicate exclusively through files written to `runs/{run_id}/artifacts/`
or `runs/{run_id}/trained_models/`. An agent must never call another agent's
functions directly. An agent must never read another agent's in-memory state.

### 3.4 Orchestrator as the Only Entry Point

`orchestrator.py` is the sole entry point for pipeline execution.
No agent, utility, or test should invoke the full pipeline except through
`orchestrator.run_pipeline(input_file, config)`.
Direct execution of individual agent scripts is permitted only for isolated
debugging and must not be relied upon in production flow.

---

## 4. Development Rules

Follow these rules on every task, without exception.

**4.1 Minimal diff.** Make the smallest change that satisfies the task.
Do not touch files unrelated to the task.

**4.2 No opportunistic refactoring.** If you notice an unrelated issue while
working on a task, note it in a comment or report it to the user. Do not fix it.

**4.3 Prefer new files over modifying existing ones.** When adding a new agent,
utility, or model, create a new file. Modify an existing file only when the task
explicitly targets that file.

**4.4 Never redesign the architecture.** Do not reorganize the folder structure,
rename artifacts, change manifest schema fields, or alter agent boundaries unless
the user explicitly instructs you to do so and references `AGENT_ARCHITECTURE.md`.

**4.5 Schema compliance is mandatory.** Every artifact written by an agent must
conform to the JSON schema specified in `AGENT_ARCHITECTURE.md` §4 for that agent.
Do not add, remove, or rename top-level fields without instruction.

**4.6 No hardcoded paths.** All file paths must be derived from the manifest or
passed as parameters. Never construct paths by string concatenation against a
hardcoded base directory.

**4.7 No hardcoded model names.** Model names in agent code must be read from
`catalogs/model_catalog.json` or from `selected_models.json`. Never write a
model name as a string literal inside agent logic.

---

## 5. Canonical Agent Execution Order

The orchestrator runs agents in this order. Agents 3 and 4 run in parallel;
all others are sequential.

```
1. IngestionAgent            → writes: dataset_profile.json
2. ProblemClassificationAgent → writes: task_spec.json
3. PreprocessingPlanningAgent } parallel → writes: preprocessing_plan.json,
4. EvaluationProtocolAgent   }            processed_data.csv,
                                          preprocessing_manifest.json,
                                          eval_protocol.json
5. ModelSelectionAgent        → writes: selected_models.json
6. TrainingAgent              → writes: training_results.json,
                                        trained_models/{model_name}/
7. EvaluationAgent            → writes: evaluation_report.json,
                                        comparison_table.json,
                                        plots/
8. ArtifactAssemblyAgent      → writes: dashboard/
```

A stage may not begin until all stages it depends on are in `"completed"` status
in the manifest. The single exception is `ArtifactAssemblyAgent`, which runs
regardless of whether `EvaluationAgent` partially succeeded, assembling whatever
artifacts are available.

---

## 6. Model Policy

### 6.1 Pool Size

The total number of model implementations in the codebase must not exceed **10**.
Before adding a new model, confirm that the catalog has room. If the pool is at
capacity, a model must be removed or replaced, not appended.

### 6.2 Selection Per Run

Every pipeline run selects **exactly 3** models. This is enforced by
`ModelSelectionAgent`. Do not modify the selection logic to allow 2 or 4 models
under any circumstance.

### 6.3 Tier Coverage

The 3 selected models must always include one model from each tier:

| Tier | Examples |
|---|---|
| `baseline` | DummyRegressor, DummyClassifier, LinearModel, ARIMA |
| `classical` | XGBoost, RandomForest, LightGBM, SVR |
| `specialized` | Chronos, LSTM |

Tier assignments are defined in `catalogs/model_catalog.json` and must not be
changed without updating both the catalog and `AGENT_ARCHITECTURE.md`.

### 6.4 Task Compatibility

A model may only be selected if its `compatible_tasks` list in `model_catalog.json`
includes the run's `task_type`. `ModelSelectionAgent` enforces this gate.
Do not bypass it.

### 6.5 Registry Requirement

A model listed in `model_catalog.json` must have a corresponding implementation
registered in `models/model_registry.py`. `ModelSelectionAgent` checks availability
at runtime and substitutes if a model cannot be imported. Substitutions must be
logged in `selected_models.json` under `substitution_reason`.

---

## 7. Testing Expectations

### 7.1 Unit Tests

Every module under `core/` must have a corresponding unit test file at
`tests/unit/test_{module_name}.py`. Tests must cover:

- Normal input → expected output
- Invalid input → expected exception or error path
- Boundary conditions (empty DataFrame, single row, all-null column)

### 7.2 Integration Tests

End-to-end pipeline tests run via `orchestrator.run_pipeline()` using the sample
datasets in `inputs/`. At minimum, one integration test must exist for each
supported `task_type`. Integration test files live at `tests/integration/`.

### 7.3 Determinism Requirement

All pipeline outputs must be deterministic given the same input file and
`config.random_seed`. Tests must assert that two runs with the same seed produce
byte-identical metric values. Non-determinism introduced by a code change is
treated as a bug.

### 7.4 No Tests Against Legacy Files

Do not write tests that import or execute legacy files. If porting logic from a
legacy file, test the new implementation, not the original.

---

## 8. When in Doubt

If you are uncertain about any of the following, **stop and ask the user** before
writing code:

- Whether a change is isolated to one agent or affects the manifest schema
- Whether a new model fits within the 10-model pool limit
- Whether a requested change would alter artifact field names or schemas
- Whether two agents need to exchange information in a way not described in
  `AGENT_ARCHITECTURE.md`
- Whether the task requires modifying the orchestrator's execution order
- Whether a fix requires changing shared utilities in `core/`

Do not infer intent for architectural decisions. Ask explicitly, reference the
relevant section of `AGENT_ARCHITECTURE.md` in your question, and wait for
confirmation before proceeding.
