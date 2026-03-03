# Implementation Rules

This document defines the rules governing all development in the `/Pipeline` directory.
It is binding on all contributors and on Claude Code.

---

## 1. Source of Truth

**`Pipeline/AGENT_ARCHITECTURE.md` is the authoritative design document.**

Every decision about agent structure, artifact schemas, folder layout, model pool
composition, evaluation protocols, and manifest format is specified there.
`Pipeline/CLAUDE.md` governs Claude Code's behavior in this repository.

When any other document conflicts with `AGENT_ARCHITECTURE.md`, `AGENT_ARCHITECTURE.md` wins.
When in doubt about design intent, read `AGENT_ARCHITECTURE.md` before writing code.

The following documents are **not authoritative** and must not drive implementation decisions:

| File | Status |
|---|---|
| `PIPELINE_WORKFLOW.md` | Legacy reference only |
| `INTEGRATION_COMPLETE.md` | Legacy reference only |
| `MODEL_TRAINING_REPORT.md` | Legacy reference only |
| `AI_MODEL_PIPELINE_REPORT_SECTION.md` | Legacy reference only |
| `CHRONOS_SETUP.md` | Legacy reference only |
| `ARCHITECTURE.md` | Legacy reference only |
| `README.md` | Legacy reference only |
| `high_level.md` | Historical context only |

---

## 2. Legacy Handling Policy

### 2.1 What "Legacy" Means

The following files are legacy. They implement an earlier, monolithic pipeline
that predates the agent-based architecture:

- `preprocessing_pipeline.py`
- `preprocessor.py`
- `run_full_pipeline.py`
- `models/trainer.py`
- `models/evaluator.py`
- `quick_test_pipeline.py`
- Anything placed under `_legacy/`

### 2.2 What "Discard" Means in Practice

"Discard" does not mean delete. It means the following constraints apply
without exception:

- **No imports.** New code under `agents/`, `core/`, or `orchestrator.py` must
  not import anything from a legacy file.
- **No execution.** No new code path may call, subprocess-launch, or otherwise
  execute a legacy file.
- **No dependency.** No new module may depend on a legacy file being present,
  absent, or in any particular state.
- **No extension.** Legacy files must not be modified to make them compatible
  with the new architecture. They are frozen as-is.

### 2.3 What You May Do With Legacy Files

- **Read them** to understand how existing logic works before reimplementing it.
- **Port logic** from a legacy file into a new agent or core utility.
- **Cite them** in comments as the source of a ported algorithm.

### 2.4 Legacy Files Must Not Influence Design Decisions

If a legacy file does something in a particular way, that is not a reason to do
it the same way in new code. The design is governed by `AGENT_ARCHITECTURE.md`.
If the architecture specifies a different approach, follow the architecture.

---

## 3. Entry Point Policy

`Pipeline/orchestrator.py` is the **sole entry point** for pipeline execution.

- No agent module may be called directly to run a full pipeline.
- No test, script, or utility may instantiate and chain agents outside of the
  orchestrator.
- Individual agent modules may expose a `run(manifest_path)` method that can be
  called in isolation for debugging. This is permitted for development purposes
  only and must never be relied upon in the production execution path.
- The public API of the orchestrator is:
  ```python
  orchestrator.run_pipeline(input_file: str, config: dict) -> str
  ```
  The return value is the path to the completed `dashboard/` bundle.

---

## 4. Implementation Philosophy

### 4.1 Vertical Slices

Implement one agent at a time, end-to-end. A completed agent reads its inputs,
produces its outputs, updates the manifest, and passes validation. Do not
implement partial agents or stub out output files. An agent is either complete
or it does not exist yet.

### 4.2 Deterministic Behavior First

Every agent must produce identical outputs given identical inputs and the same
`config.random_seed`. Determinism is not optional. Non-determinism is a bug.

Concretely:
- All train/val/test splits use `random_state=config.random_seed`.
- Optuna studies use `TPESampler(seed=config.random_seed)`.
- File sort order must be explicit, never relying on filesystem ordering.
- LLM calls are the only permitted source of non-determinism, and their outputs
  must be validated and normalized before being written to any artifact.

### 4.3 Heuristic Before LLM

When a decision can be made with deterministic heuristics, use heuristics.
LLM calls are reserved for tasks that require semantic understanding:
dataset description, target variable labeling, and preprocessing step justification.

LLM calls must never be used for:
- Task type classification (heuristic only — see `AGENT_ARCHITECTURE.md` §4 Agent 2)
- Model selection (catalog-gated, tier-enforced — see Agent 5)
- Evaluation protocol routing (lookup table — see Agent 4)
- Split strategy selection (derived from task type — see Agent 4)

Every LLM call must have a documented fallback that allows the pipeline to
continue without LLM output. Agents 1 and 3 define these fallbacks explicitly
in `AGENT_ARCHITECTURE.md`.

---

## 5. Allowed vs Disallowed Changes

### 5.1 Acceptable Changes

- Adding a new file under `agents/`, `core/`, `models/`, or `tests/`.
- Implementing a method or class specified in `AGENT_ARCHITECTURE.md`.
- Adding a new model implementation under `models/` and registering it in
  `models/model_registry.py` and `catalogs/model_catalog.json`, provided the
  pool does not exceed 10 total models.
- Writing a unit test for a `core/` module.
- Writing an integration test that invokes `orchestrator.run_pipeline()`.
- Porting logic from a legacy file into a new agent or core utility.
- Adding a field to an artifact JSON file if `AGENT_ARCHITECTURE.md` specifies
  that field and it does not yet exist in the implementation.
- Fixing a bug in a specific agent without touching other agents.

### 5.2 Prohibited Changes

- Importing a legacy file from any new code.
- Modifying an artifact schema in a way not described in `AGENT_ARCHITECTURE.md`.
- Changing the orchestrator's agent execution order without explicit instruction.
- Changing a `core/` utility's public interface without updating all callers.
- Adding a model that brings the pool above 10 without removing another.
- Selecting a number of models per run other than exactly 3.
- Bypassing the tier constraint in `ModelSelectionAgent` (baseline + classical
  + specialized must always be represented).
- Calling one agent's methods from inside another agent's module.
- Writing agents that pass in-memory state to each other instead of artifacts.
- Hardcoding model names, file paths, or metric names as string literals inside
  agent logic.
- Deleting any file unless explicitly instructed.
- Reorganizing the folder structure without explicit instruction.
- Cleaning up, reformatting, or renaming files unrelated to the current task.

---

## 6. How Claude Should Behave

### 6.1 Minimal Diffs

Make the smallest change that satisfies the stated task. If a task says
"implement the Ingestion Agent," implement the Ingestion Agent and nothing else.
Do not fix adjacent issues, improve code style in nearby files, or refactor
utilities that happen to be imported.

### 6.2 Ask Before Architectural Changes

If a task would require any of the following, stop and ask the user for
explicit confirmation before writing code:

- Changing the manifest schema (adding, removing, or renaming fields)
- Changing artifact field names or structure
- Modifying the orchestrator's execution order or parallelism
- Changing a shared utility in `core/` in a way that affects multiple agents
- Adding or removing a model from the pool
- Changing tier assignments in `catalogs/model_catalog.json`
- Any change that touches more than one agent's boundary

When asking, cite the specific section of `AGENT_ARCHITECTURE.md` that is
relevant to the question. Do not proceed based on inferred intent.

### 6.3 Never Clean Up Unless Instructed

Do not delete files, remove commented-out code, reformat existing files,
or rename variables in code you were not asked to modify.
Cleanup tasks must be explicitly requested. Unrequested cleanup is a prohibited change.

### 6.4 Do Not Invent Architecture

If the task requires a design decision not covered by `AGENT_ARCHITECTURE.md`,
do not invent a solution. Report the gap to the user, describe the options,
and wait for a decision before writing code. The architecture document must be
updated to reflect any new decision before it is implemented.

### 6.5 Verification Before Modification

Before modifying any existing file, read it in full. Do not edit based on
filename or partial context alone. Confirm the file is what you expect it to be
and that your change is consistent with the rest of its contents.
