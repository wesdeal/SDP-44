"""core/manifest.py — Manifest helper utilities for the Pipeline.

These three functions are the only permitted way to read or write job_manifest.json.
All agents must use this module; no agent may open the manifest file directly.

Schema authority: AGENT_ARCHITECTURE.md §2.
"""

import json
import os
import tempfile
import uuid
from datetime import datetime, timezone

# Canonical stage order — must match AGENT_ARCHITECTURE.md §5.
_STAGE_NAMES = [
    "ingestion",
    "problem_classification",
    "preprocessing_planning",
    "evaluation_protocol",
    "model_selection",
    "training",
    "evaluation",
    "artifact_assembly",
]

_VALID_STATUSES = frozenset(
    {"pending", "running", "completed", "failed", "partial_failure"}
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _detect_file_format(file_path: str) -> str:
    """Return 'csv', 'parquet', or 'json' from the file extension.

    Raises:
        ValueError: If the extension is not one of the three supported formats.
    """
    ext = os.path.splitext(file_path)[1].lower()
    mapping = {".csv": "csv", ".parquet": "parquet", ".json": "json"}
    fmt = mapping.get(ext)
    if fmt is None:
        raise ValueError(
            f"Unsupported file format: {ext!r}. Expected .csv, .parquet, or .json"
        )
    return fmt


def _write_atomically(path: str, data: dict) -> None:
    """Write *data* as JSON to *path* atomically (temp-file + rename).

    Creates parent directories as needed.
    """
    dir_name = os.path.dirname(path)
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def initialize_manifest(input_file: str, config: dict) -> dict:
    """Create a new run manifest, write it to disk, and return it.

    The manifest is written to::

        {runs_dir}/{run_id}/job_manifest.json

    where ``runs_dir`` is ``config.get("runs_dir", "runs")``.  Passing an
    absolute path for ``runs_dir`` in the config is strongly recommended so
    that the pipeline is not sensitive to the working directory.

    Args:
        input_file: Path to the raw input file (CSV / Parquet / JSON).
            Resolved to an absolute path before being stored.
        config: Run-level configuration dict.  Stored verbatim in the manifest.
            Recognised key: ``"runs_dir"`` (str) — base directory for run
            workspaces.  Defaults to ``"runs"`` (relative to cwd).

    Returns:
        The initialized manifest dict, identical to what is written to disk.

    Raises:
        ValueError: If *input_file* has an unsupported extension.
    """
    run_id = str(uuid.uuid4())
    now = _now_iso()

    abs_input = os.path.abspath(input_file)
    file_format = _detect_file_format(abs_input)
    original_filename = os.path.basename(abs_input)

    stages = {
        name: {
            "status": "pending",
            "started_at": None,
            "completed_at": None,
            "artifacts": {},
            "error": None,
        }
        for name in _STAGE_NAMES
    }

    manifest = {
        "run_id": run_id,
        "created_at": now,
        "updated_at": now,
        "status": "pending",
        "input": {
            "file_path": abs_input,
            "file_format": file_format,
            "original_filename": original_filename,
        },
        "stages": stages,
        "config": config,
    }

    runs_dir = config.get("runs_dir", "runs")
    manifest_path = os.path.join(runs_dir, run_id, "job_manifest.json")
    _write_atomically(manifest_path, manifest)

    return manifest


def update_stage(
    manifest_path: str,
    stage_name: str,
    status: str,
    artifacts: dict = None,
    error: str = None,
) -> None:
    """Update a single stage in the manifest on disk.

    Only the named stage and the root ``updated_at`` field are modified.
    Every other field is left exactly as found on disk.

    Timestamp rules:
    - ``started_at`` is set when *status* == ``"running"``.
    - ``completed_at`` is set when *status* is ``"completed"``, ``"failed"``,
      or ``"partial_failure"``.

    Args:
        manifest_path: Absolute or relative path to ``job_manifest.json``.
        stage_name: Key of the stage to update (must be present in the manifest).
        status: New status string.  Must be one of ``pending``, ``running``,
            ``completed``, ``failed``, ``partial_failure``.
        artifacts: If not ``None``, replaces the stage's ``artifacts`` dict.
            Pass ``None`` to leave existing artifacts untouched.
        error: If not ``None``, sets the stage's ``error`` field.

    Raises:
        FileNotFoundError: If *manifest_path* does not exist.
        KeyError: If *stage_name* is not present in ``manifest["stages"]``.
        ValueError: If *status* is not a recognised value.
    """
    if status not in _VALID_STATUSES:
        raise ValueError(
            f"Invalid status {status!r}. Must be one of {sorted(_VALID_STATUSES)}"
        )

    manifest = read_manifest(manifest_path)

    if stage_name not in manifest["stages"]:
        raise KeyError(f"Stage {stage_name!r} not found in manifest")

    now = _now_iso()
    stage = manifest["stages"][stage_name]

    stage["status"] = status

    if status == "running":
        stage["started_at"] = now
    elif status in ("completed", "failed", "partial_failure"):
        stage["completed_at"] = now

    if artifacts is not None:
        stage["artifacts"] = artifacts

    if error is not None:
        stage["error"] = error

    manifest["updated_at"] = now

    _write_atomically(manifest_path, manifest)


def read_manifest(manifest_path: str) -> dict:
    """Read and return the manifest dict from disk.

    Args:
        manifest_path: Absolute or relative path to ``job_manifest.json``.

    Returns:
        The manifest dict.

    Raises:
        FileNotFoundError: If *manifest_path* does not exist.
    """
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)
