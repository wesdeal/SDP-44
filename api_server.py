"""
api_server.py — FastAPI backend for the Pipeline dashboard.

Start:
    uvicorn api_server:app --reload --port 8000

Endpoints:
    GET /api/runs                               → list all runs, newest first
    GET /api/runs/latest                        → most recent completed run
    GET /api/runs/{run_id}                      → single run metadata
    GET /api/runs/{run_id}/artifacts/{filename} → artifact JSON file
    GET /plots/{run_id}/{filename}              → evaluation plot image
"""

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# ── Paths ──────────────────────────────────────────────────────────────────────

RUNS_DIR = Path(__file__).parent / "runs"

# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Pipeline Dashboard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)

# ── Internal helpers ───────────────────────────────────────────────────────────


def _load_manifest(run_id: str) -> dict:
    path = RUNS_DIR / run_id / "job_manifest.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    with open(path) as f:
        return json.load(f)


def _effective_status(manifest: dict) -> str:
    """
    Derive a reliable run status from stage statuses.

    The top-level manifest 'status' is sometimes stale ('pending' even when all
    stages completed). We compute it from stages instead.
    """
    stages = manifest.get("stages", {})
    if not stages:
        return manifest.get("status", "unknown")

    statuses = [s.get("status", "unknown") for s in stages.values()]

    if any(s == "failed" for s in statuses):
        return "failed"
    if any(s == "running" for s in statuses):
        return "running"
    # Consider completed when artifact_assembly finished, or all stages done
    aa = stages.get("artifact_assembly", {})
    if aa.get("status") == "completed":
        return "completed"
    if all(s == "completed" for s in statuses):
        return "completed"
    return manifest.get("status", "partial")


def _run_summary(run_id: str, manifest: dict) -> dict:
    """Build the slim summary object returned by GET /api/runs and /api/runs/latest."""
    input_info = manifest.get("input", {})
    raw_path = input_info.get("file_path", "")
    dataset_name = (
        input_info.get("original_filename")
        or (raw_path.split("/")[-1] if raw_path else None)
    )

    # Read task_type from task_spec artifact (not in manifest)
    task_type = None
    task_spec_path = RUNS_DIR / run_id / "artifacts" / "task_spec.json"
    if task_spec_path.exists():
        try:
            with open(task_spec_path) as f:
                task_type = json.load(f).get("task_type")
        except Exception:
            pass

    return {
        "run_id": run_id,
        "status": _effective_status(manifest),
        "created_at": manifest.get("created_at"),
        "updated_at": manifest.get("updated_at"),
        "dataset_name": dataset_name,
        "task_type": task_type,
    }


# ── Routes ─────────────────────────────────────────────────────────────────────


@app.get("/api/runs")
def list_runs():
    """Return all runs sorted newest-first. Excludes malformed directories."""
    if not RUNS_DIR.exists():
        return []

    results = []
    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / "job_manifest.json"
        if not manifest_path.exists():
            continue
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            results.append(_run_summary(run_dir.name, manifest))
        except Exception:
            # Include the run but annotate it as broken
            results.append({
                "run_id": run_dir.name,
                "status": "unknown",
                "created_at": None,
                "updated_at": None,
                "dataset_name": None,
                "task_type": None,
            })

    # Sort by created_at descending; null timestamps go last
    results.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return results


@app.get("/api/runs/latest")
def get_latest_run():
    """
    Return the most recent completed run.
    Falls back to the newest run of any status if no completed run exists.
    """
    runs = list_runs()
    if not runs:
        raise HTTPException(status_code=404, detail="No runs found")

    completed = [r for r in runs if r.get("status") == "completed"]
    return completed[0] if completed else runs[0]


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    """Return metadata for a single run."""
    # Guard: 'latest' is handled by a more-specific route above; if we somehow
    # hit here with 'latest' it means the dedicated route missed — redirect.
    if run_id == "latest":
        return get_latest_run()
    manifest = _load_manifest(run_id)
    return _run_summary(run_id, manifest)


@app.get("/api/runs/{run_id}/artifacts/{filename}")
def get_artifact(run_id: str, filename: str):
    """Serve an artifact JSON file for a run."""
    # Safety checks
    if "/" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid artifact path")
    if not filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Only .json artifacts are served here")

    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    path = run_dir / "artifacts" / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Artifact not found: {filename}")

    with open(path) as f:
        return JSONResponse(content=json.load(f))


@app.get("/plots/{run_id}/{filename}")
def get_plot(run_id: str, filename: str):
    """Serve a plot image for a run."""
    if "/" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    path = RUNS_DIR / run_id / "plots" / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Plot not found: {filename}")

    # Determine media type from extension
    ext = filename.rsplit(".", 1)[-1].lower()
    media_types = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "svg": "image/svg+xml"}
    media_type = media_types.get(ext, "application/octet-stream")

    return FileResponse(path, media_type=media_type)
