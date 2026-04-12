/**
 * api.js — public data-access layer for React components.
 *
 * MOCK vs. LIVE MODE
 * ──────────────────
 * Set VITE_USE_MOCK=true in your .env file (or .env.local) to force mock mode.
 * When VITE_API_BASE is empty AND VITE_USE_MOCK is not explicitly "false",
 * the app falls back to mock data automatically so development works without
 * a running backend.
 *
 * To run in live mode:
 *   VITE_API_BASE=http://localhost:8000  (or wherever your backend lives)
 *   VITE_USE_MOCK=false
 *
 * See frontend/INTEGRATION.md for the full wiring guide.
 *
 * ── BACKEND WIRING POINT ─────────────────────────────────────────────────────
 * All live-mode network calls originate from:
 *   src/services/api/dashboardFetcher.js   (parallel artifact fetches)
 *   src/services/adapters/runAdapter.js    (field normalization)
 *
 * No component file needs to change when switching from mock to live.
 * ─────────────────────────────────────────────────────────────────────────────
 */

import { mockRun } from "./mockRun.js";
import { getRunDashboardData } from "../services/api/dashboardFetcher.js";
import { adaptRawArtifacts } from "../services/adapters/runAdapter.js";

/** True when live backend is configured and mock mode is not forced. */
export const LIVE_MODE =
  !!import.meta.env.VITE_API_BASE &&
  import.meta.env.VITE_USE_MOCK !== "true";

export const API_BASE = import.meta.env.VITE_API_BASE ?? "";

/**
 * Default run ID to load when none is specified via URL or argument.
 * Override with VITE_DEFAULT_RUN_ID in your .env file.
 */
export const DEFAULT_RUN_ID =
  import.meta.env.VITE_DEFAULT_RUN_ID ?? "299970a3-0f86-4c91-bc05-4e0f3995ee43";

/**
 * getRun(runId?)
 *
 * Loads a full dashboard run. In mock mode, returns the static mockRun with
 * a small artificial delay so loading / skeleton states are exercised.
 * In live mode, fetches all artifacts in parallel and normalizes them into
 * the frontend Run shape via runAdapter.
 *
 * @param {string} [runId] — run UUID. Falls back to DEFAULT_RUN_ID.
 * @returns {Promise<Run>}
 */
export async function getRun(runId) {
  const id = runId ?? DEFAULT_RUN_ID;

  if (!LIVE_MODE) {
    // Mock mode: small delay so loading state is exercised
    await new Promise((r) => setTimeout(r, 80));
    return mockRun;
  }

  const raw = await getRunDashboardData(id, API_BASE);
  return adaptRawArtifacts(raw);
}

/**
 * getLatestRunId()
 *
 * Resolves the run ID to load when no explicit ID is provided.
 * Live mode: GET /api/runs/latest → run_id string
 * Mock mode: returns DEFAULT_RUN_ID immediately.
 *
 * @returns {Promise<string>}
 */
export async function getLatestRunId() {
  if (!LIVE_MODE) return DEFAULT_RUN_ID;
  const res = await fetch(`${API_BASE}/api/runs/latest`);
  if (!res.ok) return DEFAULT_RUN_ID;
  const data = await res.json();
  return data?.run_id ?? DEFAULT_RUN_ID;
}

/**
 * listRuns()
 *
 * Returns a summary list of available runs for a run-picker UI.
 * Live mode: GET /api/runs → [{ run_id, status, created_at, dataset_name }]
 * Mock mode: returns a single-entry list derived from mockRun.
 *
 * @returns {Promise<Array<{ run_id, status, created_at, dataset_name }>>}
 */
export async function listRuns() {
  if (!LIVE_MODE) {
    await new Promise((r) => setTimeout(r, 40));
    return [
      {
        run_id: mockRun.run_id,
        status: mockRun.status,
        created_at: mockRun.created_at,
        dataset_name: mockRun.dataset_profile?.dataset_name ?? "—",
      },
    ];
  }

  const res = await fetch(`${API_BASE}/api/runs`);
  if (!res.ok) throw new Error(`HTTP ${res.status} — /api/runs`);
  return res.json();
}

/**
 * createRun(file)
 *
 * Upload a dataset file and start a new pipeline run.
 * Always hits the real backend regardless of LIVE_MODE.
 *
 * POST /api/runs/upload  →  { run_id }
 *
 * @param {File} file — The dataset file to upload (.csv / .parquet / .json)
 * @returns {Promise<{ run_id: string }>}
 */
export async function createRun(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/api/runs/upload`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (body.detail) detail = body.detail;
    } catch (_) {}
    throw new Error(detail);
  }
  return res.json();
}

/**
 * getRunStatus(runId)
 *
 * Poll the live stage-level status of a run.
 * Always hits the real backend regardless of LIVE_MODE.
 *
 * GET /api/runs/{runId}/status  →  { run_id, status, updated_at, stages }
 *
 * @param {string} runId
 * @returns {Promise<{ run_id: string, status: string, updated_at: string, stages: object }>}
 */
export async function getRunStatus(runId) {
  const res = await fetch(`${API_BASE}/api/runs/${runId}/status`);
  if (!res.ok) throw new Error(`HTTP ${res.status} — /api/runs/${runId}/status`);
  return res.json();
}
