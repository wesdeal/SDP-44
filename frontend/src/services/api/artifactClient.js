/**
 * artifactClient.js — low-level per-artifact fetch helpers.
 *
 * Each exported function fetches one artifact from the backend and returns
 * its raw JSON. Returns null on 404 (artifact not yet written); throws on
 * other HTTP errors so the caller can surface a meaningful error state.
 *
 * URL convention:
 *   GET /api/runs/:runId/artifacts/:filename
 *
 * Override the base URL via the VITE_API_BASE environment variable.
 * Leave it empty (default) to use the Vite dev-server proxy at /api.
 */

const DEFAULT_BASE = import.meta.env.VITE_API_BASE ?? "";

function artifactUrl(runId, filename, base) {
  return `${base}/api/runs/${runId}/artifacts/${filename}`;
}

/**
 * Fetch a single artifact JSON file. Returns null on 404.
 */
async function fetchArtifact(url) {
  const res = await fetch(url);
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`HTTP ${res.status} — ${url}`);
  return res.json();
}

/**
 * Fetch the top-level run record (run_id, status, created_at, completed_at).
 * Expects GET /api/runs/:runId → { run_id, status, created_at, completed_at }
 */
export async function fetchRunMeta(runId, base = DEFAULT_BASE) {
  const res = await fetch(`${base}/api/runs/${runId}`);
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`HTTP ${res.status} — /api/runs/${runId}`);
  return res.json();
}

export async function fetchTaskSpec(runId, base = DEFAULT_BASE) {
  return fetchArtifact(artifactUrl(runId, "task_spec.json", base));
}

export async function fetchDatasetProfile(runId, base = DEFAULT_BASE) {
  return fetchArtifact(artifactUrl(runId, "dataset_profile.json", base));
}

export async function fetchEvalProtocol(runId, base = DEFAULT_BASE) {
  return fetchArtifact(artifactUrl(runId, "eval_protocol.json", base));
}

export async function fetchSelectedModels(runId, base = DEFAULT_BASE) {
  return fetchArtifact(artifactUrl(runId, "selected_models.json", base));
}

export async function fetchTrainingResults(runId, base = DEFAULT_BASE) {
  return fetchArtifact(artifactUrl(runId, "training_results.json", base));
}

export async function fetchEvaluationReport(runId, base = DEFAULT_BASE) {
  return fetchArtifact(artifactUrl(runId, "evaluation_report.json", base));
}

export async function fetchComparisonTable(runId, base = DEFAULT_BASE) {
  return fetchArtifact(artifactUrl(runId, "comparison_table.json", base));
}

export async function fetchPreprocessingPlan(runId, base = DEFAULT_BASE) {
  return fetchArtifact(artifactUrl(runId, "preprocessing_plan.json", base));
}

export async function fetchPreprocessingManifest(runId, base = DEFAULT_BASE) {
  return fetchArtifact(artifactUrl(runId, "preprocessing_manifest.json", base));
}
