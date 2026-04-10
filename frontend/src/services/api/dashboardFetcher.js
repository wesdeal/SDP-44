/**
 * dashboardFetcher.js — fetches all artifacts for a run in parallel.
 *
 * Returns a raw artifact bundle. Missing artifacts resolve to null so the
 * adapter can apply safe defaults rather than crashing.
 *
 * DATA FLOW
 * ─────────
 *   dashboardFetcher.getRunDashboardData(runId)
 *     → raw artifact bundle { run_id, task_spec, ..., comparison_table, ... }
 *   runAdapter.adaptRawArtifacts(bundle)
 *     → normalized Run object (same shape as mockRun.js)
 *   getRun() in data/api.js
 *     → passes the normalized Run to React components
 */

import {
  fetchRunMeta,
  fetchTaskSpec,
  fetchDatasetProfile,
  fetchEvalProtocol,
  fetchSelectedModels,
  fetchTrainingResults,
  fetchEvaluationReport,
  fetchComparisonTable,
  fetchPreprocessingPlan,
  fetchPreprocessingManifest,
} from "./artifactClient.js";

/**
 * Fetch all artifacts for a single run in parallel.
 *
 * @param {string} runId — pipeline run UUID
 * @param {string} [baseUrl] — backend base URL (e.g. "http://localhost:8000").
 *   Omit to use the Vite dev-server proxy at /api.
 * @returns {Promise<RawArtifactBundle>}
 */
export async function getRunDashboardData(runId, baseUrl) {
  const [
    meta,
    task_spec,
    dataset_profile,
    eval_protocol,
    selected_models,
    training_results,
    evaluation_report,
    comparison_table,
    preprocessing_plan,
    preprocessing_manifest,
  ] = await Promise.all([
    fetchRunMeta(runId, baseUrl),
    fetchTaskSpec(runId, baseUrl),
    fetchDatasetProfile(runId, baseUrl),
    fetchEvalProtocol(runId, baseUrl),
    fetchSelectedModels(runId, baseUrl),
    fetchTrainingResults(runId, baseUrl),
    fetchEvaluationReport(runId, baseUrl),
    fetchComparisonTable(runId, baseUrl),
    fetchPreprocessingPlan(runId, baseUrl),
    fetchPreprocessingManifest(runId, baseUrl),
  ]);

  return {
    run_id: runId,
    // Top-level run metadata from /api/runs/:runId
    status: meta?.status ?? null,
    created_at: meta?.created_at ?? null,
    completed_at: meta?.completed_at ?? null,
    // Individual artifact payloads
    task_spec,
    dataset_profile,
    eval_protocol,
    selected_models,
    training_results,
    evaluation_report,
    comparison_table,
    preprocessing_plan,
    preprocessing_manifest,
  };
}
