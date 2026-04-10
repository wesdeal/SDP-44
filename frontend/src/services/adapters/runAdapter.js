/**
 * runAdapter.js — normalizes raw backend artifact data into the frontend Run shape.
 *
 * DESIGN CONTRACT
 * ───────────────
 * Input:  RawArtifactBundle from dashboardFetcher.getRunDashboardData()
 * Output: Run object — same shape as mockRun.js so every existing component
 *         and runService function works without modification.
 *
 * NORMALIZATION RESPONSIBILITIES
 * ──────────────────────────────
 * 1. Metric key casing
 *    Real artifacts use lowercase keys (mae, rmse, mape, smape).
 *    The UI layer (runService.js, format.js) expects uppercase (MAE, RMSE,
 *    MAPE, sMAPE). This adapter lifts all metric keys to their UI form.
 *
 * 2. Training history structure
 *    XGBoost real shape:
 *      training_history.evals_result.validation_0.rmse[]  ← train loss
 *      training_history.evals_result.validation_1.rmse[]  ← val loss
 *    Component-expected shape (TrainingHistoryChart):
 *      training_history.validation_0[]   (flat RMSE array, train)
 *      training_history.validation_1[]   (flat RMSE array, val)
 *      training_history.metric           ("rmse")
 *
 * 3. Split counts
 *    Real artifacts store split info inside training_results.split_info
 *    as { n_train, n_val, n_test }. The UI expects eval_protocol.split_counts
 *    as { train, validation, test }. This adapter backfills the field.
 *
 * 4. Plot paths
 *    Real paths are absolute filesystem paths:
 *      /Users/.../runs/RUN_ID/plots/XGBoost_eval.png
 *    Resolved to a browser-fetchable URL:
 *      /plots/RUN_ID/XGBoost_eval.png
 *    Override the /plots prefix via VITE_PLOT_BASE_URL.
 *
 * 5. primary_metric casing
 *    Real: lowercase "mae" → UI: "MAE"
 *
 * 6. Null/missing artifacts
 *    Every artifact is optional. Missing values produce safe UI defaults.
 */

// ── Metric key normalization ──────────────────────────────────────────────────

const METRIC_UP = {
  mae: "MAE",
  rmse: "RMSE",
  mape: "MAPE",
  smape: "sMAPE",
};

/** Normalize a single metric key to the uppercase UI form. */
function upMetricKey(k) {
  return METRIC_UP[k?.toLowerCase()] ?? k;
}

/** Normalize all keys of a metrics object to uppercase UI form. */
function normalizeMetrics(raw) {
  if (!raw) return {};
  const out = {};
  for (const [k, v] of Object.entries(raw)) {
    out[upMetricKey(k)] = v;
  }
  return out;
}

// ── Plot path resolution ──────────────────────────────────────────────────────

const PLOT_BASE = import.meta.env.VITE_PLOT_BASE_URL ?? "/plots";

/**
 * Converts an absolute filesystem plot path to a browser-fetchable URL.
 *
 * Examples:
 *   "/Users/wesdeal/.../runs/abc/plots/XGBoost_eval.png"
 *   → "/plots/abc/XGBoost_eval.png"
 *
 * If rawPath is already an http URL or starts with /plots/, return as-is.
 * Returns null if path is absent or filename cannot be parsed.
 */
function resolvePlotUrl(rawPath, runId) {
  if (!rawPath) return null;
  if (rawPath.startsWith("http") || rawPath.startsWith("/plots/")) return rawPath;
  const filename = rawPath.split(/[/\\]/).pop();
  if (!filename) return null;
  return `${PLOT_BASE}/${runId}/${filename}`;
}

// ── Training history normalization ────────────────────────────────────────────

/**
 * Normalizes XGBoost's evals_result structure into flat arrays.
 *
 * Real XGBoost shape:
 *   { training_time, evals_result: { validation_0: { rmse: [...] },
 *                                    validation_1: { rmse: [...] } },
 *     n_estimators_used }
 *
 * Normalized shape (consumed by TrainingHistoryChart):
 *   { validation_0: [...],  validation_1: [...],
 *     metric: "rmse",  n_rounds: 317 }
 *
 * Returns null for models that have no iterative convergence curve
 * (RandomForest, DummyRegressor).
 */
function normalizeTrainingHistory(history) {
  if (!history) return null;

  // Already normalized (mock shape or previously adapted)
  if (Array.isArray(history.validation_1) || Array.isArray(history.validation_0)) {
    return history;
  }

  // XGBoost real shape: extract from evals_result
  const evals = history.evals_result;
  if (evals) {
    const v0 = evals.validation_0?.rmse ?? [];
    const v1 = evals.validation_1?.rmse ?? [];
    if (v0.length || v1.length) {
      return {
        ...(v0.length ? { validation_0: v0 } : {}),
        ...(v1.length ? { validation_1: v1 } : {}),
        metric: "rmse",
        n_rounds: history.n_estimators_used ?? Math.max(v0.length, v1.length),
      };
    }
  }

  // No convergence data (RandomForest n_estimators counter, DummyRegressor timer)
  return null;
}

// ── Ranking entry normalization ───────────────────────────────────────────────

function normalizeRankingEntry(entry) {
  return {
    rank: entry.rank,
    model_name: entry.model_name,
    tier: entry.tier,
    primary_metric_value: entry.primary_metric_value,
    is_best: entry.is_best ?? false,
    all_metrics: normalizeMetrics(entry.all_metrics),
    interpretation: entry.interpretation ?? null,
  };
}

// ── Main adapter ──────────────────────────────────────────────────────────────

/**
 * Normalize a raw artifact bundle into the frontend Run object.
 *
 * @param {object} raw — output of dashboardFetcher.getRunDashboardData()
 * @returns {object} Run — same shape as src/data/mockRun.js
 */
export function adaptRawArtifacts(raw) {
  const runId = raw.run_id;
  const ct = raw.comparison_table;
  const tr = raw.training_results;
  const er = raw.evaluation_report;
  const ep = raw.eval_protocol;

  // ── Split counts ─────────────────────────────────────────────────────────
  // Prefer eval_protocol.split_counts if already present; fall back to
  // training_results.split_info (real artifact location).
  const splitInfo = tr?.split_info;
  const splitCounts =
    ep?.split_counts ??
    (splitInfo
      ? {
          train: splitInfo.n_train,
          validation: splitInfo.n_val,
          test: splitInfo.n_test,
        }
      : {});

  // ── Normalized eval_protocol ─────────────────────────────────────────────
  const normalizedEp = ep
    ? {
        ...ep,
        primary_metric: upMetricKey(ep.primary_metric) ?? ep.primary_metric,
        split_counts: splitCounts,
      }
    : null;

  // ── Normalized comparison_table ──────────────────────────────────────────
  const normalizedCt = ct
    ? {
        ...ct,
        ranked_by: upMetricKey(ct.ranked_by),
        ranking: (ct.ranking ?? []).map(normalizeRankingEntry),
      }
    : null;

  // ── Normalized training_results ──────────────────────────────────────────
  const normalizedTr = tr
    ? {
        ...tr,
        models: (tr.models ?? []).map((m) => ({
          ...m,
          training_history: normalizeTrainingHistory(m.training_history),
        })),
      }
    : null;

  // ── Normalized evaluation_report ─────────────────────────────────────────
  const normalizedEr = er
    ? {
        ...er,
        primary_metric: upMetricKey(er.primary_metric) ?? er.primary_metric,
        models: (er.models ?? []).map((m) => ({
          ...m,
          metrics: normalizeMetrics(m.metrics),
          plot_path: resolvePlotUrl(m.plot_path, runId),
        })),
      }
    : null;

  return {
    run_id: runId,
    status: raw.status ?? "completed",
    created_at: raw.created_at ?? null,
    completed_at: raw.completed_at ?? null,
    task_spec: raw.task_spec ?? null,
    dataset_profile: raw.dataset_profile ?? null,
    eval_protocol: normalizedEp,
    selected_models: raw.selected_models ?? null,
    training_results: normalizedTr,
    evaluation_report: normalizedEr,
    comparison_table: normalizedCt,
    preprocessing_plan: raw.preprocessing_plan ?? null,
    preprocessing_manifest: raw.preprocessing_manifest ?? null,
    // feature_engineering: real backend derives this from preprocessing_plan.
    // Not a standalone artifact. Pass through if present; runService
    // deriveAdvancedDetails() falls back to {} when absent.
    feature_engineering: raw.feature_engineering ?? null,
    pipeline_metadata: raw.pipeline_metadata ?? null,
  };
}
