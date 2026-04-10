/**
 * runService.js — data boundary between raw pipeline artifacts and UI components.
 *
 * This module is the single place where raw Run JSON is shaped into view models.
 * All components import their data from here, not from mockRun.js or api.js.
 *
 * ┌─────────────────────────────────────────────────────────────────────┐
 * │  BACKEND WIRING POINT                                               │
 * │                                                                     │
 * │  When you connect the real backend, only two things change:         │
 * │    1. src/data/api.js  — replace getRun() body with fetch()         │
 * │    2. This file        — adjust any field-name mismatches if the    │
 * │                          real artifact shape differs from the mock  │
 * │                                                                     │
 * │  No component files need to change.                                 │
 * └─────────────────────────────────────────────────────────────────────┘
 *
 * DATA FLOW
 * ─────────
 *   api.getRun()          → raw Run object (mirrors artifact JSON)
 *   runService.*()        → normalized view-model objects
 *   React components      → consume view models, render UI
 *
 * ARTIFACT SOURCES (under runs/{run_id}/artifacts/)
 * ─────────────────────────────────────────────────
 *   task_spec.json           → run.task_spec
 *   dataset_profile.json     → run.dataset_profile
 *   eval_protocol.json       → run.eval_protocol (+ split_counts)
 *   selected_models.json     → run.selected_models
 *   training_results.json    → run.training_results
 *   evaluation_report.json   → run.evaluation_report
 *   comparison_table.json    → run.comparison_table
 *   preprocessing_plan.json  → run.preprocessing_plan
 *   preprocessing_manifest.json → run.preprocessing_manifest
 *
 * FIELD NOTES
 * ───────────
 *   - MAPE and sMAPE are stored as fractions (0.0–1.0+) in artifacts.
 *     formatMetricByName() in format.js multiplies by 100 for display.
 *   - training_history is only present for XGBoost; null for others.
 *   - inference_time_ms lives in evaluation_report, not training_results.
 *   - split_counts is a sub-object inside eval_protocol (not a separate artifact).
 *   - feature_engineering is a denormalized field on the mockRun; it is derived
 *     from preprocessing_plan in the real backend (not a separate artifact file).
 */

// ─── Run-level metadata ──────────────────────────────────────────────────────

/**
 * Derive a flat run-header object for AppShell / RunOverviewHeader.
 *
 * @param {object} run — raw Run object from getRun()
 * @returns {{ runId, datasetName, taskType, status, createdAt, completedAt }}
 */
export function deriveRunHeader(run) {
  if (!run) return null;
  return {
    runId: run.run_id ?? null,
    datasetName: run.dataset_profile?.dataset_name ?? null,
    taskType: run.task_spec?.task_type ?? null,
    status: run.status ?? "unknown",
    createdAt: run.created_at ?? null,
    completedAt: run.completed_at ?? null,
  };
}

// ─── Summary cards ────────────────────────────────────────────────────────────

/**
 * Derives the 6 summary stat cards for the run overview.
 *
 * @param {object} run
 * @returns {Array<{ label, value, sublabel, accent? }>}
 */
export function deriveSummaryStats(run) {
  if (!run) return [];
  const ts = run.task_spec ?? {};
  const ep = run.eval_protocol ?? {};
  const ranking = run.comparison_table?.ranking ?? [];
  const winner = ranking.find((r) => r.is_best) ?? ranking[0] ?? null;
  const sc = ep.split_counts ?? {};

  return [
    {
      label: "Task type",
      value: ts.task_type ?? "—",
      sublabel: ts.forecast_horizon ? `horizon · ${ts.forecast_horizon}` : null,
    },
    {
      label: "Target column",
      value: ts.target_col ?? "—",
      sublabel: ts.target_dtype ?? null,
    },
    {
      label: "Split strategy",
      value: ep.split_strategy ?? "—",
      sublabel: ep.time_col ? `time col · ${ep.time_col}` : null,
    },
    {
      label: "Test samples",
      value: sc.test != null ? sc.test.toLocaleString() : "—",
      sublabel:
        sc.train && sc.validation
          ? `train ${sc.train.toLocaleString()} · val ${sc.validation.toLocaleString()}`
          : null,
    },
    {
      label: "Primary metric",
      value: ep.primary_metric ?? "—",
      sublabel: "lower is better",
    },
    {
      label: "Best model",
      value: winner?.model_name ?? "—",
      sublabel:
        winner
          ? `${ep.primary_metric} ${winner.primary_metric_value?.toFixed(3)}`
          : null,
      accent: "best",
    },
  ];
}

// ─── Leaderboard ─────────────────────────────────────────────────────────────

/**
 * Merges comparison_table + training_results + evaluation_report into one row
 * per model, sorted by rank ascending.
 *
 * Components that were previously doing this inline:
 *   - LeaderboardTable.jsx (buildRows)
 *
 * @param {object} run
 * @returns {Array<LeaderboardRow>}
 *
 * LeaderboardRow shape:
 *   { rank, name, tier, is_best, MAE, RMSE, MAPE, sMAPE,
 *     training_seconds, inference_ms, status }
 */
export function deriveLeaderboardRows(run) {
  if (!run) return [];
  const ranking = run.comparison_table?.ranking ?? [];
  const trained = run.training_results?.models ?? [];
  const evaluated = run.evaluation_report?.models ?? [];

  return ranking
    .slice()
    .sort((a, b) => a.rank - b.rank)
    .map((r) => {
      const tr = trained.find((m) => m.name === r.model_name);
      const ev = evaluated.find((m) => m.name === r.model_name);
      return {
        rank: r.rank,
        name: r.model_name,
        tier: r.tier,
        is_best: r.is_best,
        MAE: r.all_metrics?.MAE ?? null,
        RMSE: r.all_metrics?.RMSE ?? null,
        MAPE: r.all_metrics?.MAPE ?? null,
        sMAPE: r.all_metrics?.sMAPE ?? null,
        training_seconds: tr?.training_duration_seconds ?? null,
        inference_ms: ev?.inference_time_ms ?? null,
        status: ev?.status ?? tr?.status ?? "—",
      };
    });
}

// ─── Recommendation / winner card ────────────────────────────────────────────

/**
 * Derives everything the RecommendationCard needs in one shape.
 *
 * @param {object} run
 * @returns {WinnerViewModel | null}
 *
 * WinnerViewModel shape:
 *   { name, tier, primaryMetric, primaryValue, rationale,
 *     interpretation, deltas: [{ name, rank, tier, value, pctBetter, absDelta }] }
 */
export function deriveWinner(run) {
  if (!run) return null;
  const ranking = run.comparison_table?.ranking ?? [];
  const winner = ranking.find((r) => r.is_best) ?? ranking[0];
  if (!winner) return null;

  const ep = run.eval_protocol ?? {};
  const metric = ep.primary_metric ?? "MAE";

  const selectedModel = run.selected_models?.selected_models?.find(
    (m) => m.name === winner.model_name
  );

  const deltas = ranking
    .filter((r) => r.model_name !== winner.model_name)
    .map((other) => {
      const absDelta = other.primary_metric_value - winner.primary_metric_value;
      const pctBetter =
        other.primary_metric_value > 0
          ? (absDelta / other.primary_metric_value) * 100
          : null;
      return {
        name: other.model_name,
        rank: other.rank,
        tier: other.tier,
        value: other.primary_metric_value,
        pctBetter,
        absDelta,
      };
    });

  return {
    name: winner.model_name,
    tier: winner.tier,
    primaryMetric: metric,
    primaryValue: winner.primary_metric_value,
    rationale: selectedModel?.rationale ?? null,
    interpretation: winner.interpretation ?? null,
    deltas,
  };
}

// ─── Per-model detail ─────────────────────────────────────────────────────────

/**
 * Merges 4 artifact slices into a flat per-model view model list, ordered by rank.
 * Used by ModelDetailSection.
 *
 * Components that were previously doing this inline:
 *   - ModelDetailSection.jsx (buildModelList)
 *
 * @param {object} run
 * @returns {Array<ModelDetailViewModel>}
 */
export function deriveModelList(run) {
  if (!run?.comparison_table) return [];
  return run.comparison_table.ranking.map((rank) => {
    const training = run.training_results?.models?.find(
      (m) => m.name === rank.model_name
    );
    const evaluation = run.evaluation_report?.models?.find(
      (m) => m.name === rank.model_name
    );
    const selected = run.selected_models?.selected_models?.find(
      (m) => m.name === rank.model_name
    );
    return {
      name: rank.model_name,
      rank: rank.rank,
      tier: rank.tier,
      is_best: rank.is_best,
      interpretation: rank.interpretation ?? null,
      metrics: evaluation?.metrics ?? {},
      inference_time_ms: evaluation?.inference_time_ms ?? null,
      plot_path: evaluation?.plot_path ?? null,
      hyperparameters: training?.hyperparameters ?? {},
      hyperparameter_source: training?.hyperparameter_source ?? null,
      best_val_score: training?.best_val_score ?? null,
      training_duration_seconds: training?.training_duration_seconds ?? null,
      training_history: training?.training_history ?? null,
      status: training?.status ?? null,
      model_path: training?.model_path ?? null,
      rationale: selected?.rationale ?? null,
    };
  });
}

// ─── Metric chart rows ────────────────────────────────────────────────────────

/**
 * Builds the flat rows used by MetricBarChart / MetricComparisonSection.
 * MAPE/sMAPE are kept as raw fractions here; formatMetricByName handles display.
 *
 * @param {object} run
 * @returns {Array<{ name, rank, tier, is_best, MAE, RMSE, MAPE, sMAPE,
 *                   training_duration_seconds, inference_time_ms }>}
 */
export function deriveMetricChartRows(run) {
  if (!run) return [];
  const ranking = run.comparison_table?.ranking ?? [];
  const trained = run.training_results?.models ?? [];
  const evaluated = run.evaluation_report?.models ?? [];

  return ranking
    .slice()
    .sort((a, b) => a.rank - b.rank)
    .map((r) => {
      const tr = trained.find((m) => m.name === r.model_name);
      const ev = evaluated.find((m) => m.name === r.model_name);
      return {
        name: r.model_name,
        rank: r.rank,
        tier: r.tier,
        is_best: r.is_best,
        MAE: r.all_metrics?.MAE ?? null,
        RMSE: r.all_metrics?.RMSE ?? null,
        MAPE: r.all_metrics?.MAPE ?? null,
        sMAPE: r.all_metrics?.sMAPE ?? null,
        training_duration_seconds: tr?.training_duration_seconds ?? null,
        inference_time_ms: ev?.inference_time_ms ?? null,
      };
    });
}

// ─── Advanced details ─────────────────────────────────────────────────────────

/**
 * Derives the advanced-section props for AdvancedDetailsSection.
 * Keeps raw sub-objects intact; panels destructure what they need.
 *
 * @param {object} run
 * @returns {{ evalProtocol, splitCounts, preprocessingPlan, preprocessingManifest,
 *             featureEngineering, selectedModels, pipelineMetadata }}
 */
export function deriveAdvancedDetails(run) {
  if (!run) return null;
  return {
    evalProtocol: run.eval_protocol ?? {},
    splitCounts: run.eval_protocol?.split_counts ?? {},
    preprocessingPlan: run.preprocessing_plan ?? {},
    preprocessingManifest: run.preprocessing_manifest ?? {},
    featureEngineering: run.feature_engineering ?? {},
    selectedModels: run.selected_models ?? {},
    pipelineMetadata: run.pipeline_metadata ?? {},
  };
}
