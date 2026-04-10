/**
 * Single source of truth for every metric the dashboard knows how to display.
 *
 * Each entry tells the chart layer:
 *   - what to call it (label, shortLabel, sublabel)
 *   - which artifact slice to read it from (source, key)
 *   - how to format raw values (unit) — used by formatMetricByName
 *   - which group it belongs to (prediction quality vs efficiency)
 *   - direction (lower_is_better is the only one we have today, but the
 *     field is explicit so charts never have to guess)
 */

export const METRICS = [
  {
    key: "MAE",
    label: "MAE",
    shortLabel: "MAE",
    sublabel: "Mean absolute error",
    source: "metrics",
    group: "quality",
    unit: "value",
    lower_is_better: true,
    isPrimary: true,
  },
  {
    key: "RMSE",
    label: "RMSE",
    shortLabel: "RMSE",
    sublabel: "Root mean squared error",
    source: "metrics",
    group: "quality",
    unit: "value",
    lower_is_better: true,
  },
  {
    key: "MAPE",
    label: "MAPE",
    shortLabel: "MAPE",
    sublabel: "Mean absolute % error",
    source: "metrics",
    group: "quality",
    unit: "percent",
    lower_is_better: true,
  },
  {
    key: "sMAPE",
    label: "sMAPE",
    shortLabel: "sMAPE",
    sublabel: "Symmetric MAPE",
    source: "metrics",
    group: "quality",
    unit: "percent",
    lower_is_better: true,
  },
  {
    key: "training_duration_seconds",
    label: "Training duration",
    shortLabel: "Train time",
    sublabel: "Wall-clock fit time",
    source: "training",
    group: "efficiency",
    unit: "seconds",
    lower_is_better: true,
  },
  {
    key: "inference_time_ms",
    label: "Inference time",
    shortLabel: "Inference",
    sublabel: "Per-batch test latency",
    source: "evaluation",
    group: "efficiency",
    unit: "milliseconds",
    lower_is_better: true,
  },
];

export const QUALITY_METRICS = METRICS.filter((m) => m.group === "quality");
export const EFFICIENCY_METRICS = METRICS.filter((m) => m.group === "efficiency");

export function getMetric(key) {
  return METRICS.find((m) => m.key === key) || null;
}

/**
 * Delegates to runService.deriveMetricChartRows().
 * Kept here so existing chart imports don't need to change.
 * Internal data merging lives in runService.js — the authoritative
 * data-boundary module.
 */
export { deriveMetricChartRows as buildModelMetricRows } from "../../services/runService.js";
