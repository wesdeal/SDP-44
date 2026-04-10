/**
 * format.js — single source of truth for all display formatting.
 *
 * All metric display, duration display, label normalization, and ID
 * formatting goes through this module so output is consistent across
 * charts, tables, tooltips, and detail panels.
 *
 * Rules:
 *   - MAPE and sMAPE are stored as fractions in artifacts; multiply × 100 to display.
 *   - training_duration_seconds uses formatDuration (seconds/minutes).
 *   - inference_time_ms uses formatMs (ms / µs).
 *   - Everything else uses formatMetric (fixed decimals).
 */

// ─── Core numeric formatters ─────────────────────────────────────────────────

export function formatMetric(value, opts = {}) {
  if (value == null || Number.isNaN(value)) return "—";
  const { digits = 3 } = opts;
  if (Math.abs(value) >= 10000) return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (Math.abs(value) >= 1000) return value.toFixed(0);
  return value.toFixed(digits);
}

export function formatInt(n) {
  if (n == null) return "—";
  return n.toLocaleString();
}

export function formatPercent(value, digits = 1) {
  if (value == null || Number.isNaN(value)) return "—";
  return `${value.toFixed(digits)}%`;
}

// ─── Duration / time formatters ──────────────────────────────────────────────

export function formatDuration(seconds) {
  if (seconds == null) return "—";
  if (seconds < 0.001) return `${(seconds * 1_000_000).toFixed(0)} µs`;
  if (seconds < 1) return `${(seconds * 1000).toFixed(0)} ms`;
  if (seconds < 60) return `${seconds.toFixed(2)} s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}m ${s}s`;
}

export function formatMs(ms) {
  if (ms == null || Number.isNaN(ms)) return "—";
  if (ms < 0.001) return `${(ms * 1_000_000).toFixed(0)} ps`;
  if (ms < 1) return `${(ms * 1000).toFixed(0)} µs`;
  if (ms < 1000) return `${ms.toFixed(1)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

// ─── Named-metric formatter ───────────────────────────────────────────────────

/**
 * Format a metric value by its canonical name.
 * This is the single authoritative function — use it everywhere a metric
 * value is displayed: charts, tables, tooltips, detail panels.
 *
 * MAPE / sMAPE: stored as fractions → rendered as % (e.g. 0.122 → "12.20%")
 * training_duration_seconds → formatDuration
 * inference_time_ms → formatMs
 * Everything else → fixed decimals
 */
export function formatMetricByName(value, name, opts = {}) {
  if (value == null || Number.isNaN(value)) return "—";
  const { digits = 3 } = opts;
  switch (name) {
    case "MAPE":
    case "sMAPE":
      return `${(value * 100).toFixed(2)}%`;
    case "training_duration_seconds":
      return formatDuration(value);
    case "inference_time_ms":
      return formatMs(value);
    default:
      return value.toFixed(digits);
  }
}

// ─── Label / display-name helpers ────────────────────────────────────────────

/**
 * Human-readable display label for a metric key.
 * Used in section headers, chart titles, and tooltip labels.
 */
export function metricDisplayLabel(name) {
  const MAP = {
    MAE: "MAE",
    RMSE: "RMSE",
    MAPE: "MAPE (%)",
    sMAPE: "sMAPE (%)",
    training_duration_seconds: "Train Time",
    inference_time_ms: "Inference",
  };
  return MAP[name] ?? name;
}

/**
 * Humanize a snake_case or camelCase task type string for display.
 * e.g. "time_series_forecasting" → "Time Series Forecasting"
 */
export function humanizeTaskType(t) {
  if (!t) return "—";
  return t
    .split(/[_\s]+/)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

/**
 * Humanize any snake_case string for display.
 * e.g. "forward_fill" → "Forward Fill"
 */
export function humanize(s) {
  if (!s) return "—";
  return String(s)
    .split(/[_\s]+/)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

// ─── ID helpers ──────────────────────────────────────────────────────────────

export function shortId(id, length = 8) {
  if (!id) return "—";
  return id.slice(0, length);
}

/**
 * Format a UUID for display: first 8 chars highlighted, rest dimmed.
 * Returns { head: string, tail: string } for split rendering.
 */
export function splitRunId(id) {
  if (!id) return { head: "—", tail: "" };
  return { head: id.slice(0, 8), tail: id.slice(8) };
}

// ─── Improvement / delta helpers ─────────────────────────────────────────────

/**
 * Format a percentage-improvement value between two metric readings.
 * pctImprovement = (other - winner) / other × 100
 */
export function formatPctImprovement(pct) {
  if (pct == null || Number.isNaN(pct)) return "—";
  return `${pct.toFixed(1)}% better`;
}

/**
 * Format an absolute delta between two metric values.
 * Δ = other - winner  (always positive if winner is better)
 */
export function formatAbsDelta(delta, metricName) {
  if (delta == null || Number.isNaN(delta)) return "—";
  const formatted = formatMetricByName(Math.abs(delta), metricName);
  return `Δ ${formatted}`;
}
