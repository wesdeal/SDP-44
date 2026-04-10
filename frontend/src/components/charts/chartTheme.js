/**
 * Recharts theme constants for the dark console aesthetic.
 * Colors are pulled from the global theme tokens but mirrored here as
 * literals because Recharts SVG props don't read CSS custom properties.
 */

export const CHART_COLORS = {
  bg: "#10141b",
  panel: "#161b25",
  border: "#1f2735",
  borderStrong: "#2a3447",
  grid: "#1f2735",
  axis: "#4a5466",
  text: "#7c8699",
  textHi: "#e7ecf3",
  cyan: "#22d3ee",
  amber: "#fbbf24",
  purple: "#a78bfa",
  green: "#34d399",
  red: "#f87171",
  neutral: "#7c8699",
};

/**
 * Stable per-model color so any chart that shows the 3 models uses the
 * same identity. Best/cyan, runner-up/amber, baseline/neutral.
 */
export const MODEL_COLORS = {
  XGBoost: CHART_COLORS.cyan,
  RandomForest: CHART_COLORS.amber,
  DummyRegressor: CHART_COLORS.neutral,
};

export function colorForModel(name) {
  return MODEL_COLORS[name] || CHART_COLORS.purple;
}

export const AXIS_PROPS = {
  stroke: CHART_COLORS.axis,
  tick: {
    fill: CHART_COLORS.text,
    fontSize: 11,
    fontFamily:
      '"JetBrains Mono","SF Mono","Menlo",ui-monospace,"Cascadia Code",monospace',
  },
  tickLine: false,
  axisLine: { stroke: CHART_COLORS.border },
};

export const GRID_PROPS = {
  stroke: CHART_COLORS.grid,
  strokeDasharray: "2 4",
  vertical: false,
};

export const TOOLTIP_WRAPPER_STYLE = {
  outline: "none",
};

export const TOOLTIP_CONTENT_STYLE = {
  background: "rgba(16,20,27,0.96)",
  border: "1px solid #2a3447",
  borderRadius: 6,
  padding: "8px 12px",
  fontFamily:
    '"JetBrains Mono","SF Mono","Menlo",ui-monospace,"Cascadia Code",monospace',
  fontSize: 11,
  color: "#e7ecf3",
  boxShadow: "0 8px 24px -12px rgba(0,0,0,0.6)",
};

export const TOOLTIP_LABEL_STYLE = {
  color: "#7c8699",
  fontSize: 10,
  letterSpacing: "0.08em",
  textTransform: "uppercase",
  marginBottom: 4,
};

export const TOOLTIP_ITEM_STYLE = {
  color: "#e7ecf3",
};
