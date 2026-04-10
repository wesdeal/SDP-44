import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import styles from "./TrainingHistoryChart.module.css";
import {
  AXIS_PROPS,
  GRID_PROPS,
  TOOLTIP_CONTENT_STYLE,
  TOOLTIP_WRAPPER_STYLE,
  CHART_COLORS,
} from "../charts/chartTheme.js";

/**
 * Merges validation_0 (train) and validation_1 (validation) arrays into
 * Recharts-ready data. Falls back to the legacy `validation` field if
 * validation_1 is absent (backwards compat with earlier mock shapes).
 */
function buildChartData(history) {
  const trainArr = history.validation_0 ?? [];
  const valArr = history.validation_1 ?? history.validation ?? [];
  const len = Math.max(trainArr.length, valArr.length);
  return Array.from({ length: len }, (_, i) => ({
    round: i + 1,
    ...(trainArr.length ? { train: trainArr[i] ?? null } : {}),
    ...(valArr.length ? { validation: valArr[i] ?? null } : {}),
  }));
}

const FALLBACK_NOTES = {
  RandomForest:
    "Tree ensemble — fitted in a single pass. No iterative training curve available.",
  DummyRegressor:
    "Baseline model does not expose epoch-based training history.",
};

function FallbackPanel({ model }) {
  const note =
    FALLBACK_NOTES[model.name] ??
    "No iterative training curve available for this model.";
  return (
    <div className={styles.fallback}>
      <span className={styles.fallbackGlyph}>◌</span>
      <p className={styles.fallbackNote}>{note}</p>
      {model.name === "RandomForest" && model.hyperparameters?.n_estimators && (
        <p className={styles.fallbackMeta}>
          n_estimators = {model.hyperparameters.n_estimators}
          {model.hyperparameters.max_depth != null
            ? ` · max_depth = ${model.hyperparameters.max_depth}`
            : " · max_depth = unlimited"}
        </p>
      )}
    </div>
  );
}

export default function TrainingHistoryChart({ model }) {
  const history = model.training_history;
  const hasTrainLine = !!(history?.validation_0?.length);

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <span className={styles.eyebrow}>training history</span>
        <h3 className={styles.title}>
          {history
            ? `Validation ${history.metric} · Convergence`
            : "Training History"}
        </h3>
        {history && (
          <span className={styles.subtitle}>
            lower {history.metric} is better ↓
          </span>
        )}
      </div>

      <div className={styles.body}>
        {!history ? (
          <FallbackPanel model={model} />
        ) : (
          <ResponsiveContainer width="100%" height={210}>
            <LineChart
              data={buildChartData(history)}
              margin={{ top: 8, right: 16, left: 4, bottom: 24 }}
            >
              <CartesianGrid {...GRID_PROPS} />
              <XAxis
                dataKey="round"
                {...AXIS_PROPS}
                label={{
                  value: "Boosting Round",
                  position: "insideBottom",
                  offset: -10,
                  fill: CHART_COLORS.text,
                  fontSize: 10,
                  fontFamily: AXIS_PROPS.tick.fontFamily,
                }}
                height={40}
              />
              <YAxis
                {...AXIS_PROPS}
                width={48}
                tickFormatter={(v) => v.toFixed(2)}
              />
              <Tooltip
                wrapperStyle={TOOLTIP_WRAPPER_STYLE}
                contentStyle={TOOLTIP_CONTENT_STYLE}
                labelFormatter={(v) => `Round ${v}`}
                formatter={(value, name) => [
                  value != null ? value.toFixed(4) : "—",
                  name === "train" ? "Train RMSE" : "Val RMSE",
                ]}
              />
              {hasTrainLine && (
                <Legend
                  wrapperStyle={{
                    fontFamily: AXIS_PROPS.tick.fontFamily,
                    fontSize: 10,
                    color: CHART_COLORS.text,
                  }}
                  formatter={(v) =>
                    v === "train" ? "Train RMSE" : "Validation RMSE"
                  }
                />
              )}
              {hasTrainLine && (
                <Line
                  type="monotone"
                  dataKey="train"
                  stroke={CHART_COLORS.borderStrong}
                  strokeWidth={1.5}
                  dot={false}
                  strokeDasharray="3 3"
                  name="train"
                  isAnimationActive={false}
                />
              )}
              <Line
                type="monotone"
                dataKey="validation"
                stroke={CHART_COLORS.cyan}
                strokeWidth={2}
                dot={false}
                name="validation"
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
