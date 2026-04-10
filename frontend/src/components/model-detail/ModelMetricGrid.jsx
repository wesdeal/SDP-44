import styles from "./ModelMetricGrid.module.css";
import { formatMetricByName, formatMs, formatDuration } from "../../utils/format.js";

const METRIC_DEFS = [
  { key: "MAE",   label: "MAE",       src: "metrics",  primary: true },
  { key: "RMSE",  label: "RMSE",      src: "metrics" },
  { key: "MAPE",  label: "MAPE",      src: "metrics" },
  { key: "sMAPE", label: "sMAPE",     src: "metrics" },
  { key: "training_duration_seconds", label: "Train Time", src: "direct" },
  { key: "inference_time_ms",         label: "Inference",  src: "direct" },
  { key: "best_val_score",            label: "Best Val",   src: "direct" },
];

function getValue(model, def) {
  return def.src === "metrics"
    ? (model.metrics?.[def.key] ?? null)
    : (model[def.key] ?? null);
}

function fmt(value, key) {
  if (value == null || Number.isNaN(value)) return "—";
  if (key === "training_duration_seconds") return formatDuration(value);
  if (key === "inference_time_ms") return formatMs(value);
  return formatMetricByName(value, key);
}

export default function ModelMetricGrid({ model }) {
  return (
    <div className={styles.grid}>
      {METRIC_DEFS.map((def) => (
        <div
          key={def.key}
          className={`${styles.cell} ${def.primary ? styles.primary : ""}`}
        >
          <span className={styles.label}>{def.label}</span>
          <span className={styles.value}>{fmt(getValue(model, def), def.key)}</span>
        </div>
      ))}
    </div>
  );
}
