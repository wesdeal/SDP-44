import styles from "./MetricSelector.module.css";

/**
 * Segmented chip control for switching the focused chart between metrics.
 * Stays out of the chart's way: just a row of pills with a clear active state.
 */
export default function MetricSelector({ metrics, value, onChange }) {
  return (
    <div className={styles.wrap} role="tablist" aria-label="Metric selector">
      {metrics.map((m) => {
        const active = m.key === value;
        return (
          <button
            key={m.key}
            type="button"
            role="tab"
            aria-selected={active}
            className={`${styles.pill} ${active ? styles.pillActive : ""}`}
            onClick={() => onChange(m.key)}
          >
            <span className={styles.pillLabel}>{m.shortLabel}</span>
            {active && <span className={styles.pillDot} />}
          </button>
        );
      })}
    </div>
  );
}
