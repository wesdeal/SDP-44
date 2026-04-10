import styles from "./FeatureEngineeringPanel.module.css";

const GROUP_META = {
  raw: {
    label: "Raw Signals",
    accentVar: "--text-lo",
    description: "Original sensor columns from the dataset, passed through after imputation.",
  },
  rolling_mean: {
    label: "Rolling Mean",
    accentVar: "--accent-cyan",
    description: "Exponential-window rolling averages over 6 h and 24 h periods.",
  },
  rolling_std: {
    label: "Rolling Std Dev",
    accentVar: "--accent-amber",
    description: "Rolling standard deviation — captures local volatility across the same windows.",
  },
  lag_features: {
    label: "Exogenous Lags",
    accentVar: "--accent-blue",
    description: "Historical values of the exogenous input signals at lags 1, 2, 3, 6, 12, 24.",
  },
  target_lags: {
    label: "Target Lags (OT)",
    accentVar: "--accent-purple",
    description: "Autoregressive lags of the target variable — directly encode recent output history.",
  },
};

function FeatureChip({ name, accentVar }) {
  return (
    <span
      className={styles.chip}
      style={{ "--chip-accent": `var(${accentVar})` }}
    >
      {name}
    </span>
  );
}

function FeatureGroup({ groupKey, features }) {
  const meta = GROUP_META[groupKey] || { label: groupKey, accentVar: "--text-lo", description: "" };

  return (
    <div className={styles.group} style={{ "--group-accent": `var(${meta.accentVar})` }}>
      <div className={styles.groupHeader}>
        <span className={styles.groupDot} />
        <span className={styles.groupLabel}>{meta.label}</span>
        <span className={styles.groupCount}>{features.length}</span>
      </div>
      {meta.description && (
        <p className={styles.groupDesc}>{meta.description}</p>
      )}
      <div className={styles.chipGrid}>
        {features.map((f) => (
          <FeatureChip key={f} name={f} accentVar={meta.accentVar} />
        ))}
      </div>
    </div>
  );
}

export default function FeatureEngineeringPanel({ run }) {
  const fe = run.feature_engineering;

  return (
    <div className={styles.panel}>
      {/* Summary strip */}
      <div className={styles.summaryStrip}>
        <div className={styles.summaryItem}>
          <span className={styles.summaryVal}>{fe.total_features}</span>
          <span className={styles.summaryLabel}>total features</span>
        </div>
        <div className={styles.summaryDivider} />
        <div className={styles.summaryItem}>
          <span className={styles.summaryVal}>{fe.lag_config.total_generated}</span>
          <span className={styles.summaryLabel}>lag columns</span>
        </div>
        <div className={styles.summaryDivider} />
        <div className={styles.summaryItem}>
          <span className={styles.summaryVal}>{fe.rolling_config.total_generated}</span>
          <span className={styles.summaryLabel}>rolling columns</span>
        </div>
        <div className={styles.summaryDivider} />
        <div className={styles.summaryItem}>
          <span className={styles.summaryVal}>
            ±{fe.rows_dropped_for_lags}
          </span>
          <span className={styles.summaryLabel}>rows trimmed (max lag)</span>
        </div>
      </div>

      {/* Config row */}
      <div className={styles.configRow}>
        <div className={styles.configBlock}>
          <span className={styles.configLabel}>lag windows</span>
          <div className={styles.tagRow}>
            {fe.lag_config.lags.map((l) => (
              <span key={l} className={`${styles.configTag} ${styles.tagCyan}`}>
                lag {l}
              </span>
            ))}
          </div>
        </div>
        <div className={styles.configBlock}>
          <span className={styles.configLabel}>rolling windows</span>
          <div className={styles.tagRow}>
            {fe.rolling_config.windows.map((w) => (
              <span key={w} className={`${styles.configTag} ${styles.tagAmber}`}>
                {w} h
              </span>
            ))}
          </div>
        </div>
        <div className={styles.configBlock}>
          <span className={styles.configLabel}>statistics</span>
          <div className={styles.tagRow}>
            {fe.rolling_config.statistics.map((s) => (
              <span key={s} className={`${styles.configTag} ${styles.tagDefault}`}>
                {s}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* Feature groups */}
      <div className={styles.groups}>
        {Object.entries(fe.feature_groups).map(([key, features]) => (
          <FeatureGroup key={key} groupKey={key} features={features} />
        ))}
      </div>
    </div>
  );
}
