import styles from "./SplitStrategyPanel.module.css";

function StatRow({ label, value, accent }) {
  return (
    <div className={styles.statRow}>
      <span className={styles.statLabel}>{label}</span>
      <span className={styles.statValue} style={accent ? { color: `var(--accent-${accent})` } : undefined}>
        {value}
      </span>
    </div>
  );
}

function SplitBar({ train, validation, test }) {
  const total = train + validation + test;
  const trainPct = ((train / total) * 100).toFixed(1);
  const valPct = ((validation / total) * 100).toFixed(1);
  const testPct = ((test / total) * 100).toFixed(1);

  return (
    <div className={styles.splitBarSection}>
      <div className={styles.splitBar}>
        <div
          className={`${styles.segment} ${styles.segTrain}`}
          style={{ width: `${trainPct}%` }}
          title={`Train: ${train.toLocaleString()} rows (${trainPct}%)`}
        />
        <div
          className={`${styles.segment} ${styles.segVal}`}
          style={{ width: `${valPct}%` }}
          title={`Validation: ${validation.toLocaleString()} rows (${valPct}%)`}
        />
        <div
          className={`${styles.segment} ${styles.segTest}`}
          style={{ width: `${testPct}%` }}
          title={`Test: ${test.toLocaleString()} rows (${testPct}%)`}
        />
      </div>
      <div className={styles.splitLegend}>
        <span className={`${styles.legendDot} ${styles.trainDot}`} />
        <span className={styles.legendLabel}>train</span>
        <span className={styles.legendCount}>{train.toLocaleString()}</span>
        <span className={styles.legendPct}>{trainPct}%</span>

        <span className={`${styles.legendDot} ${styles.valDot}`} />
        <span className={styles.legendLabel}>validation</span>
        <span className={styles.legendCount}>{validation.toLocaleString()}</span>
        <span className={styles.legendPct}>{valPct}%</span>

        <span className={`${styles.legendDot} ${styles.testDot}`} />
        <span className={styles.legendLabel}>test</span>
        <span className={styles.legendCount}>{test.toLocaleString()}</span>
        <span className={styles.legendPct}>{testPct}%</span>
      </div>
    </div>
  );
}

export default function SplitStrategyPanel({ run }) {
  const proto = run.eval_protocol;
  const { train, validation, test } = proto.split_counts;

  return (
    <div className={styles.panel}>
      <div className={styles.grid}>
        {/* Left column: config */}
        <div className={styles.configCol}>
          <div className={styles.subsectionLabel}>strategy</div>
          <div className={styles.strategyRow}>
            <span className={styles.strategyName}>Chronological</span>
            <span className={styles.noShufflePill}>no shuffle</span>
          </div>
          <p className={styles.strategyNote}>
            Records are split in time order without shuffling. This is the
            correct methodology for time series: the model trains only on
            past observations and is evaluated strictly on future data it
            never saw during training.
          </p>

          <div className={styles.divider} />

          <div className={styles.configGrid}>
            <StatRow label="time column" value={proto.time_col} />
            <StatRow label="ranking metric" value={proto.primary_metric} accent="amber" />
            <StatRow label="prediction type" value={proto.prediction_type} />
            <StatRow label="cross-validation" value={proto.cv?.enabled ? "enabled" : "disabled"} />
            <StatRow label="total rows" value={(train + validation + test).toLocaleString()} />
          </div>
        </div>

        {/* Right column: split bar */}
        <div className={styles.barCol}>
          <div className={styles.subsectionLabel}>partition sizes</div>
          <SplitBar train={train} validation={validation} test={test} />
        </div>
      </div>

      <div className={styles.footer}>
        <span className={styles.footerNote}>
          Fractions — train {(proto.train_fraction * 100).toFixed(0)}% ·
          val {(proto.val_fraction * 100).toFixed(0)}% ·
          test {(proto.test_fraction * 100).toFixed(0)}%
        </span>
      </div>
    </div>
  );
}
