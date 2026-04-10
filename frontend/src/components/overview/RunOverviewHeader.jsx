import styles from "./RunOverviewHeader.module.css";
import { humanizeTaskType, splitRunId } from "../../utils/format.js";

function MetaItem({ label, value, mono = true }) {
  return (
    <div className={styles.meta}>
      <div className={styles.metaLabel}>{label}</div>
      <div className={`${styles.metaValue} ${mono ? styles.mono : ""}`}>
        {value}
      </div>
    </div>
  );
}

export default function RunOverviewHeader({ run }) {
  if (!run) return null;

  const ts = run.task_spec || {};
  const ep = run.eval_protocol || {};
  const dp = run.dataset_profile || {};
  const { head, tail } = splitRunId(run.run_id);

  return (
    <header className={styles.header}>
      <div className={styles.topRow}>
        <div className={styles.titleBlock}>
          <div className={styles.eyebrow}>
            <span className={styles.dot} />
            Pipeline run · status {run.status}
          </div>
          <h1 className={styles.title}>Model Evaluation Console</h1>
          <p className={styles.subtitle}>
            Comparative report for the three models trained on this run.
            Strategy, metrics, and winner derived from the run manifest.
          </p>
        </div>
        <div className={styles.runIdBlock}>
          <div className={styles.runIdLabel}>Run ID</div>
          <div className={styles.runIdValue} title={run.run_id}>
            {head}
            <span className={styles.runIdRest}>{tail}</span>
          </div>
        </div>
      </div>

      <div className={styles.divider} />

      <div className={styles.metaGrid}>
        <MetaItem label="Dataset" value={dp.dataset_name || "—"} />
        <MetaItem label="Task type" value={humanizeTaskType(ts.task_type)} />
        <MetaItem label="Target column" value={ts.target_col || "—"} />
        <MetaItem label="Split strategy" value={ep.split_strategy || "—"} />
        <MetaItem label="Primary metric" value={ep.primary_metric || "—"} />
      </div>
    </header>
  );
}
