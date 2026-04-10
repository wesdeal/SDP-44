import styles from "./PipelineMetadataPanel.module.css";

function MetaRow({ label, value, valueClass }) {
  return (
    <div className={styles.row}>
      <span className={styles.rowLabel}>{label}</span>
      <span className={`${styles.rowVal} ${valueClass || ""}`}>{value}</span>
    </div>
  );
}

export default function PipelineMetadataPanel({ run }) {
  const m = run.pipeline_metadata;
  const taskDisplay = m.task_type.replace(/_/g, " ");

  return (
    <div className={styles.panel}>
      <div className={styles.grid}>
        {/* Left: run identity */}
        <div className={styles.section}>
          <span className={styles.sectionLabel}>run identity</span>
          <div className={styles.rows}>
            <MetaRow label="run id" value={m.run_id} />
            <MetaRow label="pipeline version" value={m.pipeline_version} />
            <MetaRow label="dataset" value={m.dataset_name} />
            <MetaRow
              label="evaluation timestamp"
              value={new Date(m.evaluation_timestamp).toLocaleString("en-US", {
                dateStyle: "medium",
                timeStyle: "short",
              })}
            />
          </div>
        </div>

        {/* Middle: task config */}
        <div className={styles.section}>
          <span className={styles.sectionLabel}>task configuration</span>
          <div className={styles.rows}>
            <MetaRow label="task type" value={taskDisplay} />
            <MetaRow label="target column" value={m.target_col} valueClass={styles.valHighlight} />
            <MetaRow label="time column" value={m.time_col} />
            <MetaRow label="primary metric" value={m.primary_metric} valueClass={styles.valAmber} />
            <MetaRow label="models tested" value={String(m.models_tested)} />
            <MetaRow label="hyperparam source" value={m.hyperparameter_source} />
          </div>
        </div>

        {/* Right: status and notes */}
        <div className={styles.section}>
          <span className={styles.sectionLabel}>system notes</span>
          <div className={styles.rows}>
            <MetaRow
              label="data mode"
              value={m.using_mock_data ? "mock (dev)" : "live backend"}
              valueClass={m.using_mock_data ? styles.valAmber : styles.valGreen}
            />
            <MetaRow label="run status" value={run.status} valueClass={styles.valGreen} />
          </div>

          <div className={styles.wiringNote}>
            <span className={styles.wiringNoteLabel}>backend wiring</span>
            <p className={styles.wiringNoteText}>{m.api_wiring_note}</p>
          </div>
        </div>
      </div>
    </div>
  );
}
