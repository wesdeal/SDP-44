import styles from "./ModelTrainingMetadata.module.css";

function KVRow({ label, value, mono }) {
  return (
    <div className={styles.kvRow}>
      <span className={styles.kvKey}>{label}</span>
      <span className={`${styles.kvVal} ${mono ? styles.mono : ""}`}>
        {value ?? "—"}
      </span>
    </div>
  );
}

export default function ModelTrainingMetadata({ model }) {
  const hp = model.hyperparameters ?? {};
  const entries = Object.entries(hp);

  return (
    <div className={styles.wrap}>
      <div className={styles.section}>
        <span className={styles.eyebrow}>training configuration</span>
        <div className={styles.infoGrid}>
          <KVRow label="Hyperparameter source" value={model.hyperparameter_source} />
          <KVRow label="Training status" value={model.status} />
          <KVRow label="Model artifact path" value={model.model_path} mono />
        </div>
      </div>

      {entries.length > 0 && (
        <div className={styles.section}>
          <span className={styles.eyebrow}>hyperparameters</span>
          <div className={styles.hpGrid}>
            {entries.map(([k, v]) => (
              <div key={k} className={styles.hpCell}>
                <span className={styles.hpKey}>{k}</span>
                <span className={styles.hpVal}>
                  {v === null ? "null" : String(v)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
