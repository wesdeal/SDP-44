import styles from "./ModelInterpretation.module.css";

const REC_CONFIG = {
  deploy:   { label: "Recommended",    cls: "deploy" },
  fallback: { label: "Viable Fallback", cls: "fallback" },
  floor:    { label: "Baseline Floor",  cls: "floor" },
};

export default function ModelInterpretation({ model }) {
  const interp = model.interpretation;
  if (!interp) return null;

  const rec = REC_CONFIG[interp.recommendation] ?? {
    label: interp.recommendation,
    cls: "fallback",
  };

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <span className={styles.eyebrow}>interpretation</span>
        <span className={`${styles.recPill} ${styles[rec.cls]}`}>
          {rec.label}
        </span>
      </div>
      <div className={styles.body}>
        <p className={styles.summary}>{interp.summary}</p>

        {interp.strengths?.length > 0 && (
          <ul className={styles.list}>
            {interp.strengths.map((s, i) => (
              <li key={i} className={`${styles.listItem} ${styles.strength}`}>
                {s}
              </li>
            ))}
          </ul>
        )}

        {interp.weaknesses?.length > 0 && (
          <ul className={styles.list}>
            {interp.weaknesses.map((s, i) => (
              <li key={i} className={`${styles.listItem} ${styles.weakness}`}>
                {s}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
