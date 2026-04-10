import styles from "./ModelPlotPanel.module.css";

const PLOT_DEFS = [
  { key: "pred_vs_actual", label: "Actual vs Predicted",   suffix: "_pred_vs_actual.png" },
  { key: "residuals",      label: "Residual Distribution", suffix: "_residuals.png" },
  { key: "trace",          label: "Prediction Trace",      suffix: "_trace.png" },
];

export default function ModelPlotPanel({ model }) {
  const base = model.name.toLowerCase().replace(/[^a-z0-9]/g, "");

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <span className={styles.eyebrow}>evaluation plots</span>
        <h3 className={styles.title}>Plot Outputs</h3>
        <span className={styles.subtitle}>
          pending backend artifact paths
        </span>
      </div>
      <div className={styles.body}>
        {PLOT_DEFS.map((def) => (
          <div key={def.key} className={styles.slot}>
            <div className={styles.slotImg}>
              <span className={styles.slotGlyph}>⬡</span>
              <span className={styles.slotLabel}>{def.label}</span>
            </div>
            <span className={styles.slotPath}>
              plots/{base}{def.suffix}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
