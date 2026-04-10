import styles from "./ModelDetailPanel.module.css";
import RankBadge from "../primitives/RankBadge.jsx";
import TierBadge from "../primitives/TierBadge.jsx";
import ModelMetricGrid from "./ModelMetricGrid.jsx";
import ModelTrainingMetadata from "./ModelTrainingMetadata.jsx";
import TrainingHistoryChart from "./TrainingHistoryChart.jsx";
import ModelPlotPanel from "./ModelPlotPanel.jsx";
import ModelInterpretation from "./ModelInterpretation.jsx";
import { colorForModel } from "../charts/chartTheme.js";

export default function ModelDetailPanel({ model }) {
  const color = colorForModel(model.name);

  return (
    <div className={styles.panel} style={{ "--model-accent": color }}>
      {/* Two-column body */}
      <div className={styles.body}>
        {/* Left: overview + metrics + interpretation */}
        <div className={styles.leftCol}>
          <div className={styles.overviewCard}>
            <div className={styles.overviewRow}>
              <RankBadge rank={model.rank} />
              <h2 className={styles.modelName}>{model.name}</h2>
              <TierBadge tier={model.tier} />
              <span className={`${styles.statusPill} ${model.status === "success" ? styles.statusOk : ""}`}>
                evaluated
              </span>
            </div>
            {model.interpretation?.role_label && (
              <p className={styles.roleLabel}>{model.interpretation.role_label}</p>
            )}
            {model.rationale && (
              <p className={styles.rationale}>{model.rationale}</p>
            )}
          </div>

          <div className={styles.subsection}>
            <span className={styles.subsectionLabel}>metrics</span>
            <ModelMetricGrid model={model} />
          </div>

          {model.interpretation && (
            <ModelInterpretation model={model} />
          )}
        </div>

        {/* Right: training history + plots */}
        <div className={styles.rightCol}>
          <TrainingHistoryChart model={model} />
          <ModelPlotPanel model={model} />
        </div>
      </div>

      {/* Footer: training configuration */}
      <div className={styles.footerRow}>
        <ModelTrainingMetadata model={model} />
      </div>
    </div>
  );
}
