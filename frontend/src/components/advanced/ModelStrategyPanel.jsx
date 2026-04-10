import styles from "./ModelStrategyPanel.module.css";
import TierBadge from "../primitives/TierBadge.jsx";

const STRATEGY_NOTES = {
  XGBoost:
    "Iterative gradient boosting with validation-set early stopping. Convergence monitored via RMSE on the held-out validation split.",
  RandomForest:
    "Bagged tree ensemble. One-shot fit — no iterative training risk. Strong comparator for gradient boosting on tabular features.",
  DummyRegressor:
    "Constant mean predictor. Zero predictive capability — establishes the performance floor any real model must surpass.",
};

const TIER_ORDER = ["baseline", "classical", "specialized"];

function ModelCard({ selected, training, evaluation }) {
  const tierColor = {
    baseline: "var(--tier-baseline)",
    classical: "var(--tier-classical)",
    specialized: "var(--tier-specialized)",
  }[selected.tier] || "var(--text-lo)";

  const strategyNote = STRATEGY_NOTES[selected.name] ?? "—";
  const valScore = training?.best_val_score != null
    ? training.best_val_score.toFixed(4)
    : "—";
  const duration = training?.training_duration_seconds != null
    ? `${training.training_duration_seconds.toFixed(3)} s`
    : "—";
  const mae = evaluation?.metrics?.MAE != null
    ? evaluation.metrics.MAE.toFixed(4)
    : "—";

  return (
    <div
      className={styles.modelCard}
      style={{ "--model-color": tierColor }}
    >
      <div className={styles.cardHeader}>
        <span className={styles.modelName}>{selected.name}</span>
        <TierBadge tier={selected.tier} />
      </div>

      <p className={styles.strategyNote}>{strategyNote}</p>

      <div className={styles.cardGrid}>
        <div className={styles.cardStat}>
          <span className={styles.cardStatLabel}>hyperparam source</span>
          <span className={styles.cardStatVal}>{training?.hyperparameter_source ?? "default"}</span>
        </div>
        <div className={styles.cardStat}>
          <span className={styles.cardStatLabel}>best val score</span>
          <span className={styles.cardStatVal}>{valScore}</span>
        </div>
        <div className={styles.cardStat}>
          <span className={styles.cardStatLabel}>training time</span>
          <span className={styles.cardStatVal}>{duration}</span>
        </div>
        <div className={styles.cardStat}>
          <span className={styles.cardStatLabel}>test MAE</span>
          <span className={`${styles.cardStatVal} ${styles.cardStatHighlight}`}>{mae}</span>
        </div>
      </div>

      <div className={styles.rationaleRow}>
        <span className={styles.rationaleLabel}>selection rationale</span>
        <p className={styles.rationaleText}>{selected.rationale}</p>
      </div>

      {/* Hyperparameters */}
      {training?.hyperparameters && (
        <div className={styles.hpSection}>
          <span className={styles.hpLabel}>hyperparameters</span>
          <div className={styles.hpGrid}>
            {Object.entries(training.hyperparameters).map(([k, v]) => (
              <div key={k} className={styles.hpRow}>
                <span className={styles.hpKey}>{k}</span>
                <span className={styles.hpVal}>
                  {v === null ? "none" : String(v)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default function ModelStrategyPanel({ run }) {
  const { selected_models, training_results, evaluation_report } = run;

  // Index training and evaluation by model name
  const trainingByName = {};
  training_results.models.forEach((m) => { trainingByName[m.name] = m; });

  const evalByName = {};
  evaluation_report.models.forEach((m) => { evalByName[m.name] = m; });

  // Sort by tier order
  const sorted = [...selected_models.selected_models].sort((a, b) => {
    return TIER_ORDER.indexOf(a.tier) - TIER_ORDER.indexOf(b.tier);
  });

  const strategy = selected_models.selection_strategy.replace(/_/g, " ");

  return (
    <div className={styles.panel}>
      {/* Strategy summary */}
      <div className={styles.strategyHeader}>
        <div className={styles.stratHeaderItem}>
          <span className={styles.stratLabel}>selection strategy</span>
          <span className={styles.stratValue}>{strategy}</span>
        </div>
        <div className={styles.stratHeaderItem}>
          <span className={styles.stratLabel}>models selected</span>
          <span className={styles.stratValue}>{sorted.length} / one per tier</span>
        </div>
        <div className={styles.stratHeaderItem}>
          <span className={styles.stratLabel}>considered</span>
          <span className={styles.stratValue}>{selected_models.models_considered.length} candidates</span>
        </div>
      </div>

      {/* Model cards */}
      <div className={styles.cardGrid}>
        {sorted.map((sel) => (
          <ModelCard
            key={sel.name}
            selected={sel}
            training={trainingByName[sel.name]}
            evaluation={evalByName[sel.name]}
          />
        ))}
      </div>

      {/* Rejected models */}
      {selected_models.models_rejected?.length > 0 && (
        <div className={styles.rejectedSection}>
          <span className={styles.rejectedLabel}>rejected models</span>
          <div className={styles.rejectedList}>
            {selected_models.models_rejected.map((m) => (
              <div key={m.name} className={styles.rejectedRow}>
                <span className={styles.rejectedName}>{m.name}</span>
                <span className={styles.rejectedReason}>{m.reason}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
