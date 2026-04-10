import styles from "./PreprocessingDetailsPanel.module.css";

const METHOD_META = {
  imputation: {
    label: "Imputation",
    accentVar: "--accent-blue",
    note: "Forward-fill preserves the last known value, maintaining temporal continuity without introducing future information.",
  },
  lag_features: {
    label: "Lag Features",
    accentVar: "--accent-cyan",
    note: "Lag columns give the model access to historical signal values as input features, encoding temporal autocorrelation explicitly.",
  },
  rolling_stats: {
    label: "Rolling Statistics",
    accentVar: "--accent-amber",
    note: "Rolling mean and std capture local trend and volatility over short (6 h) and long (24 h) windows.",
  },
  z_norm: {
    label: "Z-score Normalisation",
    accentVar: "--accent-green",
    note: "Per-column standardisation prevents large-magnitude sensor channels from dominating gradient-based models.",
  },
};

function ParamChip({ k, v }) {
  const display = Array.isArray(v) ? `[${v.join(", ")}]` : String(v);
  return (
    <span className={styles.paramChip}>
      <span className={styles.paramKey}>{k}</span>
      <span className={styles.paramVal}>{display}</span>
    </span>
  );
}

function RowDelta({ before, after }) {
  const dropped = before - after;
  return (
    <div className={styles.rowDelta}>
      <span className={styles.rowBefore}>{before.toLocaleString()} rows</span>
      <span className={styles.rowArrow}>→</span>
      <span className={styles.rowAfter}>{after.toLocaleString()} rows</span>
      {dropped > 0 && (
        <span className={styles.rowDropped}>−{dropped.toLocaleString()} dropped</span>
      )}
    </div>
  );
}

function StepCard({ plan, manifest }) {
  const meta = METHOD_META[plan.method] || { label: plan.method, accentVar: "--text-lo", note: "" };
  const params = manifest?.parameters_used ?? plan.parameters;
  const cols = manifest?.columns_affected ?? plan.applies_to;
  const displayCols = Array.isArray(cols)
    ? cols
    : typeof cols === "string"
    ? [cols]
    : [];

  return (
    <div
      className={styles.stepCard}
      style={{ "--step-accent": `var(${meta.accentVar})` }}
    >
      <div className={styles.stepHeader}>
        <span className={styles.stepOrder}>{plan.order}</span>
        <span className={styles.stepLabel}>{meta.label}</span>
        {manifest && (
          <RowDelta before={manifest.rows_before} after={manifest.rows_after} />
        )}
      </div>

      {meta.note && <p className={styles.stepNote}>{meta.note}</p>}

      <div className={styles.stepMeta}>
        {/* Parameters */}
        <div className={styles.metaBlock}>
          <span className={styles.metaBlockLabel}>parameters</span>
          <div className={styles.paramList}>
            {Object.entries(params).map(([k, v]) => (
              <ParamChip key={k} k={k} v={v} />
            ))}
          </div>
        </div>

        {/* Columns */}
        {displayCols.length > 0 && (
          <div className={styles.metaBlock}>
            <span className={styles.metaBlockLabel}>columns</span>
            <div className={styles.colList}>
              {displayCols.map((c) => (
                <span key={c} className={styles.colChip}>{c}</span>
              ))}
            </div>
          </div>
        )}

        {/* Generated columns count */}
        {manifest?.fitted_params?.generated_columns != null && (
          <div className={styles.metaBlock}>
            <span className={styles.metaBlockLabel}>generated</span>
            <span className={styles.genCount}>
              {manifest.fitted_params.generated_columns} columns
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

export default function PreprocessingDetailsPanel({ run }) {
  const { preprocessing_plan, preprocessing_manifest } = run;
  const manifestByOrder = {};
  preprocessing_manifest.steps_applied.forEach((s) => {
    manifestByOrder[s.order] = s;
  });

  return (
    <div className={styles.panel}>
      <div className={styles.intro}>
        <span className={styles.introStat}>
          {preprocessing_plan.steps.length} steps applied
        </span>
        <span className={styles.introDivider}>·</span>
        <span className={styles.introStat}>temporal order preserved</span>
        <span className={styles.introDivider}>·</span>
        <span className={styles.introStat}>plan source: {preprocessing_plan.plan_source}</span>
      </div>

      <div className={styles.steps}>
        {preprocessing_plan.steps.map((step) => (
          <StepCard
            key={step.order}
            plan={step}
            manifest={manifestByOrder[step.order]}
          />
        ))}
      </div>
    </div>
  );
}
