import styles from "./RunCompletionCard.module.css";

function SuccessIcon() {
  return (
    <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

function FailIcon() {
  return (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <line x1="15" y1="9" x2="9" y2="15" />
      <line x1="9" y1="9" x2="15" y2="15" />
    </svg>
  );
}

function ArrowIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="5" y1="12" x2="19" y2="12" />
      <polyline points="12 5 19 12 12 19" />
    </svg>
  );
}

/**
 * RunCompletionCard
 *
 * Shown when a run reaches a terminal state (completed or failed).
 *
 * Props:
 *   runId       — run UUID
 *   status      — "completed" | "failed" | "partial_failure"
 *   datasetName — filename of the uploaded dataset
 *   stages      — stage map from getRunStatus
 *   onViewResults — callback to navigate to results dashboard
 */
export default function RunCompletionCard({ runId, status, datasetName, stages = {}, onViewResults }) {
  const isFailed = status === "failed";
  const isPartial = status === "partial_failure";

  const completedCount = Object.values(stages).filter(
    (s) => s.status === "completed"
  ).length;
  const totalCount = Object.keys(stages).length || 8;

  const titleLabel = isFailed ? "Run Failed" : "Run Complete";
  const heading = isFailed
    ? "Pipeline encountered an error"
    : isPartial
    ? "Pipeline completed with warnings"
    : "Pipeline finished successfully";
  const sub = isFailed
    ? "Some stages failed. You can inspect the partial results or start a new run."
    : "All models have been evaluated. Open the dashboard to explore results.";

  return (
    <div className={`${styles.card} ${isFailed ? styles.cardFailed : ""}`}>
      {/* Icon */}
      <div className={`${styles.iconWrap} ${isFailed ? styles.iconWrapFailed : ""}`}>
        {isFailed ? <FailIcon /> : <SuccessIcon />}
      </div>

      {/* Text */}
      <div className={styles.textGroup}>
        <span className={`${styles.titleLabel} ${isFailed ? styles.titleLabelFailed : ""}`}>
          {titleLabel}
        </span>
        <h2 className={styles.title}>{heading}</h2>
        <p className={styles.subtitle}>{sub}</p>
      </div>

      {/* Meta */}
      <div className={styles.meta}>
        {datasetName && (
          <>
            <div className={styles.metaItem}>
              <span className={styles.metaKey}>Dataset</span>
              <span className={styles.metaVal}>{datasetName}</span>
            </div>
            <span className={styles.metaSep}>·</span>
          </>
        )}
        <div className={styles.metaItem}>
          <span className={styles.metaKey}>Stages</span>
          <span className={styles.metaVal}>{completedCount} / {totalCount}</span>
        </div>
        {runId && (
          <>
            <span className={styles.metaSep}>·</span>
            <div className={styles.metaItem}>
              <span className={styles.metaKey}>Run ID</span>
              <span className={styles.metaVal}>{runId.slice(0, 8)}</span>
            </div>
          </>
        )}
      </div>

      {/* CTA */}
      {!isFailed && (
        <button className={styles.ctaBtn} onClick={onViewResults}>
          Open Results Dashboard
          <span className={styles.ctaArrow}>
            <ArrowIcon />
          </span>
        </button>
      )}

      {isFailed && (
        <button className={styles.ctaBtn} onClick={onViewResults}
          style={{ borderColor: "rgba(248,113,113,0.35)", color: "var(--accent-red)",
            background: "rgba(248,113,113,0.07)" }}>
          Inspect Partial Results
          <span className={styles.ctaArrow}><ArrowIcon /></span>
        </button>
      )}
    </div>
  );
}
