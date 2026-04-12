import styles from "./PipelineStageList.module.css";

/** Canonical pipeline stage definitions — display name + manifest key + parallel flag */
const STAGE_DEFS = [
  { key: "ingestion",               label: "Ingestion",               parallel: false },
  { key: "problem_classification",  label: "Problem Classification",  parallel: false },
  { key: "preprocessing_planning",  label: "Preprocessing Planning",  parallel: true  },
  { key: "evaluation_protocol",     label: "Evaluation Protocol",     parallel: true  },
  { key: "model_selection",         label: "Model Selection",         parallel: false },
  { key: "training",                label: "Training",                parallel: false },
  { key: "evaluation",              label: "Evaluation",              parallel: false },
  { key: "artifact_assembly",       label: "Artifact Assembly",       parallel: false },
];

const TERMINAL = new Set(["completed", "failed", "partial_failure"]);

/** Format ISO timestamp delta into a human duration string. */
function elapsedLabel(started, completed) {
  if (!started) return null;
  const end = completed ? new Date(completed) : new Date();
  const ms = end - new Date(started);
  if (ms < 1000) return `${ms}ms`;
  const s = (ms / 1000).toFixed(1);
  if (ms < 60_000) return `${s}s`;
  const m = Math.floor(ms / 60_000);
  const rem = ((ms % 60_000) / 1000).toFixed(0);
  return `${m}m ${rem}s`;
}

/* ── Status indicator icon ─────────────────────────────────────────────── */

function StageIndicator({ status }) {
  if (status === "completed") {
    return (
      <div className={`${styles.indicator} ${styles.indicatorCompleted}`}>
        <svg width="10" height="8" viewBox="0 0 10 8" fill="none">
          <path d="M1 4l2.5 2.5L9 1" stroke="#34d399" strokeWidth="1.6"
            strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </div>
    );
  }
  if (status === "failed") {
    return (
      <div className={`${styles.indicator} ${styles.indicatorFailed}`}>
        <svg width="8" height="8" viewBox="0 0 8 8" fill="none">
          <path d="M1 1l6 6M7 1L1 7" stroke="#f87171" strokeWidth="1.5"
            strokeLinecap="round" />
        </svg>
      </div>
    );
  }
  if (status === "partial_failure") {
    return (
      <div className={`${styles.indicator} ${styles.indicatorPartial}`}>
        <svg width="8" height="8" viewBox="0 0 8 8" fill="none">
          <path d="M4 2v2.5M4 6h.01" stroke="#fbbf24" strokeWidth="1.5"
            strokeLinecap="round" />
        </svg>
      </div>
    );
  }
  if (status === "running") {
    return (
      <div className={styles.indicator}>
        <div className={styles.indicatorRunning} />
      </div>
    );
  }
  // pending / unknown
  return (
    <div className={styles.indicator}>
      <div className={styles.indicatorPending} />
    </div>
  );
}

/* ── Single stage row ──────────────────────────────────────────────────── */

function PipelineStageRow({ label, status, parallel, stageData }) {
  const started = stageData?.started_at ?? null;
  const completed = stageData?.completed_at ?? null;
  const error = stageData?.error ?? null;

  const rowClass = [
    styles.stageRow,
    status === "running"          ? styles.stageRowRunning   : "",
    status === "completed"        ? styles.stageRowCompleted : "",
    status === "failed"           ? styles.stageRowFailed    : "",
  ].filter(Boolean).join(" ");

  const nameClass = [
    styles.stageName,
    status === "pending" ? styles.stageNamePending : "",
  ].filter(Boolean).join(" ");

  let subText = null;
  let subClass = styles.stageSub;

  if (status === "running") {
    subText = "In progress…";
    subClass = `${styles.stageSub} ${styles.stageSubRunning}`;
  } else if (status === "completed") {
    const dur = elapsedLabel(started, completed);
    subText = dur ? `Completed in ${dur}` : "Completed";
    subClass = `${styles.stageSub} ${styles.stageSubCompleted}`;
  } else if (status === "failed") {
    subText = error ? `Failed: ${error.slice(0, 80)}` : "Failed";
    subClass = `${styles.stageSub} ${styles.stageSubFailed}`;
  } else if (status === "partial_failure") {
    subText = "Partial failure";
    subClass = `${styles.stageSub} ${styles.stageSubFailed}`;
  } else {
    subText = "Waiting";
  }

  const timing = status === "running" && started
    ? elapsedLabel(started, null)
    : null;

  return (
    <li className={rowClass}>
      <StageIndicator status={status} />
      <div className={styles.stageContent}>
        <div className={nameClass}>
          {label}
          {parallel && <span className={styles.parallelBadge}>parallel</span>}
        </div>
        {subText && <div className={subClass}>{subText}</div>}
      </div>
      {timing && <div className={styles.stageTiming}>{timing}</div>}
    </li>
  );
}

/* ── Stage list card ───────────────────────────────────────────────────── */

/**
 * PipelineStageList
 *
 * Props:
 *   stages  — object from GET /api/runs/{id}/status, keyed by stage name.
 *             { ingestion: { status, started_at, completed_at, error }, ... }
 *   runId   — run UUID, shown as subtitle
 *   datasetName — dataset filename for header annotation
 */
export default function PipelineStageList({ stages = {}, runId, datasetName }) {
  const completedCount = STAGE_DEFS.filter(
    (d) => stages[d.key]?.status === "completed"
  ).length;

  return (
    <div className={styles.card}>
      <div className={styles.cardHeader}>
        <span className={styles.cardTitle}>Pipeline Stages</span>
        <span className={styles.cardMeta}>
          {completedCount} / {STAGE_DEFS.length} completed
          {runId && ` · ${runId.slice(0, 8)}`}
        </span>
      </div>

      <ul className={styles.stageList}>
        {STAGE_DEFS.map((def) => {
          const stageData = stages[def.key] ?? null;
          const status = stageData?.status ?? "pending";
          return (
            <PipelineStageRow
              key={def.key}
              label={def.label}
              status={status}
              parallel={def.parallel}
              stageData={stageData}
            />
          );
        })}
      </ul>
    </div>
  );
}
