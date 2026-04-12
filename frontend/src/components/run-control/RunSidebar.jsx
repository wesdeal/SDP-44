import styles from "./RunSidebar.module.css";

/** Stage keys in canonical order — used for progress bar segments */
const STAGE_KEYS = [
  "ingestion", "problem_classification", "preprocessing_planning",
  "evaluation_protocol", "model_selection", "training", "evaluation",
  "artifact_assembly",
];

function formatDate(iso) {
  if (!iso) return null;
  try {
    return new Date(iso).toLocaleDateString(undefined, {
      month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
    });
  } catch {
    return null;
  }
}

/* ── Stage progress bar ──────────────────────────────────────────────────── */

function StageProgressBar({ stages }) {
  return (
    <div className={styles.stageProgress} role="progressbar" aria-label="Stage progress">
      {STAGE_KEYS.map((key) => {
        const s = stages?.[key]?.status ?? "pending";
        const segClass = [
          styles.stageProgressSegment,
          s === "completed"      ? styles.stageProgressSegmentCompleted : "",
          s === "running"        ? styles.stageProgressSegmentRunning   : "",
          s === "failed"         ? styles.stageProgressSegmentFailed    : "",
          s === "partial_failure"? styles.stageProgressSegmentFailed    : "",
        ].filter(Boolean).join(" ");
        return <div key={key} className={segClass} />;
      })}
    </div>
  );
}

/* ── Small status pill ───────────────────────────────────────────────────── */

function StatusPill({ status }) {
  const pillClass = [
    styles.statusPill,
    status === "running"         ? styles.statusPillRunning   : "",
    status === "completed"       ? styles.statusPillCompleted : "",
    status === "failed"          ? styles.statusPillFailed    : "",
    status === "partial_failure" ? styles.statusPillFailed    : "",
  ].filter(Boolean).join(" ");

  const dotClass = [
    styles.statusDot,
    status === "running" ? styles.statusDotRunning : "",
  ].filter(Boolean).join(" ");

  const label =
    status === "partial_failure" ? "partial" :
    status ?? "—";

  return (
    <span className={pillClass}>
      <span className={dotClass} />
      {label}
    </span>
  );
}

/* ── Current run card ────────────────────────────────────────────────────── */

function CurrentRunCard({ runId, runStatus, datasetName }) {
  const status = runStatus?.status ?? "running";
  const stages = runStatus?.stages ?? {};
  const displayName = datasetName ?? runStatus?.dataset_name ?? "Unknown dataset";

  return (
    <div className={styles.currentRunCard}>
      <div className={styles.currentRunTop}>
        <span className={styles.currentRunDataset} title={displayName}>
          {displayName}
        </span>
        <StatusPill status={status} />
      </div>
      <StageProgressBar stages={stages} />
      {runId && (
        <span className={styles.currentRunId}>{runId.slice(0, 8)}…</span>
      )}
    </div>
  );
}

/* ── Past runs list ──────────────────────────────────────────────────────── */

function PastRunsList({ runs, activeRunId, onSelectRun }) {
  if (!runs || runs.length === 0) {
    return <div className={styles.pastRunsEmpty}>No past runs</div>;
  }

  return (
    <div className={styles.pastRunsList}>
      {runs.map((r) => {
        const isActive = r.run_id === activeRunId;
        return (
          <button
            key={r.run_id}
            className={`${styles.pastRunItem} ${isActive ? styles.pastRunItemActive : ""}`}
            onClick={() => onSelectRun(r.run_id, r.status)}
            title={r.run_id}
          >
            <div className={styles.pastRunTop}>
              <span className={styles.pastRunDataset}>
                {r.dataset_name ?? "Unknown"}
              </span>
              <StatusPill status={r.status} />
            </div>
            <span className={styles.pastRunMeta}>
              {formatDate(r.created_at) ?? r.run_id.slice(0, 8)}
            </span>
          </button>
        );
      })}
    </div>
  );
}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */

/**
 * RunSidebar
 *
 * Props:
 *   mode          — "upload" | "running" | "completed"
 *   runId         — active run UUID (null when mode=upload)
 *   runStatus     — object from getRunStatus (null when mode=upload)
 *   datasetName   — filename of active dataset (for display)
 *   runs          — array from listRuns (past runs)
 *   activeRunId   — UUID to highlight in past runs (may equal runId)
 *   onNewRun      — callback to reset to upload mode
 *   onSelectRun   — callback(runId, status) when user clicks a past run
 */
export default function RunSidebar({
  mode,
  runId,
  runStatus,
  datasetName,
  runs = [],
  activeRunId,
  onNewRun,
  onSelectRun,
}) {
  const showCurrentRun = mode === "running" || mode === "completed";

  // Filter out the current active run from past runs to avoid duplicate
  const pastRuns = runs.filter((r) => r.run_id !== runId);

  return (
    <div className={styles.sidebar}>
      {/* Header */}
      <div className={styles.header}>
        <span className={styles.sectionLabel}>Pipeline Control</span>
        {mode !== "upload" && (
          <button className={styles.newRunBtn} onClick={onNewRun}>
            <span className={styles.newRunPlus}>+</span>
            New Run
          </button>
        )}
      </div>

      {/* Current run */}
      {showCurrentRun && (
        <div className={styles.section}>
          <div className={styles.sectionTitle}>Active Run</div>
          <CurrentRunCard
            runId={runId}
            runStatus={runStatus}
            datasetName={datasetName}
          />
        </div>
      )}

      {/* Past runs */}
      <div className={styles.section} style={{ flex: 1, overflow: "hidden",
        display: "flex", flexDirection: "column" }}>
        <div className={styles.sectionTitle}>Past Runs</div>
        <PastRunsList
          runs={pastRuns}
          activeRunId={activeRunId}
          onSelectRun={onSelectRun}
        />
      </div>
    </div>
  );
}
