import styles from "./AppShell.module.css";
import { shortId } from "../utils/format.js";

const STATUS_LABELS = {
  completed: "completed",
  running: "running",
  failed: "failed",
  unknown: "—",
};

const STATUS_STYLE = {
  completed: styles.statusCompleted,
  running: styles.statusRunning,
  failed: styles.statusFailed,
};

export default function AppShell({ run, runSelector, children }) {
  const status = run?.status ?? "loading";
  const runIdShort = run?.run_id ? shortId(run.run_id, 8) : "—";
  const dataset = run?.dataset_profile?.dataset_name ?? null;
  const taskType = run?.task_spec?.task_type ?? null;

  const statusLabel = STATUS_LABELS[status] ?? status;
  const statusClass = STATUS_STYLE[status] ?? "";

  return (
    <div className={styles.shell}>
      <header className={styles.topbar}>
        {/* Brand */}
        <div className={styles.brand}>
          <span className={styles.brandMark} aria-hidden="true" />
          <span className={styles.brandName}>Pipeline · Model Arena</span>
        </div>

        <div className={styles.divider} aria-hidden="true" />

        {/* Run metadata */}
        <div className={styles.runMeta}>
          <span className={styles.metaKey}>RUN</span>
          <strong className={styles.metaValue}>{runIdShort}</strong>
          {dataset && (
            <>
              <span className={styles.dot} aria-hidden="true">·</span>
              <span className={styles.metaValue}>{dataset}</span>
            </>
          )}
          {taskType && (
            <>
              <span className={styles.dot} aria-hidden="true">·</span>
              <span className={styles.metaValue}>{taskType}</span>
            </>
          )}
        </div>

        {runSelector && (
          <>
            <div className={styles.divider} aria-hidden="true" />
            <div className={styles.selectorWrap}>{runSelector}</div>
          </>
        )}

        <div className={styles.spacer} />

        {/* Status pill */}
        <span className={`${styles.statusPill} ${statusClass}`}>
          <span className={styles.statusDot} aria-hidden="true" />
          {statusLabel}
        </span>
      </header>

      <main className={styles.content}>{children}</main>
    </div>
  );
}
