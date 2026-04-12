import { useEffect, useRef, useState } from "react";
import { createRun, getRunStatus, listRuns } from "../data/api.js";
import RunSidebar from "../components/run-control/RunSidebar.jsx";
import UploadDropzone from "../components/run-control/UploadDropzone.jsx";
import PipelineStageList from "../components/run-control/PipelineStageList.jsx";
import RunCompletionCard from "../components/run-control/RunCompletionCard.jsx";
import styles from "./RunControlPage.module.css";

const TERMINAL = new Set(["completed", "failed", "partial_failure"]);
const POLL_INTERVAL_MS = 2000;

/* ── Run status header (shown above the stage list) ─────────────────────── */

function RunStatusHeader({ status, datasetName, runId }) {
  const statusLabel =
    status === "running"         ? "Running pipeline…" :
    status === "completed"       ? "Pipeline complete" :
    status === "failed"          ? "Pipeline failed" :
    status === "partial_failure" ? "Completed with warnings" :
    "Initializing…";

  const statusColor =
    status === "running"         ? "var(--accent-amber)" :
    status === "completed"       ? "var(--accent-green)" :
    status === "failed"          ? "var(--accent-red)" :
    status === "partial_failure" ? "var(--accent-amber)" :
    "var(--text-dim)";

  return (
    <div style={{
      display: "flex",
      flexDirection: "column",
      gap: "var(--sp-1)",
    }}>
      <span style={{
        fontFamily: "var(--font-mono)",
        fontSize: 10,
        letterSpacing: "0.18em",
        textTransform: "uppercase",
        color: "var(--text-dim)",
      }}>
        Live Execution
      </span>
      <div style={{ display: "flex", alignItems: "center", gap: "var(--sp-3)" }}>
        <h1 style={{
          margin: 0,
          fontFamily: "var(--font-sans)",
          fontSize: 20,
          fontWeight: 600,
          color: "var(--text-hi)",
        }}>
          {statusLabel}
        </h1>
        {status === "running" && (
          <span style={{
            display: "inline-block",
            width: 8,
            height: 8,
            borderRadius: "50%",
            background: "var(--accent-amber)",
            boxShadow: "0 0 8px rgba(251,191,36,0.6)",
            animation: "runPulse 1.2s ease-in-out infinite",
          }} />
        )}
      </div>
      <div style={{
        display: "flex",
        alignItems: "center",
        gap: "var(--sp-2)",
        fontFamily: "var(--font-mono)",
        fontSize: 11,
        color: "var(--text-lo)",
      }}>
        {datasetName && <span>{datasetName}</span>}
        {datasetName && runId && <span style={{ color: "var(--text-dim)" }}>·</span>}
        {runId && (
          <span style={{ color: "var(--text-dim)" }}>
            {runId.slice(0, 8)}
          </span>
        )}
      </div>
      <style>{`@keyframes runPulse { 0%,100%{opacity:.4} 50%{opacity:1} }`}</style>
    </div>
  );
}

/* ── Upload main panel ────────────────────────────────────────────────────── */

function UploadPanel({ file, onFileSelect, onStartRun, isStarting, uploadError }) {
  return (
    <div className={styles.uploadMain}>
      <div className={styles.uploadTitle}>
        <span className={styles.uploadTitleLabel}>Pipeline Control</span>
        <h1 className={styles.uploadTitleHeading}>Start a new run</h1>
        <p className={styles.uploadTitleSub}>
          Upload a dataset to begin automatic model evaluation
        </p>
      </div>

      <UploadDropzone
        file={file}
        onFileSelect={onFileSelect}
        onStartRun={onStartRun}
        isStarting={isStarting}
        error={uploadError}
      />
    </div>
  );
}

/* ── Pipeline status panel ────────────────────────────────────────────────── */

function StatusPanel({ runId, runStatus, datasetName }) {
  const status = runStatus?.status ?? "running";
  const stages = runStatus?.stages ?? {};

  return (
    <div className={styles.statusMain}>
      <RunStatusHeader status={status} datasetName={datasetName} runId={runId} />
      <PipelineStageList stages={stages} runId={runId} datasetName={datasetName} />
    </div>
  );
}

/* ── Completion panel ─────────────────────────────────────────────────────── */

function CompletionPanel({ runId, runStatus, datasetName, onViewResults }) {
  const status = runStatus?.status ?? "completed";
  const stages = runStatus?.stages ?? {};

  return (
    <div className={styles.completionMain}>
      <RunCompletionCard
        runId={runId}
        status={status}
        datasetName={datasetName}
        stages={stages}
        onViewResults={onViewResults}
      />
      {/* Keep stage list visible below for reference */}
      <PipelineStageList stages={stages} runId={runId} datasetName={datasetName} />
    </div>
  );
}

/* ── Main page ────────────────────────────────────────────────────────────── */

/**
 * RunControlPage
 *
 * Props:
 *   onNavigateToDashboard(runId) — callback to switch to the results view
 *   initialRunId — optional: if a running run_id is passed, jump to tracking mode
 */
export default function RunControlPage({ onNavigateToDashboard, initialRunId }) {
  // "upload" | "running" | "completed"
  const [mode, setMode] = useState(initialRunId ? "running" : "upload");
  const [file, setFile] = useState(null);
  const [runId, setRunId] = useState(initialRunId ?? null);
  const [runStatus, setRunStatus] = useState(null);
  const [uploadError, setUploadError] = useState(null);
  const [isStarting, setIsStarting] = useState(false);
  const [runs, setRuns] = useState([]);

  // ── Fetch past runs list ─────────────────────────────────────────────────
  function refreshRuns() {
    listRuns()
      .then(setRuns)
      .catch(() => {});
  }

  useEffect(() => {
    refreshRuns();
  }, []);

  // Refresh past runs whenever a run completes
  useEffect(() => {
    if (mode === "completed") {
      refreshRuns();
    }
  }, [mode]);

  // ── Polling ──────────────────────────────────────────────────────────────
  useEffect(() => {
    if (mode !== "running" || !runId) return;

    let cancelled = false;

    async function poll() {
      while (!cancelled) {
        try {
          const status = await getRunStatus(runId);
          if (!cancelled) {
            setRunStatus(status);
            if (TERMINAL.has(status.status)) {
              setMode("completed");
              break;
            }
          }
        } catch (_) {
          // Network error — keep polling
        }
        // Wait before next poll
        await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
      }
    }

    // Start immediately
    poll();

    return () => {
      cancelled = true;
    };
  }, [mode, runId]);

  // ── Start run ────────────────────────────────────────────────────────────
  async function handleStartRun() {
    if (!file || isStarting) return;
    setIsStarting(true);
    setUploadError(null);

    try {
      const { run_id } = await createRun(file);
      setRunId(run_id);
      setRunStatus(null);
      setMode("running");
    } catch (e) {
      setUploadError(e.message ?? "Failed to start run");
    } finally {
      setIsStarting(false);
    }
  }

  // ── New run ──────────────────────────────────────────────────────────────
  function handleNewRun() {
    setMode("upload");
    setFile(null);
    setUploadError(null);
    // Keep runId/runStatus so sidebar still shows the last run in "active" card
    // until the user starts another one
  }

  // ── Select past run ──────────────────────────────────────────────────────
  function handleSelectRun(selectedRunId, status) {
    if (status === "completed" || status === "partial_failure") {
      onNavigateToDashboard(selectedRunId);
    } else if (status === "running") {
      // Switch to tracking that run
      setRunId(selectedRunId);
      setRunStatus(null);
      setMode("running");
    }
    // failed runs: navigate to dashboard to inspect partial artifacts
    else if (status === "failed") {
      onNavigateToDashboard(selectedRunId);
    }
  }

  // ── View results ─────────────────────────────────────────────────────────
  function handleViewResults() {
    if (runId) onNavigateToDashboard(runId);
  }

  const datasetName = file?.name ?? runStatus?.dataset_name ?? null;

  return (
    <div className={styles.page}>
      {/* Sidebar */}
      <div className={styles.sidebar}>
        <RunSidebar
          mode={mode}
          runId={runId}
          runStatus={runStatus}
          datasetName={datasetName}
          runs={runs}
          activeRunId={runId}
          onNewRun={handleNewRun}
          onSelectRun={handleSelectRun}
        />
      </div>

      {/* Main content */}
      <div className={styles.main}>
        {mode === "upload" && (
          <UploadPanel
            file={file}
            onFileSelect={setFile}
            onStartRun={handleStartRun}
            isStarting={isStarting}
            uploadError={uploadError}
          />
        )}

        {mode === "running" && (
          <StatusPanel
            runId={runId}
            runStatus={runStatus}
            datasetName={datasetName}
          />
        )}

        {mode === "completed" && (
          <CompletionPanel
            runId={runId}
            runStatus={runStatus}
            datasetName={datasetName}
            onViewResults={handleViewResults}
          />
        )}
      </div>
    </div>
  );
}
