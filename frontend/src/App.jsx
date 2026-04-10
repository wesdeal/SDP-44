import { useEffect, useState } from "react";
import AppShell from "./layout/AppShell.jsx";
import RunDashboard from "./pages/RunDashboard.jsx";
import { getRun, listRuns, getLatestRunId, DEFAULT_RUN_ID, LIVE_MODE } from "./data/api.js";

/**
 * Root component.
 *
 * Run resolution order:
 *   1. ?runId= URL query param  → use as-is
 *   2. Live mode, no param      → GET /api/runs/latest → use that run_id
 *   3. Mock mode, no param      → DEFAULT_RUN_ID
 *
 * The run selector in the AppShell header populates from GET /api/runs and
 * lets the user switch between any available run without reloading the page.
 */
export default function App() {
  // null = not yet resolved (shows skeleton)
  const [runId, setRunId] = useState(null);
  const [run, setRun] = useState(null);
  const [error, setError] = useState(null);
  const [runs, setRuns] = useState([]);

  // ── Step 1: Resolve the initial run ID ────────────────────────────────────
  useEffect(() => {
    const fromUrl = new URLSearchParams(window.location.search).get("runId");
    if (fromUrl) {
      setRunId(fromUrl);
      return;
    }
    getLatestRunId()
      .then((id) => setRunId(id ?? DEFAULT_RUN_ID))
      .catch(() => setRunId(DEFAULT_RUN_ID));
  }, []);

  // ── Step 2: Populate run selector list ────────────────────────────────────
  useEffect(() => {
    listRuns()
      .then(setRuns)
      .catch(() => {}); // silently fail — selector just won't appear
  }, []);

  // ── Step 3: Fetch run data whenever runId changes ─────────────────────────
  useEffect(() => {
    if (!runId) return;
    let alive = true;
    setRun(null);
    setError(null);
    getRun(runId)
      .then((r) => { if (alive) setRun(r); })
      .catch((e) => { if (alive) setError(e); });
    return () => { alive = false; };
  }, [runId]);

  // ── Run selector handler ───────────────────────────────────────────────────
  function handleRunChange(newRunId) {
    setRunId(newRunId);
    // Sync the URL so the page is bookmarkable / shareable
    const url = new URL(window.location.href);
    url.searchParams.set("runId", newRunId);
    window.history.pushState({}, "", url.toString());
  }

  // Only show selector when we have multiple runs (live mode)
  const runSelector =
    runs.length > 0 ? (
      <RunSelector runs={runs} currentRunId={runId} onRunChange={handleRunChange} />
    ) : null;

  return (
    <AppShell run={run} runSelector={runSelector}>
      {error ? (
        <ErrorPanel error={error} />
      ) : run ? (
        <RunDashboard run={run} />
      ) : (
        <DashboardSkeleton />
      )}
    </AppShell>
  );
}

/* ── Run selector ─────────────────────────────────────────────────────────── */

function RunSelector({ runs, currentRunId, onRunChange }) {
  return (
    <select
      value={currentRunId ?? ""}
      onChange={(e) => onRunChange(e.target.value)}
      aria-label="Select run"
      style={selectorStyle}
    >
      {currentRunId && !runs.find((r) => r.run_id === currentRunId) && (
        <option value={currentRunId}>{currentRunId.slice(0, 8)}… (current)</option>
      )}
      {runs.map((r) => (
        <option key={r.run_id} value={r.run_id}>
          {r.dataset_name ?? "unknown"} · {r.run_id.slice(0, 8)} · {r.status}
        </option>
      ))}
    </select>
  );
}

const selectorStyle = {
  height: 28,
  padding: "0 28px 0 8px",
  background: "var(--bg-2)",
  border: "1px solid var(--border-strong)",
  borderRadius: "var(--r-md)",
  color: "var(--text-md)",
  fontFamily: "var(--font-mono)",
  fontSize: 11,
  letterSpacing: "0.04em",
  cursor: "pointer",
  outline: "none",
  appearance: "none",
  WebkitAppearance: "none",
  backgroundImage:
    "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' fill='none'%3E%3Cpath d='M1 1l4 4 4-4' stroke='%236b7280' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E\")",
  backgroundRepeat: "no-repeat",
  backgroundPosition: "right 8px center",
  maxWidth: 280,
  overflow: "hidden",
  textOverflow: "ellipsis",
  whiteSpace: "nowrap",
  flexShrink: 0,
};

/* ── Error panel ──────────────────────────────────────────────────────────── */

function ErrorPanel({ error }) {
  const message =
    error instanceof Error ? error.message : String(error ?? "Unknown error");
  return (
    <div style={panelStyle}>
      <div style={errorDotStyle} />
      <div>
        <div style={errorTitleStyle}>Failed to load run</div>
        <pre style={errorMessageStyle}>{message}</pre>
        <div style={errorHintStyle}>
          Check that the backend is running ({LIVE_MODE ? "live mode" : "mock mode"}) and that{" "}
          <code style={{ fontFamily: "var(--font-mono)", color: "var(--text-md)" }}>
            getRun()
          </code>{" "}
          in{" "}
          <code style={{ fontFamily: "var(--font-mono)", color: "var(--text-md)" }}>
            src/data/api.js
          </code>{" "}
          resolves correctly.
        </div>
      </div>
    </div>
  );
}

const panelStyle = {
  display: "flex",
  alignItems: "flex-start",
  gap: "var(--sp-4)",
  padding: "var(--sp-5)",
  background: "var(--bg-1)",
  border: "1px solid rgba(248, 113, 113, 0.35)",
  borderRadius: "var(--r-lg)",
  boxShadow: "0 0 24px -12px rgba(248, 113, 113, 0.3)",
};
const errorDotStyle = {
  flexShrink: 0,
  width: 8,
  height: 8,
  marginTop: 5,
  borderRadius: "50%",
  background: "var(--accent-red)",
  boxShadow: "0 0 8px rgba(248, 113, 113, 0.6)",
};
const errorTitleStyle = {
  fontFamily: "var(--font-mono)",
  fontSize: 12,
  letterSpacing: "0.12em",
  textTransform: "uppercase",
  color: "var(--accent-red)",
  marginBottom: "var(--sp-2)",
};
const errorMessageStyle = {
  fontFamily: "var(--font-mono)",
  fontSize: 13,
  color: "var(--text-md)",
  background: "var(--bg-2)",
  border: "1px solid var(--border)",
  borderRadius: "var(--r-md)",
  padding: "var(--sp-3) var(--sp-4)",
  margin: "0 0 var(--sp-3)",
  overflowX: "auto",
  whiteSpace: "pre-wrap",
  wordBreak: "break-word",
};
const errorHintStyle = {
  fontSize: 12,
  color: "var(--text-lo)",
  lineHeight: 1.5,
};

/* ── Skeleton ─────────────────────────────────────────────────────────────── */

function DashboardSkeleton() {
  return (
    <div style={skeletonStackStyle}>
      <SkeletonBlock height={148} />
      <div style={skeletonCardRowStyle}>
        {[...Array(6)].map((_, i) => (
          <SkeletonBlock key={i} height={92} />
        ))}
      </div>
      <SkeletonBlock height={180} />
      <SkeletonBlock height={180} />
      <SkeletonBlock height={360} />
      <SkeletonBlock height={300} />
    </div>
  );
}

function SkeletonBlock({ height }) {
  return (
    <div
      style={{
        height,
        background: "var(--bg-1)",
        border: "1px solid var(--border)",
        borderRadius: "var(--r-lg)",
        overflow: "hidden",
        position: "relative",
      }}
    >
      <div style={shimmerStyle} />
    </div>
  );
}

const skeletonStackStyle = {
  display: "flex",
  flexDirection: "column",
  gap: "var(--sp-6)",
};

const skeletonCardRowStyle = {
  display: "grid",
  gridTemplateColumns: "repeat(6, minmax(0, 1fr))",
  gap: "var(--sp-3)",
};

const shimmerStyle = {
  position: "absolute",
  inset: 0,
  background:
    "linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.03) 50%, transparent 100%)",
  backgroundSize: "200% 100%",
  animation: "skeleton-shimmer 1.6s ease-in-out infinite",
};

if (typeof document !== "undefined") {
  const id = "skeleton-shimmer-kf";
  if (!document.getElementById(id)) {
    const style = document.createElement("style");
    style.id = id;
    style.textContent = `
      @keyframes skeleton-shimmer {
        0%   { background-position: -200% 0; }
        100% { background-position:  200% 0; }
      }
    `;
    document.head.appendChild(style);
  }
}
