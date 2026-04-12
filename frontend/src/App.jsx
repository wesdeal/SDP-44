import { useEffect, useState } from "react";
import AppShell from "./layout/AppShell.jsx";
import RunDashboard from "./pages/RunDashboard.jsx";
import RunControlPage from "./pages/RunControlPage.jsx";
import { getRun, listRuns, getLatestRunId, DEFAULT_RUN_ID, LIVE_MODE } from "./data/api.js";

/**
 * Root component.
 *
 * Views:
 *   "control"   — Run Control / Pipeline Status page (default, no runId in URL)
 *   "dashboard" — Results dashboard for a specific run (?view=dashboard&runId=...)
 *
 * Navigation is URL-param-driven (no router library), matching the existing pattern.
 * Backwards-compat: a bare ?runId= URL (no view param) opens the dashboard directly.
 */
export default function App() {
  // ── View state ─────────────────────────────────────────────────────────
  const [view, setView] = useState(() => {
    const params = new URLSearchParams(window.location.search);
    // Bare ?runId= is treated as dashboard (backwards compat)
    if (params.get("view") === "dashboard" || params.has("runId")) return "dashboard";
    return "control";
  });

  // ── Dashboard state ────────────────────────────────────────────────────
  const [runId, setRunId] = useState(null);
  const [run, setRun] = useState(null);
  const [error, setError] = useState(null);
  const [runs, setRuns] = useState([]);

  // ── Resolve initial runId when in dashboard view ───────────────────────
  useEffect(() => {
    if (view !== "dashboard") return;

    const fromUrl = new URLSearchParams(window.location.search).get("runId");
    if (fromUrl) {
      setRunId(fromUrl);
      return;
    }
    // No runId in URL — resolve latest
    getLatestRunId()
      .then((id) => setRunId(id ?? DEFAULT_RUN_ID))
      .catch(() => setRunId(DEFAULT_RUN_ID));
  }, [view]);

  // ── Populate run selector list ─────────────────────────────────────────
  useEffect(() => {
    if (view !== "dashboard") return;
    listRuns()
      .then(setRuns)
      .catch(() => {});
  }, [view]);

  // ── Fetch run data whenever runId changes ──────────────────────────────
  useEffect(() => {
    if (!runId || view !== "dashboard") return;
    let alive = true;
    setRun(null);
    setError(null);
    getRun(runId)
      .then((r) => { if (alive) setRun(r); })
      .catch((e) => { if (alive) setError(e); });
    return () => { alive = false; };
  }, [runId, view]);

  // ── Navigation helpers ─────────────────────────────────────────────────

  /** Switch to the results dashboard for a specific run. */
  function goToDashboard(id) {
    setRunId(id);
    setRun(null);
    setError(null);
    setView("dashboard");
    const url = new URL(window.location.href);
    url.searchParams.set("view", "dashboard");
    url.searchParams.set("runId", id);
    window.history.pushState({}, "", url.toString());
  }

  /** Switch to the Run Control page. */
  function goToControl() {
    setView("control");
    const url = new URL(window.location.href);
    url.searchParams.set("view", "control");
    url.searchParams.delete("runId");
    window.history.pushState({}, "", url.toString());
  }

  // ── Run selector handler (dashboard) ──────────────────────────────────
  function handleRunChange(newRunId) {
    setRunId(newRunId);
    const url = new URL(window.location.href);
    url.searchParams.set("runId", newRunId);
    window.history.pushState({}, "", url.toString());
  }

  // ── Run selector dropdown ──────────────────────────────────────────────
  const runSelector =
    view === "dashboard" && runs.length > 0 ? (
      <RunSelector runs={runs} currentRunId={runId} onRunChange={handleRunChange} />
    ) : null;

  // ── Control link in topbar ─────────────────────────────────────────────
  const controlLink =
    view === "dashboard" ? (
      <ControlLink onClick={goToControl} />
    ) : null;

  return (
    <AppShell run={view === "dashboard" ? run : null} runSelector={runSelector}
      controlLink={controlLink}>
      {view === "control" ? (
        <RunControlPage onNavigateToDashboard={goToDashboard} />
      ) : error ? (
        <ErrorPanel error={error} />
      ) : run ? (
        <RunDashboard run={run} />
      ) : (
        <DashboardSkeleton />
      )}
    </AppShell>
  );
}

/* ── Control link ─────────────────────────────────────────────────────────── */

function ControlLink({ onClick }) {
  return (
    <button onClick={onClick} style={controlLinkStyle} title="Go to Pipeline Control">
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor"
        strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"
        style={{ marginRight: 6 }}>
        <polyline points="15 18 9 12 15 6" />
      </svg>
      Control
    </button>
  );
}

const controlLinkStyle = {
  display: "inline-flex",
  alignItems: "center",
  padding: "4px 10px",
  background: "none",
  border: "1px solid var(--border-strong)",
  borderRadius: "var(--r-md)",
  color: "var(--text-lo)",
  fontFamily: "var(--font-mono)",
  fontSize: 10,
  letterSpacing: "0.1em",
  textTransform: "uppercase",
  cursor: "pointer",
  transition: "color 150ms ease, border-color 150ms ease",
  flexShrink: 0,
};

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
