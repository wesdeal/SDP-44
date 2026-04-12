import { useRef, useState } from "react";
import styles from "./UploadDropzone.module.css";

const ACCEPTED = [".csv", ".parquet", ".json"];
const ACCEPT_ATTR = ACCEPTED.join(",");

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function validateFile(file) {
  if (!file) return null;
  const ext = "." + file.name.split(".").pop().toLowerCase();
  if (!ACCEPTED.includes(ext)) {
    return `Unsupported type "${ext}". Accepted: ${ACCEPTED.join(", ")}`;
  }
  return null;
}

/** Upload icon */
function UploadIcon() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" y1="3" x2="12" y2="15" />
    </svg>
  );
}

/** File icon */
function FileIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
      <polyline points="14 2 14 8 20 8" />
    </svg>
  );
}

/** Check icon */
function CheckIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

/** Spinner */
function Spinner() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="2" strokeLinecap="round"
      style={{ animation: "spin 0.8s linear infinite" }}>
      <path d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" opacity="0.2" />
      <path d="M12 3a9 9 0 019 9" />
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </svg>
  );
}

/**
 * UploadDropzone
 *
 * Props:
 *   file         — currently selected File object (or null)
 *   onFileSelect — callback(File | null)
 *   onStartRun   — called when user clicks "Start Run"
 *   isStarting   — bool, disables button during upload
 *   error        — string or null, backend error after attempting to start
 */
export default function UploadDropzone({ file, onFileSelect, onStartRun, isStarting, error }) {
  const inputRef = useRef(null);
  const [isOver, setIsOver] = useState(false);
  const [validationError, setValidationError] = useState(null);

  function handleDragOver(e) {
    e.preventDefault();
    setIsOver(true);
  }

  function handleDragLeave(e) {
    e.preventDefault();
    setIsOver(false);
  }

  function handleDrop(e) {
    e.preventDefault();
    setIsOver(false);
    const dropped = e.dataTransfer.files[0];
    if (!dropped) return;
    const err = validateFile(dropped);
    if (err) {
      setValidationError(err);
      return;
    }
    setValidationError(null);
    onFileSelect(dropped);
  }

  function handleInputChange(e) {
    const selected = e.target.files[0];
    if (!selected) return;
    const err = validateFile(selected);
    if (err) {
      setValidationError(err);
      e.target.value = "";
      return;
    }
    setValidationError(null);
    onFileSelect(selected);
  }

  function handleClearFile(e) {
    e.stopPropagation();
    onFileSelect(null);
    setValidationError(null);
    if (inputRef.current) inputRef.current.value = "";
  }

  const hasFile = !!file;
  const displayError = validationError || error;

  const zoneClass = [
    styles.dropzone,
    isOver   ? styles.dropzoneOver     : "",
    hasFile  ? styles.dropzoneHasFile  : "",
  ].filter(Boolean).join(" ");

  return (
    <>
      <div
        className={zoneClass}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => { if (!hasFile) inputRef.current?.click(); }}
        role="button"
        tabIndex={0}
        aria-label="Upload dataset file"
        onKeyDown={(e) => { if (e.key === "Enter" && !hasFile) inputRef.current?.click(); }}
      >
        {/* Hidden file input — triggered by onClick forwarding on the dropzone */}
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT_ATTR}
          className={styles.fileInput}
          onChange={handleInputChange}
          tabIndex={-1}
          aria-hidden
        />

        {/* Icon */}
        <div className={styles.iconWrap}>
          {hasFile ? <CheckIcon /> : <UploadIcon />}
        </div>

        {/* Text */}
        {!hasFile ? (
          <div className={styles.textGroup}>
            <span className={styles.primaryText}>
              {isOver ? "Drop to upload" : "Drag & drop or click to browse"}
            </span>
            <span className={styles.secondaryText}>Your dataset file</span>
            <span className={styles.acceptText}>CSV · PARQUET · JSON</span>
          </div>
        ) : (
          <div className={styles.filePreview}>
            <div className={styles.fileIcon}>
              <FileIcon />
            </div>
            <div className={styles.fileInfo}>
              <span className={styles.fileName}>{file.name}</span>
              <span className={styles.fileSize}>{formatBytes(file.size)}</span>
            </div>
            <button
              className={styles.fileClear}
              onClick={handleClearFile}
              aria-label="Remove file"
              title="Remove file"
            >
              ×
            </button>
          </div>
        )}
      </div>

      {/* Error */}
      {displayError && (
        <div className={styles.errorMsg}>
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor"
            strokeWidth="2" strokeLinecap="round">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
          {displayError}
        </div>
      )}

      {/* Start run button */}
      <button
        className={styles.startBtn}
        onClick={onStartRun}
        disabled={!hasFile || isStarting}
        aria-label="Start pipeline run"
      >
        {isStarting ? (
          <>
            <Spinner />
            Starting…
          </>
        ) : (
          <>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
              <polygon points="5,3 19,12 5,21" />
            </svg>
            Start Run
          </>
        )}
      </button>
    </>
  );
}
