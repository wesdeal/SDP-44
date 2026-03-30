"""agents/ingestion_agent.py — Ingestion / Metadata Agent (Phase 4.1)

Reads the raw input file, computes deterministic column profiles, detects
temporal structure, and writes dataset_profile.json to the run artifact
directory.

Architecture authority: AGENT_ARCHITECTURE.md §4 Agent 1.
No legacy files are imported.
"""

import json
import os
import tempfile
import time
import warnings
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from core.manifest import read_manifest, update_stage

# ---------------------------------------------------------------------------
# Inferred-type enum values (must match AGENT_ARCHITECTURE.md §4 Agent 1)
# ---------------------------------------------------------------------------
_TYPE_NUMERIC = "numeric"
_TYPE_CATEGORICAL = "categorical"
_TYPE_DATETIME = "datetime"
_TYPE_BOOLEAN = "boolean"
_TYPE_UNKNOWN = "unknown"

# Minimum datetime parse success rate to classify an object column as datetime
_DATETIME_PARSE_THRESHOLD = 0.8

# LLM configuration
_OPENAI_MODEL = "gpt-4o-mini"
_LLM_MAX_RETRIES = 2
_LLM_RETRY_BASE_DELAY = 1.0  # seconds; doubles each retry (1s, 2s)
_LLM_MAX_TOKENS = 1024
_LLM_SAMPLE_ROWS = 50

_LLM_ANALYSIS_REQUIRED_KEYS = frozenset({
    "dataset_description",
    "suggested_target_variable",
    "target_confidence",
    "target_reasoning",
    "known_quality_issues",
    "has_trend",
    "has_seasonality",
    "is_multivariate",
    "data_source_hint",
    "ingestion_date",
})
_TARGET_CONFIDENCE_VALUES = frozenset({"high", "medium", "low"})


class IngestionAgent:
    """Agent 1: Ingestion / Metadata Agent."""

    def run(self, manifest_path: str) -> None:
        """Profile the raw input file and write dataset_profile.json.

        Sets stage status: running → completed | failed.

        Args:
            manifest_path: Path to the run's job_manifest.json.
        """
        update_stage(manifest_path, "ingestion", "running")

        try:
            manifest = read_manifest(manifest_path)
            artifact_path = self._artifact_path(manifest_path)
            profile = self._build_profile(manifest)
            _write_json_atomic(artifact_path, profile)
            update_stage(
                manifest_path,
                "ingestion",
                "completed",
                artifacts={"dataset_profile": artifact_path},
            )
        except Exception as exc:
            update_stage(
                manifest_path,
                "ingestion",
                "failed",
                error=str(exc),
            )
            raise

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _artifact_path(manifest_path: str) -> str:
        """Return the absolute path for dataset_profile.json.

        Derived from manifest_path — no hardcoded base directory.
        """
        run_dir = os.path.dirname(os.path.abspath(manifest_path))
        return os.path.join(run_dir, "artifacts", "dataset_profile.json")

    # ------------------------------------------------------------------
    # Profile construction
    # ------------------------------------------------------------------

    def _build_profile(self, manifest: dict) -> dict:
        input_block = manifest["input"]
        file_path = input_block["file_path"]
        file_format = input_block["file_format"]
        dataset_name = input_block["original_filename"]

        df = _load_file(file_path, file_format)

        if df.shape[0] < 2 or df.shape[1] < 1:
            raise ValueError(
                f"INGESTION_EMPTY_FILE: file must have at least 2 rows and "
                f"1 column, got {df.shape[0]} rows and {df.shape[1]} columns."
            )

        columns, detected_datetime_columns = _profile_columns(df)
        llm_analysis = _get_llm_analysis_with_fallback(
            df, dataset_name, file_format, columns, detected_datetime_columns
        )

        # Final safety net: ensure suggested_target_variable names a real column.
        col_names = {c["name"] for c in columns}
        if llm_analysis.get("suggested_target_variable") not in col_names:
            llm_analysis["suggested_target_variable"] = None
            llm_analysis["target_confidence"] = "low"
            llm_analysis["target_reasoning"] = (
                "Suggested target variable not found among real column names; "
                "downgraded to null."
            )

        return {
            "run_id": manifest["run_id"],
            "dataset_name": dataset_name,
            "file_format": file_format,
            "num_rows": int(df.shape[0]),
            "num_columns": int(df.shape[1]),
            "columns": columns,
            "has_header": True,
            "detected_datetime_columns": detected_datetime_columns,
            "llm_analysis": llm_analysis,
            "profiling_completed_at": datetime.now(timezone.utc).isoformat(),
        }


# ---------------------------------------------------------------------------
# Module-level helpers (pure functions, no side effects)
# ---------------------------------------------------------------------------

def _load_file(file_path: str, file_format: str) -> pd.DataFrame:
    """Load the raw file into a DataFrame based on declared format."""
    if file_format == "csv":
        return pd.read_csv(file_path)
    if file_format == "parquet":
        return pd.read_parquet(file_path)
    if file_format == "json":
        return pd.read_json(file_path)
    raise ValueError(f"Unsupported file_format: {file_format!r}")


def _profile_columns(df: pd.DataFrame) -> tuple:
    """Return (column profiles list, detected_datetime_columns list)."""
    columns = []
    detected_datetime_columns = []

    for col in df.columns:
        series = df[col]
        dtype_str = str(series.dtype)
        inferred_type = _infer_type(series)
        is_temporal = inferred_type == _TYPE_DATETIME
        if is_temporal:
            detected_datetime_columns.append(col)

        total = len(series)
        missing_count = int(series.isna().sum())
        missing_fraction = missing_count / total if total > 0 else 0.0
        unique_count = int(series.nunique(dropna=True))

        non_null = series.dropna()
        sample_values = [_json_safe(v) for v in non_null.head(5).tolist()]

        min_val = max_val = mean_val = std_val = None
        has_negative = False
        is_monotonically_increasing = None

        if inferred_type == _TYPE_NUMERIC:
            numeric = pd.to_numeric(series, errors="coerce")
            valid = numeric.dropna()
            if len(valid) > 0:
                min_val = _safe_float(valid.min())
                max_val = _safe_float(valid.max())
                mean_val = _safe_float(valid.mean())
                std_val = _safe_float(valid.std()) if len(valid) > 1 else None
                has_negative = bool((valid < 0).any())

        if is_temporal:
            dt_series = pd.to_datetime(series, errors="coerce")
            valid_dt = dt_series.dropna()
            if len(valid_dt) > 1:
                is_monotonically_increasing = bool(valid_dt.is_monotonic_increasing)

        columns.append({
            "name": col,
            "dtype_pandas": dtype_str,
            "inferred_type": inferred_type,
            "missing_count": missing_count,
            "missing_fraction": round(missing_fraction, 6),
            "unique_count": unique_count,
            "sample_values": sample_values,
            "min": min_val,
            "max": max_val,
            "mean": mean_val,
            "std": std_val,
            "has_negative_values": has_negative,
            "is_temporal": is_temporal,
            "is_monotonically_increasing": is_monotonically_increasing,
        })

    return columns, detected_datetime_columns


def _infer_type(series: pd.Series) -> str:
    """Infer column type as one of the architecture enum values."""
    dtype = series.dtype

    if pd.api.types.is_bool_dtype(dtype):
        return _TYPE_BOOLEAN

    if pd.api.types.is_datetime64_any_dtype(dtype):
        return _TYPE_DATETIME

    if pd.api.types.is_numeric_dtype(dtype):
        return _TYPE_NUMERIC

    if dtype == object or pd.api.types.is_string_dtype(dtype):
        # Probe for datetime: parse a sample and require high success rate
        non_null = series.dropna().head(20)
        if len(non_null) > 0:
            try:
                parsed = pd.to_datetime(non_null, errors="coerce")
                if parsed.notna().mean() >= _DATETIME_PARSE_THRESHOLD:
                    return _TYPE_DATETIME
            except Exception:
                pass
        return _TYPE_CATEGORICAL

    return _TYPE_UNKNOWN


def _heuristic_llm_analysis(columns: list) -> dict:
    """Return a schema-compliant llm_analysis block using heuristics only.

    No LLM integration exists in this repository. This fallback satisfies the
    architecture requirement that the pipeline continues in heuristic-only mode
    when the LLM is unavailable (AGENT_ARCHITECTURE.md §4 Agent 1 Failure Handling).
    """
    numeric_cols = [c for c in columns if c["inferred_type"] == _TYPE_NUMERIC]
    is_multivariate = len(numeric_cols) > 1

    # Heuristic target suggestion: last numeric column, else None
    suggested_target = numeric_cols[-1]["name"] if numeric_cols else None
    target_reasoning = (
        "Heuristic-only mode (no LLM): last numeric column selected as candidate target."
        if suggested_target
        else "Heuristic-only mode (no LLM): no numeric column found; target unknown."
    )

    quality_issues = [
        f"Column '{c['name']}' has {c['missing_fraction']:.1%} missing values (likely_unusable)."
        for c in columns
        if c["missing_fraction"] > 0.9
    ]

    return {
        "dataset_description": "Heuristic-only mode: LLM analysis unavailable.",
        "suggested_target_variable": suggested_target,
        "target_confidence": "low",
        "target_reasoning": target_reasoning,
        "known_quality_issues": quality_issues,
        "has_trend": False,
        "has_seasonality": False,
        "is_multivariate": is_multivariate,
        "data_source_hint": "unknown",
        "ingestion_date": date.today().isoformat(),
    }


def _build_llm_dataset_context(
    df: pd.DataFrame,
    dataset_name: str,
    file_format: str,
    columns: list,
    detected_datetime_columns: list,
) -> str:
    """Build a compact JSON string describing the dataset for the LLM prompt.

    Sends column profiles and the first ``_LLM_SAMPLE_ROWS`` rows so the model
    can infer semantics without receiving the full file.
    """
    col_summary = [
        {
            "name": c["name"],
            "inferred_type": c["inferred_type"],
            "missing_fraction": c["missing_fraction"],
            "unique_count": c["unique_count"],
            "sample_values": c["sample_values"],
        }
        for c in columns
    ]

    sample_rows = [
        {col: _json_safe(val) for col, val in row.items()}
        for _, row in df.head(_LLM_SAMPLE_ROWS).iterrows()
    ]

    return json.dumps(
        {
            "dataset_name": dataset_name,
            "file_format": file_format,
            "num_rows": int(df.shape[0]),
            "num_columns": int(df.shape[1]),
            "detected_datetime_columns": detected_datetime_columns,
            "column_profiles": col_summary,
            "sample_rows": sample_rows,
        },
        indent=2,
    )


def _call_openai_for_llm_analysis(context_payload: str, api_key: str) -> str | None:
    """Send the dataset context to gpt-4o-mini and return the raw response text.

    Retries up to ``_LLM_MAX_RETRIES`` times with exponential backoff on any
    exception (network error, timeout, rate-limit).  Returns ``None`` if all
    attempts fail or the ``openai`` package is not installed.
    """
    try:
        import openai  # noqa: PLC0415 — intentional late import
    except ImportError:
        warnings.warn(
            "openai package not installed — ingestion agent running in heuristic-only mode.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    system_prompt = (
        "You are a data science assistant performing semantic dataset analysis. "
        "Given a dataset profile and sample rows, return ONLY a valid JSON object "
        "— no prose, no markdown fences — that conforms exactly to this schema:\n"
        "{\n"
        '  "dataset_description": "<string: what this dataset appears to represent>",\n'
        '  "suggested_target_variable": "<string column name or null>",\n'
        '  "target_confidence": "<high | medium | low>",\n'
        '  "target_reasoning": "<string: why you chose that target>",\n'
        '  "known_quality_issues": ["<string>", ...],\n'
        '  "has_trend": <true | false>,\n'
        '  "has_seasonality": <true | false>,\n'
        '  "is_multivariate": <true | false>,\n'
        '  "data_source_hint": "<string: likely domain or data source>",\n'
        '  "ingestion_date": "<YYYY-MM-DD>"\n'
        "}\n"
        "Rules:\n"
        "- suggested_target_variable must be the exact column name from the profile, or null.\n"
        "- target_confidence must be exactly 'high', 'medium', or 'low'.\n"
        "- has_trend and has_seasonality apply only to time-series datasets; use false otherwise.\n"
        "- is_multivariate is true when there are multiple numeric feature columns.\n"
        "- Output the JSON object only — no surrounding text."
    )

    user_prompt = (
        "Analyse the following dataset and return the JSON object described above:\n\n"
        + context_payload
    )

    client = openai.OpenAI(api_key=api_key)

    last_exc: Exception | None = None
    for attempt in range(_LLM_MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=_OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=_LLM_MAX_TOKENS,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < _LLM_MAX_RETRIES:
                delay = _LLM_RETRY_BASE_DELAY * (2 ** attempt)
                time.sleep(delay)

    warnings.warn(
        f"OpenAI API call failed after {_LLM_MAX_RETRIES + 1} attempt(s): {last_exc}",
        RuntimeWarning,
        stacklevel=2,
    )
    return None


def _parse_and_validate_llm_analysis(raw_text: str, col_names: set) -> dict | None:
    """Parse raw LLM text into a validated llm_analysis dict.

    Steps:
    1. Strip markdown code fences if present.
    2. Parse JSON.
    3. Check all required keys are present.
    4. Coerce and sanitize field values.
    5. Validate ``suggested_target_variable`` against real column names.

    Returns the sanitized dict, or ``None`` if parsing or key validation fails.
    """
    if not raw_text:
        return None

    text = raw_text.strip()
    if text.startswith("```"):
        lines = [ln for ln in text.splitlines() if not ln.startswith("```")]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(data, dict):
        return None

    missing = _LLM_ANALYSIS_REQUIRED_KEYS - data.keys()
    if missing:
        warnings.warn(
            f"LLM response missing required keys: {missing} — discarding.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    # Coerce target_confidence to a valid enum value.
    if data["target_confidence"] not in _TARGET_CONFIDENCE_VALUES:
        data["target_confidence"] = "low"

    # Validate suggested_target_variable against real column names.
    suggested = data.get("suggested_target_variable")
    if suggested is not None and suggested not in col_names:
        data["suggested_target_variable"] = None
        data["target_confidence"] = "low"
        data["target_reasoning"] = (
            f"LLM suggested {suggested!r} but that column does not exist in "
            "the dataset. Confidence downgraded to low."
        )

    # Coerce boolean fields.
    for bool_key in ("has_trend", "has_seasonality", "is_multivariate"):
        if not isinstance(data[bool_key], bool):
            data[bool_key] = bool(data[bool_key])

    # Coerce known_quality_issues to a list of strings.
    if not isinstance(data["known_quality_issues"], list):
        data["known_quality_issues"] = []
    else:
        data["known_quality_issues"] = [
            str(item) for item in data["known_quality_issues"]
            if isinstance(item, (str, int, float))
        ]

    # Coerce remaining string fields.
    for str_key in (
        "dataset_description", "target_reasoning", "data_source_hint", "ingestion_date"
    ):
        if not isinstance(data[str_key], str):
            data[str_key] = str(data[str_key]) if data[str_key] is not None else ""

    return data


def _resolve_api_key() -> str | None:
    """Return the OpenAI API key, checking the environment then the project .env file.

    Priority:
    1. ``OPENAI_API_KEY`` already present in the process environment.
    2. ``OPENAI_API_KEY`` defined in ``<project_root>/.env``
       (loaded non-destructively via python-dotenv).
    """
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key

    # Locate .env relative to this file: agents/ → project root
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.isfile(env_path):
        try:
            from dotenv import dotenv_values  # noqa: PLC0415
            values = dotenv_values(env_path)
            key = values.get("OPENAI_API_KEY")
            if key:
                return key
        except Exception:
            pass

    return None


def _get_llm_analysis_with_fallback(
    df: pd.DataFrame,
    dataset_name: str,
    file_format: str,
    columns: list,
    detected_datetime_columns: list,
) -> dict:
    """Return the llm_analysis block, using OpenAI when available.

    Execution order:
    1. If ``OPENAI_API_KEY`` is absent (env + .env) → heuristic fallback.
    2. Build prompt context and call OpenAI with retry.
    3. Parse and validate the response.
    4. If any step fails → heuristic fallback.

    The heuristic fallback guarantees the pipeline always continues, as required
    by AGENT_ARCHITECTURE.md §4 Agent 1 Failure Handling.
    """
    api_key = _resolve_api_key()
    if not api_key:
        warnings.warn(
            "OPENAI_API_KEY is not set — ingestion agent running in heuristic-only mode.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _heuristic_llm_analysis(columns)

    col_names = {c["name"] for c in columns}

    context_payload = _build_llm_dataset_context(
        df, dataset_name, file_format, columns, detected_datetime_columns
    )

    raw_text = _call_openai_for_llm_analysis(context_payload, api_key)
    if raw_text is None:
        warnings.warn(
            "OpenAI returned no usable response — falling back to heuristic analysis.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _heuristic_llm_analysis(columns)

    validated = _parse_and_validate_llm_analysis(raw_text, col_names)
    if validated is None:
        warnings.warn(
            "LLM response failed schema validation — falling back to heuristic analysis.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _heuristic_llm_analysis(columns)

    return validated


def _safe_float(val) -> float | None:
    """Convert to float, returning None for NaN/Inf/None."""
    try:
        f = float(val)
        return None if (np.isnan(f) or np.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _json_safe(val):
    """Convert a scalar to a JSON-serializable Python primitive."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float, str)) or val is None:
        return val
    if isinstance(val, np.bool_):
        return bool(val)
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        f = float(val)
        return None if (np.isnan(f) or np.isinf(f)) else f
    return str(val)


def _write_json_atomic(path: str, data: dict) -> None:
    """Write *data* as JSON to *path* atomically (temp-file + rename).

    Creates parent directories as needed.
    """
    dir_name = os.path.dirname(path)
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
