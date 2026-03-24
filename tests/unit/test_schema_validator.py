import json
import pytest
from core.schema_validator import validate


def _write(tmp_path, name, data):
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return str(p)


# ── dataset_profile ──────────────────────────────────────────────────────────

def test_valid_dataset_profile(tmp_path):
    path = _write(tmp_path, "dp.json", {
        "run_id": "r1",
        "num_rows": 100,
        "num_columns": 5,
        "columns": ["a", "b"],
    })
    assert validate(path, "dataset_profile") is True


def test_invalid_dataset_profile_missing_field(tmp_path):
    path = _write(tmp_path, "dp.json", {
        "run_id": "r1",
        "num_rows": 100,
        # num_columns and columns missing
    })
    assert validate(path, "dataset_profile") is False


# ── task_spec ────────────────────────────────────────────────────────────────

def test_valid_task_spec(tmp_path):
    path = _write(tmp_path, "ts.json", {
        "run_id": "r1",
        "task_type": "regression",
        "target_col": "price",
    })
    assert validate(path, "task_spec") is True


def test_invalid_task_spec_missing_field(tmp_path):
    path = _write(tmp_path, "ts.json", {
        "run_id": "r1",
        # task_type and target_col missing
    })
    assert validate(path, "task_spec") is False


# ── unknown schema ────────────────────────────────────────────────────────────

def test_unknown_schema_raises(tmp_path):
    path = _write(tmp_path, "x.json", {})
    with pytest.raises(ValueError, match="Unknown schema_name"):
        validate(path, "nonexistent_schema")


# ── file not found ────────────────────────────────────────────────────────────

def test_file_not_found_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate(str(tmp_path / "missing.json"), "dataset_profile")
