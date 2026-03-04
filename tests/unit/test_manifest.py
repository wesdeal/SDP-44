"""tests/unit/test_manifest.py — Unit tests for core/manifest.py.

Covers:
- Normal inputs → expected outputs
- Invalid inputs → expected exceptions
- Boundary conditions (missing file, unsupported extension, etc.)
- Stage isolation: update_stage must not touch other stages
- Determinism: successive 
"""

import json
import os
import uuid

import pytest

from Pipeline.core.manifest import initialize_manifest, read_manifest, update_stage
# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runs_dir(tmp_path):
    """Isolated runs directory under pytest's tmp_path."""
    d = tmp_path / "runs"
    d.mkdir()
    return str(d)


@pytest.fixture
def sample_csv(tmp_path):
    """Minimal CSV file."""
    p = tmp_path / "data.csv"
    p.write_text("col_a,col_b\n1,2\n3,4\n")
    return str(p)


@pytest.fixture
def base_config(runs_dir):
    return {
        "runs_dir": runs_dir,
        "tune_hyperparameters": True,
        "n_optuna_trials": 20,
        "random_seed": 42,
        "cpu_only": True,
    }


@pytest.fixture
def initialized(sample_csv, base_config, runs_dir):
    """Return (manifest_dict, manifest_path) for a freshly initialized run."""
    manifest = initialize_manifest(sample_csv, base_config)
    path = os.path.join(runs_dir, manifest["run_id"], "job_manifest.json")
    return manifest, path


# ---------------------------------------------------------------------------
# initialize_manifest — normal behaviour
# ---------------------------------------------------------------------------


class TestInitializeManifest:
    def test_returns_dict(self, sample_csv, base_config):
        result = initialize_manifest(sample_csv, base_config)
        assert isinstance(result, dict)

    def test_top_level_keys_present(self, sample_csv, base_config):
        m = initialize_manifest(sample_csv, base_config)
        for key in ("run_id", "created_at", "updated_at", "status", "input", "stages", "config"):
            assert key in m, f"Missing top-level key: {key!r}"

    def test_run_id_is_uuid4(self, sample_csv, base_config):
        m = initialize_manifest(sample_csv, base_config)
        parsed = uuid.UUID(m["run_id"])
        assert parsed.version == 4

    def test_unique_run_ids_across_calls(self, sample_csv, base_config):
        m1 = initialize_manifest(sample_csv, base_config)
        m2 = initialize_manifest(sample_csv, base_config)
        assert m1["run_id"] != m2["run_id"]

    def test_initial_status_is_pending(self, sample_csv, base_config):
        m = initialize_manifest(sample_csv, base_config)
        assert m["status"] == "pending"

    def test_created_at_is_valid_iso8601(self, sample_csv, base_config):
        from datetime import datetime
        m = initialize_manifest(sample_csv, base_config)
        # Should not raise
        datetime.fromisoformat(m["created_at"])

    def test_updated_at_is_valid_iso8601(self, sample_csv, base_config):
        from datetime import datetime
        m = initialize_manifest(sample_csv, base_config)
        datetime.fromisoformat(m["updated_at"])

    def test_created_at_equals_updated_at_on_init(self, sample_csv, base_config):
        m = initialize_manifest(sample_csv, base_config)
        assert m["created_at"] == m["updated_at"]

    # --- input block ---

    def test_input_file_path_is_absolute(self, sample_csv, base_config):
        m = initialize_manifest(sample_csv, base_config)
        assert os.path.isabs(m["input"]["file_path"])

    def test_input_file_format_csv(self, sample_csv, base_config):
        m = initialize_manifest(sample_csv, base_config)
        assert m["input"]["file_format"] == "csv"

    def test_input_original_filename(self, sample_csv, base_config):
        m = initialize_manifest(sample_csv, base_config)
        assert m["input"]["original_filename"] == "data.csv"

    def test_input_file_format_parquet(self, tmp_path, base_config):
        p = tmp_path / "dataset.parquet"
        p.write_bytes(b"")
        m = initialize_manifest(str(p), base_config)
        assert m["input"]["file_format"] == "parquet"

    def test_input_file_format_json(self, tmp_path, base_config):
        p = tmp_path / "dataset.json"
        p.write_text("{}")
        m = initialize_manifest(str(p), base_config)
        assert m["input"]["file_format"] == "json"

    # --- stages ---

    def test_all_eight_stages_present(self, sample_csv, base_config):
        m = initialize_manifest(sample_csv, base_config)
        expected = {
            "ingestion",
            "problem_classification",
            "preprocessing_planning",
            "evaluation_protocol",
            "model_selection",
            "training",
            "evaluation",
            "artifact_assembly",
        }
        assert set(m["stages"].keys()) == expected

    def test_all_stages_start_pending(self, sample_csv, base_config):
        m = initialize_manifest(sample_csv, base_config)
        for name, stage in m["stages"].items():
            assert stage["status"] == "pending", f"Stage {name!r} not pending"

    def test_all_stages_started_at_null(self, sample_csv, base_config):
        m = initialize_manifest(sample_csv, base_config)
        for name, stage in m["stages"].items():
            assert stage["started_at"] is None, f"Stage {name!r} started_at not null"

    def test_all_stages_completed_at_null(self, sample_csv, base_config):
        m = initialize_manifest(sample_csv, base_config)
        for name, stage in m["stages"].items():
            assert stage["completed_at"] is None, f"Stage {name!r} completed_at not null"

    def test_all_stages_artifacts_empty(self, sample_csv, base_config):
        m = initialize_manifest(sample_csv, base_config)
        for name, stage in m["stages"].items():
            assert stage["artifacts"] == {}, f"Stage {name!r} artifacts not empty"

    def test_all_stages_error_null(self, sample_csv, base_config):
        m = initialize_manifest(sample_csv, base_config)
        for name, stage in m["stages"].items():
            assert stage["error"] is None, f"Stage {name!r} error not null"

    # --- config ---

    def test_config_stored_verbatim(self, sample_csv, base_config):
        m = initialize_manifest(sample_csv, base_config)
        assert m["config"] == base_config

    # --- disk persistence ---

    def test_manifest_file_created_on_disk(self, sample_csv, base_config, runs_dir):
        m = initialize_manifest(sample_csv, base_config)
        expected = os.path.join(runs_dir, m["run_id"], "job_manifest.json")
        assert os.path.isfile(expected)

    def test_on_disk_content_matches_returned_dict(self, sample_csv, base_config, runs_dir):
        m = initialize_manifest(sample_csv, base_config)
        path = os.path.join(runs_dir, m["run_id"], "job_manifest.json")
        with open(path, encoding="utf-8") as f:
            on_disk = json.load(f)
        assert on_disk == m

    # --- invalid inputs ---

    def test_unsupported_extension_raises_value_error(self, tmp_path, base_config):
        p = tmp_path / "data.xlsx"
        p.write_bytes(b"")
        with pytest.raises(ValueError, match="Unsupported file format"):
            initialize_manifest(str(p), base_config)

    def test_no_extension_raises_value_error(self, tmp_path, base_config):
        p = tmp_path / "datafile"
        p.write_bytes(b"")
        with pytest.raises(ValueError, match="Unsupported file format"):
            initialize_manifest(str(p), base_config)


# ---------------------------------------------------------------------------
# update_stage — normal behaviour
# ---------------------------------------------------------------------------


class TestUpdateStage:
    def test_running_sets_started_at(self, initialized):
        _, path = initialized
        update_stage(path, "ingestion", "running")
        m = read_manifest(path)
        assert m["stages"]["ingestion"]["status"] == "running"
        assert m["stages"]["ingestion"]["started_at"] is not None

    def test_running_does_not_set_completed_at(self, initialized):
        _, path = initialized
        update_stage(path, "ingestion", "running")
        m = read_manifest(path)
        assert m["stages"]["ingestion"]["completed_at"] is None

    def test_completed_sets_completed_at(self, initialized):
        _, path = initialized
        update_stage(path, "ingestion", "completed")
        m = read_manifest(path)
        assert m["stages"]["ingestion"]["status"] == "completed"
        assert m["stages"]["ingestion"]["completed_at"] is not None

    def test_failed_sets_completed_at(self, initialized):
        _, path = initialized
        update_stage(path, "ingestion", "failed")
        m = read_manifest(path)
        assert m["stages"]["ingestion"]["status"] == "failed"
        assert m["stages"]["ingestion"]["completed_at"] is not None

    def test_partial_failure_sets_completed_at(self, initialized):
        _, path = initialized
        update_stage(path, "training", "partial_failure")
        m = read_manifest(path)
        assert m["stages"]["training"]["status"] == "partial_failure"
        assert m["stages"]["training"]["completed_at"] is not None

    def test_error_message_stored(self, initialized):
        _, path = initialized
        update_stage(path, "ingestion", "failed", error="disk full")
        m = read_manifest(path)
        assert m["stages"]["ingestion"]["error"] == "disk full"

    def test_artifacts_dict_stored(self, initialized):
        _, path = initialized
        arts = {"dataset_profile": "runs/abc/artifacts/dataset_profile.json"}
        update_stage(path, "ingestion", "completed", artifacts=arts)
        m = read_manifest(path)
        assert m["stages"]["ingestion"]["artifacts"] == arts

    def test_none_artifacts_leaves_existing_untouched(self, initialized):
        _, path = initialized
        arts = {"dataset_profile": "runs/abc/artifacts/dataset_profile.json"}
        update_stage(path, "ingestion", "running", artifacts=arts)
        # Second call without artifacts kwarg
        update_stage(path, "ingestion", "completed")
        m = read_manifest(path)
        assert m["stages"]["ingestion"]["artifacts"] == arts

    def test_only_named_stage_modified(self, initialized):
        m_before, path = initialized
        update_stage(path, "ingestion", "running")
        m_after = read_manifest(path)
        for name in m_before["stages"]:
            if name == "ingestion":
                continue
            assert m_before["stages"][name] == m_after["stages"][name], (
                f"Stage {name!r} was unexpectedly modified"
            )

    def test_root_updated_at_refreshed(self, initialized):
        from datetime import datetime
        m_before, path = initialized
        update_stage(path, "ingestion", "running")
        m_after = read_manifest(path)
        # Must be a valid ISO-8601 timestamp
        dt_after = datetime.fromisoformat(m_after["updated_at"])
        dt_before = datetime.fromisoformat(m_before["updated_at"])
        assert dt_after >= dt_before

    def test_root_created_at_not_modified(self, initialized):
        m_before, path = initialized
        update_stage(path, "ingestion", "running")
        m_after = read_manifest(path)
        assert m_after["created_at"] == m_before["created_at"]

    def test_sequential_transitions(self, initialized):
        """running → completed is a normal agent lifecycle."""
        _, path = initialized
        update_stage(path, "ingestion", "running")
        update_stage(path, "ingestion", "completed", artifacts={"x": "y"})
        m = read_manifest(path)
        stage = m["stages"]["ingestion"]
        assert stage["status"] == "completed"
        assert stage["started_at"] is not None
        assert stage["completed_at"] is not None
        assert stage["artifacts"] == {"x": "y"}

    def test_all_eight_stages_individually_updatable(self, initialized):
        stage_names = [
            "ingestion", "problem_classification", "preprocessing_planning",
            "evaluation_protocol", "model_selection", "training",
            "evaluation", "artifact_assembly",
        ]
        _, path = initialized
        for name in stage_names:
            update_stage(path, name, "running")
            m = read_manifest(path)
            assert m["stages"][name]["status"] == "running"

    # --- invalid inputs ---

    def test_unknown_stage_raises_key_error(self, initialized):
        _, path = initialized
        with pytest.raises(KeyError):
            update_stage(path, "nonexistent_stage", "running")

    def test_invalid_status_raises_value_error(self, initialized):
        _, path = initialized
        with pytest.raises(ValueError, match="Invalid status"):
            update_stage(path, "ingestion", "in-progress")

    def test_missing_manifest_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            update_stage(str(tmp_path / "ghost.json"), "ingestion", "running")


# ---------------------------------------------------------------------------
# read_manifest — normal behaviour
# ---------------------------------------------------------------------------


class TestReadManifest:
    def test_returns_dict(self, initialized):
        _, path = initialized
        result = read_manifest(path)
        assert isinstance(result, dict)

    def test_content_matches_initialized_manifest(self, initialized):
        m, path = initialized
        result = read_manifest(path)
        assert result == m

    def test_deterministic_successive_reads(self, initialized):
        _, path = initialized
        r1 = read_manifest(path)
        r2 = read_manifest(path)
        assert r1 == r2

    def test_reflects_update_stage_changes(self, initialized):
        _, path = initialized
        update_stage(path, "ingestion", "running")
        m = read_manifest(path)
        assert m["stages"]["ingestion"]["status"] == "running"

    # --- invalid inputs ---

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_manifest(str(tmp_path / "does_not_exist.json"))
