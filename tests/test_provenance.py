"""
tests/test_provenance.py

Unit tests for the provenance tracking module.
"""

import pytest

from fq_ssiuu.provenance import (
    _sha256_file,
    complete_run,
    create_run,
    fail_run,
    generate_run_id,
    load_provenance,
    save_provenance,
)


class TestRunId:
    """Tests for run ID generation."""

    def test_unique_ids(self):
        """Two generated IDs should never collide."""
        ids = {generate_run_id() for _ in range(100)}
        assert len(ids) == 100

    def test_id_format(self):
        """Run ID should match YYYYMMDD-HHMMSS-hex8 format."""
        run_id = generate_run_id()
        parts = run_id.split("-")
        assert len(parts) == 3
        assert len(parts[0]) == 8  # YYYYMMDD
        assert len(parts[1]) == 6  # HHMMSS
        assert len(parts[2]) == 8  # UUID hex


class TestCreateRun:
    """Tests for create_run()."""

    def test_creates_running_context(self):
        """New run should have 'running' status."""
        ctx = create_run("c2_npo", seed=42)
        assert ctx.status == "running"
        assert ctx.condition_id == "c2_npo"
        assert ctx.seed == 42
        assert ctx.start_time != ""
        assert ctx.run_id != ""

    def test_config_snapshot(self):
        """Config should be stored in the context."""
        config = {"lr": 1e-4, "epochs": 5}
        ctx = create_run("c2_npo", seed=42, config=config)
        assert ctx.config_snapshot == config


class TestCompleteRun:
    """Tests for complete_run()."""

    def test_sets_completed_status(self, tmp_path):
        """Completing a run should set status and end_time."""
        ctx = create_run("c2_npo", seed=42)
        # Create a dummy artifact
        artifact = tmp_path / "output.json"
        artifact.write_text('{"loss": 0.5}')

        ctx = complete_run(ctx, str(tmp_path))
        assert ctx.status == "completed"
        assert ctx.end_time != ""
        assert "output.json" in ctx.artifact_checksums

    def test_cannot_complete_non_running(self, tmp_path):
        """Cannot complete a run that is not 'running'."""
        ctx = create_run("c2_npo", seed=42)
        ctx.status = "failed"
        with pytest.raises(ValueError):
            complete_run(ctx, str(tmp_path))


class TestFailRun:
    """Tests for fail_run()."""

    def test_records_failure(self):
        """Failing a run should set status and reason."""
        ctx = create_run("c2_npo", seed=42)
        ctx = fail_run(ctx, "OOM at step 50")
        assert ctx.status == "failed"
        assert ctx.failure_reason == "OOM at step 50"
        assert ctx.end_time != ""


class TestSaveLoad:
    """Tests for provenance persistence."""

    def test_roundtrip(self, tmp_path):
        """Save and load should produce identical records."""
        ctx = create_run("c2_npo", seed=42, config={"lr": 1e-4})
        path = str(tmp_path / "provenance.json")
        save_provenance(ctx, path)

        loaded = load_provenance(path)
        assert loaded.run_id == ctx.run_id
        assert loaded.condition_id == ctx.condition_id
        assert loaded.seed == ctx.seed

    def test_overwrite_prevention(self, tmp_path):
        """Should refuse to overwrite existing provenance."""
        ctx = create_run("c2_npo", seed=42)
        path = str(tmp_path / "provenance.json")
        save_provenance(ctx, path)

        with pytest.raises(FileExistsError):
            save_provenance(ctx, path)

    def test_allow_overwrite(self, tmp_path):
        """Should overwrite when explicitly allowed."""
        ctx = create_run("c2_npo", seed=42)
        path = str(tmp_path / "provenance.json")
        save_provenance(ctx, path)
        # Should not raise
        save_provenance(ctx, path, allow_overwrite=True)

    def test_no_hardcoded_metrics(self, tmp_path):
        """Provenance records should never contain experiment metrics."""
        ctx = create_run("c2_npo", seed=42)
        path = str(tmp_path / "provenance.json")
        save_provenance(ctx, path)

        with open(path) as f:
            content = f.read()
        # Provenance should not contain metric-like fields
        for field in ["forget_quality", "model_utility", "loss", "accuracy"]:
            assert field not in content


class TestChecksum:
    """Tests for SHA-256 checksum."""

    def test_deterministic(self, tmp_path):
        """Same content should produce same checksum."""
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("hello")
        f2.write_text("hello")
        assert _sha256_file(str(f1)) == _sha256_file(str(f2))

    def test_different_content(self, tmp_path):
        """Different content should produce different checksums."""
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("hello")
        f2.write_text("world")
        assert _sha256_file(str(f1)) != _sha256_file(str(f2))
