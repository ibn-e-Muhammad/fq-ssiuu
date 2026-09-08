"""
tests/test_thesis_update.py

Unit tests for thesis update packet generator.
"""

import json

import pytest

from fq_ssiuu.thesis_update import (
    generate_thesis_packet,
    load_thesis_packet,
    save_thesis_packet,
)


class TestGeneratePacket:
    """Tests for generate_thesis_packet()."""

    def test_required_fields(self):
        """Packet should have all required identification fields."""
        packet = generate_thesis_packet(
            run_id="20260903-120000-abcd1234",
            condition_id="c2_npo",
            git_commit="abc123def",
        )
        assert packet.run_id == "20260903-120000-abcd1234"
        assert packet.condition_id == "c2_npo"
        assert packet.git_commit == "abc123def"
        assert packet.timestamp != ""

    def test_optional_fields_default_empty(self):
        """Optional fields should default to empty collections."""
        packet = generate_thesis_packet(
            run_id="test",
            condition_id="c2_npo",
            git_commit="abc",
        )
        assert packet.affected_sections == []
        assert packet.metrics == {}
        assert packet.claims_supported == []
        assert packet.claims_unsupported == []
        assert packet.limitations == []

    def test_no_hardcoded_metrics(self):
        """Packet should not contain hardcoded metric values."""
        packet = generate_thesis_packet(
            run_id="test",
            condition_id="c2_npo",
            git_commit="abc",
            metrics={"forget_quality": 0.5, "model_utility": 0.8},
        )
        # Metrics should be exactly what was passed, nothing extra
        assert set(packet.metrics.keys()) == {"forget_quality", "model_utility"}


class TestSaveLoad:
    """Tests for packet persistence."""

    def test_roundtrip(self, tmp_path):
        """Save and load should produce identical packets."""
        packet = generate_thesis_packet(
            run_id="test-run",
            condition_id="c2_npo",
            git_commit="abc123",
            metrics={"fq": 0.5},
            claims_supported=["H1: QRA manifests"],
            limitations=["single seed"],
        )
        path = str(tmp_path / "packet.json")
        save_thesis_packet(packet, path)

        loaded = load_thesis_packet(path)
        assert loaded.run_id == packet.run_id
        assert loaded.metrics == packet.metrics
        assert loaded.claims_supported == packet.claims_supported

    def test_file_not_found(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_thesis_packet("/nonexistent/path.json")

    def test_valid_json_output(self, tmp_path):
        """Output should be valid JSON."""
        packet = generate_thesis_packet(
            run_id="test",
            condition_id="c2",
            git_commit="abc",
        )
        path = str(tmp_path / "packet.json")
        save_thesis_packet(packet, path)

        with open(path) as f:
            data = json.load(f)
        assert "run_id" in data
        assert "condition_id" in data
        assert "timestamp" in data
