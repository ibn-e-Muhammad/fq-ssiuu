"""
tests/test_environment.py

Unit tests for environment preflight checks and manifest capture.
"""

import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

from fq_ssiuu.environment import (
    PreflightResult,
    _check_cuda,
    _check_disk_space,
    _check_python_version,
    capture_manifest,
    preflight_check,
    save_manifest,
)


class TestPreflight:
    """Tests for preflight_check()."""

    def test_python_version_passes(self):
        """Current Python should pass the version check."""
        result = _check_python_version()
        # We're running on Python 3.10+, so this should pass
        assert result.passed is True

    def test_cuda_uses_documented_total_memory_property(self):
        cuda = SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda i: "Tesla T4",
            get_device_properties=lambda i: SimpleNamespace(total_memory=15_636_037_632),
        )
        with patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=cuda)}):
            result = _check_cuda()
        assert result.passed is True
        assert "Tesla T4" in result.message

    def test_disk_space_check(self):
        """Disk space check should return a valid result."""
        result = _check_disk_space(".")
        assert isinstance(result.passed, bool)
        assert result.check_name == "Disk space"

    def test_preflight_returns_tuple(self):
        """preflight_check should return (bool, list)."""
        all_passed, results = preflight_check()
        assert isinstance(all_passed, bool)
        assert isinstance(results, list)
        assert all(isinstance(r, PreflightResult) for r in results)


class TestManifest:
    """Tests for capture_manifest()."""

    def test_manifest_has_required_fields(self):
        """Manifest should have all required fields populated."""
        manifest = capture_manifest()
        assert manifest.timestamp != ""
        assert manifest.python_version != ""
        assert manifest.platform_info != ""
        assert isinstance(manifest.packages, dict)

    def test_manifest_never_contains_secrets(self):
        """Manifest should list env var NAMES, never VALUES."""
        with patch.dict("os.environ", {"HF_TOKEN": "secret_value_123"}):
            manifest = capture_manifest()
            manifest_str = json.dumps(manifest.__dict__)
            assert "secret_value_123" not in manifest_str
            # But the name should be in env_vars_set
            assert "HF_TOKEN" in manifest.env_vars_set

    def test_save_manifest(self, tmp_path):
        """save_manifest should write valid JSON."""
        manifest = capture_manifest()
        output = str(tmp_path / "manifest.json")
        save_manifest(manifest, output)

        with open(output) as f:
            data = json.load(f)
        assert "timestamp" in data
        assert "python_version" in data
        assert "packages" in data
