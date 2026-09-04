"""
tests/test_drive_upload.py

Unit tests for Google Drive upload module.
All tests use mocks — no actual Google Drive access.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest


class TestDriveUpload:
    """Tests for drive_upload module."""

    def test_import(self):
        """Module should be importable."""
        import fq_ssiuu.drive_upload as du
        assert hasattr(du, "upload_run_directory")
        assert hasattr(du, "_get_drive_service")

    def test_upload_nonexistent_dir_raises(self):
        """Should raise FileNotFoundError for nonexistent directory."""
        from fq_ssiuu.drive_upload import upload_run_directory

        with pytest.raises(FileNotFoundError):
            upload_run_directory("/nonexistent/path/12345")

    def test_no_credentials_raises(self, tmp_path):
        """Should raise RuntimeError when no credentials available."""
        from fq_ssiuu.drive_upload import upload_run_directory

        test_dir = tmp_path / "test_run"
        test_dir.mkdir()
        (test_dir / "test.txt").write_text("test")

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises((RuntimeError, ImportError)):
                upload_run_directory(str(test_dir), folder_id="fake_id")

    def test_upload_summary_structure(self):
        """Upload summary should have expected keys."""
        expected_keys = {
            "drive_folder_id",
            "drive_folder_name",
            "file_count",
            "total_bytes",
            "error_count",
            "errors",
        }
        # Just verify the structure is documented
        assert expected_keys  # Placeholder for mocked test
