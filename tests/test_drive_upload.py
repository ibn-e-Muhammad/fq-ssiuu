"""
tests/test_drive_upload.py

Unit tests for Google Drive upload module.
All tests use mocks — no actual Google Drive access.
"""

import os
from unittest.mock import patch

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

    @pytest.mark.parametrize("fail_second", [False, True])
    def test_upload_summary_accounts_for_success_and_failure(self, tmp_path, fail_second):
        from fq_ssiuu import drive_upload as du

        first = tmp_path / "a.txt"
        second = tmp_path / "b.txt"
        first.write_bytes(b"abc")
        second.write_bytes(b"12345")
        outcomes = [{"id": "a", "size": "3"}, {"id": "b", "size": "5"}]
        if fail_second:
            outcomes[1] = RuntimeError("synthetic upload failure")
        with (
            patch.object(du, "_get_drive_service", return_value=object()) as service,
            patch.object(du, "_create_drive_folder", return_value="run-folder") as folder,
            patch.object(du, "_upload_file_with_retry", side_effect=outcomes) as upload,
        ):
            result = du.upload_run_directory(str(tmp_path), folder_id="parent", run_id="run-test")
            folder.assert_called_once_with(service.return_value, "run-test", "parent")
            assert [call.args[1] for call in upload.call_args_list] == [str(first), str(second)]
        assert result["drive_folder_id"] == "run-folder"
        assert result["drive_folder_name"] == "run-test"
        assert result["file_count"] == (1 if fail_second else 2)
        assert result["total_bytes"] == (3 if fail_second else 8)
        assert result["error_count"] == int(fail_second)
        assert len(result["errors"]) == int(fail_second)
        if fail_second:
            assert "synthetic upload failure" in result["errors"][0]
        assert first.read_bytes() == b"abc" and second.read_bytes() == b"12345"
