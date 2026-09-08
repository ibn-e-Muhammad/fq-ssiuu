"""Behavioral tests of immutable output sealing and credential exclusion."""

import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from scripts.archive_run import digest, seal_run


def test_archive_preserves_raw_bytes_and_every_hash(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "stdout.log").write_bytes(b"failed run\nraw trailing space \n")
    manifest = seal_run(raw, tmp_path / "sealed", "fixture-run-1")
    assert manifest["artifacts"][0]["sha256"] == digest(raw / "stdout.log")
    with tarfile.open(tmp_path / "sealed/artifacts.tar") as archive:
        assert archive.extractfile("stdout.log").read() == (raw / "stdout.log").read_bytes()
    assert json.loads((tmp_path / "sealed/ARCHIVE_MANIFEST.json").read_text()) == manifest
    with pytest.raises(FileExistsError):
        seal_run(raw, tmp_path / "sealed", "fixture-run-1")


@pytest.mark.parametrize("unsafe", ["file", "symlink", "token"])
def test_unsafe_outputs_block_before_archive_creation(tmp_path, unsafe):
    raw = tmp_path / "raw"
    raw.mkdir()
    if unsafe == "file":
        (raw / ".env").write_text("fixture")
    elif unsafe == "symlink":
        (raw / "escape").symlink_to(tmp_path / "outside")
    else:
        # Construct a fake pattern across the scanner's read boundary.
        (raw / "output.log").write_bytes(b" " * (1024 * 1024 - 12) + b"hf_" + b"a" * 32)
    with pytest.raises(ValueError):
        seal_run(raw, tmp_path / "sealed", "fixture-run-2")
    assert not (tmp_path / "sealed").exists()


def test_test_package_survives_an_installed_package_with_same_name(tmp_path):
    foreign = tmp_path / "tests"
    foreign.mkdir()
    (foreign / "__init__.py").write_text("raise AssertionError('foreign tests imported')\n")
    root = Path(__file__).resolve().parents[1]
    code = "import sys;sys.path.append(sys.argv[1]);import tests.og001_fixtures;print('local fixtures imported')"
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)], cwd=root, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "local fixtures imported"
