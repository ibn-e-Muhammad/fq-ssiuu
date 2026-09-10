"""Validate source entry points without model execution or external services."""

import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_candidate_validation_entrypoint_runs_directly():
    result = subprocess.run(
        [sys.executable, "experiments/reproductions/NPO/validate_upstream.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--openunlearning" in result.stdout


def test_importing_colab_helper_has_no_setup_side_effects(capsys):
    before = list(sys.path)
    runpy.run_path(str(ROOT / "scripts/colab_mount.py"), run_name="import_probe")
    assert sys.path == before
    assert capsys.readouterr().out == ""


def test_historical_pilot_cli_stops_before_any_model_access():
    result = subprocess.run(
        [sys.executable, "experiments/reproductions/QRA/historical/pilot_p1.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Historical pilot disabled" in result.stderr
