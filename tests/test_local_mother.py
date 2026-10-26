"""Exercise local orchestration and failure preservation without model downloads."""

import importlib.util
import json
import os
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.reproductions.NPO import local_inputs

spec = importlib.util.spec_from_file_location(
    "local_mother", Path(__file__).resolve().parents[1] / "notebooks/Local/mother_runner.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize("failure", [False, True])
def test_whole_child_is_invoked_and_sealed_even_on_failure(tmp_path, monkeypatch, failure):
    repo = runner.ROOT
    # Temporary output root but actual recipe/requirements are copied as source fixtures.
    for relative in ["configs/runs/work_unit_d1_npo_v2.json", "requirements/local-windows.in"]:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((repo / relative).read_bytes())
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "source", lambda: {"snapshot_commit_sha": "a" * 40})
    monkeypatch.setattr(runner.platform, "system", lambda: "Windows")
    monkeypatch.setattr(runner, "prepare", lambda *a: {"records": []})
    monkeypatch.setattr(runner, "setup", lambda *a: sys.executable)
    calls = []

    def child(name, argv, raw, env):
        calls.append(argv)
        assert env["CUDA_VISIBLE_DEVICES"] == "2"
        output = raw / "NPO"
        output.mkdir()
        (output / "partial.log").write_text("evidence")
        if failure:
            raise RuntimeError("CUDA out of memory")
        (output / "RESULT.json").write_text(
            json.dumps({"status": "PASS_LOCAL_RUNTIME_CRITERIA_ARCHIVAL_PENDING"})
        )

    monkeypatch.setattr(runner, "command", child)
    args = SimpleNamespace(
        models=tmp_path / "models", data=tmp_path / "data", gpu=2, resume_checkpoint=None
    )
    assert runner.execute(args) == int(failure)
    assert len(calls) == 1 and "--local-models" in calls[0]
    run = next((tmp_path / "results/local").iterdir())
    result = json.loads((run / "raw/run_result.json").read_text())
    assert result["training_attempted"]
    with tarfile.open(run / "sealed/artifacts.tar") as archive:
        assert archive.extractfile("NPO/partial.log").read() == b"evidence"


def test_real_child_failure_retains_logs_and_returncode(tmp_path):
    with pytest.raises(RuntimeError, match="exit code 7"):
        runner.command(
            "failure",
            [sys.executable, "-c", "import sys;print('partial');sys.exit(7)"],
            tmp_path,
            os.environ.copy(),
        )
    assert "partial" in (tmp_path / "failure.stdout.log").read_text()
    assert json.loads((tmp_path / "failure.execution.json").read_text())["returncode"] == 7


def test_manual_input_hash_mismatch_and_missing_file(tmp_path):
    file = tmp_path / "input.json"
    file.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        local_inputs.verify(file, "0" * 64)
    with pytest.raises(FileNotFoundError):
        local_inputs.verify(tmp_path / "absent")


def test_setup_orders_cuda_before_deepspeed_and_records_environment(tmp_path, monkeypatch):
    calls = []
    target = tmp_path / "experiments/reproductions/NPO/runtime_sources_v1.json"
    target.parent.mkdir(parents=True)
    target.write_bytes(
        (runner.ROOT / "experiments/reproductions/NPO/runtime_sources_v1.json").read_bytes()
    )
    monkeypatch.setattr(runner, "command", lambda name, *a: calls.append(name))
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    runner.install_environment(tmp_path, {})
    assert calls.index("torch") < calls.index("deepspeed") < calls.index("project")


def test_local_memory_observation_has_no_historical_cap(monkeypatch):
    memory_path = Path(__file__).resolve().parents[1] / "experiments/reproductions/NPO/memory.py"
    memory_spec = importlib.util.spec_from_file_location("local_memory_test", memory_path)
    memory = importlib.util.module_from_spec(memory_spec)
    with monkeypatch.context() as patcher:
        patcher.setitem(sys.modules, "torch", SimpleNamespace())
        memory_spec.loader.exec_module(memory)
    monitor = memory.MemoryMonitor.__new__(memory.MemoryMonitor)
    monitor.errors = []
    monitor.limit = None
    sample = {
        k: 24_000_000_000
        for k in ("peak_allocated_bytes", "peak_reserved_bytes", "sampled_peak_device_used_bytes")
    }
    monitor.snapshot = lambda: sample
    assert monitor.observe() == sample
    monitor.limit = 16_000_000_000
    with pytest.raises(MemoryError):
        monitor.observe()
