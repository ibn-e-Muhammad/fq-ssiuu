"""Notebook lifecycle and live child diagnostics; no training or model downloads."""

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from notebooks.Local import live_output
from notebooks.Local import mother_runner as runner

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def local_run(tmp_path, monkeypatch):
    for relative in ("configs/runs/work_unit_d1_npo_v2.json", "requirements/local-windows.in"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / relative).read_bytes())
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "source", lambda: {"snapshot_commit_sha": "a" * 40})
    monkeypatch.setattr(runner.platform, "system", lambda: "Windows")
    monkeypatch.setattr(runner, "prepare", lambda *args: {"records": []})
    monkeypatch.setattr(runner, "setup", lambda *args: sys.executable)
    return runner.LocalRun(
        SimpleNamespace(
            models=tmp_path / "models", data=tmp_path / "data", gpu=0, resume_checkpoint=None
        )
    )


def test_cell_order_and_repeated_stage_are_rejected(local_run):
    with pytest.raises(RuntimeError, match="cannot run now"):
        local_run.train()
    local_run.initialize()
    with pytest.raises(RuntimeError, match="cannot run now"):
        local_run.initialize()
    assert local_run.next_stage == "inputs"
    assert local_run.finish() == 1
    assert local_run.result["status"] == "CANCELLED"


@pytest.mark.parametrize("error", [FileNotFoundError("missing manual input"), KeyboardInterrupt()])
def test_failure_and_interrupt_finalize_once(local_run, monkeypatch, error):
    local_run.initialize()

    def fail(*args):
        raise error

    monkeypatch.setattr(runner, "prepare", fail)
    with pytest.raises(type(error)):
        local_run.verify_inputs()
    assert local_run.finished
    assert local_run.result["phase"] == "inputs"
    assert not local_run.result["training_attempted"]
    assert local_run.finish() == 1
    manifest = json.loads((local_run.run / "sealed/ARCHIVE_MANIFEST.json").read_text())
    for item in manifest["artifacts"]:
        assert (
            hashlib.sha256((local_run.raw / item["path"]).read_bytes()).hexdigest()
            == item["sha256"]
        )


def test_archive_failure_keeps_raw_and_nonzero_exit(local_run, monkeypatch):
    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(runner, "seal_run", fail)
    assert local_run.finish() == 1
    assert (local_run.raw / "run_result.json").is_file()
    assert json.loads((local_run.run / "archive_status.json").read_text())["error"] == "disk full"


def test_live_stdout_stderr_before_child_exit(tmp_path, capsys):
    release = tmp_path / "release"
    errors = []
    code = (
        "import pathlib,sys,time; print('LIVE stdout ✓',flush=True); "
        "print('LIVE stderr',file=sys.stderr,flush=True); "
        f"p=pathlib.Path({str(release)!r}); "
        "\nwhile not p.exists(): time.sleep(.02)"
    )

    def run_child():
        try:
            runner.command("live", [sys.executable, "-c", code], tmp_path, os.environ.copy())
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=run_child)
    worker.start()
    text = ""
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            text += capsys.readouterr().out
            if "LIVE stdout ✓\nLIVE stderr" in text:
                break
            time.sleep(0.05)
        assert "LIVE stdout ✓\nLIVE stderr" in text
        assert worker.is_alive(), "Child must still be running when output is displayed"
        active = json.loads((tmp_path / "active_command.json").read_text())
        assert active["status"] == "RUNNING" and active["pid"] > 0
    finally:
        release.touch()
        worker.join(timeout=10)
    assert not worker.is_alive() and not errors
    assert (tmp_path / "live.stdout.log").read_text() == "LIVE stdout ✓\n"
    assert (tmp_path / "live.stderr.log").read_text() == "LIVE stderr\n"


def test_startup_failure_is_recorded(tmp_path):
    with pytest.raises(FileNotFoundError):
        runner.command("missing", [str(tmp_path / "no-python")], tmp_path, os.environ.copy())
    record = json.loads((tmp_path / "missing.execution.json").read_text())
    assert record["error_type"] == "FileNotFoundError"
    assert json.loads((tmp_path / "active_command.json").read_text())["status"] == "FAILED"


def test_interrupt_stops_child_before_return(tmp_path, monkeypatch):
    pid_file = tmp_path / "pid"
    original_sleep = time.sleep

    def interrupt(delay):
        deadline = time.monotonic() + 10
        while not pid_file.exists() and time.monotonic() < deadline:
            original_sleep(0.02)
        monkeypatch.setattr(live_output.time, "sleep", original_sleep)
        raise KeyboardInterrupt

    monkeypatch.setattr(live_output.time, "sleep", interrupt)
    with pytest.raises(KeyboardInterrupt):
        runner.command(
            "cancel",
            [
                sys.executable,
                "-c",
                "import os,pathlib,time; "
                f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(120)",
            ],
            tmp_path,
            os.environ.copy(),
        )
    record = json.loads((tmp_path / "cancel.execution.json").read_text())
    assert record["returncode"] is not None
    assert record["error_type"] == "KeyboardInterrupt"


def test_evidence_display_preserves_runtime_files(tmp_path, capsys):
    training = tmp_path / "NPO/training"
    training.mkdir(parents=True)
    forward = training / "forward_observations.jsonl"
    original = b'{"microbatch":0,"phase":"FORWARD_COMPLETE_BEFORE_BACKWARD"}\n'
    forward.write_bytes(original)
    trace = tmp_path / "NPO/traceback.log"
    trace.write_text('File "training_loop.py", line 94\nCUDA out of memory\n')
    watcher = live_output.EvidenceDisplay(tmp_path)
    watcher.poll()
    output = capsys.readouterr().out
    assert "training_loop.py" in output and "FORWARD_COMPLETE_BEFORE_BACKWARD" in output
    watcher.poll()
    assert capsys.readouterr().out == ""
    assert forward.read_bytes() == original


@pytest.mark.parametrize("failure,setup_only", [(False, False), (True, False), (False, True)])
def test_notebook_executes_in_process_kernel_with_fixture_child(tmp_path, failure, setup_only):
    nbformat = pytest.importorskip("nbformat")
    pytest.importorskip("ipykernel")

    fixture = tmp_path / "fixture repo with spaces"
    for relative in (
        ".gitignore",
        "LOCAL_MOTHER_RUNNER.ipynb",
        "notebooks/Local/mother_runner.py",
        "configs/runs/work_unit_d1_npo_v2.json",
        "requirements/local-windows.in",
    ):
        target = fixture / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    child = fixture / "experiments/reproductions/NPO/train_runtime.py"
    child.parent.mkdir(parents=True)
    child.write_text(
        "import json,pathlib,sys\n"
        "output=pathlib.Path(sys.argv[sys.argv.index('--output')+1]); output.mkdir()\n"
        "print('fixture NPO stdout',flush=True)\n"
        "print('fixture NPO stderr',file=sys.stderr,flush=True)\n"
        + (
            "(output/'traceback.log').write_text('fixture training_loop.py:94 CUDA out of memory')\n"
            "sys.exit(7)\n"
            if failure
            else "(output/'RESULT.json').write_text(json.dumps({'status':"
            "'PASS_LOCAL_RUNTIME_CRITERIA_ARCHIVAL_PENDING'}))\n"
        )
    )
    for argv in (
        ["git", "init", "-q"],
        ["git", "add", "."],
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
    ):
        subprocess.run(argv, cwd=fixture, check=True, capture_output=True)
    notebook = nbformat.read(ROOT / "LOCAL_MOTHER_RUNNER.ipynb", as_version=4)
    nbformat.validate(notebook)
    assert all(
        not c.get("outputs") and c.get("execution_count") is None
        for c in notebook.cells
        if c.cell_type == "code"
    )
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, "LOCAL_MOTHER_RUNNER.ipynb", "exec")
            cell.source = cell.source.replace(
                "REPO_ROOT = None", f"REPO_ROOT = Path({str(fixture)!r})"
            )
            if setup_only:
                cell.source = cell.source.replace("SETUP_ONLY = False", "SETUP_ONLY = True")
    prelude = nbformat.v4.new_code_cell(f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
from pathlib import Path
from notebooks.Local import mother_runner as fixture_runner
fixture_runner.ROOT = Path({str(fixture)!r})
fixture_runner.platform.system = lambda: "Windows"
fixture_runner.prepare = lambda *args: {{"records": []}}
def fixture_setup(raw, env, mode="auto"):
    fixture_runner.command("fixture_setup", [sys.executable, "-c", "print('fixture installation')"], raw, env)
    return Path(sys.executable)
fixture_runner.setup = fixture_setup
""")
    notebook.cells.insert(0, prelude)
    # A fresh process with an actual IPython/Jupyter in-process kernel needs no
    # network sockets, and prevents fixture patches leaking into other tests.
    input_path, output_path = tmp_path / "input.ipynb", tmp_path / "executed.ipynb"
    nbformat.write(notebook, input_path)
    driver = """
import sys
import nbformat
from ipykernel.inprocess.manager import InProcessKernelManager
nb = nbformat.read(sys.argv[1], as_version=4)
manager = InProcessKernelManager()
manager.start_kernel()
client = manager.client()
client.start_channels()
try:
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        client.execute(cell.source)
        reply = client.get_shell_msg(timeout=30)
        cell.execution_count = reply["content"]["execution_count"]
        cell.outputs = []
        while client.iopub_channel.msg_ready():
            message = client.get_iopub_msg(timeout=5)
            if message["msg_type"] in ("stream", "error", "display_data", "execute_result"):
                cell.outputs.append(nbformat.v4.output_from_msg(message))
    nbformat.write(nb, sys.argv[2])
finally:
    client.stop_channels()
    manager.shutdown_kernel()
"""
    subprocess.run(
        [sys.executable, "-c", driver, str(input_path), str(output_path)],
        cwd=fixture,
        check=True,
        capture_output=True,
        timeout=60,
    )
    executed = nbformat.read(output_path, as_version=4)
    outputs = [o for c in executed.cells if c.cell_type == "code" for o in c.get("outputs", [])]
    errors = [o for o in outputs if o.output_type == "error"]
    assert len(errors) == (2 if failure else 0)
    text = "".join(o.get("text", "") for o in outputs)
    assert "fixture installation" in text
    if not setup_only:
        assert "fixture NPO stdout" in text and "fixture NPO stderr" in text
    else:
        assert "fixture NPO stdout" not in text
    if failure:
        assert "fixture training_loop.py:94 CUDA out of memory" in text
    run = next((fixture / "results/local").iterdir())
    result = json.loads((run / "raw/run_result.json").read_text())
    assert result["status"] == (
        "PASS_LOCAL_ENVIRONMENT" if setup_only else "FAIL" if failure else "PASS_LOCAL_PIPELINE"
    )
    assert result["training_attempted"] is not setup_only
    assert (run / "sealed/ARCHIVE_MANIFEST.json").is_file()
    assert executed.cells[-1].outputs[-1].output_type == "stream"


def test_zero_exit_without_child_acceptance_is_failure(local_run, monkeypatch):
    local_run.initialize()
    local_run.verify_inputs()
    local_run.setup_environment()

    def child(name, argv, raw, env):
        output = raw / "NPO"
        output.mkdir()
        (output / "RESULT.json").write_text('{"status":"FAIL_PARTIAL_PRESERVED"}')

    monkeypatch.setattr(runner, "command", child)
    with pytest.raises(RuntimeError, match="acceptance checks"):
        local_run.train()
    assert local_run.result["status"] == "FAIL"
    assert local_run.result["training_attempted"]
    assert local_run.finished


def test_windows_interrupt_requests_entire_owned_process_tree(monkeypatch):
    calls = []
    monkeypatch.setattr(live_output, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(live_output.subprocess, "run", lambda argv, **kwargs: calls.append(argv))
    child = SimpleNamespace(pid=1234, poll=lambda: 1, wait=lambda **kwargs: calls.append("wait"))
    live_output.stop_child(child)
    assert calls == [["taskkill", "/PID", "1234", "/T", "/F"], "wait"]
