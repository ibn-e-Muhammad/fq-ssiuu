"""Persistent environment behavior without installing or importing GPU packages."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from notebooks.Local import environment
from notebooks.Local import mother_runner as runner
from scripts import check_local_environment as probe

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def lifecycle(tmp_path):
    root = tmp_path / "repository"
    for name in environment.CONTRACT_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    python = root / ".venv-local/Scripts/python.exe"
    calls = []
    report = {
        "status": "PASS",
        "inventory": {"torch": "2.10.0+cu128"},
        "python": "3.12.14",
        "prefix": str(root / ".venv-local"),
        "project_root": str(root),
    }

    def install(raw, env):
        calls.append("INSTALL")
        python.parent.mkdir(parents=True, exist_ok=True)
        python.touch()

    def command(name, argv, raw, env):
        calls.append(name)
        assert env["PIP_NO_INDEX"] == "1"
        assert "install" not in [str(x) for x in argv]
        if name == "environment_probe":
            (raw / "environment_probe.json").write_text(json.dumps(report))

    count = 0

    def run(mode="auto"):
        nonlocal count
        count += 1
        raw = tmp_path / f"attempt-{count}"
        raw.mkdir()
        return environment.ensure_environment(root, raw, {}, command, install, mode)

    return SimpleNamespace(
        root=root,
        python=python,
        calls=calls,
        run=run,
        report=report,
        receipt=python.parents[1] / environment.RECEIPT_NAME,
    )


def test_first_setup_installs_once_and_next_attempts_never_install(lifecycle):
    assert lifecycle.run() == lifecycle.python
    assert lifecycle.calls.count("INSTALL") == 1
    lifecycle.run()
    lifecycle.run("check")
    assert lifecycle.calls.count("INSTALL") == 1
    assert lifecycle.calls.count("environment_probe") == 3
    assert lifecycle.receipt.is_file()


def test_legacy_existing_environment_adopted_without_installation(lifecycle):
    lifecycle.python.parent.mkdir(parents=True)
    lifecycle.python.touch()
    lifecycle.run()
    assert "INSTALL" not in lifecycle.calls
    assert lifecycle.receipt.is_file()


def test_check_mode_missing_environment_never_creates_or_installs(lifecycle):
    with pytest.raises(RuntimeError, match="No local environment"):
        lifecycle.run("check")
    assert not lifecycle.python.parents[1].exists()
    assert not lifecycle.calls


def test_requirements_change_requires_explicit_repair(lifecycle):
    lifecycle.run()
    requirements = lifecycle.root / "requirements/local-windows.in"
    requirements.write_text(requirements.read_text() + "\n# changed contract\n")
    with pytest.raises(RuntimeError, match="requirements or verifier changed"):
        lifecycle.run()
    assert lifecycle.calls.count("INSTALL") == 1
    lifecycle.run("repair")
    assert lifecycle.calls.count("INSTALL") == 2
    lifecycle.run()
    assert lifecycle.calls.count("INSTALL") == 2


def test_dependency_drift_rejected_without_installation(lifecycle):
    lifecycle.run()
    lifecycle.report["inventory"]["extra-package"] = "1.0"
    with pytest.raises(RuntimeError, match="Environment drift"):
        lifecycle.run()
    assert lifecycle.calls.count("INSTALL") == 1
    lifecycle.run("repair")
    assert lifecycle.calls.count("INSTALL") == 2


def test_failed_initial_validation_leaves_no_success_marker_or_auto_retry(lifecycle):
    lifecycle.report["status"] = "FAIL"
    with pytest.raises(RuntimeError, match="probe failed"):
        lifecycle.run()
    assert not lifecycle.receipt.exists()
    with pytest.raises(RuntimeError, match="probe failed"):
        lifecycle.run()
    assert lifecycle.calls.count("INSTALL") == 1
    assert not (lifecycle.python.parents[1] / "fq-setup.lock").exists()
    lifecycle.report["status"] = "PASS"
    lifecycle.run("repair")
    assert lifecycle.receipt.is_file()


def test_failed_repair_invalidates_previous_success_receipt(lifecycle):
    lifecycle.run()
    lifecycle.report["status"] = "FAIL"
    with pytest.raises(RuntimeError):
        lifecycle.run("repair")
    assert not lifecycle.receipt.exists()


def test_incomplete_environment_requires_explicit_repair(lifecycle):
    lifecycle.python.parents[1].mkdir()
    with pytest.raises(RuntimeError, match="incomplete setup"):
        lifecycle.run()
    assert "INSTALL" not in lifecycle.calls
    lifecycle.run("repair")
    assert lifecycle.calls.count("INSTALL") == 1


def test_existing_lock_prevents_mutation_and_is_not_deleted(lifecycle):
    lifecycle.python.parents[1].mkdir()
    lock = lifecycle.python.parents[1] / "fq-setup.lock"
    lock.write_text("another process")
    with pytest.raises(RuntimeError, match="already locked"):
        lifecycle.run("repair")
    assert lock.read_text() == "another process"
    assert not lifecycle.calls


def test_malformed_receipt_requires_repair(lifecycle):
    lifecycle.run()
    lifecycle.receipt.write_text("not JSON")
    with pytest.raises(RuntimeError, match="Invalid environment receipt"):
        lifecycle.run()
    assert lifecycle.calls.count("INSTALL") == 1
    lifecycle.run("repair")
    assert lifecycle.receipt.is_file()


def test_setup_only_does_not_require_inputs_or_invoke_npo(tmp_path, monkeypatch):
    for name in ("requirements/local-windows.in", "configs/runs/work_unit_d1_npo_v2.json"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "source", lambda: {"snapshot_commit_sha": "a" * 40})
    monkeypatch.setattr(runner.platform, "system", lambda: "Windows")

    def forbidden(*args):
        raise AssertionError("Model/data access or NPO invocation during setup-only")

    monkeypatch.setattr(runner, "prepare", forbidden)
    monkeypatch.setattr(runner, "command", forbidden)
    modes = []
    monkeypatch.setattr(
        runner, "setup", lambda raw, env, mode: modes.append(mode) or sys.executable
    )
    args = SimpleNamespace(
        models=tmp_path / "missing",
        data=tmp_path / "missing",
        gpu=0,
        resume_checkpoint=None,
        setup_only=True,
        environment="check",
    )
    assert runner.execute(args) == 0
    assert modes == ["check"]
    result_file = next((tmp_path / "results/local").glob("*/raw/run_result.json"))
    result = json.loads(result_file.read_text())
    assert result["status"] == "PASS_LOCAL_ENVIRONMENT"
    assert not result["training_attempted"]


@pytest.fixture
def probe_context(monkeypatch):
    pin = json.loads((ROOT / "experiments/reproductions/NPO/runtime_sources_v1.json").read_text())[
        "deepspeed"
    ]
    inventory = {}
    for line in (ROOT / "requirements/local-windows.in").read_text().splitlines():
        if line and not line.startswith("#"):
            parts = line.split("==")
            inventory[parts[0]] = parts[1] if len(parts) > 1 else "1.0"
    inventory.update(deepspeed=pin["version"], **{"fq-ssiuu": "1.0"})
    monkeypatch.setattr(probe.platform, "system", lambda: "Windows")
    monkeypatch.setattr(probe.sys, "prefix", str(ROOT / ".venv-local"))
    monkeypatch.setattr(
        probe.metadata,
        "distributions",
        lambda: [
            SimpleNamespace(metadata={"Name": name}, version=version)
            for name, version in inventory.items()
        ],
    )
    direct = {
        "deepspeed": {"url": pin["url"], "archive_info": {"hashes": {"sha256": pin["sha256"]}}},
        "fq-ssiuu": {"url": ROOT.as_uri(), "dir_info": {"editable": True}},
    }
    monkeypatch.setattr(
        probe.metadata,
        "distribution",
        lambda name: SimpleNamespace(read_text=lambda path: json.dumps(direct[name])),
    )
    # No GPU package is imported or downloaded; only probe contracts are exercised.
    monkeypatch.setattr(probe.importlib, "import_module", lambda name: None)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            version=SimpleNamespace(cuda="12.8"), cuda=SimpleNamespace(is_available=lambda: False)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "bitsandbytes.optim",
        SimpleNamespace(PagedAdamW32bit=type("PagedAdamW32bit", (), {})),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForCausalLM=type("AutoModelForCausalLM", (), {}),
            AutoTokenizer=type("AutoTokenizer", (), {}),
        ),
    )
    return inventory, direct


def test_probe_records_cuda_unavailable_without_vram_gate(probe_context):
    report = probe.verify(ROOT)
    assert report["status"] == "PASS" and report["cuda_available"] is False


def test_probe_rejects_core_version_drift(probe_context):
    inventory, _ = probe_context
    inventory["torch"] = "2.10.0"
    with pytest.raises(RuntimeError, match="mismatched requirement: torch"):
        probe.verify(ROOT)


def test_probe_rejects_unverified_deepspeed_source(probe_context):
    _, direct = probe_context
    direct["deepspeed"]["archive_info"]["hashes"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="pinned-source receipt"):
        probe.verify(ROOT)
