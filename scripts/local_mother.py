"""Native Windows local entry point for bounded setup, tests, and D1 readiness.

This is an infrastructure runner. It does not modify or bypass the frozen D1
recipe, the Kaggle runner, or the NPO training implementation.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "configs/runs/work_unit_d1_npo_v2.json"
TESTS = ("tests/test_mother_runner.py", "tests/test_run_archiving.py")


def git_value(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def source_identity():
    commit = git_value("rev-parse", "HEAD")
    tree = git_value("rev-parse", "HEAD^{tree}")
    if git_value("status", "--porcelain"):
        raise ValueError(
            "Source checkout must be clean; use an output directory outside the repository"
        )
    if not all(re.fullmatch(r"[0-9a-f]{40}", x) for x in (commit, tree)):
        raise ValueError("Invalid source identity")
    return {"source_origin_sha": commit, "source_tree_sha": tree, "snapshot_commit_sha": commit}


def validate_recipe():
    # Validation is deferred so the runner can report hardware before installing ML packages.
    recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
    for key in ("train_config", "eval_config", "overlay", "source_spec", "runtime_sources"):
        relative = Path(recipe[key])
        target = (ROOT / relative).resolve()
        if relative.is_absolute() or not target.is_relative_to(ROOT) or not target.is_file():
            raise ValueError(f"Unsafe or missing recipe file: {key}")
        if hashlib.sha256(target.read_bytes()).hexdigest() != recipe[key + "_sha256"]:
            raise ValueError(f"D1 recipe hash mismatch: {key}")
    if recipe.get("condition") != "d1-npo-source-faithful-1b-v2":
        raise ValueError("Unexpected D1 condition")
    return recipe


def system_ram_bytes():
    if os.name != "nt":
        return None
    import ctypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("load", ctypes.c_ulong),
            ("total_physical", ctypes.c_ulonglong),
            ("available_physical", ctypes.c_ulonglong),
            ("total_page", ctypes.c_ulonglong),
            ("available_page", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("available_virtual", ctypes.c_ulonglong),
            ("available_extended", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("Cannot read Windows RAM capacity")
    return status.total_physical


def gpu_inventory():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    if result.returncode:
        raise RuntimeError("nvidia-smi failed")
    devices = []
    for line in result.stdout.splitlines():
        name, memory = line.rsplit(",", 1)
        devices.append({"name": name.strip(), "vram_mib": int(memory.strip())})
    if not devices:
        raise RuntimeError("No NVIDIA GPU detected")
    return devices


def preflight(output_root):
    checks = {
        "os": platform.system(),
        "python": platform.python_version(),
        "free_disk_bytes": shutil.disk_usage(output_root).free,
        "ram_bytes": system_ram_bytes(),
    }
    blockers = []
    try:
        checks["gpus"] = gpu_inventory()
    except (FileNotFoundError, OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        checks["gpu_probe_error"] = type(exc).__name__
        blockers.append("NVIDIA GPU probe did not succeed")
    if checks["os"] != "Windows":
        blockers.append("Native Windows host required for this local runner")
    if sys.version_info[:2] not in ((3, 11), (3, 12)):
        blockers.append("Use Python 3.11 or 3.12")
    if checks["free_disk_bytes"] < 70_000_000_000:
        blockers.append("At least 70 GB free disk is required for cache, raw output and archive")
    if len(checks.get("gpus", [])) != 1:
        blockers.append("D1 needs exactly one selected CUDA device")
    if checks.get("gpus") and "T4" not in checks["gpus"][0]["name"]:
        blockers.append("Current D1 runtime only accepts Tesla T4")
    if checks["os"] == "Windows":
        blockers.append("Current D1 runtime initializes NCCL; native Windows PyTorch has no NCCL")
    if checks.get("gpus") and checks["gpus"][0]["vram_mib"] < 16_000:
        blockers.append(
            "Current D1 already failed first backward on a 16 GB T4; smaller VRAM is unvalidated"
        )
    if checks["ram_bytes"] is not None and checks["ram_bytes"] < 16_000_000_000:
        blockers.append(
            "Less than 16 GB system RAM; full 1B optimizer/checkpoint workflow is unvalidated"
        )
    # The current recipe also fixes a Kaggle overlay, scratch paths and version pins.
    blockers.append("No validated native Windows D1 dependency/runtime route exists yet")
    return {"checks": checks, "d1_blockers": blockers, "d1_ready": False}


def run_logged(argv, folder, name):
    with (
        (folder / f"{name}.stdout.log").open("xb") as stdout,
        (folder / f"{name}.stderr.log").open("xb") as stderr,
    ):
        completed = subprocess.run(argv, cwd=ROOT, stdout=stdout, stderr=stderr, check=False)
    return {"argv": argv, "returncode": completed.returncode}


def venv_python():
    return ROOT / ".venv" / "Scripts" / "python.exe"


def execute(action, output_root, run_id):
    output_root = Path(output_root).resolve()
    if output_root.is_relative_to(ROOT):
        raise ValueError("Run outputs must be outside the source checkout")
    output_root.mkdir(parents=True, exist_ok=True)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{5,120}", run_id):
        raise ValueError("Invalid run ID")
    output = output_root / run_id
    output.mkdir(exist_ok=False)
    record = {
        "run_id": run_id,
        "action": action,
        "started_utc": datetime.now(UTC).isoformat(),
        "status": "FAIL",
        "scientific_experiment_executed": False,
    }
    try:
        record["provenance"] = source_identity()
        recipe = validate_recipe()
        record["recipe_sha256"] = hashlib.sha256(RECIPE.read_bytes()).hexdigest()
        record["condition"] = recipe["condition"]
        report = preflight(output_root)
        (output / "preflight.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        if action == "check":
            record["status"] = "PREFLIGHT_COMPLETE"
        elif action == "d1":
            record["status"] = "BLOCKED"
            record["reasons"] = report["d1_blockers"]
        elif action == "bootstrap":
            if platform.system() != "Windows":
                raise RuntimeError("Environment bootstrap requires native Windows")
            if sys.version_info[:2] not in ((3, 11), (3, 12)):
                raise RuntimeError("Bootstrap requires Python 3.11 or 3.12")
            if not venv_python().is_file():
                record["venv"] = run_logged(
                    [sys.executable, "-m", "venv", str(ROOT / ".venv")], output, "venv"
                )
                if record["venv"]["returncode"]:
                    raise RuntimeError("Virtual environment creation failed")
            record["install"] = run_logged(
                [str(venv_python()), "-m", "pip", "install", "-e", ".[dev]"], output, "install"
            )
            if record["install"]["returncode"]:
                raise RuntimeError("Portable development environment installation failed")
            record["pip_check"] = run_logged(
                [str(venv_python()), "-m", "pip", "check"], output, "pip_check"
            )
            if record["pip_check"]["returncode"]:
                raise RuntimeError("pip check failed")
            record["status"] = "PORTABLE_ENV_READY"
        elif action == "test":
            if platform.system() != "Windows":
                raise RuntimeError("Local integration testing requires native Windows")
            if not venv_python().is_file():
                raise RuntimeError("Run bootstrap first to create .venv")
            record["tests"] = run_logged(
                [str(venv_python()), "-m", "pytest", *TESTS, "-q"], output, "tests"
            )
            record["status"] = (
                "PORTABLE_TESTS_PASS" if record["tests"]["returncode"] == 0 else "FAIL"
            )
        else:
            raise ValueError("Unknown action")
    except Exception as exc:
        record["error_type"] = type(exc).__name__
        record["error"] = str(exc)
    finally:
        record["finished_utc"] = datetime.now(UTC).isoformat()
        (output / "run_result.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
    return output, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "bootstrap", "test", "d1"))
    parser.add_argument("--output-root", type=Path, default=Path.home() / "fq-ssiuu-runs")
    parser.add_argument("--run-id", default="local-" + uuid.uuid4().hex)
    args = parser.parse_args()
    output, record = execute(args.action, args.output_root, args.run_id)
    print(json.dumps({"output": str(output), "status": record["status"]}, indent=2))
    return (
        0
        if record["status"] in {"PREFLIGHT_COMPLETE", "PORTABLE_ENV_READY", "PORTABLE_TESTS_PASS"}
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
