"""Set up native Windows dependencies and execute the complete existing NPO D1 pipeline."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import traceback
import uuid
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from experiments.reproductions.NPO.local_inputs import prepare  # noqa: E402
from scripts.archive_run import seal_run  # noqa: E402


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def command(name, argv, raw, env):
    print(f"{name}: logs in {raw}", flush=True)
    record = {"argv": [str(x) for x in argv], "started_utc": datetime.now(UTC).isoformat()}
    try:
        with (
            (raw / f"{name}.stdout.log").open("xb") as out,
            (raw / f"{name}.stderr.log").open("xb") as err,
        ):
            child = subprocess.run(
                record["argv"], cwd=ROOT, env=env, stdout=out, stderr=err, check=False
            )
        record["returncode"] = child.returncode
        if child.returncode:
            raise RuntimeError(
                f"{name} failed with exit code {child.returncode}; see {name}.stderr.log"
            )
    except BaseException as error:
        record["error_type"] = type(error).__name__
        raise
    finally:
        record["finished_utc"] = datetime.now(UTC).isoformat()
        write(raw / f"{name}.execution.json", record)


def source():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    if git("status", "--porcelain"):
        raise ValueError("Commit source changes before a recorded NPO run")
    commit = git("rev-parse", "HEAD")
    return {
        "source_origin_sha": commit,
        "snapshot_commit_sha": commit,
        "source_tree_sha": git("rev-parse", "HEAD^{tree}"),
    }


def setup(raw, env):
    python = ROOT / ".venv-local" / "Scripts" / "python.exe"
    if not python.is_file():
        command("venv", [sys.executable, "-m", "venv", ROOT / ".venv-local"], raw, env)
    command(
        "build_tools",
        [python, "-m", "pip", "install", "pip", "setuptools", "wheel", "packaging", "ninja"],
        raw,
        env,
    )
    command(
        "torch",
        [
            python,
            "-m",
            "pip",
            "install",
            "torch==2.10.0+cu128",
            "--index-url",
            "https://download.pytorch.org/whl/cu128",
        ],
        raw,
        env,
    )
    command(
        "dependencies",
        [python, "-m", "pip", "install", "-r", ROOT / "requirements/local-windows.in"],
        raw,
        env,
    )
    deepspeed_pin = json.loads(
        (ROOT / "experiments/reproductions/NPO/runtime_sources_v1.json").read_text()
    )["deepspeed"]
    deepspeed_source = deepspeed_pin["url"] + "#sha256=" + deepspeed_pin["sha256"]
    command(
        "deepspeed",
        [python, "-m", "pip", "install", "--no-build-isolation", deepspeed_source],
        raw,
        env,
    )
    command(
        "project",
        [
            python,
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "-c",
            ROOT / "requirements/local-windows.in",
            "-e",
            ROOT,
        ],
        raw,
        env,
    )
    command("pip_check", [python, "-m", "pip", "check"], raw, env)
    command("inventory", [python, "-m", "pip", "freeze", "--all"], raw, env)
    return python


def execute(args):
    run_id = "local-" + datetime.now(UTC).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
    run = ROOT / "results" / "local" / run_id
    raw = run / "raw"
    raw.mkdir(parents=True, exist_ok=False)
    result = {
        "run_id": run_id,
        "status": "FAIL",
        "phase": "source",
        "training_attempted": False,
        "started_utc": datetime.now(UTC).isoformat(),
        "execution_profile": "local-native-windows-v1",
    }
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        DS_BUILD_OPS="0",
        PYTHONHASHSEED="1001",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
    )
    # D1 uses logical device zero; the user selects its physical device in a subprocess.
    env["FQ_LOCAL_GPU_SELECTION"] = str(args.gpu)
    try:
        provenance = source()
        write(raw / "source_provenance.json", provenance)
        write(
            raw / "host.json",
            {
                "os": platform.platform(),
                "python": sys.version,
                "selected_gpu": args.gpu,
                "memory_admission_gate": False,
            },
        )
        if platform.system() != "Windows":
            raise RuntimeError("Use native Windows for this launcher")
        if sys.version_info[:2] not in ((3, 11), (3, 12)):
            raise RuntimeError("Use Python 3.11 or 3.12")
        recipe = ROOT / "configs/runs/work_unit_d1_npo_v2.json"
        for name, path in [
            ("recipe", recipe),
            ("requirements", ROOT / "requirements/local-windows.in"),
        ]:
            write(
                raw / f"{name}.json",
                {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "text": path.read_text(),
                },
            )
        result["phase"] = "inputs"
        inputs = prepare(args.models, args.data)
        write(raw / "input_files.json", inputs["records"])
        result["phase"] = "environment"
        python = setup(raw, env)
        result["phase"] = "training"
        result["training_attempted"] = True
        argv = [
            python,
            ROOT / "experiments/reproductions/NPO/train_runtime.py",
            "--recipe",
            recipe,
            "--output",
            raw / "NPO",
            "--run-id",
            run_id,
            "--provenance",
            raw / "source_provenance.json",
            "--local-models",
            args.models.resolve(),
            "--local-data",
            args.data.resolve(),
        ]
        if args.resume_checkpoint:
            argv += ["--resume-checkpoint", args.resume_checkpoint.resolve()]
        command("npo", argv, raw, env)
        child = json.loads((raw / "NPO/RESULT.json").read_text())
        if child.get("status") != "PASS_LOCAL_RUNTIME_CRITERIA_ARCHIVAL_PENDING":
            raise RuntimeError("NPO did not complete its acceptance checks")
        result["status"] = "PASS_LOCAL_PIPELINE"
    except KeyboardInterrupt:
        result["status"] = "CANCELLED"
    except Exception as error:
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
        (raw / "traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        result["finished_utc"] = datetime.now(UTC).isoformat()
        write(raw / "run_result.json", result)
        try:
            seal_run(raw, run / "sealed", run_id)
            write(run / "archive_status.json", {"status": "SEALED_LOCAL", "remote_backup": False})
        except Exception as error:
            write(
                run / "archive_status.json",
                {"status": "FAIL_RAW_PRESERVED", "error_type": type(error).__name__},
            )
            result["archive_failed"] = True
    print(
        json.dumps(
            {"status": result["status"], "output": str(run), "error": result.get("error")}, indent=2
        )
    )
    return (
        0 if result["status"] == "PASS_LOCAL_PIPELINE" and not result.get("archive_failed") else 1
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=ROOT / "models/NPO")
    parser.add_argument("--data", type=Path, default=ROOT / "data/NPO")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error("--gpu must be nonnegative")
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
