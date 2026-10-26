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
from notebooks.Local.environment import ensure_environment  # noqa: E402
from notebooks.Local.live_output import stream_command  # noqa: E402
from scripts.archive_run import seal_run  # noqa: E402


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def command(name, argv, raw, env):
    return stream_command(name, argv, raw, env, ROOT, write)


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


def install_environment(raw, env):
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
    return python


def setup(raw, env, mode="auto"):
    return ensure_environment(ROOT, raw, env, command, install_environment, mode)


class LocalRun:
    """The same CLI workflow, exposed as guarded notebook stages."""

    def __init__(self, args):
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
        self.args, self.run_id, self.run, self.raw = args, run_id, run, raw
        self.result, self.env = result, env
        self.next_stage = "source"
        self.finished = False
        self.event("created", "Run directory created")

    def event(self, status, message):
        entry = {
            "utc": datetime.now(UTC).isoformat(),
            "phase": self.result["phase"],
            "status": status,
            "message": message,
        }
        with (self.raw / "steps.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(entry) + "\n")
        # Atomic current-stage receipt survives a killed notebook kernel.
        temp = self.raw / "active_step.tmp"
        temp.write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
        temp.replace(self.raw / "active_step.json")
        print(f"[{entry['utc']}] {status.upper()} {entry['phase']}: {message}", flush=True)

    def stage(self, name, following, action):
        if self.finished or self.next_stage != name:
            raise RuntimeError(
                f"Stage {name} cannot run now; next={self.next_stage}, "
                f"finished={self.finished}. Do not rerun cells in this attempt."
            )
        self.result["phase"] = name
        self.event("start", name)
        try:
            action()
        except BaseException as error:
            self.result["status"] = "CANCELLED" if isinstance(error, KeyboardInterrupt) else "FAIL"
            self.result["error_type"] = type(error).__name__
            self.result["error"] = str(error)
            (self.raw / "traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
            self.event("failed", f"{type(error).__name__}: {error}")
            self.finish()
            raise
        self.event("complete", name)
        self.next_stage = following

    def initialize(self):
        self.stage("source", "inputs", self._initialize)

    def _initialize(self):
        args, raw = self.args, self.raw
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
        self.recipe = recipe

    def verify_inputs(self):
        self.stage("inputs", "environment", self._verify_inputs)

    def _verify_inputs(self):
        if getattr(self.args, "setup_only", False):
            self.event("skipped", "Setup-only: model and dataset files are not required")
            return
        inputs = prepare(self.args.models, self.args.data)
        write(self.raw / "input_files.json", inputs["records"])

    def setup_environment(self):
        def action():
            self.python = setup(self.raw, self.env, getattr(self.args, "environment", "auto"))

        self.stage("environment", "training", action)

    def train(self):
        self.stage("training", "archive", self._train)

    def _train(self):
        if getattr(self.args, "setup_only", False):
            self.result["status"] = "PASS_LOCAL_ENVIRONMENT"
            self.event("skipped", "Setup-only: no NPO subprocess or experiment was run")
            return
        args, raw, env = self.args, self.raw, self.env
        result, python, recipe, run_id = self.result, self.python, self.recipe, self.run_id
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

    def finish(self):
        if self.finished:
            return self.exit_code
        if self.next_stage != "archive" and "error_type" not in self.result:
            self.result["status"] = "CANCELLED"
            self.result["error"] = "Finalized before all stages completed"
        # Keep the failing phase in run_result; archiving is reported separately.
        self.event("finalizing", "Writing final result and sealing outputs")
        self.result["finished_utc"] = datetime.now(UTC).isoformat()
        write(self.raw / "run_result.json", self.result)
        archive_failed = False
        try:
            seal_run(self.raw, self.run / "sealed", self.run_id)
            write(
                self.run / "archive_status.json", {"status": "SEALED_LOCAL", "remote_backup": False}
            )
        except (Exception, KeyboardInterrupt) as error:
            archive_failed = True
            write(
                self.run / "archive_status.json",
                {
                    "status": "FAIL_RAW_PRESERVED",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            print(f"ARCHIVE FAILED: {error}; raw outputs preserved at {self.raw}", flush=True)
        self.exit_code = (
            0
            if self.result["status"] in {"PASS_LOCAL_PIPELINE", "PASS_LOCAL_ENVIRONMENT"}
            and not archive_failed
            else 1
        )
        self.finished = True
        # Do not modify raw after sealing; manifest hashes must remain valid.
        print(
            json.dumps(
                {
                    "status": self.result["status"],
                    "output": str(self.run),
                    "error": self.result.get("error"),
                    "exit_code": self.exit_code,
                },
                indent=2,
            ),
            flush=True,
        )
        return self.exit_code


def execute(args):
    run = LocalRun(args)
    try:
        run.initialize()
        run.verify_inputs()
        run.setup_environment()
        run.train()
    except (Exception, KeyboardInterrupt):
        # Each failed stage has already recorded the traceback and sealed its outputs.
        return run.finish()
    return run.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=ROOT / "models/NPO")
    parser.add_argument("--data", type=Path, default=ROOT / "data/NPO")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Prepare/check the environment without model/data files or training",
    )
    parser.add_argument(
        "--environment",
        choices=("auto", "check", "repair"),
        default="auto",
        help="auto installs only when absent; check never installs; repair explicitly reconciles dependencies",
    )
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error("--gpu must be nonnegative")
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
