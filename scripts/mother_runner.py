"""Run a pinned research recipe on one T4 and preserve its complete output record."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.archive_run import seal_run  # noqa: E402
from scripts.validate_og001_artifacts import validate_setup_artifacts  # noqa: E402


def write_new(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def checked_path(root, name):
    path = (Path(root) / name).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
        raise ValueError("Recipe path must be a file within the verified source")
    return path


def load_recipe(path):
    recipe = json.loads(Path(path).read_text())
    if recipe.get("schema_version") != "1.0" or recipe.get("kind") not in {
        "integration",
        "pilot",
        "experiment",
    }:
        raise ValueError("Unknown recipe schema or run kind")
    if type(recipe.get("seed")) is not int or not 0 <= recipe["seed"] < 2**32:
        raise ValueError("Explicit integer seed required")
    if recipe["kind"] != "integration":
        # Each accepted method supplies its own adapter recipe. Do not relabel
        # a historical surrogate as an accepted method or invent missing inputs.
        spec = json.loads(checked_path(ROOT, recipe["method_spec"]).read_text())
        if spec.get("status") != "ACCEPTED_FOR_PILOT":
            raise ValueError("Method conformance has not been accepted for pilot execution")
        for asset in recipe["inputs"]:
            if asset["type"] not in {"model", "dataset"} or len(asset["revision"]) != 40:
                raise ValueError("Model/data inputs require immutable revisions")
        if recipe["kind"] == "experiment":
            raise ValueError("Confirmatory execution is unavailable until approved protocol freeze")
        checked_path(ROOT, recipe["entrypoint"])
    return recipe


def run_command(name, argv, output, env, timeout):
    started = time.monotonic()
    record = {"argv": argv, "started_utc": datetime.now(UTC).isoformat()}
    try:
        with (
            (output / f"{name}_stdout.log").open("xb") as stdout,
            (output / f"{name}_stderr.log").open("xb") as stderr,
        ):
            result = subprocess.run(
                argv, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, timeout=timeout, check=False
            )
        record["returncode"] = result.returncode
    except subprocess.TimeoutExpired:
        record.update(returncode=124, termination="TIMEOUT")
    finally:
        record["elapsed_seconds"] = time.monotonic() - started
        write_new(output / f"{name}_execution.json", record)
    return record["returncode"]


def execute(recipe_path, output, provenance, run_id):
    recipe_path, output = Path(recipe_path), Path(output)
    recipe = load_recipe(recipe_path)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "recipe.json", recipe)
    write_new(output / "source_provenance.json", provenance)
    start = {
        "run_id": run_id,
        "kind": recipe["kind"],
        "seed": recipe["seed"],
        "started_utc": datetime.now(UTC).isoformat(),
        "status": "RUNNING",
        "recipe_sha256": hashlib.sha256(recipe_path.read_bytes()).hexdigest(),
    }
    write_new(output / "run_started.json", start)
    result = {
        **start,
        "status": "FAIL",
        "scientific_experiment_executed": False,
        "supported_claims": [],
        "unsupported_claims": ["Scientific hypothesis confirmation"],
    }
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="0",
        PYTHONHASHSEED=str(recipe["seed"]),
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
    )
    phase = "environment"
    try:
        # Credentials are passed only to operations explicitly declaring them.
        # The integration recipe declares none and never reads local/Kaggle secrets.
        if recipe.get("required_secrets"):
            sys.path.insert(0, str(ROOT / "src"))
            from fq_ssiuu.secrets import get_secret

            for name in recipe["required_secrets"]:
                env[name] = get_secret(name)
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
        if head != provenance["snapshot_commit_sha"]:
            raise ValueError("Checkout differs from verified source provenance")
        code = run_command(
            "environment",
            [
                sys.executable,
                str(ROOT / "scripts/kaggle_setup.py"),
                "--apply",
                "--overlay",
                str(checked_path(ROOT, recipe["overlay_lock"])),
                "--output-dir",
                str(output / "environment"),
            ],
            output,
            env,
            1200,
        )
        if code:
            raise RuntimeError("Environment setup failed; see environment logs")
        acceptance = validate_setup_artifacts(
            output / "environment", head, ROOT / "requirements/kaggle-native-baseline.json"
        )
        write_new(output / "environment_acceptance.json", acceptance)
        result["environment_run_id"] = acceptance["run_id"]
        if recipe["kind"] == "integration":
            phase = "integration"
            env.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            codes = {
                "notebooks": run_command(
                    "notebooks",
                    [
                        sys.executable,
                        str(ROOT / "scripts/check_notebooks.py"),
                        "--output",
                        str(output / "notebook_validation.json"),
                    ],
                    output,
                    env,
                    120,
                ),
                "tests": run_command(
                    "tests",
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "tests",
                        "-m",
                        "not requires_model",
                        "-ra",
                        "--junitxml=" + str(output / "tests.xml"),
                    ],
                    output,
                    env,
                    1200,
                ),
            }
            result["returncodes"] = codes
            if any(codes.values()):
                raise RuntimeError("Integration failed; see tests and notebooks output")
            result["supported_claims"] = ["Validated environment and non-model integration recipe"]
        else:
            phase = "pilot"
            # The accepted adapter owns upstream-specific input loading, seeding,
            # training/checkpoint/resume and measured GPU memory. It receives the
            # exact recipe and isolated output namespace rather than hidden defaults.
            code = run_command(
                "pilot",
                [
                    sys.executable,
                    str(checked_path(ROOT, recipe["entrypoint"])),
                    "--recipe",
                    str(recipe_path.resolve()),
                    "--output",
                    str(output / "pilot"),
                ],
                output,
                env,
                recipe["timeout_seconds"],
            )
            result["scientific_experiment_executed"] = True
            if code:
                raise RuntimeError("Pilot adapter failed; see pilot output")
            result["supported_claims"] = [
                "Pilot execution; scientific interpretation requires result review"
            ]
        result["status"] = "PASS"
    except Exception as exc:
        result.update(failure_phase=phase, failure_type=type(exc).__name__)
    finally:
        result["finished_utc"] = datetime.now(UTC).isoformat()
        write_new(output / "run_result.json", result)
        seal_run(output, output.parent / (run_id + "-sealed"), run_id)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    outcome = execute(
        args.recipe, args.output, json.loads(Path(args.provenance).read_text()), args.run_id
    )
    print(json.dumps(outcome, indent=2))
    raise SystemExit(0 if outcome["status"] == "PASS" else 1)
