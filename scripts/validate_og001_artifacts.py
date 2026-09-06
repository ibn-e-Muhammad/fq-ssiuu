"""Derive OG-001 acceptance from hash-verified raw records, never a PASS label alone."""

import hashlib
import json
from pathlib import Path

from scripts.kaggle_setup import (
    compare_environment_transition,
    generate_overlay_lock,
    parse_freeze_mapping,
    validate_inventory_observation,
)
from scripts.pip_consistency import compare_checks, load_record
from scripts.verify_kaggle_env import REQUIRED_RUNTIME_CHECKS
from scripts.verify_pip_report import verify_report

REQUIRED = {
    "setup_status.json",
    "MANIFEST.json",
    "before_manifest.json",
    "after_manifest.json",
    "before_freeze.log",
    "after_freeze.log",
    "verifier_execution.json",
    "verifier_stdout.log",
    "verifier_stderr.log",
    "kaggle_env_verification.json",
    "overlay_lock.json",
    "kaggle-overlay.lock",
    "pip_install_report.json",
    "pip_dry_run_report.json",
    "install_verification_report.json",
    "verification_report.json",
    "pip_consistency.json",
    "environment_transition.json",
    "native_baseline.json",
    "before_distributions.json",
    "after_distributions.json",
    *{
        f"pip_check_{stage}{suffix}"
        for stage in ("before", "after", "verifier")
        for suffix in (".json", "_stdout.log", "_stderr.log")
    },
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_setup_artifacts(output, expected_sha, baseline_path=None):
    output = Path(output).resolve()
    for name in sorted(REQUIRED):
        if not (output / name).is_file():
            raise ValueError(f"Missing setup evidence: {name}")

    def read(name):
        return json.loads((output / name).read_text())

    manifest = read("MANIFEST.json")
    entries = manifest["artifacts"]
    paths = [item["path"] for item in entries]
    actual = {str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()}
    if len(set(paths)) != len(paths) or set(paths) != actual - {"MANIFEST.json"}:
        raise ValueError("Artifact manifest does not cover every evidence file exactly once")
    if not REQUIRED - {"MANIFEST.json"} <= set(paths):
        raise ValueError("Mandatory evidence absent from hashed manifest")
    for item in entries:
        original = output / item["path"]
        path = original.resolve()
        if not path.is_relative_to(output) or original.is_symlink() or not path.is_file():
            raise ValueError("Unsafe evidence path")
        if path.stat().st_size != item["size_bytes"] or digest(path) != item["sha256"]:
            raise ValueError(f"Artifact hash mismatch: {item['path']}")
    run_id = manifest["run_id"]
    if manifest.get("git_sha") != expected_sha or manifest.get("execution_status") != "PASS":
        raise ValueError("Setup manifest identity/status mismatch")
    for name, field in [
        ("setup_status.json", "git_sha"),
        ("before_manifest.json", "git_commit"),
        ("after_manifest.json", "git_commit"),
        ("verifier_execution.json", "setup_git_sha"),
        ("kaggle_env_verification.json", "git_sha"),
        ("overlay_lock.json", "git_sha"),
        ("environment_transition.json", "git_sha"),
        ("pip_consistency.json", "git_sha"),
    ]:
        value = read(name)
        if value.get("run_id") != run_id or value.get(field) != expected_sha:
            raise ValueError(f"Source/run linkage mismatch: {name}")
    if baseline_path is not None and digest(baseline_path) != digest(
        output / "native_baseline.json"
    ):
        raise ValueError("Recorded native baseline differs from verified source baseline")
    baseline = read("native_baseline.json")
    report = read("pip_install_report.json")
    for filename in ("pip_install_report.json", "pip_dry_run_report.json"):
        valid, errors, _ = verify_report(read(filename), baseline)
        if not valid:
            raise ValueError(f"Resolver policy failed: {filename}: {errors}")
    names = [p["metadata"]["name"] for p in report["install"]]
    checks = {
        stage: load_record(
            output / f"pip_check_{stage}.json", stage=stage, run_id=run_id, git_sha=expected_sha
        )
        for stage in ("before", "after", "verifier")
    }
    consistency = compare_checks(checks["before"], checks["after"], names)
    if consistency["status"] != "PASS" or consistency != read("pip_consistency.json"):
        raise ValueError(
            "Differential consistency failed or derived record disagrees with raw checks"
        )
    verifier_consistency = compare_checks(checks["before"], checks["verifier"], names)
    if verifier_consistency["status"] != "PASS":
        raise ValueError("Verifier pip consistency failed")
    status = read("setup_status.json")
    if status.get("overall_status") != "PASS" or status.get("mode") != "APPLY":
        raise ValueError("Setup did not record APPLY PASS")
    for key in (
        "dry_run_exit_code",
        "install_exit_code",
        "editable_exit_code",
        "verifier_exit_code",
    ):
        if type(status.get(key)) is not int or status[key] != 0:
            raise ValueError(f"Setup operation failed: {key}")
    if (
        status.get("pip_check_exit_code") != checks["after"]["returncode"]
        or status.get("pip_consistency_state") != consistency["state"]
    ):
        raise ValueError("Setup pip outcome disagrees with observed consistency")
    snapshots = {stage: read(f"{stage}_manifest.json") for stage in ("before", "after")}
    for stage, snapshot in snapshots.items():
        if snapshot.get("snapshot_status") != "PASS" or snapshot.get("stage") != stage:
            raise ValueError(f"Invalid {stage} snapshot")
        if snapshot.get("git_dirty") is not False:
            raise ValueError(f"Source was dirty in {stage} snapshot")
    inventories = {}
    for stage in ("before", "after"):
        inventory = read(f"{stage}_distributions.json")
        if (
            inventory.get("schema_version") != "1.1"
            or inventory.get("status") != "PASS"
            or inventory.get("source") != "importlib.metadata.distributions"
            or inventory.get("stage") != stage
            or inventory.get("run_id") != run_id
            or inventory.get("git_sha") != expected_sha
        ):
            raise ValueError("Invalid complete distribution inventory")
        validate_inventory_observation(inventory)
        packages = inventory.get("packages")
        if (
            not isinstance(packages, dict)
            or not packages
            or any(
                not isinstance(k, str) or not isinstance(v, str) or not k or not v
                for k, v in packages.items()
            )
        ):
            raise ValueError("Malformed distribution inventory")
        freeze = parse_freeze_mapping((output / f"{stage}_freeze.log").read_text())
        if (
            any(packages.get(k) != v for k, v in freeze.items())
            or inventory.get("freeze_disagreements") != []
        ):
            raise ValueError("Distribution inventory disagrees with raw freeze")
        if snapshots[stage].get("installed_packages") != packages:
            raise ValueError("Snapshot does not contain observed complete inventory")
        inventories[stage] = packages
    transition = compare_environment_transition(
        snapshots["before"],
        snapshots["after"],
        inventories["before"],
        inventories["after"],
        baseline,
        report,
        read("before_distributions.json")["distributions"],
        read("after_distributions.json")["distributions"],
    )
    if transition["status"] != "PASS" or transition != read("environment_transition.json"):
        raise ValueError("Protected/native platform transition failed or record mismatch")
    verifier = read("kaggle_env_verification.json")
    if verifier.get("statuses", {}).get("overall") != "PASS":
        raise ValueError("Runtime verifier failed")
    observed = verifier.get("checks", {})
    if any(observed.get(key, {}).get("status") != "PASS" for key in REQUIRED_RUNTIME_CHECKS):
        raise ValueError("Substantive runtime check missing or failed")
    if observed["pip_check"].get("observed") != verifier_consistency:
        raise ValueError("Verifier consistency record disagrees with raw checks")
    memory = observed["peak_memory"]["observed"]
    if not (
        0 < memory["peak_allocated_bytes"] <= memory["peak_reserved_bytes"] < 16_000_000_000
        and memory["device"] == "cuda:0"
    ):
        raise ValueError("Peak memory exceeds selected-device budget or invalid observation")
    if (
        observed["cuda_device_count"]["observed"] != 1
        or observed["cuda_visible_devices_var"]["observed"] != "0"
        or observed["nf4_quantise_dequantise"]["observed"]["quant_type"] != "nf4"
    ):
        raise ValueError("Single-device or genuine NF4 observation mismatch")
    execution = read("verifier_execution.json")
    if (
        execution.get("verdict") != "PASS"
        or execution.get("verifier_returncode") != 0
        or execution.get("verifier_git_sha") != expected_sha
        or execution.get("run_id_valid") is not True
        or execution.get("git_sha_valid") is not True
    ):
        raise ValueError("Verifier execution failed")
    for field, name in [
        ("stdout_log", "verifier_stdout.log"),
        ("stderr_log", "verifier_stderr.log"),
        ("verification_report", "kaggle_env_verification.json"),
    ]:
        if execution[field].get("path") != name or execution[field].get("sha256") != digest(
            output / name
        ):
            raise ValueError("Verifier execution hash linkage mismatch")
    lock = read("overlay_lock.json")
    expected_lock, _ = generate_overlay_lock(
        inventories["before"], inventories["after"], baseline, run_id, expected_sha
    )
    if lock.get("lock_entries") != expected_lock["lock_entries"]:
        raise ValueError("Overlay lock disagrees with observed package delta")
    pins = {
        line.split("#")[0].strip()
        for line in (output / "kaggle-overlay.lock").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    if pins != {
        f"{e['normalized_name']}=={e['after_version']}" for e in expected_lock["lock_entries"]
    }:
        raise ValueError("Text overlay lock disagrees with structured lock")
    for key, filename in [
        ("pip_dry_run_report", "pip_dry_run_report.json"),
        ("pip_install_report", "pip_install_report.json"),
        ("pip_consistency", "pip_consistency.json"),
        ("environment_transition", "environment_transition.json"),
    ]:
        if lock.get("report_hashes", {}).get(key) != digest(output / filename):
            raise ValueError("Overlay lock evidence hash mismatch")
    return {
        "status": "PASS",
        "run_id": run_id,
        "git_sha": expected_sha,
        "pip_consistency_state": consistency["state"],
        "artifact_count": len(entries),
        "peak_memory": memory,
        "scientific_experiment_executed": False,
    }
