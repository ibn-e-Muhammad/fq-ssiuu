#!/usr/bin/env python3
"""
kaggle_setup.py — Orchestration script for Kaggle overlay dependency installation.

Fail-closed implementation:
- Preview is the default mode.
- Requires explicit --apply flag for installation.
- --apply and --preview are mutually exclusive.
- --report-file may only be used with --preview for offline testing.
- Refuses to operate unless CUDA_VISIBLE_DEVICES=0.
- Refuses to overwrite existing output directory.
- Strictly validates baseline structure using validate_baseline before all actions.
- Captures genuine before/after/post-attempt environment manifests using common run_id.
- Validates dry-run pip report and actual installation report against requirements/kaggle-native-baseline.json.
- Installs repository editable with --no-deps only after successful validation.
- Runs post-install verifier (scripts/verify_kaggle_env.py) with common run_id and checks run_id, git_sha, and overall status.
- Generates verifier_execution.json artifact recording verifier execution details.
- Generates overlay lock using production helper generate_overlay_lock only after all checks pass.
- Creates atomic failure artifacts and setup_status.json even when setup fails.
- Never edits .bashrc, runs pip uninstall, or downloads models/datasets.
"""

import os
import sys
import json
import uuid
import hashlib
import tempfile
import subprocess
import argparse
import importlib.metadata
import platform
from datetime import datetime, timezone

# Ensure project root is in sys.path and resolve paths relative to REPO_ROOT
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from scripts.verify_pip_report import verify_report, validate_baseline, normalize_name, atomic_write_json
except ImportError:
    from verify_pip_report import verify_report, validate_baseline, normalize_name, atomic_write_json

from scripts.pip_consistency import capture_pip_check, compare_checks, validate_record
from scripts.verify_kaggle_env import REQUIRED_RUNTIME_CHECKS

def atomic_write_text(filepath: str, content: str) -> None:
    """Atomically write text content to file."""
    dir_name = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(dir_name, exist_ok=True)
    temp_file = tempfile.NamedTemporaryFile(mode="w", dir=dir_name, delete=False, encoding="utf-8")
    try:
        temp_file.write(content)
        temp_file.flush()
        os.fsync(temp_file.fileno())
        temp_file.close()
        os.replace(temp_file.name, filepath)
    except Exception:
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)
        raise

def file_sha256(filepath: str) -> str:
    """Calculate SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def get_git_info() -> tuple[str, bool, bool]:
    """Returns (git_commit_sha, is_dirty, success)."""
    try:
        sha_res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT)
        if sha_res.returncode != 0:
            return "UNKNOWN", True, False
        sha = sha_res.stdout.strip()

        status_res = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=REPO_ROOT)
        if status_res.returncode != 0:
            return sha, True, False
        is_dirty = bool(status_res.stdout.strip())
        return sha, is_dirty, True
    except Exception:
        return "UNKNOWN", True, False

def get_pip_freeze() -> tuple[str, int]:
    """Runs pip freeze and returns (stdout_text, returncode)."""
    res = subprocess.run([sys.executable, "-m", "pip", "freeze", "--all"], capture_output=True, text=True)
    return res.stdout, res.returncode

def parse_freeze_mapping(freeze_text: str) -> dict[str, str]:
    """Parses pip freeze output into normalized_name -> version mapping."""
    installed = {}
    if freeze_text:
        for line in freeze_text.splitlines():
            line = line.strip()
            if "==" in line:
                parts = line.split("==")
                installed[normalize_name(parts[0])] = parts[1]
    return installed


def get_distribution_inventory() -> dict[str, str]:
    """Include direct-reference and editable distributions omitted by freeze pins."""
    installed = {}
    for distribution in importlib.metadata.distributions():
        name, version = distribution.metadata.get("Name"), distribution.version
        if not name or not version:
            raise ValueError("Installed distribution has missing name/version metadata")
        name = normalize_name(name)
        if name in installed and installed[name] != version:
            raise ValueError(f"Conflicting installed distribution metadata: {name}")
        installed[name] = version
    if not installed:
        raise ValueError("Installed distribution inventory is empty")
    return dict(sorted(installed.items()))


def capture_distribution_inventory(out_dir, stage, run_id, git_sha, freeze_text):
    installed = get_distribution_inventory()
    disagreements = [name for name, version in parse_freeze_mapping(freeze_text).items()
                     if installed.get(name) != version]
    record = {"schema_version": "1.0", "stage": stage, "run_id": run_id, "git_sha": git_sha,
              "source": "importlib.metadata.distributions", "packages": installed,
              "freeze_disagreements": disagreements, "status": "FAIL" if disagreements else "PASS"}
    atomic_write_json(os.path.join(out_dir, f"{stage}_distributions.json"), record)
    return record

def compare_versions(b_ver: str, a_ver: str) -> str:
    """
    PEP 440-aware version comparison.
    Returns: 'upgraded', 'downgraded', 'unchanged', or 'changed-unordered'.
    """
    if b_ver == a_ver:
        return "unchanged"

    try:
        from packaging.version import Version, InvalidVersion
    except ImportError:
        raise RuntimeError("PEP 440 comparison support (packaging.version) is unavailable.")

    try:
        v_b = Version(b_ver)
        v_a = Version(a_ver)
        if v_a > v_b:
            return "upgraded"
        elif v_a < v_b:
            return "downgraded"
        else:
            return "unchanged"
    except InvalidVersion:
        return "changed-unordered"

def validate_protected_packages_unchanged(
    before_installed: dict[str, str],
    after_installed: dict[str, str],
    baseline_data: dict
) -> None:
    """
    Asserts that no protected package (hard-protected compute or native stability)
    was added, removed, upgraded, downgraded, or reinstalled with a different version.
    Raises ValueError if any protected package was mutated.
    """
    hard_protected = {normalize_name(k): (k, v) for k, v in baseline_data.get("hard_protected_compute_stack", {}).items()}
    native_stability = {normalize_name(k): (k, v) for k, v in baseline_data.get("native_stability_stack", {}).items()}
    all_protected = {**hard_protected, **native_stability}

    mutations = []
    for norm_p, (orig_name, exp_ver) in all_protected.items():
        b_ver = before_installed.get(norm_p)
        a_ver = after_installed.get(norm_p)
        if b_ver != a_ver:
            mutations.append(f"{orig_name} changed between before ({b_ver}) and after ({a_ver})")
        elif b_ver is not None and b_ver != exp_ver:
            mutations.append(f"{orig_name} before version ({b_ver}) != expected ({exp_ver})")
        elif a_ver is not None and a_ver != exp_ver:
            mutations.append(f"{orig_name} after version ({a_ver}) != expected ({exp_ver})")

    if mutations:
        raise ValueError(f"Protected package mutation detected: {', '.join(mutations)}")

def create_environment_snapshot(
    stage: str,
    run_id: str,
    git_sha: str,
    git_dirty: bool,
    baseline_data: dict,
    freeze_text: str | None = None,
    installed_inventory: dict | None = None
) -> dict:
    """Creates a comprehensive, structured environment snapshot for before/after manifests."""
    timestamp = datetime.now(timezone.utc).isoformat()
    cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES")

    if freeze_text is not None:
        freeze_code = 0
    else:
        freeze_text, freeze_code = get_pip_freeze()

    installed = installed_inventory if installed_inventory is not None else parse_freeze_mapping(freeze_text)

    hard_stack = baseline_data.get("hard_protected_compute_stack", {}) if isinstance(baseline_data, dict) else {}
    native_stack = baseline_data.get("native_stability_stack", {}) if isinstance(baseline_data, dict) else {}
    all_protected = {**hard_stack, **native_stack}

    protected_obs = {}
    protected_valid = True
    for pkg_raw, exp_ver in all_protected.items():
        norm_p = normalize_name(pkg_raw)
        actual_ver = installed.get(norm_p)
        if not actual_ver:
            try:
                actual_ver = importlib.metadata.version(pkg_raw)
            except Exception:
                actual_ver = None

        is_match = (actual_ver == exp_ver)
        if not is_match:
            protected_valid = False
        protected_obs[norm_p] = {
            "name": pkg_raw,
            "expected_version": exp_ver,
            "actual_version": actual_ver,
            "match": is_match
        }

    overlay_obs = {}
    overlay_candidates = ["bitsandbytes", "captum", "hydra-core", "lm-eval", "jsonlines"]
    for ov in overlay_candidates:
        norm_ov = normalize_name(ov)
        ver = installed.get(norm_ov)
        if not ver:
            try:
                ver = importlib.metadata.version(ov)
            except Exception:
                ver = None
        overlay_obs[norm_ov] = ver

    torch_version = "UNAVAILABLE"
    torch_cuda_version = "UNAVAILABLE"
    cudnn_version = "UNAVAILABLE"
    cuda_device_count = 0
    selected_gpu_name = "UNAVAILABLE"
    compute_capability = None
    total_memory_bytes = None
    driver_version = None
    try:
        driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "--id=0"],
                                capture_output=True, text=True)
        if driver.returncode == 0:
            driver_version = driver.stdout.strip()
    except OSError:
        pass

    try:
        import torch
        torch_version = torch.__version__
        torch_cuda_version = getattr(torch.version, "cuda", "UNAVAILABLE")
        if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "version"):
            try:
                cudnn_version = str(torch.backends.cudnn.version())
            except Exception:
                cudnn_version = "UNAVAILABLE"

        cuda_device_count = torch.cuda.device_count()
        if torch.cuda.is_available() and cuda_device_count > 0:
            selected_gpu_name = torch.cuda.get_device_name(0)
            compute_capability = list(torch.cuda.get_device_capability(0))
            total_memory_bytes = torch.cuda.get_device_properties(0).total_memory
    except Exception:
        pass

    snapshot_status = "PASS" if (protected_valid and freeze_code == 0) else "FAIL"

    return {
        "schema_version": "1.0",
        "stage": stage,
        "run_id": run_id,
        "utc_timestamp": timestamp,
        "git_commit": git_sha,
        "git_dirty": git_dirty,
        "python_version": sys.version,
        "platform": {"system": platform.system(), "release": platform.release(), "version": platform.version()},
        "machine": platform.machine(),
        "driver_version": driver_version,
        "installed_packages": installed,
        "cuda_visible_devices": cuda_vis,
        "protected_packages": protected_obs,
        "overlay_packages": overlay_obs,
        "pytorch_version": torch_version,
        "pytorch_cuda_version": torch_cuda_version,
        "cudnn_version": cudnn_version,
        "cuda_device_count": cuda_device_count,
        "selected_gpu_name": selected_gpu_name,
        "compute_capability": compute_capability,
        "total_memory_bytes": total_memory_bytes,
        "snapshot_status": snapshot_status
    }

def create_verifier_execution_record(
    out_dir: str,
    run_id: str,
    setup_git_sha: str,
    verifier_git_sha: str | None,
    verifier_returncode: int,
    verifier_overall_status: str | None,
    run_id_valid: bool,
    git_sha_valid: bool,
    verdict: str
) -> dict:
    """Writes verifier_execution.json recording detailed verifier execution metadata."""
    stdout_path = os.path.join(out_dir, "verifier_stdout.log")
    stderr_path = os.path.join(out_dir, "verifier_stderr.log")
    env_rep_path = os.path.join(out_dir, "kaggle_env_verification.json")

    record = {
        "schema_version": "1.0",
        "run_id": run_id,
        "setup_git_sha": setup_git_sha,
        "verifier_git_sha": verifier_git_sha,
        "verifier_returncode": verifier_returncode,
        "verifier_overall_status": verifier_overall_status,
        "stdout_log": {
            "path": "verifier_stdout.log" if os.path.exists(stdout_path) else None,
            "sha256": file_sha256(stdout_path) if os.path.exists(stdout_path) else None
        },
        "stderr_log": {
            "path": "verifier_stderr.log" if os.path.exists(stderr_path) else None,
            "sha256": file_sha256(stderr_path) if os.path.exists(stderr_path) else None
        },
        "verification_report": {
            "path": "kaggle_env_verification.json" if os.path.exists(env_rep_path) else None,
            "sha256": file_sha256(env_rep_path) if os.path.exists(env_rep_path) else None
        },
        "run_id_valid": run_id_valid,
        "git_sha_valid": git_sha_valid,
        "verdict": verdict
    }
    atomic_write_json(os.path.join(out_dir, "verifier_execution.json"), record)
    return record

def generate_overlay_lock(
    before_installed: dict[str, str],
    after_installed: dict[str, str],
    baseline_data: dict,
    run_id: str,
    git_sha: str,
    report_hashes: dict | None = None
) -> tuple[dict, str]:
    """
    Production helper to compute overlay lock entries.
    Validates that zero protected packages were mutated; raises ValueError if any protected package changed.
    Excludes protected compute stack, unchanged native stability stack, and editable project.
    Includes newly added, upgraded, downgraded, and changed-unordered non-protected packages.
    Returns (lock_dict, lock_text_content).
    """
    # 1. Mandatory protected package integrity validation
    validate_protected_packages_unchanged(before_installed, after_installed, baseline_data)

    hard_protected = {normalize_name(k) for k in baseline_data.get("hard_protected_compute_stack", {}).keys()}
    native_stability = {normalize_name(k) for k in baseline_data.get("native_stability_stack", {}).keys()}
    all_protected = hard_protected | native_stability

    editable_project = normalize_name("fq-ssiuu")

    all_pkgs = sorted(list(set(before_installed.keys()) | set(after_installed.keys())))
    lock_entries = []

    for pkg_norm in all_pkgs:
        if pkg_norm == editable_project or pkg_norm in all_protected:
            continue

        b_ver = before_installed.get(pkg_norm)
        a_ver = after_installed.get(pkg_norm)

        if b_ver == a_ver:
            continue

        if b_ver is None and a_ver is not None:
            op = "added"
        elif b_ver is not None and a_ver is None:
            raise ValueError(f"Cannot produce installable overlay lock after removal of {pkg_norm}")
        else:
            op = compare_versions(b_ver, a_ver)
            if op == "unchanged":
                continue

        lock_entries.append({
            "normalized_name": pkg_norm,
            "before_version": b_ver,
            "after_version": a_ver,
            "operation": op
        })

    lock_dict = {
        "schema_version": "1.0",
        "run_id": run_id,
        "git_sha": git_sha,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "report_hashes": report_hashes or {},
        "lock_entries": lock_entries
    }

    lock_lines = [
        f"# FQ-SSIUU Overlay Lock (Generated: {lock_dict['created_utc']})",
        f"# Run ID: {run_id}",
        f"# Git SHA: {git_sha}",
        "# Non-protected overlay dependencies (added/upgraded/downgraded):"
    ]
    for entry in lock_entries:
        lock_lines.append(
            f"{entry['normalized_name']}=={entry['after_version']}  "
            f"# operation: {entry['operation']}, before: {entry['before_version']}"
        )

    return lock_dict, "\n".join(lock_lines) + "\n"


def compare_environment_transition(before, after, before_installed, after_installed, baseline, report):
    """Reject protected/platform drift and unreported changes, recording all deltas."""
    fields = ("python_version", "platform", "machine", "driver_version", "cuda_visible_devices",
              "pytorch_version", "pytorch_cuda_version", "cudnn_version", "cuda_device_count",
              "selected_gpu_name", "compute_capability", "total_memory_bytes")
    platform_drift = {key: {"before": before.get(key), "after": after.get(key)}
                      for key in fields if before.get(key) != after.get(key)}
    errors = []
    if platform_drift:
        errors.append("Native platform changed during setup")
    try:
        validate_protected_packages_unchanged(before_installed, after_installed, baseline)
    except ValueError as exc:
        errors.append(str(exc))
    allowed = {normalize_name(p["metadata"]["name"]): p["metadata"]["version"] for p in report["install"]}
    changes = []
    for name in sorted(before_installed.keys() | after_installed.keys()):
        b, a = before_installed.get(name), after_installed.get(name)
        if a == b:
            continue
        changes.append({"name": name, "before": b, "after": a})
        if a is None or (name != "fq-ssiuu" and allowed.get(name) != a):
            errors.append(f"Unreported distribution change: {name}")
    for name, version in allowed.items():
        if after_installed.get(name) != version:
            errors.append(f"Installed distribution disagrees with actual pip report: {name}")
    return {"schema_version": "1.0", "run_id": before["run_id"], "git_sha": before["git_commit"],
            "status": "FAIL" if errors else "PASS", "errors": errors,
            "platform_drift": platform_drift, "package_changes": changes,
            "baseline_platform_observation": {key: before.get(key) for key in fields}}

def generate_manifest(out_dir: str, run_id: str, git_sha: str, status: str, start_time: str) -> dict:
    """Generates atomic MANIFEST.json for all artifacts in out_dir."""
    end_time = datetime.now(timezone.utc).isoformat()
    files_entry = []
    if os.path.exists(out_dir):
        for root, _, filenames in os.walk(out_dir):
            for fname in sorted(filenames):
                if fname == "MANIFEST.json":
                    continue
                fpath = os.path.join(root, fname)
                rel_path = os.path.relpath(fpath, out_dir)
                size_bytes = os.path.getsize(fpath)
                h = file_sha256(fpath)
                files_entry.append({
                    "path": rel_path,
                    "size_bytes": size_bytes,
                    "sha256": h
                })

    manifest = {
        "schema_version": "1.0",
        "run_id": run_id,
        "git_sha": git_sha,
        "execution_status": status,
        "start_utc_timestamp": start_time,
        "end_utc_timestamp": end_time,
        "artifacts": files_entry
    }
    atomic_write_json(os.path.join(out_dir, "MANIFEST.json"), manifest)
    return manifest

def finalize_setup_failure(
    out_dir: str,
    run_id: str,
    git_sha: str,
    git_dirty: bool,
    baseline_data: dict | None,
    start_time: str,
    status_payload: dict
) -> None:
    """
    Common failure-finalization function that writes all available failure artifacts:
    - setup_status.json
    - post_attempt_manifest.json (when baseline is available)
    - MANIFEST.json (final manifest with execution_status FAIL)
    """
    atomic_write_json(os.path.join(out_dir, "setup_status.json"), status_payload)
    if baseline_data and isinstance(baseline_data, dict):
        try:
            post_manifest = create_environment_snapshot("post_attempt", run_id, git_sha, git_dirty, baseline_data)
            atomic_write_json(os.path.join(out_dir, "post_attempt_manifest.json"), post_manifest)
        except Exception:
            pass

    generate_manifest(out_dir, run_id, git_sha, "FAIL", start_time)

def main():
    start_time = datetime.now(timezone.utc).isoformat()
    parser = argparse.ArgumentParser(description="Kaggle Environment Overlay Setup (Fail-Closed)")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--apply", action="store_true", help="Explicitly apply the overlay installation.")
    mode_group.add_argument("--preview", action="store_true", help="Run in preview mode (dry-run report validation only). Default if omitted.")

    parser.add_argument("--overlay", help="Path to kaggle-overlay.in file.")
    parser.add_argument("--baseline", help="Path to baseline JSON file.")
    parser.add_argument("--report-file", help="Optional path to pre-generated pip report JSON file (preview mode only).")
    parser.add_argument("--output-dir", help="Optional override directory for setup artifacts.")

    args = parser.parse_args()

    # CLI Safety: --apply and --report-file are incompatible
    if args.apply and args.report_file:
        sys.stderr.write("ERROR: --report-file cannot be used in --apply mode. Apply mode must generate a fresh pip report itself.\n")
        sys.exit(1)

    mode = "APPLY" if args.apply else "PREVIEW"
    print(f"=== KAGGLE OVERLAY SETUP (MODE: {mode}) ===")

    # 1. Single-device CUDA environment enforcement (Required for BOTH preview and apply)
    cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_vis != "0":
        sys.stderr.write(
            f"ERROR: CUDA_VISIBLE_DEVICES must be set to '0' (single-device enforcement). "
            f"Observed: '{cuda_vis}'\n"
        )
        sys.exit(1)

    # Git provenance check
    git_sha, git_dirty, git_success = get_git_info()
    if not git_success:
        sys.stderr.write("ERROR: Git provenance check failed! Cannot verify Git SHA or dirty state.\n")
        sys.exit(1)

    run_id = str(uuid.uuid4())

    # Path resolution relative to REPO_ROOT
    baseline_path = args.baseline if args.baseline else os.path.join(REPO_ROOT, "requirements/kaggle-native-baseline.json")
    overlay_path = args.overlay if args.overlay else os.path.join(REPO_ROOT, "requirements/kaggle-overlay.in")

    out_dir = args.output_dir if args.output_dir else os.path.join(REPO_ROOT, f"results/kaggle_setup_{run_id}")

    # Refuse to overwrite existing output directory
    if os.path.exists(out_dir):
        sys.stderr.write(f"ERROR: Output directory already exists: {out_dir}\n")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=False)

    # Structural baseline validation FIRST
    baseline_data = None
    if not os.path.exists(baseline_path):
        sys.stderr.write(f"ERROR: Baseline file not found: {baseline_path}\n")
        finalize_setup_failure(out_dir, run_id, git_sha, git_dirty, None, start_time, {"overall_status": "FAIL", "reason": "Baseline file missing"})
        sys.exit(1)

    try:
        with open(baseline_path, "r", encoding="utf-8") as f:
            baseline_data = json.load(f)
        b_valid, b_errors = validate_baseline(baseline_data)
        if not b_valid:
            sys.stderr.write(f"ERROR: Baseline structural validation failed:\n")
            for err in b_errors:
                sys.stderr.write(f"  - {err}\n")
            finalize_setup_failure(out_dir, run_id, git_sha, git_dirty, baseline_data, start_time, {"overall_status": "FAIL", "reason": "Baseline structural validation failed", "errors": b_errors})
            sys.exit(1)
    except Exception as e:
        sys.stderr.write(f"ERROR: Failed to load or parse baseline JSON: {e}\n")
        finalize_setup_failure(out_dir, run_id, git_sha, git_dirty, None, start_time, {"overall_status": "FAIL", "reason": f"Baseline JSON unparseable: {e}"})
        sys.exit(1)

    # Preserve the exact baseline configuration with the run for portable replay.
    with open(baseline_path, "rb") as handle:
        with open(os.path.join(out_dir, "native_baseline.json"), "wb") as target:
            target.write(handle.read())

    # Check pip freeze returncode
    before_freeze_text, freeze_code = get_pip_freeze()
    if freeze_code != 0:
        sys.stderr.write(f"ERROR: pip freeze failed with returncode {freeze_code}.\n")
        finalize_setup_failure(out_dir, run_id, git_sha, git_dirty, baseline_data, start_time, {"overall_status": "FAIL", "reason": "pip freeze failed"})
        sys.exit(1)

    atomic_write_text(os.path.join(out_dir, "before_freeze.log"), before_freeze_text)

    # Genuine Before-Environment Manifest
    try:
        before_inventory = capture_distribution_inventory(out_dir, "before", run_id, git_sha, before_freeze_text)
    except (OSError, ValueError) as exc:
        finalize_setup_failure(out_dir, run_id, git_sha, git_dirty, baseline_data, start_time,
                               {"overall_status": "FAIL", "reason": f"Before inventory failed: {exc}"})
        sys.exit(1)
    before_manifest = create_environment_snapshot("before", run_id, git_sha, git_dirty, baseline_data,
                                                  freeze_text=before_freeze_text,
                                                  installed_inventory=before_inventory["packages"])
    if before_inventory["status"] != "PASS":
        before_manifest["snapshot_status"] = "FAIL"
    atomic_write_json(os.path.join(out_dir, "before_manifest.json"), before_manifest)

    if before_manifest["snapshot_status"] != "PASS":
        sys.stderr.write("ERROR: Environment before-manifest failed baseline validation! Setup stopped.\n")
        finalize_setup_failure(out_dir, run_id, git_sha, git_dirty, baseline_data, start_time, {"overall_status": "FAIL", "reason": "Before-manifest failed baseline validation"})
        sys.exit(1)

    # Capture native diagnostics before any overlay or editable installation.
    try:
        pip_before = capture_pip_check(out_dir, "before", run_id, git_sha)
        validate_record(pip_before, stage="before", run_id=run_id, git_sha=git_sha)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        finalize_setup_failure(out_dir, run_id, git_sha, git_dirty, baseline_data, start_time,
                               {"overall_status": "FAIL", "reason": f"Invalid native pip baseline: {exc}"})
        sys.exit(1)

    # Dry-run pip report generation & validation
    dry_run_exit_code = None
    if args.report_file:
        report_file = args.report_file
        print(f"Using pre-generated pip report: {report_file}")
        dry_run_exit_code = 0
    else:
        report_file = os.path.join(out_dir, "pip_dry_run_report.json")
        dry_run_cmd = [
            sys.executable, "-m", "pip", "install",
            "--dry-run",
            "--report", report_file,
            "-r", overlay_path
        ]
        print(f"Generating dry-run pip report against {overlay_path}...")
        res_dry = subprocess.run(dry_run_cmd, capture_output=True, text=True)
        dry_run_exit_code = res_dry.returncode
        atomic_write_text(os.path.join(out_dir, "dry_run_stdout.log"), res_dry.stdout)
        atomic_write_text(os.path.join(out_dir, "dry_run_stderr.log"), res_dry.stderr)

        if dry_run_exit_code != 0 or not os.path.exists(report_file):
            sys.stderr.write(f"ERROR: Failed to generate dry-run pip report:\n{res_dry.stderr}\n")
            finalize_setup_failure(out_dir, run_id, git_sha, git_dirty, baseline_data, start_time, {"overall_status": "FAIL", "dry_run_exit_code": dry_run_exit_code})
            sys.exit(1)

    try:
        with open(report_file, "r", encoding="utf-8") as f:
            report_data = json.load(f)
    except Exception as e:
        sys.stderr.write(f"ERROR: Failed to load pip report JSON: {e}\n")
        finalize_setup_failure(out_dir, run_id, git_sha, git_dirty, baseline_data, start_time, {"overall_status": "FAIL", "dry_run_exit_code": dry_run_exit_code})
        sys.exit(1)

    is_valid, violations, summary = verify_report(report_data, baseline_data, allow_no_ops=False)
    summary["run_id"] = run_id
    atomic_write_json(os.path.join(out_dir, "verification_report.json"), summary)

    if not is_valid:
        sys.stderr.write("ERROR: Dry-run pip report validation failed! Stopped setup.\n")
        for v in violations:
            sys.stderr.write(f"  - {v}\n")
        finalize_setup_failure(out_dir, run_id, git_sha, git_dirty, baseline_data, start_time, {"overall_status": "FAIL", "dry_run_exit_code": dry_run_exit_code})
        sys.exit(1)

    print("Dry-run pip report validation PASSED cleanly.")

    if not args.apply:
        print("\n[PREVIEW MODE COMPLETE] No packages were installed or modified.")
        atomic_write_json(os.path.join(out_dir, "setup_status.json"), {"overall_status": "PASS", "mode": "PREVIEW", "dry_run_exit_code": dry_run_exit_code})
        generate_manifest(out_dir, run_id, git_sha, "PASS", start_time)
        print(f"Artifacts saved to: {out_dir}")
        sys.exit(0)

    # 4. Apply mode execution
    print("\nApplying overlay installation...")
    install_report_file = os.path.join(out_dir, "pip_install_report.json")
    install_cmd = [
        sys.executable, "-m", "pip", "install",
        "--report", install_report_file,
        "-r", overlay_path
    ]
    res_inst = subprocess.run(install_cmd, capture_output=True, text=True)
    install_exit_code = res_inst.returncode
    atomic_write_text(os.path.join(out_dir, "install_stdout.log"), res_inst.stdout)
    atomic_write_text(os.path.join(out_dir, "install_stderr.log"), res_inst.stderr)

    if install_exit_code != 0 or not os.path.exists(install_report_file):
        sys.stderr.write(f"ERROR: Overlay installation failed or missing pip_install_report.json:\n{res_inst.stderr}\n")
        finalize_setup_failure(out_dir, run_id, git_sha, git_dirty, baseline_data, start_time, {"overall_status": "FAIL", "mode": "APPLY", "install_exit_code": install_exit_code})
        sys.exit(1)

    # Validate actual installation report
    try:
        with open(install_report_file, "r", encoding="utf-8") as f:
            inst_report_data = json.load(f)
        inst_valid, inst_violations, inst_summary = verify_report(inst_report_data, baseline_data, allow_no_ops=False)
        inst_summary["run_id"] = run_id
        atomic_write_json(os.path.join(out_dir, "install_verification_report.json"), inst_summary)
        if not inst_valid:
            sys.stderr.write("ERROR: Actual installation report violated protected package baseline! Setup failed.\n")
            finalize_setup_failure(out_dir, run_id, git_sha, git_dirty, baseline_data, start_time, {"overall_status": "FAIL", "mode": "APPLY", "install_exit_code": install_exit_code})
            sys.exit(1)
    except Exception as e:
        sys.stderr.write(f"ERROR: Failed to validate actual install report: {e}\n")
        finalize_setup_failure(out_dir, run_id, git_sha, git_dirty, baseline_data, start_time, {"overall_status": "FAIL", "mode": "APPLY", "install_exit_code": install_exit_code})
        sys.exit(1)

    print("Installing repository editable with --no-deps...")
    editable_cmd = [sys.executable, "-m", "pip", "install", "--no-deps", "-e", REPO_ROOT]
    res_edit = subprocess.run(editable_cmd, capture_output=True, text=True)
    editable_exit_code = res_edit.returncode
    atomic_write_text(os.path.join(out_dir, "editable_stdout.log"), res_edit.stdout)
    atomic_write_text(os.path.join(out_dir, "editable_stderr.log"), res_edit.stderr)

    if editable_exit_code != 0:
        sys.stderr.write(f"ERROR: Editable installation failed:\n{res_edit.stderr}\n")
        finalize_setup_failure(out_dir, run_id, git_sha, git_dirty, baseline_data, start_time, {
            "overall_status": "FAIL", "mode": "APPLY",
            "install_exit_code": install_exit_code, "editable_exit_code": editable_exit_code
        })
        sys.exit(1)

    # Record consistency before running the independent runtime verifier. A
    # dependency failure does not suppress runtime/after-state evidence.
    try:
        pip_after = capture_pip_check(out_dir, "after", run_id, git_sha)
        consistency = compare_checks(pip_before, pip_after,
                                     [p["normalized_name"] for p in inst_summary["packages"]])
        check_code = pip_after["returncode"]
    except (OSError, KeyError, TypeError, ValueError) as exc:
        check_code = None
        consistency = {"status": "FAIL", "state": "INVALID_EVIDENCE", "error": str(exc),
                       "run_id": run_id, "git_sha": git_sha}
    atomic_write_json(os.path.join(out_dir, "pip_consistency.json"), consistency)

    # 6. Post-install verifier (pass common run_id)
    print("Running post-install environment verifier (scripts/verify_kaggle_env.py)...")
    verifier_path = os.path.join(REPO_ROOT, "scripts/verify_kaggle_env.py")
    verify_env_cmd = [sys.executable, verifier_path, "--baseline", baseline_path, "--output-dir", out_dir,
                      "--run-id", run_id, "--pip-baseline", os.path.join(out_dir, "pip_check_before.json"),
                      "--install-report", install_report_file]
    try:
        res_env = subprocess.run(verify_env_cmd, capture_output=True, text=True)
    except OSError as exc:
        res_env = subprocess.CompletedProcess(verify_env_cmd, 1, "", f"Verifier could not start: {exc}")
    verifier_exit_code = res_env.returncode

    atomic_write_text(os.path.join(out_dir, "verifier_stdout.log"), res_env.stdout)
    atomic_write_text(os.path.join(out_dir, "verifier_stderr.log"), res_env.stderr)

    env_report_file = os.path.join(out_dir, "kaggle_env_verification.json")
    v_git_sha = None
    overall_status = None
    v_run_id = None
    runtime_checks_valid = False

    if os.path.exists(env_report_file):
        try:
            with open(env_report_file, "r", encoding="utf-8") as f:
                env_rep = json.load(f)
            v_run_id = env_rep.get("run_id")
            v_git_sha = env_rep.get("git_sha")
            overall_status = env_rep.get("statuses", {}).get("overall")
            runtime_checks_valid = all(env_rep.get("checks", {}).get(name, {}).get("status") == "PASS"
                                       for name in REQUIRED_RUNTIME_CHECKS)
        except Exception:
            pass

    run_id_valid = (v_run_id == run_id)
    git_sha_valid = (v_git_sha == git_sha)
    verifier_passed = ((verifier_exit_code == 0) and (overall_status == "PASS")
                       and run_id_valid and git_sha_valid and runtime_checks_valid)

    v_verdict = "PASS" if verifier_passed else "FAIL"

    create_verifier_execution_record(
        out_dir=out_dir,
        run_id=run_id,
        setup_git_sha=git_sha,
        verifier_git_sha=v_git_sha,
        verifier_returncode=verifier_exit_code,
        verifier_overall_status=overall_status,
        run_id_valid=run_id_valid,
        git_sha_valid=git_sha_valid,
        verdict=v_verdict
    )

    # Single final freeze call and snapshot
    final_freeze_text, final_freeze_code = get_pip_freeze()
    atomic_write_text(os.path.join(out_dir, "after_freeze.log"), final_freeze_text)

    # Write genuine after_manifest.json using captured final_freeze_text
    try:
        after_inventory = capture_distribution_inventory(out_dir, "after", run_id, git_sha, final_freeze_text)
    except (OSError, ValueError) as exc:
        after_inventory = {"status": "FAIL", "packages": {}, "error": str(exc)}
        atomic_write_json(os.path.join(out_dir, "after_distributions.json"), after_inventory)
    after_manifest = create_environment_snapshot("after", run_id, git_sha, git_dirty, baseline_data,
                                                 freeze_text=final_freeze_text,
                                                 installed_inventory=after_inventory["packages"])
    after_manifest["freeze_returncode"] = final_freeze_code
    if final_freeze_code != 0 or after_inventory["status"] != "PASS":
        after_manifest["snapshot_status"] = "FAIL"
    atomic_write_json(os.path.join(out_dir, "after_manifest.json"), after_manifest)
    before_installed = before_inventory["packages"]
    after_installed = after_inventory["packages"]
    transition = compare_environment_transition(before_manifest, after_manifest, before_installed,
                                                after_installed, baseline_data, inst_report_data)
    atomic_write_json(os.path.join(out_dir, "environment_transition.json"), transition)
    substantive = {"pip_consistency": consistency["status"],
                   "verifier": "PASS" if verifier_passed else "FAIL",
                   "after_snapshot": after_manifest.get("snapshot_status"),
                   "environment_transition": transition["status"]}
    if any(value != "PASS" for value in substantive.values()):
        finalize_setup_failure(out_dir, run_id, git_sha, git_dirty, baseline_data, start_time, {
            "overall_status": "FAIL", "mode": "APPLY", "run_id": run_id, "git_sha": git_sha,
            "reason": "Post-install substantive checks failed", "substantive_checks": substantive,
            "install_exit_code": install_exit_code, "editable_exit_code": editable_exit_code,
            "pip_check_exit_code": check_code, "pip_consistency_state": consistency["state"],
            "verifier_exit_code": verifier_exit_code, "final_freeze_exit_code": final_freeze_code
        })
        sys.exit(1)

    # Generate overlay lock only after every substantive check passes.

    report_hashes = {
        "pip_dry_run_report": file_sha256(report_file) if os.path.exists(report_file) else None,
        "pip_install_report": file_sha256(install_report_file) if os.path.exists(install_report_file) else None
    }
    report_hashes["pip_consistency"] = file_sha256(os.path.join(out_dir, "pip_consistency.json"))
    report_hashes["environment_transition"] = file_sha256(os.path.join(out_dir, "environment_transition.json"))

    try:
        lock_dict, lock_text = generate_overlay_lock(
            before_installed=before_installed,
            after_installed=after_installed,
            baseline_data=baseline_data,
            run_id=run_id,
            git_sha=git_sha,
            report_hashes=report_hashes
        )
        atomic_write_json(os.path.join(out_dir, "overlay_lock.json"), lock_dict)
        atomic_write_text(os.path.join(out_dir, "kaggle-overlay.lock"), lock_text)
    except Exception as e:
        sys.stderr.write(f"ERROR: Failed to generate overlay lock: {e}\n")
        finalize_setup_failure(out_dir, run_id, git_sha, git_dirty, baseline_data, start_time, {
            "overall_status": "FAIL", "mode": "APPLY",
            "reason": f"Overlay lock generation failed: {e}"
        })
        sys.exit(1)

    atomic_write_json(os.path.join(out_dir, "setup_status.json"), {
        "overall_status": "PASS", "mode": "APPLY", "run_id": run_id, "git_sha": git_sha,
        "dry_run_exit_code": dry_run_exit_code,
        "install_exit_code": install_exit_code,
        "editable_exit_code": editable_exit_code,
        "pip_check_exit_code": check_code,
        "pip_consistency_state": consistency["state"],
        "substantive_checks": substantive,
        "verifier_exit_code": verifier_exit_code
    })

    generate_manifest(out_dir, run_id, git_sha, "PASS", start_time)
    print(f"\n=== OVERLAY SETUP COMPLETE (RUN ID: {run_id}) ===")
    print(f"Artifacts and overlay lock saved to: {out_dir}")

if __name__ == "__main__":
    main()
