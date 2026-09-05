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
- Captures structured before-environment manifest and validates baseline before generating pip report.
- Validates dry-run pip report and actual installation report against requirements/kaggle-native-baseline.json.
- Installs repository editable with --no-deps only after successful validation.
- Runs post-install verifier (scripts/verify_kaggle_env.py) and aborts on any failure.
- Generates overlay lock only after all checks pass.
- Creates atomic failure artifacts even when setup fails.
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
from datetime import datetime, timezone

# Ensure project root is in sys.path and resolve paths relative to REPO_ROOT
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from scripts.verify_pip_report import verify_report, normalize_name, atomic_write_json
except ImportError:
    from verify_pip_report import verify_report, normalize_name, atomic_write_json

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

def get_git_info() -> tuple[str, bool]:
    """Returns (git_commit_sha, is_dirty)."""
    try:
        sha_res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT)
        sha = sha_res.stdout.strip() if sha_res.returncode == 0 else "UNKNOWN"
        status_res = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=REPO_ROOT)
        is_dirty = bool(status_res.stdout.strip()) if status_res.returncode == 0 else False
        return sha, is_dirty
    except Exception:
        return "UNKNOWN", False

def get_pip_freeze() -> str:
    res = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    return res.stdout

def run_pip_check() -> tuple[int, str]:
    res = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    return res.returncode, res.stdout + res.stderr

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

    git_sha, git_dirty = get_git_info()
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

    # 2. Before-environment verification & manifest
    if not os.path.exists(baseline_path):
        sys.stderr.write(f"ERROR: Baseline file not found: {baseline_path}\n")
        generate_manifest(out_dir, run_id, git_sha, "FAIL", start_time)
        sys.exit(1)

    try:
        with open(baseline_path, "r", encoding="utf-8") as f:
            baseline_data = json.load(f)
    except Exception as e:
        sys.stderr.write(f"ERROR: Failed to load baseline JSON: {e}\n")
        generate_manifest(out_dir, run_id, git_sha, "FAIL", start_time)
        sys.exit(1)

    before_freeze = get_pip_freeze()
    atomic_write_text(os.path.join(out_dir, "before_freeze.log"), before_freeze)

    # Compare before-environment against baseline protected stacks
    before_checks = {}
    before_valid = True
    hard_stack = baseline_data.get("hard_protected_compute_stack", {})
    native_stack = baseline_data.get("native_stability_stack", {})

    installed_before = {}
    for line in before_freeze.splitlines():
        line = line.strip()
        if "==" in line:
            parts = line.split("==")
            installed_before[normalize_name(parts[0])] = parts[1]

    all_protected = {**hard_stack, **native_stack}
    for pkg_raw, exp_ver in all_protected.items():
        norm_p = normalize_name(pkg_raw)
        actual_ver = installed_before.get(norm_p)
        if not actual_ver:
            try:
                actual_ver = importlib.metadata.version(pkg_raw)
            except Exception:
                actual_ver = None

        is_match = (actual_ver == exp_ver)
        before_checks[norm_p] = {
            "name": pkg_raw,
            "expected_version": exp_ver,
            "actual_version": actual_ver,
            "match": is_match
        }
        if not is_match:
            before_valid = False
            sys.stderr.write(f"BEFORE-INSTALL BASELINE MISMATCH: '{pkg_raw}' expected '{exp_ver}', observed '{actual_ver}'\n")

    before_manifest = {
        "run_id": run_id,
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_sha,
        "git_dirty": git_dirty,
        "python_version": sys.version,
        "cuda_visible_devices": cuda_vis,
        "baseline_check_passed": before_valid,
        "protected_packages": before_checks
    }
    atomic_write_json(os.path.join(out_dir, "before_manifest.json"), before_manifest)

    if not before_valid:
        sys.stderr.write("ERROR: Environment before-manifest failed baseline validation! Setup stopped.\n")
        generate_manifest(out_dir, run_id, git_sha, "FAIL", start_time)
        sys.exit(1)

    # 3. Dry-run pip report generation/validation
    if args.report_file:
        report_file = args.report_file
        print(f"Using pre-generated pip report: {report_file}")
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
        atomic_write_text(os.path.join(out_dir, "dry_run_stdout.log"), res_dry.stdout)
        atomic_write_text(os.path.join(out_dir, "dry_run_stderr.log"), res_dry.stderr)

        if res_dry.returncode != 0 or not os.path.exists(report_file):
            sys.stderr.write(f"ERROR: Failed to generate dry-run pip report:\n{res_dry.stderr}\n")
            generate_manifest(out_dir, run_id, git_sha, "FAIL", start_time)
            sys.exit(1)

    try:
        with open(report_file, "r", encoding="utf-8") as f:
            report_data = json.load(f)
    except Exception as e:
        sys.stderr.write(f"ERROR: Failed to load pip report JSON: {e}\n")
        generate_manifest(out_dir, run_id, git_sha, "FAIL", start_time)
        sys.exit(1)

    is_valid, violations, summary = verify_report(report_data, baseline_data, allow_no_ops=False)
    atomic_write_json(os.path.join(out_dir, "verification_report.json"), summary)

    if not is_valid:
        sys.stderr.write("ERROR: Dry-run pip report validation failed! Stopped setup.\n")
        for v in violations:
            sys.stderr.write(f"  - {v}\n")
        generate_manifest(out_dir, run_id, git_sha, "FAIL", start_time)
        sys.exit(1)

    print("Dry-run pip report validation PASSED cleanly.")

    if not args.apply:
        print("\n[PREVIEW MODE COMPLETE] No packages were installed or modified.")
        manifest = generate_manifest(out_dir, run_id, git_sha, "PASS", start_time)
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
    atomic_write_text(os.path.join(out_dir, "install_stdout.log"), res_inst.stdout)
    atomic_write_text(os.path.join(out_dir, "install_stderr.log"), res_inst.stderr)

    if res_inst.returncode != 0:
        sys.stderr.write(f"ERROR: Overlay installation failed:\n{res_inst.stderr}\n")
        generate_manifest(out_dir, run_id, git_sha, "FAIL", start_time)
        sys.exit(1)

    # Validate actual installation report
    if os.path.exists(install_report_file):
        try:
            with open(install_report_file, "r", encoding="utf-8") as f:
                inst_report_data = json.load(f)
            inst_valid, inst_violations, inst_summary = verify_report(inst_report_data, baseline_data, allow_no_ops=False)
            atomic_write_json(os.path.join(out_dir, "install_verification_report.json"), inst_summary)
            if not inst_valid:
                sys.stderr.write("ERROR: Actual installation report violated protected package baseline! Setup failed.\n")
                generate_manifest(out_dir, run_id, git_sha, "FAIL", start_time)
                sys.exit(1)
        except Exception as e:
            sys.stderr.write(f"ERROR: Failed to validate actual install report: {e}\n")
            generate_manifest(out_dir, run_id, git_sha, "FAIL", start_time)
            sys.exit(1)

    print("Installing repository editable with --no-deps...")
    editable_cmd = [sys.executable, "-m", "pip", "install", "--no-deps", "-e", REPO_ROOT]
    res_edit = subprocess.run(editable_cmd, capture_output=True, text=True)
    atomic_write_text(os.path.join(out_dir, "editable_stdout.log"), res_edit.stdout)
    atomic_write_text(os.path.join(out_dir, "editable_stderr.log"), res_edit.stderr)

    if res_edit.returncode != 0:
        sys.stderr.write(f"ERROR: Editable installation failed:\n{res_edit.stderr}\n")
        generate_manifest(out_dir, run_id, git_sha, "FAIL", start_time)
        sys.exit(1)

    # 5. After-environment manifest & pip check
    after_freeze = get_pip_freeze()
    atomic_write_text(os.path.join(out_dir, "after_freeze.log"), after_freeze)

    check_code, check_output = run_pip_check()
    atomic_write_text(os.path.join(out_dir, "pip_check.log"), check_output)

    if check_code != 0:
        sys.stderr.write(f"ERROR: pip check failed after installation:\n{check_output}\n")
        generate_manifest(out_dir, run_id, git_sha, "FAIL", start_time)
        sys.exit(1)

    # 6. Post-install verifier
    print("Running post-install environment verifier (scripts/verify_kaggle_env.py)...")
    verifier_path = os.path.join(REPO_ROOT, "scripts/verify_kaggle_env.py")
    verify_env_cmd = [sys.executable, verifier_path, "--output-dir", out_dir]
    res_env = subprocess.run(verify_env_cmd, capture_output=True, text=True)

    if res_env.returncode != 0:
        sys.stderr.write(f"ERROR: Post-install verifier exited non-zero ({res_env.returncode}). Setup failed.\n")
        generate_manifest(out_dir, run_id, git_sha, "FAIL", start_time)
        sys.exit(1)

    # Verify report overall status is PASS
    env_report_file = os.path.join(out_dir, "kaggle_env_verification.json")
    if not os.path.exists(env_report_file):
        sys.stderr.write("ERROR: Post-install verification report missing. Setup failed.\n")
        generate_manifest(out_dir, run_id, git_sha, "FAIL", start_time)
        sys.exit(1)

    try:
        with open(env_report_file, "r", encoding="utf-8") as f:
            env_rep = json.load(f)
        overall_status = env_rep.get("statuses", {}).get("overall")
        if overall_status != "PASS":
            sys.stderr.write(f"ERROR: Post-install verifier reported '{overall_status}' (expected 'PASS'). Setup failed.\n")
            generate_manifest(out_dir, run_id, git_sha, "FAIL", start_time)
            sys.exit(1)
    except Exception as e:
        sys.stderr.write(f"ERROR: Failed to parse post-install verification report: {e}\n")
        generate_manifest(out_dir, run_id, git_sha, "FAIL", start_time)
        sys.exit(1)

    # 7. Generate overlay lock (ONLY after all checks pass)
    installed_after = {}
    for line in after_freeze.splitlines():
        line = line.strip()
        if "==" in line:
            parts = line.split("==")
            installed_after[normalize_name(parts[0])] = parts[1]

    newly_installed = {}
    for pkg_norm, ver in installed_after.items():
        if pkg_norm not in installed_before:
            newly_installed[pkg_norm] = ver

    overlay_lock_content = [
        f"# FQ-SSIUU Overlay Lock (Generated: {datetime.now(timezone.utc).isoformat()})",
        f"# Run ID: {run_id}",
        f"# Git SHA: {git_sha}",
        "# Direct and transitive overlay packages:"
    ]
    for pkg_norm in sorted(newly_installed.keys()):
        overlay_lock_content.append(f"{pkg_norm}=={newly_installed[pkg_norm]}")

    atomic_write_text(os.path.join(out_dir, "kaggle-overlay.lock"), "\n".join(overlay_lock_content) + "\n")

    lock_dict = {
        "schema_version": "1.0",
        "run_id": run_id,
        "git_sha": git_sha,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "newly_installed_overlay_packages": newly_installed
    }
    atomic_write_json(os.path.join(out_dir, "overlay_lock.json"), lock_dict)

    generate_manifest(out_dir, run_id, git_sha, "PASS", start_time)
    print(f"\n=== OVERLAY SETUP COMPLETE (RUN ID: {run_id}) ===")
    print(f"Artifacts and overlay lock saved to: {out_dir}")

if __name__ == "__main__":
    main()
