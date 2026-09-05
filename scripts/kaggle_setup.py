#!/usr/bin/env python3
"""
kaggle_setup.py — Orchestration script for Kaggle overlay dependency installation.

Preview is the default mode. Requires explicit --apply flag for installation.
Refuses to operate unless CUDA_VISIBLE_DEVICES=0.
Validates dry-run pip report against requirements/kaggle-native-baseline.json.
Installs repository editable with --no-deps only after successful validation.
Never edits .bashrc, runs pip uninstall, or downloads models/datasets.
"""

import os
import sys
import json
import uuid
import subprocess
import argparse
from datetime import datetime, timezone

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from scripts.verify_pip_report import verify_report
except ImportError:
    from verify_pip_report import verify_report

def get_pip_freeze() -> str:
    res = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    return res.stdout

def run_pip_check() -> tuple[int, str]:
    res = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    return res.returncode, res.stdout + res.stderr

def main():
    parser = argparse.ArgumentParser(description="Kaggle Environment Overlay Setup")
    parser.add_argument("--apply", action="store_true", help="Explicitly apply the overlay installation. Defaults to PREVIEW if omitted.")
    parser.add_argument("--preview", action="store_true", help="Run in preview mode (dry-run report validation only).")
    parser.add_argument("--overlay", default="requirements/kaggle-overlay.in", help="Path to kaggle-overlay.in file.")
    parser.add_argument("--baseline", default="requirements/kaggle-native-baseline.json", help="Path to baseline JSON file.")
    parser.add_argument("--report-file", help="Optional path to existing/pre-generated pip report JSON file for offline/fast preview.")
    parser.add_argument("--output-dir", help="Optional override directory for setup artifacts.")

    args = parser.parse_args()

    mode = "APPLY" if args.apply else "PREVIEW"
    print(f"=== KAGGLE OVERLAY SETUP (MODE: {mode}) ===")

    # 1. Single-device CUDA environment check
    cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_vis != "0":
        sys.stderr.write(
            f"ERROR: CUDA_VISIBLE_DEVICES must be set to '0' (single-device enforcement). "
            f"Observed: '{cuda_vis}'\n"
        )
        sys.exit(1)

    run_id = str(uuid.uuid4())
    out_dir = args.output_dir or f"results/kaggle_setup_{run_id}"
    os.makedirs(out_dir, exist_ok=True)

    # 2. Before-environment manifest
    before_freeze = get_pip_freeze()
    with open(os.path.join(out_dir, "before_freeze.log"), "w", encoding="utf-8") as f:
        f.write(before_freeze)

    # 3. Dry-run pip report generation
    if args.report_file:
        report_file = args.report_file
        print(f"Using pre-generated pip report: {report_file}")
    else:
        report_file = os.path.join(out_dir, "pip_dry_run_report.json")
        dry_run_cmd = [
            sys.executable, "-m", "pip", "install",
            "--dry-run",
            "--report", report_file,
            "-r", args.overlay
        ]
        print(f"Generating dry-run pip report against {args.overlay}...")
        res = subprocess.run(dry_run_cmd, capture_output=True, text=True)
        if res.returncode != 0 or not os.path.exists(report_file):
            sys.stderr.write(f"ERROR: Failed to generate dry-run pip report:\n{res.stderr}\n")
            sys.exit(1)

    # 4. Validate pip dry-run report
    with open(args.baseline, "r", encoding="utf-8") as f:
        baseline_data = json.load(f)
    with open(report_file, "r", encoding="utf-8") as f:
        report_data = json.load(f)

    is_valid, violations, summary = verify_report(report_data, baseline_data)
    with open(os.path.join(out_dir, "verification_report.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if not is_valid:
        sys.stderr.write("ERROR: Dry-run pip report validation failed! Stopped setup.\n")
        for v in violations:
            sys.stderr.write(f"  - {v}\n")
        sys.exit(1)

    print("Dry-run pip report validation PASSED cleanly.")

    if not args.apply:
        print("\n[PREVIEW MODE COMPLETE] No packages were installed or modified.")
        print(f"Artifacts saved to: {out_dir}")
        sys.exit(0)

    # 5. Apply overlay installation (Only when --apply is explicitly specified)
    print("\nApplying overlay installation...")
    install_cmd = [sys.executable, "-m", "pip", "install", "-r", args.overlay]
    res_inst = subprocess.run(install_cmd, capture_output=True, text=True)
    if res_inst.returncode != 0:
        sys.stderr.write(f"ERROR: Overlay installation failed:\n{res_inst.stderr}\n")
        sys.exit(1)

    print("Installing repository editable with --no-deps...")
    editable_cmd = [sys.executable, "-m", "pip", "install", "--no-deps", "-e", "."]
    res_edit = subprocess.run(editable_cmd, capture_output=True, text=True)
    if res_edit.returncode != 0:
        sys.stderr.write(f"ERROR: Editable installation failed:\n{res_edit.stderr}\n")
        sys.exit(1)

    # 6. After-environment manifest & pip check
    after_freeze = get_pip_freeze()
    with open(os.path.join(out_dir, "after_freeze.log"), "w", encoding="utf-8") as f:
        f.write(after_freeze)

    check_code, check_output = run_pip_check()
    with open(os.path.join(out_dir, "pip_check.log"), "w", encoding="utf-8") as f:
        f.write(check_output)

    if check_code != 0:
        sys.stderr.write(f"ERROR: pip check failed after installation:\n{check_output}\n")
        sys.exit(1)

    # 7. Post-install verifier
    print("Running post-install environment verifier (scripts/verify_kaggle_env.py)...")
    verify_env_cmd = [sys.executable, "scripts/verify_kaggle_env.py", "--output-dir", out_dir]
    res_env = subprocess.run(verify_env_cmd, capture_output=True, text=True)

    # 8. Generate overlay-only lock
    overlay_lock = os.path.join(out_dir, "kaggle-overlay.lock")
    with open(overlay_lock, "w", encoding="utf-8") as f:
        f.write(f"# FQ-SSIUU Overlay Lock (Generated: {datetime.now(timezone.utc).isoformat()})\n")
        f.write(f"# Run ID: {run_id}\n")
        # Record only installed packages that were in overlay
        for line in after_freeze.splitlines():
            pkg_name = line.split("==")[0].lower()
            if pkg_name in ["bitsandbytes", "captum", "hydra-core", "lm-eval", "jsonlines"]:
                f.write(line + "\n")

    print(f"\n=== OVERLAY SETUP COMPLETE (RUN ID: {run_id}) ===")
    print(f"Artifacts and overlay lock saved to: {out_dir}")

if __name__ == "__main__":
    main()
