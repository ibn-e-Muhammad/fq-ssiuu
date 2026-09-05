#!/usr/bin/env python3
"""
verify_pip_report.py — Fail-closed verification guard for pip dry-run and install reports.

Validates a pip --report JSON file against requirements/kaggle-native-baseline.json.
Rejects any proposed installation, replacement, upgrade, downgrade, re-installation,
or same-version modification involving hard-protected or native-stability packages.

Uses standard library only.
"""

import os
import sys
import json
import re
import argparse
import tempfile

def normalize_name(name: str) -> str:
    """Normalize distribution name per PEP 503 / packaging rules."""
    return re.sub(r"[-_.]+", "-", name).lower()

def atomic_write_json(filepath: str, data: dict, indent: int = 2) -> None:
    """Atomically write JSON data to file to prevent partial writes."""
    dir_name = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(dir_name, exist_ok=True)
    temp_file = tempfile.NamedTemporaryFile(mode="w", dir=dir_name, delete=False, encoding="utf-8")
    try:
        json.dump(data, temp_file, indent=indent)
        temp_file.flush()
        os.fsync(temp_file.fileno())
        temp_file.close()
        os.replace(temp_file.name, filepath)
    except Exception:
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)
        raise

def validate_baseline(baseline_data: dict) -> tuple[bool, list[str]]:
    """Strictly validates baseline dictionary structure and required keys."""
    errors = []
    if not isinstance(baseline_data, dict):
        return False, ["Baseline data must be a dictionary."]

    hard_stack = baseline_data.get("hard_protected_compute_stack")
    if not isinstance(hard_stack, dict) or not hard_stack:
        errors.append("Baseline must contain a non-empty 'hard_protected_compute_stack' dictionary.")
    else:
        for k, v in hard_stack.items():
            if not isinstance(k, str) or not isinstance(v, str) or not k.strip() or not v.strip():
                errors.append(f"Invalid entry in hard_protected_compute_stack: {k}={v}")

    native_stack = baseline_data.get("native_stability_stack")
    if not isinstance(native_stack, dict) or not native_stack:
        errors.append("Baseline must contain a non-empty 'native_stability_stack' dictionary.")
    else:
        for k, v in native_stack.items():
            if not isinstance(k, str) or not isinstance(v, str) or not k.strip() or not v.strip():
                errors.append(f"Invalid entry in native_stability_stack: {k}={v}")

    if hard_stack and isinstance(hard_stack, dict):
        norm_hard = {normalize_name(k) for k in hard_stack.keys()}
        required = {"torch", "torchvision", "torchaudio", "triton"}
        missing_req = required - norm_hard
        if missing_req:
            errors.append(f"Baseline hard_protected_compute_stack missing required packages: {sorted(list(missing_req))}")

    return len(errors) == 0, errors

def verify_report(report_data: dict, baseline_data: dict, allow_no_ops: bool = False) -> tuple[bool, list[str], dict]:
    """
    Verifies pip report against baseline data.
    By default (allow_no_ops=False), fail closed and reject any protected package proposal,
    including same-version re-installations.
    Returns (is_valid, violation_list, summary_dict).
    """
    violations = []
    processed_pkgs = []

    b_valid, b_errors = validate_baseline(baseline_data)
    if not b_valid:
        summary = {
            "status": "FAIL",
            "total_proposed_packages": 0,
            "violations_count": len(b_errors),
            "violations": b_errors,
            "packages": []
        }
        return False, b_errors, summary

    hard_protected = {normalize_name(k): v for k, v in baseline_data["hard_protected_compute_stack"].items()}
    native_stability = {normalize_name(k): v for k, v in baseline_data["native_stability_stack"].items()}
    all_protected = {**hard_protected, **native_stability}

    if not isinstance(report_data, dict):
        violations.append("Malformed pip report: report data must be a dictionary.")
        summary = {"status": "FAIL", "total_proposed_packages": 0, "violations_count": 1, "violations": violations, "packages": []}
        return False, violations, summary

    if "install" not in report_data:
        violations.append("Malformed pip report: missing 'install' field.")
        summary = {"status": "FAIL", "total_proposed_packages": 0, "violations_count": 1, "violations": violations, "packages": []}
        return False, violations, summary

    install_list = report_data.get("install")
    if not isinstance(install_list, list):
        violations.append("Malformed pip report: 'install' field must be a list.")
        summary = {"status": "FAIL", "total_proposed_packages": 0, "violations_count": 1, "violations": violations, "packages": []}
        return False, violations, summary

    for item in install_list:
        if not isinstance(item, dict):
            violations.append("Malformed pip report: item in 'install' is not a dictionary.")
            continue

        metadata = item.get("metadata")
        raw_name = None
        raw_version = None

        if isinstance(metadata, dict):
            raw_name = metadata.get("name")
            raw_version = metadata.get("version")

        if not raw_name:
            raw_name = item.get("name")
        if not raw_version:
            raw_version = item.get("version")

        if not raw_name or not raw_version or not isinstance(raw_name, str) or not isinstance(raw_version, str) or not raw_name.strip() or not raw_version.strip():
            violations.append("Malformed pip report: missing or empty package name or version.")
            continue

        raw_name = raw_name.strip()
        raw_version = raw_version.strip()
        norm_name = normalize_name(raw_name)
        processed_pkgs.append({"name": raw_name, "normalized_name": norm_name, "version": raw_version})

        if norm_name in all_protected:
            expected_ver = all_protected[norm_name]
            if raw_version != expected_ver:
                violations.append(
                    f"PROTECTED PACKAGE VERSION MISMATCH: Proposed version '{raw_version}' for protected package '{raw_name}' "
                    f"conflicts with baseline version '{expected_ver}'."
                )
            else:
                if not allow_no_ops:
                    violations.append(
                        f"PROTECTED PACKAGE SAME-VERSION REJECTION: Proposed operation for protected package '{raw_name}' "
                        f"(version '{raw_version}') is strictly rejected by fail-closed policy."
                    )

    is_valid = (len(violations) == 0)
    summary = {
        "status": "PASS" if is_valid else "FAIL",
        "total_proposed_packages": len(install_list),
        "violations_count": len(violations),
        "violations": violations,
        "packages": processed_pkgs
    }
    return is_valid, violations, summary

def main():
    parser = argparse.ArgumentParser(description="Verify pip --report JSON against Kaggle baseline policy (fail-closed).")
    parser.add_argument("--report", required=True, help="Path to pip --report JSON file.")
    parser.add_argument("--baseline", default="requirements/kaggle-native-baseline.json", help="Path to baseline JSON file.")
    parser.add_argument("--output-json", help="Optional path to write JSON validation result.")

    args = parser.parse_args()

    summary = None

    if not os.path.exists(args.baseline):
        err_msg = f"Baseline file not found: {args.baseline}"
        sys.stderr.write(f"ERROR: {err_msg}\n")
        summary = {"status": "FAIL", "violations_count": 1, "violations": [err_msg], "packages": []}
        if args.output_json:
            atomic_write_json(args.output_json, summary)
        sys.exit(1)

    if not os.path.exists(args.report):
        err_msg = f"Pip report file not found: {args.report}"
        sys.stderr.write(f"ERROR: {err_msg}\n")
        summary = {"status": "FAIL", "violations_count": 1, "violations": [err_msg], "packages": []}
        if args.output_json:
            atomic_write_json(args.output_json, summary)
        sys.exit(1)

    try:
        with open(args.baseline, "r", encoding="utf-8") as f:
            baseline_data = json.load(f)
    except Exception as e:
        err_msg = f"Failed to parse baseline JSON: {e}"
        sys.stderr.write(f"ERROR: {err_msg}\n")
        summary = {"status": "FAIL", "violations_count": 1, "violations": [err_msg], "packages": []}
        if args.output_json:
            atomic_write_json(args.output_json, summary)
        sys.exit(1)

    is_valid, violations, summary = verify_report(report_data, baseline_data, allow_no_ops=False)

    if args.output_json:
        atomic_write_json(args.output_json, summary)

    print(f"=== PIP REPORT VERIFICATION: {summary['status']} ===")
    print(f"Total proposed operations: {summary['total_proposed_packages']}")
    if violations:
        print(f"VIOLATIONS DETECTED ({len(violations)}):")
        for v in violations:
            print(f"  - {v}")
        sys.exit(1)
    else:
        print("Verification passed cleanly: No protected or native-stability packages modified.")
        sys.exit(0)

if __name__ == "__main__":
    main()
