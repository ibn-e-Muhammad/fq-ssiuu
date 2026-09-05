#!/usr/bin/env python3
"""
verify_pip_report.py — Non-mutating verification guard for pip dry-run reports.

Validates a pip --report JSON file against requirements/kaggle-native-baseline.json.
Rejects any proposed installation, replacement, upgrade, downgrade, or modification
involving hard-protected or native-stability packages.

Uses standard library only.
"""

import os
import sys
import json
import re
import argparse

def normalize_name(name: str) -> str:
    """Normalize distribution name per PEP 503 / packaging rules."""
    return re.sub(r"[-_.]+", "-", name).lower()

def verify_report(report_data: dict, baseline_data: dict, allow_no_ops: bool = True) -> tuple[bool, list[str], dict]:
    """
    Verifies pip report against baseline data.
    Returns (is_valid, violation_list, summary_dict).
    """
    violations = []
    processed_pkgs = []

    # Extract protected stacks
    hard_protected = {normalize_name(k): v for k, v in baseline_data.get("hard_protected_compute_stack", {}).items()}
    native_stability = {normalize_name(k): v for k, v in baseline_data.get("native_stability_stack", {}).items()}
    all_protected = {**hard_protected, **native_stability}

    install_list = report_data.get("install", [])
    if not isinstance(install_list, list):
        return False, ["Malformed pip report: 'install' field must be a list."], {}

    for item in install_list:
        if not isinstance(item, dict):
            return False, ["Malformed pip report: item in 'install' is not a dictionary."], {}

        # Extract metadata
        metadata = item.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        raw_name = metadata.get("name") or item.get("name")
        raw_version = metadata.get("version") or item.get("version")

        if not raw_name or not raw_version:
            return False, ["Malformed pip report: missing package name or version."], {}

        norm_name = normalize_name(raw_name)
        processed_pkgs.append({"name": raw_name, "normalized_name": norm_name, "version": raw_version})

        if norm_name in all_protected:
            expected_ver = all_protected[norm_name]
            if raw_version != expected_ver:
                violations.append(
                    f"PROTECTED PACKAGE VIOLATION: Proposed change for '{raw_name}' version '{raw_version}' "
                    f"conflicts with baseline protected version '{expected_ver}'."
                )
            elif not allow_no_ops:
                violations.append(
                    f"PROTECTED PACKAGE PROPOSAL: Proposed re-installation/operation for protected package '{raw_name}' "
                    f"(version '{raw_version}') is rejected under strict policy."
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
    parser = argparse.ArgumentParser(description="Verify pip --report JSON against Kaggle baseline policy.")
    parser.add_argument("--report", required=True, help="Path to pip --report JSON file.")
    parser.add_argument("--baseline", default="requirements/kaggle-native-baseline.json", help="Path to baseline JSON file.")
    parser.add_argument("--output-json", help="Optional path to write JSON validation result.")
    parser.add_argument("--strict-no-ops", action="store_true", help="Reject even no-op same-version proposals for protected packages.")

    args = parser.parse_args()

    if not os.path.exists(args.baseline):
        sys.stderr.write(f"ERROR: Baseline file not found: {args.baseline}\n")
        sys.exit(1)

    if not os.path.exists(args.report):
        sys.stderr.write(f"ERROR: Pip report file not found: {args.report}\n")
        sys.exit(1)

    try:
        with open(args.baseline, "r", encoding="utf-8") as f:
            baseline_data = json.load(f)
    except Exception as e:
        sys.stderr.write(f"ERROR: Failed to parse baseline JSON: {e}\n")
        sys.exit(1)

    try:
        with open(args.report, "r", encoding="utf-8") as f:
            report_data = json.load(f)
    except Exception as e:
        sys.stderr.write(f"ERROR: Failed to parse pip report JSON: {e}\n")
        sys.exit(1)

    is_valid, violations, summary = verify_report(report_data, baseline_data, allow_no_ops=not args.strict_no_ops)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

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
