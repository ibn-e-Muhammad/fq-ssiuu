"""
test_kaggle_dependency_policy.py — Pure infrastructure unit tests for Kaggle dependency policy.

Uses standard library unittest only.
Does NOT import torch, transformers, or scientific modules.
"""

import os
import sys
import json
import re
import tempfile
import unittest
import subprocess

from scripts.verify_pip_report import verify_report, normalize_name

class TestKaggleDependencyPolicy(unittest.TestCase):

    def setUp(self):
        self.baseline_path = "requirements/kaggle-native-baseline.json"
        self.overlay_path = "requirements/kaggle-overlay.in"
        with open(self.baseline_path, "r", encoding="utf-8") as f:
            self.baseline_data = json.load(f)

        # Create a mock report file for fast offline testing
        self.mock_report = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json")
        json.dump({
            "install": [
                {"metadata": {"name": "bitsandbytes", "version": "0.45.0"}},
                {"metadata": {"name": "captum", "version": "0.7.0"}}
            ]
        }, self.mock_report)
        self.mock_report.close()

    def tearDown(self):
        if os.path.exists(self.mock_report.name):
            os.remove(self.mock_report.name)

    def test_normalized_names(self):
        self.assertEqual(normalize_name("huggingface_hub"), "huggingface-hub")
        self.assertEqual(normalize_name("HuggingFace-Hub"), "huggingface-hub")
        self.assertEqual(normalize_name("hydra_core"), "hydra-core")
        self.assertEqual(normalize_name("bitsandbytes"), "bitsandbytes")

    def test_allowed_new_overlay_package_accepted(self):
        report = {
            "install": [
                {"metadata": {"name": "bitsandbytes", "version": "0.45.0"}},
                {"metadata": {"name": "captum", "version": "0.7.0"}}
            ]
        }
        is_valid, violations, summary = verify_report(report, self.baseline_data)
        self.assertTrue(is_valid)
        self.assertEqual(len(violations), 0)
        self.assertEqual(summary["status"], "PASS")

    def test_protected_package_different_version_rejected(self):
        report = {
            "install": [
                {"metadata": {"name": "torch", "version": "2.5.0"}}
            ]
        }
        is_valid, violations, summary = verify_report(report, self.baseline_data)
        self.assertFalse(is_valid)
        self.assertGreater(len(violations), 0)
        self.assertIn("PROTECTED PACKAGE VIOLATION", violations[0])

    def test_protected_package_same_version_strict_rejection(self):
        report = {
            "install": [
                {"metadata": {"name": "torch", "version": "2.10.0+cu128"}}
            ]
        }
        is_valid, violations, summary = verify_report(report, self.baseline_data, allow_no_ops=False)
        self.assertFalse(is_valid)
        self.assertGreater(len(violations), 0)
        self.assertIn("PROTECTED PACKAGE PROPOSAL", violations[0])

    def test_native_stability_package_change_rejected(self):
        report = {
            "install": [
                {"metadata": {"name": "transformers", "version": "4.46.0"}}
            ]
        }
        is_valid, violations, summary = verify_report(report, self.baseline_data)
        self.assertFalse(is_valid)
        self.assertGreater(len(violations), 0)
        self.assertIn("PROTECTED PACKAGE VIOLATION", violations[0])

    def test_malformed_pip_report_failure(self):
        malformed_report = {"install": "not_a_list"}
        is_valid, violations, summary = verify_report(malformed_report, self.baseline_data)
        self.assertFalse(is_valid)
        self.assertIn("Malformed pip report", violations[0])

    def test_missing_baseline_failure(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = "."
        res = subprocess.run(
            [sys.executable, "scripts/verify_pip_report.py", "--report", self.mock_report.name, "--baseline", "non_existent_file.json"],
            capture_output=True, text=True, env=env
        )
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("Baseline file not found", res.stderr)

    def test_overlay_file_contains_no_protected_or_native_stability_packages(self):
        hard_protected = {normalize_name(k) for k in self.baseline_data.get("hard_protected_compute_stack", {}).keys()}
        native_stability = {normalize_name(k) for k in self.baseline_data.get("native_stability_stack", {}).keys()}
        all_protected = hard_protected | native_stability

        with open(self.overlay_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        overlay_pkgs = set()
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            pkg = re.split(r"[<>=!~]", line)[0].strip()
            overlay_pkgs.add(normalize_name(pkg))

        forbidden = overlay_pkgs & all_protected
        self.assertEqual(forbidden, set(), f"Overlay file contains forbidden protected packages: {forbidden}")

    def test_single_device_requirement_enforced_by_setup(self):
        # Run kaggle_setup.py without CUDA_VISIBLE_DEVICES=0 -> must fail exit code 1
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0,1"
        env["PYTHONPATH"] = "."
        res = subprocess.run(
            [sys.executable, "scripts/kaggle_setup.py", "--preview", "--report-file", self.mock_report.name],
            capture_output=True, text=True, env=env
        )
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("CUDA_VISIBLE_DEVICES must be set to '0'", res.stderr)

    def test_preview_is_default_and_apply_requires_explicit_flag(self):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["PYTHONPATH"] = "."
        # Run kaggle_setup.py with no flags -> default to PREVIEW
        res = subprocess.run(
            [sys.executable, "scripts/kaggle_setup.py", "--report-file", self.mock_report.name],
            capture_output=True, text=True, env=env
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("MODE: PREVIEW", res.stdout)
        self.assertIn("[PREVIEW MODE COMPLETE]", res.stdout)

if __name__ == "__main__":
    unittest.main()
