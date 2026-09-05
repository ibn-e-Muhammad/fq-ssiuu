"""
test_kaggle_dependency_policy.py — Pure infrastructure unit tests for Kaggle dependency policy.

Uses standard library unittest only.
Does NOT import torch, transformers, or scientific modules.
"""

import os
import sys
import json
import re
import shutil
import tempfile
import unittest
import subprocess

from scripts.verify_pip_report import verify_report, normalize_name, validate_baseline
from scripts.verify_kaggle_env import check_env

class TestKaggleDependencyPolicy(unittest.TestCase):

    def setUp(self):
        self.baseline_path = "requirements/kaggle-native-baseline.json"
        self.overlay_path = "requirements/kaggle-overlay.in"
        with open(self.baseline_path, "r", encoding="utf-8") as f:
            self.baseline_data = json.load(f)

        self.temp_dir = tempfile.mkdtemp()
        self.mock_report_path = os.path.join(self.temp_dir, "mock_report.json")
        with open(self.mock_report_path, "w", encoding="utf-8") as f:
            json.dump({
                "install": [
                    {"metadata": {"name": "bitsandbytes", "version": "0.45.0"}},
                    {"metadata": {"name": "captum", "version": "0.7.0"}}
                ]
            }, f)

        # Create a mock baseline matching mock freeze for CLI preview tests
        local_hard = {k: "1.0.0" for k in self.baseline_data.get("hard_protected_compute_stack", {})}
        local_native = {k: "1.0.0" for k in self.baseline_data.get("native_stability_stack", {})}

        self.mock_baseline_data = {
            "schema_version": "1.0",
            "python_version": sys.version.split()[0],
            "hard_protected_compute_stack": local_hard,
            "native_stability_stack": local_native
        }
        self.mock_baseline_path = os.path.join(self.temp_dir, "mock_baseline.json")
        with open(self.mock_baseline_path, "w", encoding="utf-8") as f:
            json.dump(self.mock_baseline_data, f)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    # 1. Same-version hard-protected proposal rejected by default
    def test_01_same_version_hard_protected_rejected(self):
        report = {"install": [{"metadata": {"name": "torch", "version": "2.10.0+cu128"}}]}
        is_valid, violations, summary = verify_report(report, self.baseline_data)
        self.assertFalse(is_valid)
        self.assertGreater(len(violations), 0)
        self.assertIn("PROTECTED PACKAGE SAME-VERSION REJECTION", violations[0])

    # 2. Same-version native-stability proposal rejected by default
    def test_02_same_version_native_stability_rejected(self):
        report = {"install": [{"metadata": {"name": "transformers", "version": "5.0.0"}}]}
        is_valid, violations, summary = verify_report(report, self.baseline_data)
        self.assertFalse(is_valid)
        self.assertGreater(len(violations), 0)
        self.assertIn("PROTECTED PACKAGE SAME-VERSION REJECTION", violations[0])

    # 3. Missing protected stack rejected
    def test_03_missing_protected_stack_rejected(self):
        bad_baseline = dict(self.baseline_data)
        del bad_baseline["hard_protected_compute_stack"]
        is_valid, errors = validate_baseline(bad_baseline)
        self.assertFalse(is_valid)
        self.assertTrue(any("hard_protected_compute_stack" in e for e in errors))

    # 4. Empty protected stack rejected
    def test_04_empty_protected_stack_rejected(self):
        bad_baseline = dict(self.baseline_data)
        bad_baseline["hard_protected_compute_stack"] = {}
        is_valid, errors = validate_baseline(bad_baseline)
        self.assertFalse(is_valid)
        self.assertTrue(any("hard_protected_compute_stack" in e for e in errors))

    # 5. Missing native-stability stack rejected
    def test_05_missing_native_stability_stack_rejected(self):
        bad_baseline = dict(self.baseline_data)
        del bad_baseline["native_stability_stack"]
        is_valid, errors = validate_baseline(bad_baseline)
        self.assertFalse(is_valid)
        self.assertTrue(any("native_stability_stack" in e for e in errors))

    # 6. Missing/malformed pip report rejected
    def test_06_malformed_pip_report_rejected(self):
        malformed = {"install": "not_a_list"}
        is_valid, violations, summary = verify_report(malformed, self.baseline_data)
        self.assertFalse(is_valid)
        self.assertIn("Malformed pip report", violations[0])

    # 7. --apply and --preview together rejected
    def test_07_apply_and_preview_together_rejected(self):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0"
        res = subprocess.run(
            [sys.executable, "scripts/kaggle_setup.py", "--apply", "--preview"],
            capture_output=True, text=True, env=env
        )
        self.assertNotEqual(res.returncode, 0)

    # 8. --apply with --report-file rejected
    def test_08_apply_with_report_file_rejected(self):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0"
        res = subprocess.run(
            [sys.executable, "scripts/kaggle_setup.py", "--apply", "--report-file", self.mock_report_path],
            capture_output=True, text=True, env=env
        )
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("--report-file cannot be used in --apply mode", res.stderr)

    # 9. Preview remains default
    def test_09_preview_remains_default(self):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["PYTHONPATH"] = "."
        out_dir = os.path.join(self.temp_dir, "preview_test")

        mock_freeze_lines = [f"{k}==1.0.0" for k in self.mock_baseline_data["hard_protected_compute_stack"]] + \
                            [f"{k}==1.0.0" for k in self.mock_baseline_data["native_stability_stack"]]
        mock_freeze_text = "\n".join(mock_freeze_lines)

        code = f"""
import sys, os, json
import scripts.kaggle_setup as ks

ks.get_pip_freeze = lambda: {repr(mock_freeze_text)}
sys.argv = ['kaggle_setup.py', '--baseline', {repr(self.mock_baseline_path)}, '--report-file', {repr(self.mock_report_path)}, '--output-dir', {repr(out_dir)}]
ks.main()
"""
        res = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, env=env
        )
        self.assertEqual(res.returncode, 0, f"stdout: {res.stdout}\nstderr: {res.stderr}")
        self.assertIn("MODE: PREVIEW", res.stdout)
        self.assertIn("[PREVIEW MODE COMPLETE]", res.stdout)

    # 10. Preview requires CUDA_VISIBLE_DEVICES=0
    def test_10_preview_requires_cuda_0(self):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "1"
        res = subprocess.run(
            [sys.executable, "scripts/kaggle_setup.py", "--preview", "--report-file", self.mock_report_path],
            capture_output=True, text=True, env=env
        )
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("CUDA_VISIBLE_DEVICES must be set to '0'", res.stderr)

    # 11. Apply requires CUDA_VISIBLE_DEVICES=0
    def test_11_apply_requires_cuda_0(self):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0,1"
        res = subprocess.run(
            [sys.executable, "scripts/kaggle_setup.py", "--apply"],
            capture_output=True, text=True, env=env
        )
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("CUDA_VISIBLE_DEVICES must be set to '0'", res.stderr)

    # 12. Post-install verifier non-zero prevents success
    def test_12_post_install_verifier_failure_prevents_success(self):
        res = check_env(self.baseline_path)
        self.assertNotEqual(res["statuses"]["overall"], "PASS")

    # 13. No lock is generated after verifier failure
    def test_13_no_lock_generated_after_failure(self):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0"
        out_dir = os.path.join(self.temp_dir, "bad_baseline_test")
        res = subprocess.run(
            [sys.executable, "scripts/kaggle_setup.py", "--preview", "--baseline", "non_existent.json", "--output-dir", out_dir],
            capture_output=True, text=True, env=env
        )
        self.assertNotEqual(res.returncode, 0)
        lock_file = os.path.join(out_dir, "kaggle-overlay.lock")
        self.assertFalse(os.path.exists(lock_file))

    # 14. Lock includes newly installed transitive packages
    def test_14_lock_includes_transitive_overlay_packages(self):
        installed_before = {"torch": "2.10.0+cu128"}
        installed_after = {"torch": "2.10.0+cu128", "bitsandbytes": "0.45.0", "captum": "0.7.0"}
        newly_installed = {k: v for k, v in installed_after.items() if k not in installed_before}
        self.assertIn("bitsandbytes", newly_installed)
        self.assertIn("captum", newly_installed)
        self.assertNotIn("torch", newly_installed)

    # 15. NF4 verifier source contains explicit quantize_nf4 and dequantize_nf4
    def test_15_nf4_verifier_source_has_explicit_apis(self):
        with open("scripts/verify_kaggle_env.py", "r", encoding="utf-8") as f:
            code = f.read()
        self.assertIn("quantize_nf4", code)
        self.assertIn("dequantize_nf4", code)

    # 16. NF4 verifier contains no FP4 fallback
    def test_16_nf4_verifier_source_has_no_fp4_fallback(self):
        with open("scripts/verify_kaggle_env.py", "r", encoding="utf-8") as f:
            code = f.read()
        self.assertNotIn("quantize_fp4", code)
        self.assertNotIn("dequantize_fp4", code)
        self.assertNotIn('code_type="fp4"', code)

    # 17. SDPA is actually called
    def test_17_sdpa_is_actually_called(self):
        with open("scripts/verify_kaggle_env.py", "r", encoding="utf-8") as f:
            code = f.read()
        self.assertIn("scaled_dot_product_attention(", code)
        self.assertIn("out_sdpa = torch.nn.functional.scaled_dot_product_attention", code)

    # 18. Missing baseline produces non-zero verifier exit
    def test_18_missing_baseline_verifier_exit_nonzero(self):
        res = subprocess.run(
            [sys.executable, "scripts/verify_kaggle_env.py", "--baseline", "missing_file.json"],
            capture_output=True, text=True
        )
        self.assertNotEqual(res.returncode, 0)

    # 19. Overall PARTIAL produces non-zero exit
    def test_19_overall_partial_produces_nonzero_exit(self):
        env_res = {
            "statuses": {"overall": "PARTIAL"},
            "checks": {"c1": {"status": "PASS"}, "c2": {"status": "UNAVAILABLE"}}
        }
        self.assertNotEqual(env_res["statuses"]["overall"], "PASS")

    # 20. README and Makefile preview commands set CUDA_VISIBLE_DEVICES=0
    def test_20_readme_and_makefile_preview_commands_set_cuda_0(self):
        with open("README.md", "r", encoding="utf-8") as f:
            readme_text = f.read()
        with open("Makefile", "r", encoding="utf-8") as f:
            makefile_text = f.read()
        self.assertIn("CUDA_VISIBLE_DEVICES=0 python3 scripts/kaggle_setup.py --preview", readme_text)
        self.assertIn("CUDA_VISIBLE_DEVICES=0 python3 scripts/kaggle_setup.py --preview", makefile_text)

    # 21. Local install targets require an active virtual environment
    def test_21_local_install_targets_require_virtual_env(self):
        with open("Makefile", "r", encoding="utf-8") as f:
            makefile_text = f.read()
        self.assertIn("VIRTUAL_ENV", makefile_text)
        self.assertIn("ERROR: No active virtual environment detected.", makefile_text)

    # 22. Overlay file contains no protected/native-stability package
    def test_22_overlay_file_contains_no_protected_or_native_stability_packages(self):
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

if __name__ == "__main__":
    unittest.main()
