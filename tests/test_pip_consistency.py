"""Offline acceptance tests: real policy and orchestration, mocked host boundaries."""
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts import kaggle_setup as setup
from scripts import pip_consistency as pc
from scripts import verify_kaggle_env as verifier

SHA = "a" * 40
DIRTY = ("bigframes 2.39.0 requires google-cloud-bigquery-storage, which is not installed.\n"
         "google-colab 1.0.0 has requirement pandas==2.2.2, but you have pandas 2.3.3.\n")


def record(text=DIRTY, stage="before", code=1):
    return pc.make_record(text, "", code, stage, "test-run", SHA)


class TestPipConsistency(unittest.TestCase):
    def test_unchanged_diagnostics_accept_explicit_dirty_state(self):
        before = record()
        after = record("\n".join(reversed(DIRTY.strip().splitlines())) + "\n", "after")
        result = pc.compare_checks(before, after)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["state"], "BASELINE_DIRTY_NO_REGRESSION")
        self.assertEqual(len(result["unchanged"]), 2)
        self.assertEqual(result["introduced"], [])

    def test_clean_and_resolved_diagnostics(self):
        for before in (record(), record(pc.CLEAN + "\n", code=0)):
            result = pc.compare_checks(before, record(pc.CLEAN + "\n", "after", 0))
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["state"], "CLEAN")
            self.assertEqual(len(result["resolved"]), len(before["diagnostics"]))

    def test_recorded_interpreter_is_portable_but_cannot_change_during_run(self):
        before, after = record(), record(stage="after")
        before["command"][0] = after["command"][0] = "/recorded-kaggle/python"
        self.assertEqual(pc.compare_checks(before, after)["status"], "PASS")
        after["command"][0] = "/different/python"
        self.assertEqual(pc.compare_checks(before, after)["status"], "FAIL")

    def test_new_diagnostic_or_version_change_fails(self):
        for after in (DIRTY + "native-new 1 requires absent, which is not installed.\n",
                      DIRTY.replace("2.3.3", "2.3.4")):
            result = pc.compare_checks(record(), record(after, "after"))
            self.assertEqual(result["status"], "FAIL")
            self.assertTrue(result["introduced"])

    def test_overlay_roots_transitives_and_dependency_intersections_fail(self):
        for text, names in [("captum 0.7.0 requires absent, which is not installed.\n", []),
                            ("some-transitive 1 requires absent, which is not installed.\n", ["some_transitive"]),
                            ("native 1 has requirement bitsandbytes<0.45, but you have bitsandbytes 0.45.5.\n", [])]:
            result = pc.compare_checks(record(text), record(text, "after"), names)
            self.assertEqual(result["status"], "FAIL")
            self.assertTrue(result["overlay_diagnostics"])

    def test_unknown_output_and_inconsistent_returncodes_fail_closed(self):
        for stdout, stderr, code in [("", "", 1), (DIRTY, "", 0), (pc.CLEAN, "", 1),
                                     ("traceback", "", 1), (DIRTY, "warning", 1),
                                     (DIRTY, "", 2), (pc.CLEAN, "", True), ("", "", 0)]:
            with self.subTest(code=code, stdout=stdout):
                with self.assertRaises(ValueError):
                    pc.normalize_diagnostics(stdout, stderr, code)

    def test_malformed_or_forged_baseline_records_fail_closed(self):
        baseline = record()
        bad = [None, {}, {**baseline, "diagnostics": []}, {**baseline, "stdout_sha256": "0" * 64},
               {**baseline, "git_sha": None}, {**baseline, "schema_version": "bad"},
               {**baseline, "returncode": True}, {**baseline, "stage": "after"}]
        for before in bad:
            with self.subTest(before=before):
                self.assertEqual(pc.compare_checks(before, record(stage="after"))["status"], "FAIL")
        for field, value in [("run_id", "other"), ("git_sha", "b" * 40),
                             ("utc_timestamp", "2000-01-01T00:00:00+00:00")]:
            after = record(stage="after")
            after[field] = value
            self.assertEqual(pc.compare_checks(baseline, after)["status"], "FAIL")

    def test_raw_bytes_are_preserved_and_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = DIRTY.replace("\n", "\r\n").encode()
            with patch.object(pc.subprocess, "run", return_value=subprocess.CompletedProcess([], 1, raw, b"")):
                captured = pc.capture_pip_check(directory, "before", "test-run", SHA)
            path = Path(directory, "pip_check_before_stdout.log")
            self.assertEqual(path.read_bytes(), raw)
            self.assertEqual(captured["stdout_sha256"], hashlib.sha256(raw).hexdigest())
            pc.load_record(Path(directory, "pip_check_before.json"), run_id="test-run", git_sha=SHA)
            path.write_bytes(raw + b"changed")
            with self.assertRaises(ValueError):
                pc.load_record(Path(directory, "pip_check_before.json"))


class TestDifferentialSetup(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.baseline = json.loads(Path(setup.REPO_ROOT, "requirements/kaggle-native-baseline.json").read_text())
        self.versions = {**self.baseline["hard_protected_compute_stack"], **self.baseline["native_stability_stack"]}
        self.before = "\n".join(f"{k}=={v}" for k, v in self.versions.items())
        self.after = self.before + "\nbitsandbytes==0.45.5\nnew-transitive==1.0\n"
        self.install = {"install": [{"metadata": {"name": n, "version": v}}
                                    for n, v in [("bitsandbytes", "0.45.5"), ("new-transitive", "1.0")]]}

    def run_setup(self, after_pip=DIRTY, verifier_code=0, after_freeze=None, before_record=None):
        out = self.root / "out"
        calls = []
        pip_calls = 0
        def external(cmd, **kwargs):
            nonlocal pip_calls
            calls.append(cmd)
            if cmd == [sys.executable, "-m", "pip", "check"]:
                pip_calls += 1
                text = DIRTY if pip_calls == 1 else after_pip
                return subprocess.CompletedProcess(cmd, 1, text.encode(), b"")
            if "--report" in cmd:
                Path(cmd[cmd.index("--report") + 1]).write_text(json.dumps(self.install))
            if len(cmd) > 1 and cmd[1].endswith("verify_kaggle_env.py"):
                run_id = cmd[cmd.index("--run-id") + 1]
                Path(out, "kaggle_env_verification.json").write_text(json.dumps({
                    "run_id": run_id, "git_sha": SHA,
                    "statuses": {"overall": "PASS" if verifier_code == 0 else "FAIL"},
                    "checks": {name: {"status": "PASS"} for name in setup.REQUIRED_RUNTIME_CHECKS}}))
                return subprocess.CompletedProcess(cmd, verifier_code, "runtime attempted", "")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        def snapshot(stage, run_id, sha, dirty, baseline, freeze_text=None, installed_inventory=None):
            return {"stage": stage, "run_id": run_id, "git_commit": sha, "snapshot_status": "PASS",
                    "python_version": "3.12.13", "driver_version": "580.159.04"}
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}), \
             patch.object(setup, "get_git_info", return_value=(SHA, False, True)), \
             patch.object(setup, "get_pip_freeze", side_effect=[(self.before, 0),
                         (self.after if after_freeze is None else after_freeze, 0), (self.after, 0)]), \
             patch.object(setup, "get_distribution_inventory", side_effect=[self.versions,
                         setup.parse_freeze_mapping(self.after if after_freeze is None else after_freeze)]), \
             patch.object(setup, "create_environment_snapshot", side_effect=snapshot), \
             patch.object(setup.subprocess, "run", side_effect=external), \
             patch.object(sys, "argv", ["setup", "--apply", "--output-dir", str(out)]):
            if before_record is None:
                try:
                    setup.main()
                    code = 0
                except SystemExit as exc:
                    code = exc.code
            else:
                with patch.object(setup, "capture_pip_check", return_value=before_record):
                    with self.assertRaises(SystemExit) as exc:
                        setup.main()
                    code = exc.exception.code
        return code, out, calls

    def test_dirty_baseline_reaches_verifier_and_generates_lock_after_pass(self):
        code, out, calls = self.run_setup()
        self.assertEqual(code, 0)
        self.assertTrue(any("verify_kaggle_env.py" in str(c) for c in calls))
        status = json.loads((out / "setup_status.json").read_text())
        self.assertEqual(status["pip_check_exit_code"], 1)
        self.assertEqual(status["pip_consistency_state"], "BASELINE_DIRTY_NO_REGRESSION")
        self.assertIn("new-transitive==1.0", (out / "kaggle-overlay.lock").read_text())
        self.assertTrue((out / "after_manifest.json").is_file())

    def test_new_diagnostic_still_runs_verifier_but_never_generates_lock(self):
        code, out, calls = self.run_setup(after_pip=DIRTY + "native 1 requires absent, which is not installed.\n")
        self.assertEqual(code, 1)
        self.assertTrue(any("verify_kaggle_env.py" in str(c) for c in calls))
        for name in ("after_manifest.json", "verifier_execution.json", "pip_consistency.json", "MANIFEST.json"):
            self.assertTrue((out / name).is_file(), name)
        self.assertFalse((out / "overlay_lock.json").exists())

    def test_verifier_failure_preserves_after_manifest_and_no_lock(self):
        code, out, _ = self.run_setup(verifier_code=1)
        self.assertEqual(code, 1)
        self.assertTrue((out / "after_manifest.json").exists())
        self.assertFalse((out / "kaggle-overlay.lock").exists())

    def test_protected_drift_cannot_get_lock_even_with_verifier_pass(self):
        code, out, _ = self.run_setup(after_freeze=self.after.replace("transformers==5.0.0", "transformers==5.0.1"))
        self.assertEqual(code, 1)
        transition = json.loads((out / "environment_transition.json").read_text())
        self.assertEqual(transition["status"], "FAIL")
        self.assertFalse((out / "overlay_lock.json").exists())

    def test_unreported_change_cannot_get_lock(self):
        code, out, _ = self.run_setup(after_freeze=self.after + "unreported==1.0\n")
        self.assertEqual(code, 1)
        self.assertFalse((out / "overlay_lock.json").exists())

    def test_malformed_native_baseline_stops_before_install(self):
        code, out, calls = self.run_setup(before_record={})
        self.assertEqual(code, 1)
        self.assertEqual(calls, [])
        self.assertFalse((out / "overlay_lock.json").exists())

    def test_native_platform_drift_fails(self):
        before = {"run_id": "r", "git_commit": SHA, "driver_version": "1"}
        after = {**before, "driver_version": "2"}
        result = setup.compare_environment_transition(before, after, self.versions, self.versions,
                                                       self.baseline, {"install": []})
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("driver_version", result["platform_drift"])

    def test_lock_rejects_removed_package(self):
        with self.assertRaises(ValueError):
            setup.generate_overlay_lock({**self.versions, "removed": "1"}, self.versions,
                                        self.baseline, "r", SHA)

    def test_direct_reference_distribution_version_and_removal_are_observed(self):
        # Raw freeze's URL cannot supply the installed version; metadata does.
        distribution = type("Distribution", (), {"metadata": {"Name": "google-colab"}, "version": "1.0.0"})()
        with patch.object(setup.importlib.metadata, "distributions", return_value=[distribution]):
            self.assertEqual(setup.get_distribution_inventory(), {"google-colab": "1.0.0"})
        before = {**self.versions, "google-colab": "1.0.0"}
        snapshot = {"run_id": "r", "git_commit": SHA}
        for after in ({**self.versions, "google-colab": "1.0.1"}, self.versions):
            result = setup.compare_environment_transition(snapshot, snapshot, before, after,
                                                           self.baseline, {"install": []})
            self.assertEqual(result["status"], "FAIL")
            self.assertTrue(any(c["name"] == "google-colab" for c in result["package_changes"]))


class TestRuntimeDifferential(unittest.TestCase):
    def run_verifier(self, pip_text=DIRTY, peak=2048, nf4_type="nf4"):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = json.loads(Path(setup.REPO_ROOT, "requirements/kaggle-native-baseline.json").read_text())
            baseline["python_version"] = sys.version.split()[0]
            (root / "baseline.json").write_text(json.dumps(baseline))
            (root / "install.json").write_text(json.dumps({"install": []}))
            before = pc.make_record(DIRTY, "", 1, "before", "test-run", SHA)
            (root / "pip_check_before.json").write_text(json.dumps(before))
            (root / "pip_check_before_stdout.log").write_bytes(DIRTY.encode())
            (root / "pip_check_before_stderr.log").write_bytes(b"")
            torch = MagicMock()
            torch.__version__ = baseline["hard_protected_compute_stack"]["torch"]
            torch.version.cuda = baseline["cuda_build_version"]
            torch.cuda.device_count.return_value = 1
            torch.cuda.is_available.return_value = True
            torch.cuda.get_device_name.return_value = "Tesla T4"
            torch.cuda.get_device_capability.return_value = (7, 5)
            torch.cuda.get_device_properties.return_value.total_memory = 15_636_037_632
            torch.cuda.max_memory_allocated.return_value = 1024
            torch.cuda.max_memory_reserved.return_value = peak
            torch.all.return_value.item.return_value = True
            bnb = MagicMock()
            bnb.functional.quantize_nf4.return_value = (MagicMock(), type("State", (), {"quant_type": nf4_type})())
            modules = {"torch": torch, "bitsandbytes": bnb, "bitsandbytes.functional": bnb.functional,
                       **{name: MagicMock() for name in ("captum", "hydra", "lm_eval", "jsonlines")}}
            versions = {**baseline["hard_protected_compute_stack"], **baseline["native_stability_stack"]}
            with patch.dict(sys.modules, modules), patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}), \
                 patch.object(verifier.importlib.metadata, "version", side_effect=versions.__getitem__), \
                 patch.object(verifier.subprocess, "run", return_value=subprocess.CompletedProcess([], 1, pip_text.encode(), b"")):
                result = verifier.check_env(str(root / "baseline.json"), run_id="test-run", git_sha=SHA,
                                             pip_baseline_path=root / "pip_check_before.json",
                                             install_report_path=root / "install.json", output_dir=root)
            self.assertTrue((root / "pip_check_verifier_stdout.log").is_file())
            torch.cuda.reset_peak_memory_stats.assert_called_once_with(0)
            bnb.functional.quantize_nf4.assert_called_once()
            bnb.functional.dequantize_nf4.assert_called_once()
            torch.nn.functional.scaled_dot_product_attention.assert_called_once()
            return result

    def test_dirty_native_baseline_passes_independent_runtime_policy(self):
        result = self.run_verifier()
        self.assertEqual(result["statuses"]["overall"], "PASS")
        self.assertEqual(result["checks"]["pip_check"]["observed"]["state"], "BASELINE_DIRTY_NO_REGRESSION")

    def test_new_diagnostic_fails_despite_successful_smokes(self):
        result = self.run_verifier(pip_text=DIRTY + "native 1 requires other, which is not installed.\n")
        self.assertEqual(result["statuses"]["overall"], "FAIL")
        self.assertEqual(result["checks"]["nf4_quantise_dequantise"]["status"], "PASS")

    def test_peak_budget_boundary_fails(self):
        result = self.run_verifier(peak=16_000_000_000)
        self.assertEqual(result["statuses"]["overall"], "FAIL")
        self.assertEqual(result["checks"]["peak_memory"]["status"], "FAIL")

    def test_non_nf4_state_fails_even_with_finite_output(self):
        result = self.run_verifier(nf4_type="wrong-type")
        self.assertEqual(result["statuses"]["overall"], "FAIL")
        self.assertEqual(result["checks"]["nf4_quantise_dequantise"]["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
