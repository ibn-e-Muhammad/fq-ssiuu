#!/usr/bin/env python3
"""
test_kaggle_dependency_policy.py — Behavioral unit tests for Kaggle overlay dependency infrastructure.

Offline behavioral testing without live hardware or network calls:
- PEP 440 version delta comparison
- Protected package mutation validation
- Exit code mapping for verifier statuses
- NF4 and SDPA production smoke helpers
- Subprocess error handling (git, pip freeze, verifier, missing install report)
- Verification execution record (verifier_execution.json) and linkage assertions
- Single run ID enforcement across manifests and lock files
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
import subprocess
from unittest.mock import MagicMock, patch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import scripts.kaggle_setup as ks
import scripts.verify_kaggle_env as ve
import scripts.verify_kaggle_env as ve
import scripts.verify_pip_report as vp
from scripts.verify_pip_report import normalize_name

class TestKaggleDependencyPolicyBehavioral(unittest.TestCase):

    def test_verifier_entrypoint_writes_report_with_real_control_flow(self):
        """Stub external boundaries, not check_env or its production helpers."""
        fake_torch = MagicMock()
        fake_torch.__version__ = self.mock_baseline_data['hard_protected_compute_stack']['torch']
        fake_torch.version.cuda = '12.8'
        fake_torch.cuda.max_memory_allocated.return_value = 1024
        fake_torch.cuda.max_memory_reserved.return_value = 2048
        fake_torch.cuda.device_count.return_value = 1
        fake_torch.cuda.is_available.return_value = True
        fake_torch.cuda.get_device_name.return_value = 'Tesla T4'
        fake_torch.cuda.get_device_capability.return_value = (7, 5)
        fake_torch.cuda.get_device_properties.return_value.total_memory = 15 * 1024**3
        fake_torch.all.return_value.item.return_value = True
        fake_bnb = MagicMock()
        fake_bnb.functional.quantize_nf4.return_value = (MagicMock(), type('State', (), {'quant_type': 'nf4'})())
        versions = {**self.mock_baseline_data['hard_protected_compute_stack'],
                    **self.mock_baseline_data['native_stability_stack']}
        modules = {'torch': fake_torch, 'bitsandbytes': fake_bnb,
                   'bitsandbytes.functional': fake_bnb.functional,
                   **{name: MagicMock() for name in ('captum', 'hydra', 'lm_eval', 'jsonlines')}}
        def external(cmd, **kwargs):
            if cmd[:3] == ['git', 'rev-parse', 'HEAD']:
                return subprocess.CompletedProcess(cmd, 0, 'a' * 40 + '\n', '')
            if cmd == [sys.executable, '-m', 'pip', 'check']:
                return subprocess.CompletedProcess(cmd, 0, 'No broken requirements found.' + chr(10), '')
            raise AssertionError(f'Unexpected external operation: {cmd}')
        out = os.path.join(self.temp_dir, 'verifier')
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}), \
             patch.object(ve.subprocess, 'run', side_effect=external), \
             patch.object(ve.importlib.metadata, 'version', side_effect=versions.__getitem__), \
             patch.object(sys, 'argv', ['verify', '--baseline', self.mock_baseline_path,
                                       '--output-dir', out, '--run-id', 'entrypoint-test']):
            with self.assertRaises(SystemExit) as result:
                ve.main()
        self.assertEqual(result.exception.code, 0)
        with open(os.path.join(out, 'kaggle_env_verification.json')) as handle:
            report = json.load(handle)
        self.assertEqual(report['git_sha'], 'a' * 40)
        self.assertEqual(report['run_id'], 'entrypoint-test')
        self.assertEqual(report['statuses']['overall'], 'PASS')
        self.assertEqual(report['checks']['pip_check']['status'], 'PASS')

    def test_report_cli_reads_valid_and_malformed_reports(self):
        for content, expected in [(' {"install": []}', 0), ('invalid-json', 1)]:
            with self.subTest(content=content):
                report = os.path.join(self.temp_dir, 'pip-report.json')
                output = os.path.join(self.temp_dir, 'report-result.json')
                with open(report, 'w') as handle:
                    handle.write(content)
                with patch.object(sys, 'argv', ['verify', '--baseline', self.mock_baseline_path,
                                               '--report', report, '--output-json', output]):
                    with self.assertRaises(SystemExit) as result:
                        vp.main()
                self.assertEqual(result.exception.code, expected)
                with open(output) as handle:
                    self.assertEqual(json.load(handle)['status'], 'PASS' if expected == 0 else 'FAIL')

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_kaggle_policy_")
        self.mock_baseline_data = {
            "schema_version": "1.0",
            "environment_name": "kaggle-gpu-t4-test",
            "python_version": sys.version.split()[0],
            "cuda_build_version": "12.8",
            "hard_protected_compute_stack": {
                "torch": "2.10.0+cu128",
                "torchvision": "0.25.0+cu128",
                "torchaudio": "2.10.0+cu128",
                "triton": "3.6.0"
            },
            "native_stability_stack": {
                "transformers": "4.46.3",
                "datasets": "3.1.0",
                "accelerate": "1.1.1",
                "peft": "0.13.2",
                "huggingface-hub": "0.26.2",
                "numpy": "1.26.4",
                "scipy": "1.14.1",
                "pandas": "2.2.3",
                "omegaconf": "2.3.0",
                "tokenizers": "0.20.3",
                "safetensors": "0.4.5"
            }
        }
        self.mock_baseline_path = os.path.join(self.temp_dir, "kaggle-native-baseline.json")
        with open(self.mock_baseline_path, "w", encoding="utf-8") as f:
            json.dump(self.mock_baseline_data, f, indent=2)

        self.overlay_path = os.path.join(self.temp_dir, "kaggle-overlay.in")
        with open(self.overlay_path, "w", encoding="utf-8") as f:
            f.write("bitsandbytes==0.45.0\n")

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    # 1. FAIL, PARTIAL, and UNAVAILABLE map to non-zero exit
    def test_01_exit_code_mapping_non_zero_for_non_pass(self):
        self.assertEqual(ve.exit_code_for_status("FAIL"), 1)
        self.assertEqual(ve.exit_code_for_status("PARTIAL"), 1)
        self.assertEqual(ve.exit_code_for_status("UNAVAILABLE"), 1)

    # 2. Only PASS maps to zero
    def test_02_exit_code_mapping_zero_only_for_pass(self):
        self.assertEqual(ve.exit_code_for_status("PASS"), 0)

    # 3. The NF4 helper calls quantize_nf4 and dequantize_nf4 and never calls FP4
    def test_03_nf4_helper_invokes_quantize_and_dequantize_nf4(self):
        fake_torch = MagicMock()
        fake_torch.float32 = "float32"

        fake_tensor = MagicMock()
        fake_tensor.abs.return_value.mean.return_value.item.return_value = 0.001
        fake_tensor.abs.return_value.max.return_value.item.return_value = 0.01
        fake_torch.randn.return_value = fake_tensor
        fake_torch.all.return_value.item.return_value = True

        fake_bnb = MagicMock()
        fake_state = MagicMock()
        fake_state.quant_type = "nf4"
        fake_bnb.quantize_nf4.return_value = (fake_tensor, fake_state)
        fake_bnb.dequantize_nf4.return_value = fake_tensor

        # Ensure no FP4 functions are called or present
        del fake_bnb.quantize_fp4
        del fake_bnb.dequantize_fp4

        res = ve.run_nf4_smoke(fake_torch, fake_bnb)
        fake_bnb.quantize_nf4.assert_called_once()
        fake_bnb.dequantize_nf4.assert_called_once()
        self.assertEqual(res["status"], "PASS")

    # 4. The SDPA helper invokes the supplied scaled-dot-product-attention implementation
    def test_04_sdpa_helper_invokes_scaled_dot_product_attention(self):
        fake_torch = MagicMock()
        fake_torch.float16 = "float16"

        fake_out = MagicMock()
        fake_torch.randn.return_value = fake_out
        fake_torch.nn.functional.scaled_dot_product_attention.return_value = fake_out
        fake_torch.all.return_value.item.return_value = True

        res = ve.run_sdpa_smoke(fake_torch)
        fake_torch.nn.functional.scaled_dot_product_attention.assert_called_once()
        self.assertEqual(res["status"], "PASS")

    # 5. transformers changing from the protected baseline causes lock generation to fail
    def test_05_transformers_mutation_causes_lock_generation_failure(self):
        before = {"torch": "2.10.0+cu128", "transformers": "4.46.3"}
        after = {"torch": "2.10.0+cu128", "transformers": "4.47.0"} # Mutated transformers
        with self.assertRaises(ValueError) as ctx:
            ks.generate_overlay_lock(before, after, self.mock_baseline_data, "test_run", "test_sha")
        self.assertIn("Protected package mutation detected", str(ctx.exception))
        self.assertIn("transformers", str(ctx.exception))

    # 6. datasets changing from the protected baseline causes lock generation to fail
    def test_06_datasets_mutation_causes_lock_generation_failure(self):
        before = {"torch": "2.10.0+cu128", "datasets": "3.1.0"}
        after = {"torch": "2.10.0+cu128", "datasets": "3.2.0"} # Mutated datasets
        with self.assertRaises(ValueError) as ctx:
            ks.generate_overlay_lock(before, after, self.mock_baseline_data, "test_run", "test_sha")
        self.assertIn("Protected package mutation detected", str(ctx.exception))
        self.assertIn("datasets", str(ctx.exception))

    # 7. 0.9.0 → 0.10.0 is upgraded
    def test_07_version_comparison_0_9_0_to_0_10_0_is_upgraded(self):
        self.assertEqual(ks.compare_versions("0.9.0", "0.10.0"), "upgraded")

    # 8. 1.10.0 → 1.9.0 is downgraded
    def test_08_version_comparison_1_10_0_to_1_9_0_is_downgraded(self):
        self.assertEqual(ks.compare_versions("1.10.0", "1.9.0"), "downgraded")

    # 9. A final pip freeze failure prevents lock generation and final PASS
    def test_09_final_pip_freeze_failure_prevents_lock_generation_and_pass(self):
        out_dir = os.path.join(self.temp_dir, "test_09_out")
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["PYTHONPATH"] = "."

        mock_freeze = "\n".join([f"{k}=={v}" for k, v in self.mock_baseline_data["hard_protected_compute_stack"].items()] +
                                [f"{k}=={v}" for k, v in self.mock_baseline_data["native_stability_stack"].items()]) + "\nbitsandbytes==0.45.0"

        code = f"""
import sys, subprocess, json, os
import scripts.kaggle_setup as ks
import scripts.verify_kaggle_env as ve

freeze_count = 0
def mock_freeze():
    global freeze_count
    freeze_count += 1
    if freeze_count >= 2:
        return ("", 1) # Final freeze fails
    return ({repr(mock_freeze)}, 0)

ks.get_distribution_inventory = lambda: {repr({**self.mock_baseline_data['hard_protected_compute_stack'], **self.mock_baseline_data['native_stability_stack'], 'bitsandbytes': '0.45.0'})}
ks.get_git_info = lambda: ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", False, True)
ks.get_pip_freeze = mock_freeze

def mock_run(cmd, capture_output=True, text=True):
    if cmd == [sys.executable, '-m', 'pip', 'check']:
        return subprocess.CompletedProcess(cmd, 0, 'No broken requirements found.' + chr(10), '')
    if len(cmd) > 1 and 'verify_kaggle_env.py' in cmd[1]:
        v_rep = {repr(out_dir)} + '/kaggle_env_verification.json'
        run_id = cmd[cmd.index('--run-id')+1] if '--run-id' in cmd else 'fake_run_id'
        git_sha = cmd[cmd.index('--git-sha')+1] if '--git-sha' in cmd else 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
        with open(v_rep, 'w') as f:
            json.dump({{'run_id': run_id, 'git_sha': git_sha, 'statuses': {{'overall': 'PASS'}}, 'checks': {{name: {{'status': 'PASS'}} for name in ve.REQUIRED_RUNTIME_CHECKS}}}}, f)
        class Res:
            returncode = 0
            stdout = 'verifier pass'
            stderr = ''
        return Res()

    if '--report' in cmd:
        rep_path = cmd[cmd.index('--report')+1]
        os.makedirs(os.path.dirname(rep_path), exist_ok=True)
        with open(rep_path, 'w') as f:
            json.dump({{'install': [{{'metadata': {{'name': 'bitsandbytes', 'version': '0.45.0'}}}}]}}, f)

    class Res:
        returncode = 0
        stdout = 'ok'
        stderr = ''
    return Res()

subprocess.run = mock_run
sys.argv = ['kaggle_setup.py', '--apply', '--baseline', {repr(self.mock_baseline_path)}, '--output-dir', {repr(out_dir)}]
ks.main()
"""
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        self.assertNotEqual(res.returncode, 0)
        self.assertFalse(os.path.exists(os.path.join(out_dir, "kaggle-overlay.lock")))
        status_file = os.path.join(out_dir, "setup_status.json")
        self.assertTrue(os.path.exists(status_file))
        with open(status_file) as f:
            st = json.load(f)
        self.assertEqual(st["overall_status"], "FAIL")

    # 10. An after_manifest with snapshot_status != PASS prevents lock generation
    def test_10_after_manifest_non_pass_prevents_lock_generation(self):
        out_dir = os.path.join(self.temp_dir, "test_10_out")
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["PYTHONPATH"] = "."

        mock_freeze_before = "\n".join([f"{k}=={v}" for k, v in self.mock_baseline_data["hard_protected_compute_stack"].items()] +
                                       [f"{k}=={v}" for k, v in self.mock_baseline_data["native_stability_stack"].items()]) + "\nbitsandbytes==0.45.0"
        mock_freeze_after = mock_freeze_before.replace("4.46.3", "4.47.0") # Mutates transformers

        code = f"""
import sys, subprocess, json, os
import scripts.kaggle_setup as ks
import scripts.verify_kaggle_env as ve

freeze_count = 0
def mock_freeze():
    global freeze_count
    freeze_count += 1
    if freeze_count >= 2:
        return ({repr(mock_freeze_after)}, 0)
    return ({repr(mock_freeze_before)}, 0)

ks.get_distribution_inventory = lambda: {repr({**self.mock_baseline_data['hard_protected_compute_stack'], **self.mock_baseline_data['native_stability_stack'], 'bitsandbytes': '0.45.0'})}
ks.get_git_info = lambda: ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", False, True)
ks.get_pip_freeze = mock_freeze

def mock_run(cmd, capture_output=True, text=True):
    if cmd == [sys.executable, '-m', 'pip', 'check']:
        return subprocess.CompletedProcess(cmd, 0, 'No broken requirements found.' + chr(10), '')
    if len(cmd) > 1 and 'verify_kaggle_env.py' in cmd[1]:
        v_rep = {repr(out_dir)} + '/kaggle_env_verification.json'
        run_id = cmd[cmd.index('--run-id')+1] if '--run-id' in cmd else 'fake_run_id'
        git_sha = cmd[cmd.index('--git-sha')+1] if '--git-sha' in cmd else 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
        with open(v_rep, 'w') as f:
            json.dump({{'run_id': run_id, 'git_sha': git_sha, 'statuses': {{'overall': 'PASS'}}, 'checks': {{name: {{'status': 'PASS'}} for name in ve.REQUIRED_RUNTIME_CHECKS}}}}, f)
        class Res:
            returncode = 0
            stdout = 'verifier pass'
            stderr = ''
        return Res()

    if '--report' in cmd:
        rep_path = cmd[cmd.index('--report')+1]
        os.makedirs(os.path.dirname(rep_path), exist_ok=True)
        with open(rep_path, 'w') as f:
            json.dump({{'install': [{{'metadata': {{'name': 'bitsandbytes', 'version': '0.45.0'}}}}]}}, f)

    class Res:
        returncode = 0
        stdout = 'ok'
        stderr = ''
    return Res()

subprocess.run = mock_run
sys.argv = ['kaggle_setup.py', '--apply', '--baseline', {repr(self.mock_baseline_path)}, '--output-dir', {repr(out_dir)}]
ks.main()
"""
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        self.assertNotEqual(res.returncode, 0)
        self.assertFalse(os.path.exists(os.path.join(out_dir, "kaggle-overlay.lock")))

    # 11. Verifier run-ID mismatch produces all required failure artifacts
    def test_11_verifier_run_id_mismatch_produces_all_required_failure_artifacts(self):
        out_dir = os.path.join(self.temp_dir, "test_11_out")
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["PYTHONPATH"] = "."

        mock_freeze = "\n".join([f"{k}=={v}" for k, v in self.mock_baseline_data["hard_protected_compute_stack"].items()] +
                                [f"{k}=={v}" for k, v in self.mock_baseline_data["native_stability_stack"].items()]) + "\nbitsandbytes==0.45.0"

        code = f"""
import sys, subprocess, json, os
import scripts.kaggle_setup as ks
import scripts.verify_kaggle_env as ve

ks.get_distribution_inventory = lambda: {repr({**self.mock_baseline_data['hard_protected_compute_stack'], **self.mock_baseline_data['native_stability_stack'], 'bitsandbytes': '0.45.0'})}
ks.get_git_info = lambda: ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", False, True)
ks.get_pip_freeze = lambda: ({repr(mock_freeze)}, 0)

def mock_run(cmd, capture_output=True, text=True):
    if cmd == [sys.executable, '-m', 'pip', 'check']:
        return subprocess.CompletedProcess(cmd, 0, 'No broken requirements found.' + chr(10), '')
    if len(cmd) > 1 and 'verify_kaggle_env.py' in cmd[1]:
        v_rep = {repr(out_dir)} + '/kaggle_env_verification.json'
        git_sha = cmd[cmd.index('--git-sha')+1] if '--git-sha' in cmd else 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
        with open(v_rep, 'w') as f:
            json.dump({{'run_id': 'mismatched_run_id', 'git_sha': git_sha, 'statuses': {{'overall': 'PASS'}}, 'checks': {{name: {{'status': 'PASS'}} for name in ve.REQUIRED_RUNTIME_CHECKS}}}}, f)
        class Res:
            returncode = 0
            stdout = 'verifier pass'
            stderr = ''
        return Res()

    if '--report' in cmd:
        rep_path = cmd[cmd.index('--report')+1]
        os.makedirs(os.path.dirname(rep_path), exist_ok=True)
        with open(rep_path, 'w') as f:
            json.dump({{'install': [{{'metadata': {{'name': 'bitsandbytes', 'version': '0.45.0'}}}}]}}, f)

    class Res:
        returncode = 0
        stdout = 'ok'
        stderr = ''
    return Res()

subprocess.run = mock_run
sys.argv = ['kaggle_setup.py', '--apply', '--baseline', {repr(self.mock_baseline_path)}, '--output-dir', {repr(out_dir)}]
ks.main()
"""
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        self.assertNotEqual(res.returncode, 0)
        self.assertFalse(os.path.exists(os.path.join(out_dir, "kaggle-overlay.lock")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "setup_status.json")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "post_attempt_manifest.json")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "verifier_execution.json")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "MANIFEST.json")))

    # 12. Verifier Git-SHA mismatch produces all required failure artifacts
    def test_12_verifier_git_sha_mismatch_produces_all_required_failure_artifacts(self):
        out_dir = os.path.join(self.temp_dir, "test_12_out")
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["PYTHONPATH"] = "."

        mock_freeze = "\n".join([f"{k}=={v}" for k, v in self.mock_baseline_data["hard_protected_compute_stack"].items()] +
                                [f"{k}=={v}" for k, v in self.mock_baseline_data["native_stability_stack"].items()]) + "\nbitsandbytes==0.45.0"

        code = f"""
import sys, subprocess, json, os
import scripts.kaggle_setup as ks
import scripts.verify_kaggle_env as ve

ks.get_distribution_inventory = lambda: {repr({**self.mock_baseline_data['hard_protected_compute_stack'], **self.mock_baseline_data['native_stability_stack'], 'bitsandbytes': '0.45.0'})}
ks.get_git_info = lambda: ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", False, True)
ks.get_pip_freeze = lambda: ({repr(mock_freeze)}, 0)

def mock_run(cmd, capture_output=True, text=True):
    if cmd == [sys.executable, '-m', 'pip', 'check']:
        return subprocess.CompletedProcess(cmd, 0, 'No broken requirements found.' + chr(10), '')
    if len(cmd) > 1 and 'verify_kaggle_env.py' in cmd[1]:
        v_rep = {repr(out_dir)} + '/kaggle_env_verification.json'
        run_id = cmd[cmd.index('--run-id')+1] if '--run-id' in cmd else 'fake_run_id'
        with open(v_rep, 'w') as f:
            json.dump({{'run_id': run_id, 'git_sha': 'mismatched_sha', 'statuses': {{'overall': 'PASS'}}, 'checks': {{name: {{'status': 'PASS'}} for name in ve.REQUIRED_RUNTIME_CHECKS}}}}, f)
        class Res:
            returncode = 0
            stdout = 'verifier pass'
            stderr = ''
        return Res()

    if '--report' in cmd:
        rep_path = cmd[cmd.index('--report')+1]
        os.makedirs(os.path.dirname(rep_path), exist_ok=True)
        with open(rep_path, 'w') as f:
            json.dump({{'install': [{{'metadata': {{'name': 'bitsandbytes', 'version': '0.45.0'}}}}]}}, f)

    class Res:
        returncode = 0
        stdout = 'ok'
        stderr = ''
    return Res()

subprocess.run = mock_run
sys.argv = ['kaggle_setup.py', '--apply', '--baseline', {repr(self.mock_baseline_path)}, '--output-dir', {repr(out_dir)}]
ks.main()
"""
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        self.assertNotEqual(res.returncode, 0)
        self.assertFalse(os.path.exists(os.path.join(out_dir, "kaggle-overlay.lock")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "setup_status.json")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "post_attempt_manifest.json")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "verifier_execution.json")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "MANIFEST.json")))

    # 13. Verifier status PARTIAL produces all required failure artifacts
    def test_13_verifier_status_partial_produces_all_required_failure_artifacts(self):
        out_dir = os.path.join(self.temp_dir, "test_13_out")
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["PYTHONPATH"] = "."

        mock_freeze = "\n".join([f"{k}=={v}" for k, v in self.mock_baseline_data["hard_protected_compute_stack"].items()] +
                                [f"{k}=={v}" for k, v in self.mock_baseline_data["native_stability_stack"].items()]) + "\nbitsandbytes==0.45.0"

        code = f"""
import sys, subprocess, json, os
import scripts.kaggle_setup as ks
import scripts.verify_kaggle_env as ve

ks.get_distribution_inventory = lambda: {repr({**self.mock_baseline_data['hard_protected_compute_stack'], **self.mock_baseline_data['native_stability_stack'], 'bitsandbytes': '0.45.0'})}
ks.get_git_info = lambda: ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", False, True)
ks.get_pip_freeze = lambda: ({repr(mock_freeze)}, 0)

def mock_run(cmd, capture_output=True, text=True):
    if cmd == [sys.executable, '-m', 'pip', 'check']:
        return subprocess.CompletedProcess(cmd, 0, 'No broken requirements found.' + chr(10), '')
    if len(cmd) > 1 and 'verify_kaggle_env.py' in cmd[1]:
        v_rep = {repr(out_dir)} + '/kaggle_env_verification.json'
        run_id = cmd[cmd.index('--run-id')+1] if '--run-id' in cmd else 'fake_run_id'
        git_sha = cmd[cmd.index('--git-sha')+1] if '--git-sha' in cmd else 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
        with open(v_rep, 'w') as f:
            json.dump({{'run_id': run_id, 'git_sha': git_sha, 'statuses': {{'overall': 'PARTIAL'}}}}, f)
        class Res:
            returncode = 0
            stdout = 'verifier partial'
            stderr = ''
        return Res()

    if '--report' in cmd:
        rep_path = cmd[cmd.index('--report')+1]
        os.makedirs(os.path.dirname(rep_path), exist_ok=True)
        with open(rep_path, 'w') as f:
            json.dump({{'install': [{{'metadata': {{'name': 'bitsandbytes', 'version': '0.45.0'}}}}]}}, f)

    class Res:
        returncode = 0
        stdout = 'ok'
        stderr = ''
    return Res()

subprocess.run = mock_run
sys.argv = ['kaggle_setup.py', '--apply', '--baseline', {repr(self.mock_baseline_path)}, '--output-dir', {repr(out_dir)}]
ks.main()
"""
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        self.assertNotEqual(res.returncode, 0)
        self.assertFalse(os.path.exists(os.path.join(out_dir, "kaggle-overlay.lock")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "setup_status.json")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "post_attempt_manifest.json")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "verifier_execution.json")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "MANIFEST.json")))

    # 14. verifier_execution.json contains actual returncode, statuses, hashes, and linkage verdicts
    def test_14_verifier_execution_json_contains_returncode_statuses_hashes_and_verdicts(self):
        out_dir = os.path.join(self.temp_dir, "test_14_out")
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["PYTHONPATH"] = "."

        mock_freeze = "\n".join([f"{k}=={v}" for k, v in self.mock_baseline_data["hard_protected_compute_stack"].items()] +
                                [f"{k}=={v}" for k, v in self.mock_baseline_data["native_stability_stack"].items()]) + "\nbitsandbytes==0.45.0"

        code = f"""
import sys, subprocess, json, os
import scripts.kaggle_setup as ks
import scripts.verify_kaggle_env as ve

ks.get_distribution_inventory = lambda: {repr({**self.mock_baseline_data['hard_protected_compute_stack'], **self.mock_baseline_data['native_stability_stack'], 'bitsandbytes': '0.45.0'})}
ks.get_git_info = lambda: ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", False, True)
ks.get_pip_freeze = lambda: ({repr(mock_freeze)}, 0)

def mock_run(cmd, capture_output=True, text=True):
    if cmd == [sys.executable, '-m', 'pip', 'check']:
        return subprocess.CompletedProcess(cmd, 0, 'No broken requirements found.' + chr(10), '')
    if len(cmd) > 1 and 'verify_kaggle_env.py' in cmd[1]:
        v_rep = {repr(out_dir)} + '/kaggle_env_verification.json'
        run_id = cmd[cmd.index('--run-id')+1] if '--run-id' in cmd else 'fake_run_id'
        git_sha = cmd[cmd.index('--git-sha')+1] if '--git-sha' in cmd else 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
        with open(v_rep, 'w') as f:
            json.dump({{'run_id': run_id, 'git_sha': git_sha, 'statuses': {{'overall': 'PASS'}}, 'checks': {{name: {{'status': 'PASS'}} for name in ve.REQUIRED_RUNTIME_CHECKS}}}}, f)
        class Res:
            returncode = 0
            stdout = 'verifier pass stdout'
            stderr = 'verifier pass stderr'
        return Res()

    if '--report' in cmd:
        rep_path = cmd[cmd.index('--report')+1]
        os.makedirs(os.path.dirname(rep_path), exist_ok=True)
        with open(rep_path, 'w') as f:
            json.dump({{'install': [{{'metadata': {{'name': 'bitsandbytes', 'version': '0.45.0'}}}}]}}, f)

    class Res:
        returncode = 0
        stdout = 'ok'
        stderr = ''
    return Res()

subprocess.run = mock_run
sys.argv = ['kaggle_setup.py', '--apply', '--baseline', {repr(self.mock_baseline_path)}, '--output-dir', {repr(out_dir)}]
ks.main()
"""
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        self.assertEqual(res.returncode, 0)
        v_exec_path = os.path.join(out_dir, "verifier_execution.json")
        self.assertTrue(os.path.exists(v_exec_path))

        with open(v_exec_path) as f:
            ve_data = json.load(f)

        self.assertEqual(ve_data["schema_version"], "1.0")
        self.assertEqual(ve_data["verifier_returncode"], 0)
        self.assertEqual(ve_data["verifier_overall_status"], "PASS")
        self.assertTrue(ve_data["run_id_valid"])
        self.assertTrue(ve_data["git_sha_valid"])
        self.assertEqual(ve_data["verdict"], "PASS")
        self.assertIsNotNone(ve_data["stdout_log"]["sha256"])
        self.assertIsNotNone(ve_data["stderr_log"]["sha256"])

    # 15. Invalid baseline stops execution before pip
    def test_15_invalid_baseline_stops_execution_before_pip(self):
        invalid_baseline_path = os.path.join(self.temp_dir, "invalid_baseline.json")
        with open(invalid_baseline_path, "w", encoding="utf-8") as f:
            json.dump({"schema_version": "1.0"}, f)

        out_dir = os.path.join(self.temp_dir, "test_15_out")
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["PYTHONPATH"] = "."

        res = subprocess.run(
            [sys.executable, "scripts/kaggle_setup.py", "--preview", "--baseline", invalid_baseline_path, "--output-dir", out_dir],
            capture_output=True, text=True, env=env
        )
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("Baseline structural validation failed", res.stderr)

    # 16. Git provenance failure stops execution
    def test_16_git_provenance_failure_stops_execution(self):
        out_dir = os.path.join(self.temp_dir, "test_16_out")
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["PYTHONPATH"] = "."

        code = f"""
import sys
import scripts.kaggle_setup as ks
import scripts.verify_kaggle_env as ve

ks.get_distribution_inventory = lambda: {repr({**self.mock_baseline_data['hard_protected_compute_stack'], **self.mock_baseline_data['native_stability_stack'], 'bitsandbytes': '0.45.0'})}
ks.get_git_info = lambda: ("UNKNOWN", True, False)
sys.argv = ['kaggle_setup.py', '--preview', '--baseline', {repr(self.mock_baseline_path)}, '--output-dir', {repr(out_dir)}]
ks.main()
"""
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("Git provenance check failed", res.stderr)

    # 17. Failure of pip freeze stops execution
    def test_17_pip_freeze_failure_stops_execution(self):
        out_dir = os.path.join(self.temp_dir, "test_17_out")
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["PYTHONPATH"] = "."

        code = f"""
import sys
import scripts.kaggle_setup as ks
import scripts.verify_kaggle_env as ve

ks.get_distribution_inventory = lambda: {repr({**self.mock_baseline_data['hard_protected_compute_stack'], **self.mock_baseline_data['native_stability_stack'], 'bitsandbytes': '0.45.0'})}
ks.get_git_info = lambda: ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", False, True)
ks.get_pip_freeze = lambda: ("", 1)
sys.argv = ['kaggle_setup.py', '--preview', '--baseline', {repr(self.mock_baseline_path)}, '--output-dir', {repr(out_dir)}]
ks.main()
"""
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("pip freeze failed", res.stderr)

    # 18. Missing pip_install_report.json is rejected
    def test_18_missing_pip_install_report_rejected(self):
        out_dir = os.path.join(self.temp_dir, "test_18_out")
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["PYTHONPATH"] = "."

        mock_freeze = "\n".join([f"{k}=={v}" for k, v in self.mock_baseline_data["hard_protected_compute_stack"].items()] +
                                [f"{k}=={v}" for k, v in self.mock_baseline_data["native_stability_stack"].items()]) + "\nbitsandbytes==0.45.0"

        code = f"""
import sys, subprocess, json, os
import scripts.kaggle_setup as ks
import scripts.verify_kaggle_env as ve

ks.get_distribution_inventory = lambda: {repr({**self.mock_baseline_data['hard_protected_compute_stack'], **self.mock_baseline_data['native_stability_stack'], 'bitsandbytes': '0.45.0'})}
ks.get_git_info = lambda: ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", False, True)
ks.get_pip_freeze = lambda: ({repr(mock_freeze)}, 0)

def mock_run(cmd, capture_output=True, text=True):
    if cmd == [sys.executable, '-m', 'pip', 'check']:
        return subprocess.CompletedProcess(cmd, 0, 'No broken requirements found.' + chr(10), '')
    class Res:
        returncode = 0
        stdout = 'mock stdout'
        stderr = 'mock stderr'
    for i, arg in enumerate(cmd):
        if arg == '--report':
            rep_path = cmd[i+1]
            with open(rep_path, 'w') as f:
                json.dump({{'install': [{{'metadata': {{'name': 'bitsandbytes', 'version': '0.45.0'}}}}]}}, f)
    if 'install' in cmd and '--report' in cmd and 'pip_install_report.json' in cmd[cmd.index('--report')+1]:
        rep_path = cmd[cmd.index('--report')+1]
        if os.path.exists(rep_path):
            os.remove(rep_path)
    return Res()

subprocess.run = mock_run
sys.argv = ['kaggle_setup.py', '--apply', '--baseline', {repr(self.mock_baseline_path)}, '--output-dir', {repr(out_dir)}]
ks.main()
"""
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("missing pip_install_report.json", res.stderr)

    # 19. Lock helper includes newly installed transitive packages
    def test_19_lock_helper_includes_newly_installed_transitive_packages(self):
        before = {"torch": "2.10.0+cu128", "transformers": "4.46.3"}
        after = {
            "torch": "2.10.0+cu128",
            "transformers": "4.46.3",
            "bitsandbytes": "0.45.0",
            "transitive-dep": "1.2.3"
        }
        lock_dict, lock_text = ks.generate_overlay_lock(before, after, self.mock_baseline_data, "test_run", "test_sha")
        entries = {e["normalized_name"]: e for e in lock_dict["lock_entries"]}
        self.assertIn("bitsandbytes", entries)
        self.assertIn("transitive-dep", entries)
        self.assertEqual(entries["bitsandbytes"]["operation"], "added")
        self.assertEqual(entries["transitive-dep"]["operation"], "added")
        self.assertIn("transitive-dep==1.2.3", lock_text)

    # 20. Lock helper excludes all protected packages
    def test_20_lock_helper_excludes_all_protected_packages(self):
        before = {"torch": "2.10.0+cu128", "transformers": "4.46.3"}
        after = {"torch": "2.10.0+cu128", "transformers": "4.46.3", "bitsandbytes": "0.45.0"}
        lock_dict, lock_text = ks.generate_overlay_lock(before, after, self.mock_baseline_data, "test_run", "test_sha")
        locked_names = [e["normalized_name"] for e in lock_dict["lock_entries"]]
        self.assertNotIn("torch", locked_names)
        self.assertNotIn("transformers", locked_names)
        self.assertIn("bitsandbytes", locked_names)

if __name__ == "__main__":
    unittest.main()
