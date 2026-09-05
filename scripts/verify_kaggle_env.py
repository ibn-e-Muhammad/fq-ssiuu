#!/usr/bin/env python3
"""
verify_kaggle_env.py — Fail-closed Kaggle post-install environment verification script.

Verifies:
- Python version matches baseline expectation
- CUDA_VISIBLE_DEVICES=0 and exactly 1 visible CUDA device inside process
- Tesla T4 identity, compute capability 7.5, VRAM < 16 GiB
- Real FP16 CUDA matrix multiplication and CUDA synchronization
- Real SDPA CUDA execution via torch.nn.functional.scaled_dot_product_attention
- Baseline protected stack integrity (hard-protected and native-stability)
- Overlay module imports (bitsandbytes, captum, hydra, lm_eval, jsonlines)
- Clean pip check
- Genuine CUDA NF4 quantise/dequantise test strictly via bitsandbytes.functional.quantize_nf4 / dequantize_nf4
- NO FP4 fallback or fp4 code type usage allowed

Outputs structured JSON report. Exits code 0 ONLY when overall status is PASS.
Exits non-zero for FAIL, PARTIAL, or UNAVAILABLE.
"""

import os
import sys
import json
import uuid
import tempfile
import subprocess
import argparse
import importlib.metadata
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from scripts.verify_pip_report import validate_baseline, normalize_name, atomic_write_json
except ImportError:
    from verify_pip_report import validate_baseline, normalize_name, atomic_write_json

def run_pip_check() -> tuple[int, str]:
    res = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    return res.returncode, res.stdout + res.stderr

def get_git_sha() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT)
        return res.stdout.strip() if res.returncode == 0 else "UNKNOWN"
    except Exception:
        return "UNKNOWN"

def check_env(baseline_path: str, run_id: str | None = None, git_sha: str | None = None) -> dict:
    if run_id is None:
        run_id = str(uuid.uuid4())
    if git_sha is None:
        git_sha = get_git_sha()

    results = {
        "schema_version": "1.0",
        "verifier_version": "1.0",
        "run_id": run_id,
        "git_sha": git_sha,
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
        "statuses": {},
        "checks": {}
    }

    # 1. Baseline file verification
    bdata = None
    if not os.path.exists(baseline_path):
        results["checks"]["baseline_file"] = {
            "status": "FAIL",
            "reason": f"Baseline file missing: {baseline_path}"
        }
    else:
        try:
            with open(baseline_path, "r", encoding="utf-8") as f:
                bdata = json.load(f)
            b_valid, b_errors = validate_baseline(bdata)
            if not b_valid:
                results["checks"]["baseline_file"] = {
                    "status": "FAIL",
                    "reason": f"Baseline structural validation failed: {b_errors}"
                }
            else:
                results["checks"]["baseline_file"] = {"status": "PASS", "path": baseline_path}
        except Exception as e:
            results["checks"]["baseline_file"] = {
                "status": "FAIL",
                "reason": f"Baseline file unparseable: {e}"
            }

    # 2. Python version check
    expected_py = bdata.get("python_version") if (bdata and isinstance(bdata, dict)) else None
    actual_py = sys.version.split()[0]
    py_pass = False
    if expected_py and isinstance(expected_py, str):
        # Match major.minor or exact version
        exp_mm = ".".join(expected_py.split(".")[:2])
        act_mm = ".".join(actual_py.split(".")[:2])
        py_pass = (act_mm == exp_mm)

    results["checks"]["python_version"] = {
        "status": "PASS" if py_pass else "FAIL",
        "observed": {"actual": actual_py, "expected": expected_py}
    }

    # 3. Environment variables & CUDA visibility
    cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    env_vis_pass = (cuda_vis == "0")
    results["checks"]["cuda_visible_devices_var"] = {
        "status": "PASS" if env_vis_pass else "FAIL",
        "observed": cuda_vis
    }

    # 4. PyTorch & CUDA device checks
    try:
        import torch

        # Check PyTorch version matching baseline hard_protected
        exp_torch_ver = bdata.get("hard_protected_compute_stack", {}).get("torch") if (bdata and isinstance(bdata, dict)) else None
        torch_ver_pass = (torch.__version__ == exp_torch_ver) if exp_torch_ver else False
        results["checks"]["pytorch_version"] = {
            "status": "PASS" if torch_ver_pass else "FAIL",
            "observed": {"actual": torch.__version__, "expected": exp_torch_ver}
        }

        dev_count = torch.cuda.device_count()
        dev_count_pass = (dev_count == 1)
        results["checks"]["cuda_device_count"] = {
            "status": "PASS" if dev_count_pass else "FAIL",
            "observed": dev_count
        }

        if torch.cuda.is_available() and dev_count > 0:
            dev_name = torch.cuda.get_device_name(0)
            dev_cap = torch.cuda.get_device_capability(0)
            vram_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)

            t4_pass = ("Tesla T4" in dev_name or "T4" in dev_name) and dev_cap == (7, 5) and (vram_gib < 16.0)
            results["checks"]["t4_hardware"] = {
                "status": "PASS" if t4_pass else "FAIL",
                "observed": {
                    "name": dev_name,
                    "compute_capability": list(dev_cap),
                    "total_memory_gib": round(vram_gib, 2)
                }
            }

            # Real FP16 CUDA calculation & synchronization
            try:
                x = torch.randn(10, 10, dtype=torch.float16, device="cuda:0")
                y = torch.matmul(x, x)
                torch.cuda.synchronize()
                is_finite = bool(torch.all(torch.isfinite(y)).item())
                results["checks"]["fp16_operation"] = {
                    "status": "PASS" if is_finite else "FAIL",
                    "observed": {
                        "shape": list(y.shape),
                        "dtype": str(y.dtype),
                        "device": str(y.device),
                        "is_finite": is_finite
                    }
                }
            except Exception as e:
                results["checks"]["fp16_operation"] = {
                    "status": "FAIL",
                    "exception": str(e)
                }

            # Real SDPA CUDA calculation & synchronization
            try:
                q = torch.randn(1, 4, 16, 32, dtype=torch.float16, device="cuda:0")
                k = torch.randn(1, 4, 16, 32, dtype=torch.float16, device="cuda:0")
                v = torch.randn(1, 4, 16, 32, dtype=torch.float16, device="cuda:0")
                out_sdpa = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                torch.cuda.synchronize()
                is_sdpa_finite = bool(torch.all(torch.isfinite(out_sdpa)).item())
                results["checks"]["sdpa_operation"] = {
                    "status": "PASS" if is_sdpa_finite else "FAIL",
                    "observed": {
                        "shape": list(out_sdpa.shape),
                        "dtype": str(out_sdpa.dtype),
                        "device": str(out_sdpa.device),
                        "is_finite": is_sdpa_finite
                    }
                }
            except Exception as e:
                results["checks"]["sdpa_operation"] = {
                    "status": "FAIL",
                    "exception": str(e)
                }
        else:
            results["checks"]["t4_hardware"] = {"status": "FAIL", "reason": "CUDA not available in process"}
            results["checks"]["fp16_operation"] = {"status": "FAIL", "reason": "CUDA not available in process"}
            results["checks"]["sdpa_operation"] = {"status": "FAIL", "reason": "CUDA not available in process"}

    except ImportError:
        results["checks"]["pytorch"] = {"status": "FAIL", "reason": "PyTorch not installed"}

    # 5. Baseline protected stack integrity (Hard-protected + Native-stability)
    if bdata and isinstance(bdata, dict) and results["checks"].get("baseline_file", {}).get("status") == "PASS":
        protected_pass = True
        protected_obs = {}
        all_protected = {
            **bdata.get("hard_protected_compute_stack", {}),
            **bdata.get("native_stability_stack", {})
        }
        for pkg, expected_ver in all_protected.items():
            try:
                actual_ver = importlib.metadata.version(pkg)
                is_m = (actual_ver == expected_ver)
                protected_obs[pkg] = {"expected": expected_ver, "actual": actual_ver, "match": is_m}
                if not is_m:
                    protected_pass = False
            except Exception as e:
                protected_obs[pkg] = {"expected": expected_ver, "actual": None, "error": str(e), "match": False}
                protected_pass = False

        results["checks"]["protected_stack_integrity"] = {
            "status": "PASS" if protected_pass else "FAIL",
            "observed": protected_obs
        }

    # 6. Overlay import checks (Must be present and importable post-install)
    overlay_imports = {
        "bitsandbytes": "bitsandbytes",
        "captum": "captum",
        "hydra": "hydra-core",
        "lm_eval": "lm-eval",
        "jsonlines": "jsonlines"
    }
    import_obs = {}
    import_pass = True
    for mod_name, pkg_name in overlay_imports.items():
        try:
            __import__(mod_name)
            import_obs[mod_name] = "PASS"
        except Exception as e:
            import_obs[mod_name] = f"FAIL ({e})"
            import_pass = False

    results["checks"]["overlay_imports"] = {
        "status": "PASS" if import_pass else "FAIL",
        "observed": import_obs
    }

    # 7. Genuine NF4 quantise/dequantise check strictly via bitsandbytes
    try:
        import bitsandbytes.functional as F_bnb
        import torch

        if torch.cuda.is_available():
            # Verify explicit quantize_nf4 and dequantize_nf4 exist
            has_quant_nf4 = hasattr(F_bnb, "quantize_nf4")
            has_dequant_nf4 = hasattr(F_bnb, "dequantize_nf4")

            if not (has_quant_nf4 and has_dequant_nf4):
                results["checks"]["nf4_quantise_dequantise"] = {
                    "status": "FAIL",
                    "reason": "Explicit bitsandbytes quantize_nf4 or dequantize_nf4 API not found"
                }
            else:
                x_fp32 = torch.randn(128, 128, dtype=torch.float32, device="cuda:0")
                out_q, state = F_bnb.quantize_nf4(x_fp32)
                x_rec = F_bnb.dequantize_nf4(out_q, state)
                torch.cuda.synchronize()

                is_rec_finite = bool(torch.all(torch.isfinite(x_rec)).item())
                mean_err = float((x_fp32 - x_rec).abs().mean().item())
                max_err = float((x_fp32 - x_rec).abs().max().item())
                quant_type = getattr(state, "quant_type", None)

                nf4_valid = is_rec_finite and (quant_type == "nf4")
                results["checks"]["nf4_quantise_dequantise"] = {
                    "status": "PASS" if nf4_valid else "FAIL",
                    "observed": {
                        "shape": list(out_q.shape),
                        "device": str(out_q.device),
                        "quant_dtype": str(out_q.dtype),
                        "rec_dtype": str(x_rec.dtype),
                        "quant_type": quant_type,
                        "is_rec_finite": is_rec_finite,
                        "mean_reconstruction_error": mean_err,
                        "max_reconstruction_error": max_err
                    }
                }
        else:
            results["checks"]["nf4_quantise_dequantise"] = {
                "status": "FAIL",
                "reason": "CUDA not available for NF4 test"
            }
    except Exception as e:
        results["checks"]["nf4_quantise_dequantise"] = {
            "status": "FAIL",
            "reason": f"bitsandbytes NF4 execution failed: {e}"
        }

    # 8. Pip check
    pip_code, pip_out = run_pip_check()
    results["checks"]["pip_check"] = {
        "status": "PASS" if pip_code == 0 else "FAIL",
        "observed": pip_out.strip()
    }

    # Calculate overall status: Must be PASS only if EVERY check is PASS
    all_statuses = [v.get("status") for v in results["checks"].values()]
    if all(s == "PASS" for s in all_statuses):
        results["statuses"]["overall"] = "PASS"
    elif "FAIL" in all_statuses:
        results["statuses"]["overall"] = "FAIL"
    else:
        results["statuses"]["overall"] = "UNAVAILABLE"

    return results

def main():
    parser = argparse.ArgumentParser(description="Verify Kaggle environment after overlay setup (Fail-Closed).")
    parser.add_argument("--baseline", help="Path to baseline JSON file.")
    parser.add_argument("--output-dir", help="Directory to save kaggle_env_verification.json")
    parser.add_argument("--run-id", help="Common run ID string across setup session")

    args = parser.parse_args()

    baseline_path = args.baseline if args.baseline else os.path.join(REPO_ROOT, "requirements/kaggle-native-baseline.json")
    run_id = args.run_id if args.run_id else str(uuid.uuid4())

    res = check_env(baseline_path, run_id=run_id)
    overall = res["statuses"]["overall"]

    print("=== KAGGLE ENVIRONMENT VERIFICATION SUMMARY ===")
    print(f"Overall Status: {overall}")
    for k, v in res["checks"].items():
        print(f"  {k}: {v['status']}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out_file = os.path.join(args.output_dir, "kaggle_env_verification.json")
        atomic_write_json(out_file, res)

    if overall != "PASS":
        sys.stderr.write(f"ERROR: Environment verification failed with overall status: '{overall}'\n")
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
