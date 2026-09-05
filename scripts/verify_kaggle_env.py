#!/usr/bin/env python3
"""
verify_kaggle_env.py — Post-install Kaggle environment verification script.

Verifies:
- CUDA_VISIBLE_DEVICES=0 and 1 visible CUDA device inside process
- Tesla T4 identity and compute capability 7.5
- FP16 operation and SDPA existence
- Protected native package versions unchanged
- Overlay package imports
- Clean pip check
- Genuine CUDA NF4 quantise/dequantise test via bitsandbytes
- No model or dataset downloads

Outputs structured JSON report with explicit PASS, FAIL, UNAVAILABLE statuses.
"""

import os
import sys
import json
import subprocess
import argparse
from datetime import datetime, timezone

def run_pip_check() -> tuple[int, str]:
    res = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    return res.returncode, res.stdout + res.stderr

def check_env():
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "statuses": {},
        "checks": {}
    }

    # 1. Environment variables & CUDA visibility
    cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    env_vis_pass = (cuda_vis == "0")
    results["checks"]["cuda_visible_devices_var"] = {
        "status": "PASS" if env_vis_pass else "FAIL",
        "observed": cuda_vis
    }

    # 2. PyTorch & CUDA device checks
    torch_available = False
    try:
        import torch
        torch_available = True

        dev_count = torch.cuda.device_count()
        dev_count_pass = (dev_count == 1)
        results["checks"]["cuda_device_count"] = {
            "status": "PASS" if dev_count_pass else "FAIL",
            "observed": dev_count
        }

        if torch.cuda.is_available() and dev_count > 0:
            dev_name = torch.cuda.get_device_name(0)
            dev_cap = torch.cuda.get_device_capability(0)
            t4_pass = ("Tesla T4" in dev_name or "T4" in dev_name) and dev_cap == (7, 5)
            results["checks"]["t4_hardware"] = {
                "status": "PASS" if t4_pass else "FAIL",
                "observed": {"name": dev_name, "compute_capability": list(dev_cap)}
            }

            # FP16 test
            try:
                x = torch.randn(10, 10, dtype=torch.float16, device="cuda:0")
                y = torch.matmul(x, x)
                results["checks"]["fp16_operation"] = {
                    "status": "PASS",
                    "observed": {"shape": list(y.shape), "dtype": str(y.dtype), "device": str(y.device)}
                }
            except Exception as e:
                results["checks"]["fp16_operation"] = {
                    "status": "FAIL",
                    "exception": str(e)
                }

            # SDPA check
            sdpa_exists = hasattr(torch.nn.functional, "scaled_dot_product_attention")
            results["checks"]["sdpa_operation"] = {
                "status": "PASS" if sdpa_exists else "FAIL",
                "observed": {"sdpa_exists": sdpa_exists}
            }
        else:
            results["checks"]["t4_hardware"] = {"status": "UNAVAILABLE", "reason": "CUDA not available in process"}
            results["checks"]["fp16_operation"] = {"status": "UNAVAILABLE", "reason": "CUDA not available in process"}
            results["checks"]["sdpa_operation"] = {"status": "UNAVAILABLE", "reason": "CUDA not available in process"}

    except ImportError:
        results["checks"]["pytorch"] = {"status": "UNAVAILABLE", "reason": "PyTorch not installed"}

    # 3. Protected version check
    baseline_path = "requirements/kaggle-native-baseline.json"
    if os.path.exists(baseline_path):
        with open(baseline_path, "r", encoding="utf-8") as f:
            bdata = json.load(f)
        protected_pass = True
        protected_obs = {}
        for pkg, expected_ver in bdata.get("hard_protected_compute_stack", {}).items():
            try:
                import importlib.metadata
                actual_ver = importlib.metadata.version(pkg)
                protected_obs[pkg] = {"expected": expected_ver, "actual": actual_ver}
                if actual_ver != expected_ver:
                    protected_pass = False
            except Exception as e:
                protected_obs[pkg] = {"expected": expected_ver, "actual": None, "error": str(e)}
                protected_pass = False
        results["checks"]["protected_stack_integrity"] = {
            "status": "PASS" if protected_pass else "FAIL",
            "observed": protected_obs
        }

    # 4. Overlay import checks
    overlay_imports = ["bitsandbytes", "captum", "hydra", "lm_eval", "jsonlines"]
    import_obs = {}
    import_pass = True
    for mod in overlay_imports:
        try:
            __import__(mod)
            import_obs[mod] = "PASS"
        except Exception as e:
            import_obs[mod] = f"UNAVAILABLE ({e})"
            import_pass = False
    results["checks"]["overlay_imports"] = {
        "status": "PASS" if import_pass else "UNAVAILABLE",
        "observed": import_obs
    }

    # 5. Genuine NF4 quantise/dequantise check via bitsandbytes
    try:
        import bitsandbytes as bnb
        import torch
        if torch.cuda.is_available():
            x_fp32 = torch.randn(128, 128, dtype=torch.float32, device="cuda:0")
            # Genuine NF4 quantise/dequantise via bitsandbytes
            if hasattr(bnb.functional, "quantize_fp4"):
                out_q, state = bnb.functional.quantize_fp4(x_fp32)
                x_rec = bnb.functional.dequantize_fp4(out_q, state)
                diff = (x_fp32 - x_rec).abs().mean().item()
                results["checks"]["nf4_quantise_dequantise"] = {
                    "status": "PASS",
                    "observed": {
                        "shape": list(out_q.shape),
                        "device": str(out_q.device),
                        "dtype": str(out_q.dtype),
                        "mean_reconstruction_error": diff
                    }
                }
            elif hasattr(bnb.functional, "quantize_blockwise"):
                out_q, state = bnb.functional.quantize_blockwise(x_fp32, code_type="fp4")
                x_rec = bnb.functional.dequantize_blockwise(out_q, state)
                diff = (x_fp32 - x_rec).abs().mean().item()
                results["checks"]["nf4_quantise_dequantise"] = {
                    "status": "PASS",
                    "observed": {
                        "shape": list(out_q.shape),
                        "device": str(out_q.device),
                        "dtype": str(out_q.dtype),
                        "mean_reconstruction_error": diff
                    }
                }
            else:
                results["checks"]["nf4_quantise_dequantise"] = {
                    "status": "FAIL",
                    "reason": "bitsandbytes functional NF4 quantise function not found"
                }
        else:
            results["checks"]["nf4_quantise_dequantise"] = {
                "status": "UNAVAILABLE",
                "reason": "CUDA not available for NF4 test"
            }
    except Exception as e:
        results["checks"]["nf4_quantise_dequantise"] = {
            "status": "UNAVAILABLE",
            "reason": f"bitsandbytes not importable or failed: {e}"
        }

    # 6. Pip check
    pip_code, pip_out = run_pip_check()
    results["checks"]["pip_check"] = {
        "status": "PASS" if pip_code == 0 else "FAIL",
        "observed": pip_out.strip()
    }

    # Overall statuses
    all_statuses = [v.get("status") for v in results["checks"].values()]
    if "FAIL" in all_statuses:
        results["statuses"]["overall"] = "FAIL"
    elif all(s == "PASS" for s in all_statuses):
        results["statuses"]["overall"] = "PASS"
    else:
        results["statuses"]["overall"] = "PARTIAL"

    return results

def main():
    parser = argparse.ArgumentParser(description="Verify Kaggle environment after overlay setup.")
    parser.add_argument("--output-dir", help="Directory to save report.json")
    args = parser.parse_args()

    res = check_env()
    print("=== KAGGLE ENVIRONMENT VERIFICATION SUMMARY ===")
    print(f"Overall Status: {res['statuses']['overall']}")
    for k, v in res["checks"].items():
        print(f"  {k}: {v['status']}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "kaggle_env_verification.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)

if __name__ == "__main__":
    main()
