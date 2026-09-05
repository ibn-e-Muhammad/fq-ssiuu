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
- Differential pip consistency; dirty native baseline is not global cleanliness
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
import re
import importlib.metadata
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from scripts.verify_pip_report import validate_baseline, normalize_name, atomic_write_json, verify_report
except ImportError:
    from verify_pip_report import validate_baseline, normalize_name, atomic_write_json, verify_report

from scripts.pip_consistency import capture_pip_check, compare_checks, load_record, validate_record

REQUIRED_RUNTIME_CHECKS = ("git_provenance", "baseline_file", "python_version", "pytorch_version",
                          "cuda_build", "cuda_visible_devices_var", "cuda_device_count", "t4_hardware",
                          "fp16_operation", "sdpa_operation", "protected_stack_integrity", "overlay_imports",
                          "nf4_quantise_dequantise", "peak_memory", "pip_check")

def exit_code_for_status(overall_status: str) -> int:
    """Returns exit code 0 if overall_status is 'PASS', else 1."""
    return 0 if overall_status == "PASS" else 1

def get_git_sha() -> str:
    """Resolve the executed checkout; an unavailable identity cannot pass."""
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True, check=False)
        sha = result.stdout.strip()
        if result.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", sha):
            return sha
    except OSError:
        pass
    return "UNKNOWN"

def run_sdpa_smoke(torch_mod) -> dict:
    """Invokes scaled_dot_product_attention using the given torch module."""
    try:
        q = torch_mod.randn(1, 4, 16, 32, dtype=torch_mod.float16, device="cuda:0")
        k = torch_mod.randn(1, 4, 16, 32, dtype=torch_mod.float16, device="cuda:0")
        v = torch_mod.randn(1, 4, 16, 32, dtype=torch_mod.float16, device="cuda:0")
        out_sdpa = torch_mod.nn.functional.scaled_dot_product_attention(q, k, v)
        torch_mod.cuda.synchronize()
        is_sdpa_finite = bool(torch_mod.all(torch_mod.isfinite(out_sdpa)).item())
        return {
            "status": "PASS" if is_sdpa_finite else "FAIL",
            "observed": {
                "shape": list(out_sdpa.shape),
                "dtype": str(out_sdpa.dtype),
                "device": str(out_sdpa.device),
                "is_finite": is_sdpa_finite
            }
        }
    except Exception as e:
        return {
            "status": "FAIL",
            "exception": str(e)
        }

def run_nf4_smoke(torch_mod, bnb_functional) -> dict:
    """Invokes quantize_nf4 and dequantize_nf4 using given torch and bitsandbytes modules."""
    try:
        has_quant_nf4 = hasattr(bnb_functional, "quantize_nf4")
        has_dequant_nf4 = hasattr(bnb_functional, "dequantize_nf4")

        if not (has_quant_nf4 and has_dequant_nf4):
            return {
                "status": "FAIL",
                "reason": "Explicit bitsandbytes quantize_nf4 or dequantize_nf4 API not found"
            }

        x_fp32 = torch_mod.randn(128, 128, dtype=torch_mod.float32, device="cuda:0")
        out_q, state = bnb_functional.quantize_nf4(x_fp32)
        x_rec = bnb_functional.dequantize_nf4(out_q, state)
        torch_mod.cuda.synchronize()

        is_rec_finite = bool(torch_mod.all(torch_mod.isfinite(x_rec)).item())
        mean_err = float((x_fp32 - x_rec).abs().mean().item())
        max_err = float((x_fp32 - x_rec).abs().max().item())
        quant_type = getattr(state, "quant_type", None)

        nf4_valid = is_rec_finite and (quant_type == "nf4")
        return {
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
    except Exception as e:
        return {
            "status": "FAIL",
            "reason": f"bitsandbytes NF4 execution failed: {e}"
        }

def check_env(baseline_path: str, run_id: str | None = None, git_sha: str | None = None,
              pip_baseline_path=None, install_report_path=None, output_dir=None) -> dict:
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
    results["checks"]["git_provenance"] = {
        "status": "PASS" if re.fullmatch(r"[0-9a-f]{40}", git_sha) else "FAIL",
        "observed": git_sha,
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
        expected_cuda = bdata.get("cuda_build_version") if isinstance(bdata, dict) else None
        actual_cuda = torch.version.cuda
        results["checks"]["cuda_build"] = {
            "status": "PASS" if expected_cuda and actual_cuda == expected_cuda else "FAIL",
            "observed": {"actual": actual_cuda, "expected": expected_cuda}}

        dev_count = torch.cuda.device_count()
        dev_count_pass = (dev_count == 1)
        results["checks"]["cuda_device_count"] = {
            "status": "PASS" if dev_count_pass else "FAIL",
            "observed": dev_count
        }

        if torch.cuda.is_available() and dev_count > 0:
            torch.cuda.synchronize(0)
            torch.cuda.reset_peak_memory_stats(0)
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
            results["checks"]["sdpa_operation"] = run_sdpa_smoke(torch)
        else:
            results["checks"]["t4_hardware"] = {"status": "FAIL", "reason": "CUDA not available in process"}
            results["checks"]["fp16_operation"] = {"status": "FAIL", "reason": "CUDA not available in process"}
            results["checks"]["sdpa_operation"] = {"status": "FAIL", "reason": "CUDA not available in process"}

    except Exception as exc:
        results["checks"]["pytorch"] = {"status": "FAIL", "reason": f"PyTorch check failed: {exc}"}

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
            results["checks"]["nf4_quantise_dequantise"] = run_nf4_smoke(torch, F_bnb)
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

    # Peak covers this process's FP16, SDPA, imports and NF4 smoke operations.
    # It is an environment smoke observation, not a model-training memory claim.
    try:
        import torch
        torch.cuda.synchronize(0)
        allocated = int(torch.cuda.max_memory_allocated(0))
        reserved = int(torch.cuda.max_memory_reserved(0))
        total = int(torch.cuda.get_device_properties(0).total_memory)
        results["checks"]["peak_memory"] = {
            "status": "PASS" if 0 < allocated <= reserved < 16_000_000_000 and reserved <= total else "FAIL",
            "observed": {"device": "cuda:0", "peak_allocated_bytes": allocated,
                         "peak_reserved_bytes": reserved, "device_total_bytes": total,
                         "budget_bytes": 16_000_000_000, "scope": "verifier smoke process"}}
    except Exception as exc:
        results["checks"]["peak_memory"] = {"status": "FAIL", "reason": str(exc)}

    # Independently rerun pip check after smoke operations, revalidate raw
    # native baseline and actual report, then derive the same strict policy.
    try:
        pip_current = capture_pip_check(output_dir, "verifier", run_id, git_sha)
        validate_record(pip_current, stage="verifier", run_id=run_id, git_sha=git_sha)
        if pip_baseline_path is None and install_report_path is None:
            consistency = {"status": "PASS" if pip_current["returncode"] == 0 else "FAIL",
                           "state": "CLEAN" if pip_current["returncode"] == 0 else "BASELINE_REQUIRED",
                           "after_returncode": pip_current["returncode"]}
        else:
            baseline_record = load_record(pip_baseline_path, stage="before", run_id=run_id, git_sha=git_sha)
            with open(install_report_path, encoding="utf-8") as handle:
                report = json.load(handle)
            valid, errors, summary = verify_report(report, bdata)
            if not valid:
                raise ValueError(f"Verifier resolver policy failed: {errors}")
            consistency = compare_checks(baseline_record, pip_current,
                                         [p["normalized_name"] for p in summary["packages"]])
        results["checks"]["pip_check"] = {"status": consistency["status"], "observed": consistency}
    except (OSError, KeyError, TypeError, ValueError) as exc:
        results["checks"]["pip_check"] = {"status": "FAIL", "reason": str(exc)}

    for name in REQUIRED_RUNTIME_CHECKS:
        results["checks"].setdefault(name, {"status": "FAIL", "reason": "Required check was not completed"})

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
    parser.add_argument("--pip-baseline", help="Raw-linked pre-install pip check JSON record")
    parser.add_argument("--install-report", help="Actual pip install report for overlay ownership")

    args = parser.parse_args()

    baseline_path = args.baseline if args.baseline else os.path.join(REPO_ROOT, "requirements/kaggle-native-baseline.json")
    run_id = args.run_id if args.run_id else str(uuid.uuid4())

    try:
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
        res = check_env(baseline_path, run_id=run_id, pip_baseline_path=args.pip_baseline,
                        install_report_path=args.install_report, output_dir=args.output_dir)
    except Exception as exc:
        res = {
            "schema_version": "1.0", "run_id": run_id, "git_sha": get_git_sha(),
            "statuses": {"overall": "FAIL"},
            "checks": {"verifier_execution": {"status": "FAIL",
                "reason": f"{type(exc).__name__}: {exc}"}},
        }
    overall = res["statuses"]["overall"]

    print("=== KAGGLE ENVIRONMENT VERIFICATION SUMMARY ===")
    print(f"Overall Status: {overall}")
    for k, v in res["checks"].items():
        print(f"  {k}: {v['status']}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out_file = os.path.join(args.output_dir, "kaggle_env_verification.json")
        atomic_write_json(out_file, res)

    sys.exit(exit_code_for_status(overall))

if __name__ == "__main__":
    main()
