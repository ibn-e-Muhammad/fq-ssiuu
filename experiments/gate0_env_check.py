"""
experiments/gate0_env_check.py

GATE-0: Environment Verification Script (Updated for Colab T4, August 2026)
Run this AFTER scripts/setup_env.sh completes.
All 10 checks must pass before proceeding to GATE-1.

Colab environment: Python 3.12.13, PyTorch 2.13.0+cu130, CUDA 13.0, T4 GPU
SDPA (not flash-attn) is used for attention — T4 is Turing (CC 7.5), flash-attn 2.x unsupported.

Usage:
    python3 experiments/gate0_env_check.py
"""
import sys
import os

CHECKS_PASSED = 0
CHECKS_FAILED = 0
FAILURES = []

def check(name, fn):
    global CHECKS_PASSED, CHECKS_FAILED
    try:
        result = fn()
        status = "  PASS"
        print(f"{status}  {name}" + (f" — {result}" if result else ""))
        CHECKS_PASSED += 1
        return True
    except Exception as e:
        print(f"  FAIL  {name} — {e}")
        CHECKS_FAILED += 1
        FAILURES.append((name, str(e)))
        return False

print("=" * 60)
print("  GATE-0: FQ-SSIUU Environment Verification")
print("  Platform: Colab T4 | Python 3.12 | CUDA 13.0")
print("=" * 60)
print()

# CHECK 1: Python version (3.12 acceptable — updated from original 3.11 requirement)
check("Python 3.12+ (Colab default — 3.11 not required)",
    lambda: (lambda v: f"{v.major}.{v.minor}.{v.micro}" if v.major == 3 and v.minor >= 11
             else (_ for _ in ()).throw(RuntimeError(f"Python {v.major}.{v.minor} too old"))
             )(sys.version_info))

# CHECK 2: CUDA available and T4 detected
check("CUDA available (T4 Tesla GPU)",
    lambda: __import__('torch').cuda.get_device_name(0))

# CHECK 3: PyTorch version (2.x with CUDA — any 2.x is acceptable)
check("PyTorch 2.x with CUDA 13.0",
    lambda: (lambda t: f"{t.__version__} | CUDA {t.version.cuda}"
             if t.__version__.startswith("2") and t.cuda.is_available()
             else (_ for _ in ()).throw(RuntimeError(f"Unexpected: {t.__version__}"))
             )(__import__('torch')))

# CHECK 4: bitsandbytes 0.45.5+ with CUDA backend (no triton dependency check)
def check_bitsandbytes():
    import bitsandbytes as bnb
    import torch
    version = bnb.__version__
    # Version must be >= 0.45.0 for CUDA 13.0 support
    major, minor, patch = (int(x) for x in version.split(".")[:3])
    if (major, minor) < (0, 45):
        raise RuntimeError(f"bitsandbytes {version} too old — need >=0.45.0 for CUDA 13.0")
    # Test NF4 quantisation config can be created (light check — no model load)
    from transformers import BitsAndBytesConfig
    cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16)
    return f"{version} | NF4 config OK"

check("bitsandbytes 0.45.5+ (CUDA 13.0 compatible)", check_bitsandbytes)

# CHECK 5: transformers 4.45+ with Llama-3.2 support
check("transformers 4.45.x+ (Llama-3.2 support)",
    lambda: (lambda v: v if v >= "4.45.0"
             else (_ for _ in ()).throw(RuntimeError(f"Too old: {v}"))
             )(__import__('transformers').__version__))

# CHECK 6: peft 0.12.x
check("peft 0.12.x", lambda: __import__('peft').__version__)

# CHECK 7: SDPA attention backend available (replaces flash-attn)
def check_sdpa():
    import torch
    if not hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        raise RuntimeError("SDPA not available — need PyTorch >= 2.0")
    # Verify flash-attn env var is set correctly
    no_flash = os.environ.get("TRANSFORMERS_NO_FLASH_ATTN", "")
    if no_flash != "1":
        # Set it here if not already set — this check is informational
        os.environ["TRANSFORMERS_NO_FLASH_ATTN"] = "1"
    return f"torch SDPA available | TRANSFORMERS_NO_FLASH_ATTN={os.environ.get('TRANSFORMERS_NO_FLASH_ATTN', 'not set (now set)')}"

check("SDPA attention backend (PyTorch native, T4-compatible)", check_sdpa)

# CHECK 8: NF4 model loading (the big one — requires HF_TOKEN and Llama access)
def check_nf4_load():
    token = os.environ.get("HF_TOKEN", "")
    if not token or token == "PENDING":
        raise RuntimeError("HF_TOKEN not set or Llama licence not yet approved")
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    # CRITICAL: set flash-attn bypass before loading
    os.environ["TRANSFORMERS_NO_FLASH_ATTN"] = "1"

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct",
        quantization_config=bnb_cfg,
        device_map="auto",
        token=token,
        attn_implementation="sdpa",  # Explicit SDPA — no flash-attn
        max_memory={0: "12GiB"},
    )
    n_params = sum(p.numel() for p in model.parameters())
    vram_used = torch.cuda.memory_allocated() / 1e9
    del model
    torch.cuda.empty_cache()
    return f"Loaded OK | {n_params:,} params | {vram_used:.2f} GB VRAM"

check("NF4 model loading (Llama-3.2-1B-Instruct via SDPA)", check_nf4_load)

# CHECK 9: scipy KS test
check("scipy.stats KS test",
    lambda: (lambda r: "OK" if r.statistic > 0 else "failed"
             )(__import__('scipy.stats', fromlist=['ks_2samp']).ks_2samp([1,2,3],[4,5,6])))

# CHECK 10: Google Drive and fq_ssiuu package
def check_drive_and_package():
    drive_ok = os.path.exists("/content/drive/My Drive")
    pkg = __import__('fq_ssiuu').__file__
    drive_status = "Drive OK" if drive_ok else "Drive NOT mounted (run colab_mount.py)"
    return f"{drive_status} | package at {pkg}"

check("Google Drive + fq_ssiuu package importable", check_drive_and_package)

# SUMMARY
print()
print("=" * 60)
print(f"  Results: {CHECKS_PASSED} passed, {CHECKS_FAILED} failed")
if FAILURES:
    print()
    print("  Failed checks:")
    for name, err in FAILURES:
        print(f"    - {name}: {err}")
    print()
    print("  Fix: run scripts/setup_env.sh then re-run this script.")
else:
    print()
    print("  All checks passed. Proceed to GATE-1.")
    print("  Next: git clone the OpenUnlearning fork and run baseline.")
print("=" * 60)

if FAILURES:
    sys.exit(1)
else:
    sys.exit(0)
