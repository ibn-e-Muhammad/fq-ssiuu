"""
experiments/gate0_env_check.py

GATE-0: Environment Verification Script
Run this on Colab/Kaggle BEFORE any experiment. All 14 checks must pass.

Usage:
    python experiments/gate0_env_check.py

Or via Makefile:
    make gate0
"""
import sys
import os
import importlib
import subprocess

REQUIRED_PYTHON = (3, 11)
CHECKS_PASSED = 0
CHECKS_FAILED = 0
FAILURES = []

def check(name, fn):
    global CHECKS_PASSED, CHECKS_FAILED
    try:
        result = fn()
        print(f"  PASS  {name}" + (f" — {result}" if result else ""))
        CHECKS_PASSED += 1
        return True
    except Exception as e:
        print(f"  FAIL  {name} — {e}")
        CHECKS_FAILED += 1
        FAILURES.append((name, str(e)))
        return False

print("=" * 60)
print("  GATE-0: FQ-SSIUU Environment Verification")
print("=" * 60)
print()

# CHECK 1: Python version
check("Python version is 3.11.x",
    lambda: (lambda v: f"{v.major}.{v.minor}.{v.micro}" if (v.major, v.minor) >= REQUIRED_PYTHON
             else (_ for _ in ()).throw(RuntimeError(f"Python {v.major}.{v.minor} — need 3.11+"))
             )(sys.version_info))

# CHECK 2: CUDA available
check("CUDA available (GPU detected)",
    lambda: __import__('torch').cuda.get_device_name(0))

# CHECK 3: PyTorch version
check("PyTorch 2.4.x installed",
    lambda: (lambda t: t.__version__ if t.__version__.startswith("2.4") else
             (_ for _ in ()).throw(RuntimeError(f"Got {t.__version__}"))
             )(__import__('torch')))

# CHECK 4: bitsandbytes
check("bitsandbytes 0.42.x installed",
    lambda: __import__('bitsandbytes').__version__)

# CHECK 5: transformers
check("transformers 4.46.x installed",
    lambda: __import__('transformers').__version__)

# CHECK 6: peft
check("peft 0.12.x installed",
    lambda: __import__('peft').__version__)

# CHECK 7: NF4 model loading (requires HF_TOKEN and Llama licence)
def check_nf4_load():
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError("HF_TOKEN not set — set it before running this check")
    if token == "PENDING":
        raise RuntimeError("Llama licence not yet approved — this check is deferred")
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    import torch
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct",
        quantization_config=bnb, device_map="auto", token=token,
        max_memory={0: "8GiB"}
    )
    n_params = sum(p.numel() for p in model.parameters())
    del model
    torch.cuda.empty_cache()
    return f"Loaded 1B model, {n_params:,} params"

check("NF4 model loading (Llama-3.2-1B-Instruct)", check_nf4_load)

# CHECK 8: peft LoRA wrapping
check("peft LoRA adapter wrapping",
    lambda: (lambda: (
        __import__('peft').LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj"],
                                      task_type="CAUSAL_LM"),
        "LoraConfig created OK"
    )[-1])())

# CHECK 9: captum
check("captum 0.7.x installed",
    lambda: __import__('captum').__version__)

# CHECK 10: scipy KS test
check("scipy KS test functional",
    lambda: (lambda: (
        __import__('scipy.stats', fromlist=['ks_2samp']).ks_2samp([1,2,3], [4,5,6]),
        "KS test runs"
    )[-1])())

# CHECK 11: Google Drive accessible (Colab only)
def check_gdrive():
    gdrive = "/content/drive/My Drive/fq-ssiuu-research"
    if not os.path.exists("/content/drive"):
        raise RuntimeError("Not in Colab or Drive not mounted — run colab_mount.py first")
    if not os.path.exists(gdrive):
        raise RuntimeError(f"Drive mounted but {gdrive} not found — create it manually")
    return f"Drive OK: {gdrive}"

check("Google Drive accessible with correct folder structure", check_gdrive)

# CHECK 12: ruff linter
check("ruff linter available",
    lambda: subprocess.run(["ruff", "--version"], capture_output=True, text=True,
                           check=True).stdout.strip())

# CHECK 13: pytest available
check("pytest available",
    lambda: subprocess.run(["pytest", "--version"], capture_output=True, text=True,
                           check=True).stdout.strip())

# CHECK 14: fq_ssiuu package importable
check("fq_ssiuu package importable (pip install -e . was run)",
    lambda: __import__('fq_ssiuu').__file__)

# SUMMARY
print()
print("=" * 60)
print(f"  Results: {CHECKS_PASSED} passed, {CHECKS_FAILED} failed")
if FAILURES:
    print()
    print("  Failed checks:")
    for name, err in FAILURES:
        print(f"    - {name}: {err}")
print("=" * 60)

if CHECKS_FAILED > 0:
    sys.exit(1)
else:
    print("  All checks passed. Environment is ready.")
    sys.exit(0)
