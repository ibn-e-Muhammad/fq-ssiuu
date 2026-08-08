# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Pilot P1 + P2: QRA Existence Test and Attribution Feasibility
#
# ## IMPORTANT: Run on SINGLE T4 GPU (not dual)
#
# - **P1**: Tests whether QRA manifests at 1B scale using NF4 bitsandbytes (NOT GPTQ/AWQ)
# - **P2**: Tests SSIUU attribution feasibility within 16 GB VRAM
# - **Estimated total time**: 30-60 minutes
# - **Results saved to**: /kaggle/working/results/
#
# ### Before running:
# 1. Set accelerator to **GPU T4 x1** (single GPU)
# 2. Ensure **Internet** is toggled ON
# 3. Replace `YOUR_HF_TOKEN` with your actual HuggingFace token in each cell

# %% [markdown]
# ## Cell 1: Setup and Dependencies

# %%
import os
import subprocess

# ── Configuration ────────────────────────────────────────────
HF_TOKEN = "YOUR_HF_TOKEN"  # <-- REPLACE THIS
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["TRANSFORMERS_NO_FLASH_ATTN"] = "1"
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "offline"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("=== Cloning Repositories ===")

# Clone fq-ssiuu
if not os.path.exists("/kaggle/working/fq-ssiuu"):
    subprocess.run(
        "git clone --branch pilot/p1-p2 https://github.com/ibn-e-Muhammad/fq-ssiuu.git /kaggle/working/fq-ssiuu",
        shell=True, check=True,
    )
else:
    subprocess.run(
        "cd /kaggle/working/fq-ssiuu && git fetch origin && git checkout pilot/p1-p2",
        shell=True, check=True,
    )

# Clone OpenUnlearning
if not os.path.exists("/kaggle/working/open-unlearning"):
    subprocess.run(
        "git clone --branch fq-ssiuu-integration https://github.com/ibn-e-Muhammad/open-unlearning.git /kaggle/working/open-unlearning",
        shell=True, check=True,
    )
else:
    subprocess.run(
        "cd /kaggle/working/open-unlearning && git fetch origin && git checkout fq-ssiuu-integration",
        shell=True, check=True,
    )

print("\n=== Patching Dependencies ===")

# Remove toxic version pins
subprocess.run("sed -i '/torch==/d' /kaggle/working/open-unlearning/requirements.txt", shell=True)
subprocess.run("sed -i '/torchvision==/d' /kaggle/working/open-unlearning/requirements.txt", shell=True)
subprocess.run("sed -i '/bitsandbytes==/d' /kaggle/working/open-unlearning/requirements.txt", shell=True)
subprocess.run("sed -i 's/hf_vlms/huggingface/g' /kaggle/working/open-unlearning/src/evals/lm_eval.py", shell=True)

# Install OpenUnlearning
subprocess.run("pip install -e /kaggle/working/open-unlearning -q", shell=True)
subprocess.run("pip install lm-eval==0.4.4 bitsandbytes==0.45.5 captum -q", shell=True)
subprocess.run("pip uninstall timm flash-attn -y -q 2>/dev/null || true", shell=True)

# Install fq-ssiuu (no deps — we already installed everything)
subprocess.run("pip install -e /kaggle/working/fq-ssiuu --no-deps -q", shell=True)

print("\n=== Downloading Retain Logs ===")
subprocess.run(
    "cd /kaggle/working/open-unlearning && python setup_data.py --eval_logs",
    shell=True,
)

# Fallback: direct download
retain_logs = "/kaggle/working/open-unlearning/saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json"
if not os.path.exists(retain_logs):
    print("setup_data.py didn't produce the file. Trying direct download...")
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="open-unlearning/eval",
        allow_patterns="*.json",
        repo_type="dataset",
        local_dir="/kaggle/working/open-unlearning/saves/eval",
    )

assert os.path.exists(retain_logs), f"FATAL: Retain logs not found at {retain_logs}"
print(f"✅ Retain logs ready: {retain_logs}")
print("\n=== Setup Complete ===")

# %% [markdown]
# ## Cell 2: Verify Setup

# %%
import torch
print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB")

import bitsandbytes
print(f"bitsandbytes: {bitsandbytes.__version__}")

try:
    import captum
    print(f"captum: {captum.__version__}")
except ImportError:
    print("WARNING: captum not installed. P2 will fail. Run: pip install captum")

# Verify fq-ssiuu import
from fq_ssiuu.evaluation.qra_pipeline import test_q_int4_formula
test_q_int4_formula()
print("✅ qra_pipeline self-test passed")

# Verify retain logs
retain_logs = "/kaggle/working/open-unlearning/saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json"
assert os.path.exists(retain_logs), f"CRITICAL: Retain logs not found"
print("✅ Retain logs: OK")

print("\n✅ All checks passed. Ready for P1.")

# %% [markdown]
# ## Cell 3: Download Checkpoint

# %%
import os
from huggingface_hub import snapshot_download

CKPT = "/kaggle/working/checkpoints/gate1_npo"
os.makedirs(CKPT, exist_ok=True)

safetensors = os.path.join(CKPT, "model.safetensors")
if os.path.exists(safetensors):
    size_gb = os.path.getsize(safetensors) / (1024**3)
    print(f"✅ Checkpoint already present ({size_gb:.2f} GB)")
else:
    print("Downloading checkpoint from HuggingFace Hub...")
    snapshot_download(
        repo_id="Xafire/GATE1-NPO-Baseline",
        local_dir=CKPT,
        token=os.environ["HF_TOKEN"],
    )
    size_gb = os.path.getsize(safetensors) / (1024**3)
    print(f"✅ Checkpoint downloaded ({size_gb:.2f} GB)")

# Verify
assert os.path.exists(os.path.join(CKPT, "config.json")), "Missing config.json"
assert os.path.exists(os.path.join(CKPT, "tokenizer.json")), "Missing tokenizer.json"
print(f"✅ Checkpoint verified at: {CKPT}")

# %% [markdown]
# ## Cell 4: Run Pilot P1 — QRA Existence Test
#
# This runs three evaluations (FP16, INT8, NF4) sequentially.
# Each evaluation takes approximately 5-10 minutes.

# %%
import os, sys, subprocess

os.environ["HF_TOKEN"] = HF_TOKEN  # Ensure token is set
os.environ["TRANSFORMERS_NO_FLASH_ATTN"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONPATH"] = "/kaggle/working/fq-ssiuu/src:/kaggle/working/open-unlearning/src:" + os.environ.get("PYTHONPATH", "")

CKPT = "/kaggle/working/checkpoints/gate1_npo"
OU_PATH = "/kaggle/working/open-unlearning"
OUTPUT_DIR = "/kaggle/working/results/pilot_p1"

print("=== PILOT P1: QRA EXISTENCE TEST ===\n")

result = subprocess.run(
    [
        sys.executable,
        "/kaggle/working/fq-ssiuu/experiments/pilot_p1_qra_check.py",
        "--open_unlearning_path", OU_PATH,
        "--checkpoint_path", CKPT,
        "--drive_output", OUTPUT_DIR,
    ],
    capture_output=False,
    text=True,
    env=os.environ,
)

print(f"\n=== P1 Exit Code: {result.returncode} ===")

# Show results if available
import json
results_path = os.path.join(OUTPUT_DIR, "pilot_p1_results.json")
if os.path.exists(results_path):
    with open(results_path) as f:
        p1 = json.load(f)
    print(f"\nQ-INT4 score: {p1.get('q_int4_score', 'N/A')}")
    print(f"Verdict: {p1.get('verdict', 'N/A')}")
else:
    # Also check /tmp
    tmp_path = "/tmp/pilot_p1_results.json"
    if os.path.exists(tmp_path):
        import shutil
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        shutil.copy(tmp_path, results_path)
        with open(tmp_path) as f:
            p1 = json.load(f)
        print(f"\nQ-INT4 score: {p1.get('q_int4_score', 'N/A')}")
        print(f"Verdict: {p1.get('verdict', 'N/A')}")
    else:
        print("\n⚠️ Results file not found. Check output above for errors.")

# %% [markdown]
# ## Cell 5: Run Pilot P2 — Attribution Feasibility Test
#
# Tests whether captum's LayerIntegratedGradients can run within T4 VRAM limits.
# Takes approximately 5-15 minutes.

# %%
import os, sys, subprocess

os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["TRANSFORMERS_NO_FLASH_ATTN"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

CKPT = "/kaggle/working/checkpoints/gate1_npo"
OUTPUT_DIR = "/kaggle/working/results/pilot_p2"

print("=== PILOT P2: SSIUU ATTRIBUTION FEASIBILITY ===\n")

result = subprocess.run(
    [
        sys.executable,
        "/kaggle/working/fq-ssiuu/experiments/pilot_p2_attribution.py",
        "--checkpoint_path", CKPT,
        "--drive_output", OUTPUT_DIR,
    ],
    capture_output=False,
    text=True,
    env=os.environ,
)

print(f"\n=== P2 Exit Code: {result.returncode} ===")

# Show results
import json
results_path = os.path.join(OUTPUT_DIR, "pilot_p2_results.json")
if os.path.exists(results_path):
    with open(results_path) as f:
        p2 = json.load(f)
    print(f"\nVerdict: {p2.get('verdict', 'N/A')}")
    print(f"Peak VRAM: {p2.get('vram_peak_during_ig_gb', 'N/A')} GB")
    print(f"Recommended method: {p2.get('recommended_method', 'N/A')}")
else:
    tmp_path = "/tmp/pilot_p2_results.json"
    if os.path.exists(tmp_path):
        import shutil
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        shutil.copy(tmp_path, results_path)
        with open(tmp_path) as f:
            p2 = json.load(f)
        print(f"\nVerdict: {p2.get('verdict', 'N/A')}")
        print(f"Peak VRAM: {p2.get('vram_peak_during_ig_gb', 'N/A')} GB")
    else:
        print("\n⚠️ Results file not found. Check output above for errors.")

# %% [markdown]
# ## Cell 6: Final Summary
#
# Copy the output below and paste it back to the agent.

# %%
import json, os

print("=" * 64)
print("    PILOT P1 + P2 EXECUTION SUMMARY")
print("=" * 64)

# P1 Results
p1_path = "/kaggle/working/results/pilot_p1/pilot_p1_results.json"
if os.path.exists(p1_path):
    with open(p1_path) as f:
        p1 = json.load(f)
    print(f"\n--- P1: QRA Existence Test ---")
    print(f"Q-INT4 score: {p1.get('q_int4_score', 'N/A')}")
    print(f"Q-INT8 score: {p1.get('q_int8_score', 'N/A')}")
    print(f"Verdict: {p1.get('verdict', 'N/A')}")
    rp = p1.get('results_by_precision', {})
    for prec in ['fp16', 'int8', 'nf4']:
        if prec in rp:
            r = rp[prec]
            print(f"  {prec.upper()}: FQ={r.get('forget_quality', 'N/A')}, MU={r.get('model_utility', 'N/A')}")
else:
    print("\nP1 results not found.")

# P2 Results
p2_path = "/kaggle/working/results/pilot_p2/pilot_p2_results.json"
if os.path.exists(p2_path):
    with open(p2_path) as f:
        p2 = json.load(f)
    print(f"\n--- P2: Attribution Feasibility ---")
    print(f"Verdict: {p2.get('verdict', 'N/A')}")
    print(f"Peak VRAM: {p2.get('vram_peak_during_ig_gb', 'N/A')} GB")
    print(f"Recommended: {p2.get('recommended_method', 'N/A')}")
else:
    print("\nP2 results not found.")

print("\n" + "=" * 64)
print("Copy ALL output from Cells 4, 5, and above, then paste to the agent.")
print("=" * 64)
