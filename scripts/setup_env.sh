#!/usr/bin/env bash
# scripts/setup_env.sh
# FQ-SSIUU: Environment setup for Google Colab T4 (August 2026 environment)
#
# Colab actual environment: Python 3.12.13, PyTorch 2.13.0+cu130, CUDA 13.0, T4 GPU
# Strategy: use Colab's pre-installed PyTorch (do NOT reinstall torch).
#           flash-attn is INCOMPATIBLE with T4 — do not install.
#           Use SDPA (PyTorch native attention) instead.
#
# Usage: Run as a %%bash cell in Colab after mounting Drive and cloning the repo.
# DO NOT run with set -e in Colab (source command behaviour differs).

echo "======================================================"
echo " FQ-SSIUU Environment Setup (Colab T4, Aug 2026)"
echo " Python: $(python3 --version 2>&1)"
echo " CUDA:   $(python3 -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'check torch')"
echo " PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not installed')"
echo "======================================================"

# Step 1: Verify existing PyTorch (do NOT reinstall)
echo "[1/6] Verifying PyTorch and CUDA..."
python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available — check runtime type'
print(f'  PyTorch {torch.__version__} | CUDA {torch.version.cuda} | GPU: {torch.cuda.get_device_name(0)}')
print(f'  Using existing PyTorch installation — no reinstall needed.')
"

# Step 2: Upgrade bitsandbytes to version with CUDA 13.0 support
# bitsandbytes 0.42.0 fails with triton 3.x; 0.45.5 supports CUDA 13.0 natively.
echo "[2/6] Upgrading bitsandbytes to 0.45.5 (CUDA 13.0 compatible)..."
pip install bitsandbytes==0.45.5 -q --upgrade
python3 -c "
import bitsandbytes as bnb
print(f'  bitsandbytes {bnb.__version__} OK')
# Verify CUDA backend loaded
try:
    import bitsandbytes.cuda_setup.main as setup
    print('  CUDA backend: loaded')
except Exception as e:
    print(f'  CUDA backend check: {e} (may be OK — test with model load)')
"

# Step 3: Set flash-attn bypass — REQUIRED for T4 (Turing GPU)
# T4 is Compute Capability 7.5. flash-attn 2.x requires CC 8.0+ (Ampere).
# We use PyTorch's native SDPA which supports T4 fully.
echo "[3/6] Configuring attention backend (SDPA — T4 compatible)..."
export TRANSFORMERS_NO_FLASH_ATTN=1
export TRANSFORMERS_ATTN_IMPLEMENTATION=sdpa
# Make these persistent for the session
echo "export TRANSFORMERS_NO_FLASH_ATTN=1" >> ~/.bashrc
echo "export TRANSFORMERS_ATTN_IMPLEMENTATION=sdpa" >> ~/.bashrc
# Uninstall flash-attn if it was accidentally installed
pip uninstall flash-attn -y -q 2>/dev/null || true
echo "  flash-attn bypassed. Using torch.nn.functional.scaled_dot_product_attention (SDPA)."

# Step 4: HuggingFace ecosystem
# transformers 4.45.x already installed on Colab — verify and upgrade if needed
echo "[4/6] Verifying HuggingFace stack..."
pip install "transformers>=4.45.0" peft==0.12.0 trl==0.11.0 "accelerate>=0.34.0" datasets -q --upgrade
python3 -c "
import transformers, peft, trl
print(f'  transformers {transformers.__version__} | peft {peft.__version__} | trl {trl.__version__}')
"

# Step 5: captum, scipy, and utilities
echo "[5/6] Installing captum and scientific stack..."
pip install captum==0.7.0 "scipy>=1.12.0" pandas matplotlib seaborn -q
pip install omegaconf typer rich tqdm jsonlines pyyaml -q
python3 -c "
import captum, scipy
print(f'  captum {captum.__version__} | scipy {scipy.__version__}')
"

# Step 6: Install the fq_ssiuu package in editable mode
echo "[6/6] Installing fq_ssiuu package..."
REPO_PATH="${REPO_PATH:-/content/fq-ssiuu}"
pip install -e "$REPO_PATH" --no-deps -q
python3 -c "import fq_ssiuu; print(f'  fq_ssiuu package importable: {fq_ssiuu.__file__}')"

echo ""
echo "======================================================"
echo " Setup complete."
echo " IMPORTANT: TRANSFORMERS_NO_FLASH_ATTN=1 is set."
echo " Run: python3 experiments/gate0_env_check.py"
echo "======================================================"
