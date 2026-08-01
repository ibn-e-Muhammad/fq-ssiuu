#!/usr/bin/env bash
# scripts/setup_env.sh
# FQ-SSIUU: Full environment setup script
# Run this script ONCE on a fresh Colab/Kaggle session before any experiment.
# Platform: Ubuntu 26.04, CUDA 12.1, Python 3.11
# IMPORTANT: Install order matters. Do not reorder.

set -euo pipefail

echo "======================================================"
echo " FQ-SSIUU Environment Setup"
echo " Python: $(python3.11 --version 2>&1)"
echo " CUDA:   $(nvcc --version 2>&1 | head -1 || echo 'nvcc not found — check CUDA')"
echo "======================================================"

# Step 1: Core ML framework (MUST be first)
echo "[1/8] Installing PyTorch 2.4.1 with CUDA 12.1..."
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121 \
    --quiet
python3.11 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'  PyTorch {torch.__version__} OK | CUDA available: {torch.cuda.is_available()}')"

# Step 2: Flash attention (requires --no-build-isolation)
echo "[2/8] Installing flash-attn 2.6.3..."
pip install flash-attn==2.6.3 --no-build-isolation --quiet || {
    echo "  WARNING: flash-attn failed. Setting TRANSFORMERS_NO_FLASH_ATTN=1 as fallback."
    export TRANSFORMERS_NO_FLASH_ATTN=1
    echo "  export TRANSFORMERS_NO_FLASH_ATTN=1  # flash-attn unavailable" >> ~/.bashrc
}

# Step 3: bitsandbytes (MUST come after torch)
echo "[3/8] Installing bitsandbytes 0.42.0..."
pip install bitsandbytes==0.42.0 --quiet
python3.11 -c "import bitsandbytes; print(f'  bitsandbytes {bitsandbytes.__version__} OK')"

# Step 4: HuggingFace ecosystem
echo "[4/8] Installing HuggingFace stack..."
pip install \
    transformers==4.46.0 \
    datasets==2.21.0 \
    accelerate==0.34.0 \
    peft==0.12.0 \
    trl==0.11.0 \
    huggingface_hub==0.25.0 \
    --quiet
python3.11 -c "import transformers; print(f'  transformers {transformers.__version__} OK')"

# Step 5: captum for attribution
echo "[5/8] Installing captum 0.7.0..."
pip install captum==0.7.0 --quiet
python3.11 -c "import captum; print(f'  captum {captum.__version__} OK')"

# Step 6: Scientific computing and evaluation
echo "[6/8] Installing scientific computing stack..."
pip install \
    numpy==1.26.4 \
    scipy==1.14.0 \
    pandas==2.2.0 \
    matplotlib==3.9.0 \
    seaborn==0.13.2 \
    omegaconf==2.3.0 \
    lm-eval==0.4.4 \
    --quiet

# Step 7: CLI and utilities
echo "[7/8] Installing CLI and utilities..."
pip install \
    typer==0.12.0 \
    rich==13.7.0 \
    tqdm==4.66.0 \
    jsonlines==4.0.0 \
    pyyaml==6.0.1 \
    --quiet

# Step 8: Testing and quality
echo "[8/8] Installing testing tools..."
pip install pytest==8.3.0 pytest-cov==5.0.0 ruff==0.4.0 --quiet

# Pin exact versions
echo "Pinning exact versions to requirements_locked.txt..."
pip freeze > requirements_locked.txt
echo "  Saved $(wc -l < requirements_locked.txt) packages to requirements_locked.txt"

echo ""
echo "======================================================"
echo " Setup complete. Run: python experiments/gate0_env_check.py"
echo "======================================================"
