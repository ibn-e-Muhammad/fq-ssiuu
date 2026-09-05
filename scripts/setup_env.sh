#!/usr/bin/env bash
set -euo pipefail

echo "=========================================================================="
echo "ERROR: scripts/setup_env.sh is DEPRECATED and INACTIVE (August 2026 legacy)."
echo "This script was Colab-specific and performed unsafe environment mutations."
echo ""
echo "For generic local development (requires active virtual environment):"
echo "  python3 -m venv .venv"
echo "  source .venv/bin/activate"
echo "  python3 -m pip install -e .[quantisation,attribution,evaluation,dev]"
echo ""
echo "For Kaggle T4 execution:"
echo "  CUDA_VISIBLE_DEVICES=0 python3 scripts/kaggle_setup.py --preview"
echo "  CUDA_VISIBLE_DEVICES=0 python3 scripts/kaggle_setup.py --apply"
echo "=========================================================================="
exit 1
