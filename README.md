# FQ-SSIUU: Verifiable LLM Unlearning Against Quantisation Recovery Attacks

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11/3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://python.org)
[![Status: In Development](https://img.shields.io/badge/status-in%20development-orange.svg)]()

## Overview

FQ-SSIUU is a machine unlearning method for LLMs that combines Negative Preference Optimisation (NPO), SSIUU-style attribution-guided regularisation, and Fake-Quantisation QAT (FQ-QAT) with Straight-Through Estimators to produce unlearning results that are verifiable and durable at INT4 deployment precision.

This repository accompanies our paper:

> **FQ-SSIUU: Verifiable LLM Unlearning Durable Under INT4 Quantisation**
> [Authors], [Year]
> [arXiv link — to be added]

## The Problem We Solve

Standard machine unlearning methods pass BF16 evaluation audits but fail when the deployed model is quantised to INT4 (the Quantisation Recovery Attack, or QRA). We empirically characterise this at Llama-3.2-1B scale and evaluate our method against this attack.

## Installation & Setup

### Generic Local Development
```bash
# Install package with all local development dependency groups
pip install -e .[quantisation,attribution,evaluation,dev]

# Or using Makefile
make setup-dev
```

### Kaggle Execution Overlay (OG-001)
For canonical Kaggle T4 execution, the environment uses a protected decoupled architecture. See `requirements/README.md` and `handbook/KAGGLE_ENVIRONMENT_ARCHITECTURE.md`.
```bash
# Preview dry-run report validation (non-mutating)
python scripts/kaggle_setup.py --preview

# Apply overlay installation (requires explicit --apply and CUDA_VISIBLE_DEVICES=0)
CUDA_VISIBLE_DEVICES=0 python scripts/kaggle_setup.py --apply
```

## Unlearning Durability Toolkit (UDT)

The UDT CLI generates a structured JSON durability report:

```bash
unlearn-toolkit certify \
  --model checkpoints/c6_fq_ssiuu/seed42 \
  --base-model meta-llama/Llama-3.2-1B-Instruct \
  --forget-set data/processed/tofu_forget05.jsonl \
  --output certificates/my_certificate.json
```

## Repository Structure

See [handbook/KAGGLE_ENVIRONMENT_ARCHITECTURE.md](handbook/KAGGLE_ENVIRONMENT_ARCHITECTURE.md) and [requirements/README.md](requirements/README.md) for the complete dependency and execution architecture.

## References

- DurableUn (arXiv:2605.02196)
- SSIUU (arXiv:2509.22263)
- QUAIL (arXiv:2601.15538)
- OpenUnlearning (arXiv:2506.12618)
- TOFU (Maini et al., COLM 2024)

## Licence

MIT. See [LICENSE](LICENSE).
