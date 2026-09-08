# FQ-SSIUU: Verifiable Machine Unlearning Against Quantisation Recovery Attacks in Large Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11/3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://python.org)
[![Status: In Development](https://img.shields.io/badge/status-in%20development-orange.svg)]()

## Overview

FQ-SSIUU is a proposed machine unlearning framework under active implementation and evaluation for LLMs. It combines Negative Preference Optimisation (NPO), SSIUU-style attribution-guided regularisation, and Fake-Quantisation QAT (FQ-QAT) with Straight-Through Estimators aiming to produce unlearning results that are verifiable and resilient against Quantisation Recovery Attacks (QRA) at INT4 deployment precision.

## Scientific Research Status

> [!IMPORTANT]
> This research project is currently in the **implementation and environment verification stage**.
> - **Hypothesis Status:** Proposed framework undergoing systematic evaluation. Full experimental hypothesis is NOT yet verified.
> - **Pilot Results:** Preliminary pilot tests demonstrate technical feasibility only.
> - **Scientific Blockers (Unresolved):**
>   - **RG-001:** SSIUU attribution gradient flow and loss formulation.
>   - **RG-002:** Training-time FQ vs deployment-time bitsandbytes NF4 quantisation location mismatch.
>   - **RG-003:** QUAIL objective fidelity and retain loss integration.
>   - **RG-004:** TOFU Forget Quality interpretation.
> - **Execution Target:** Canonical NVIDIA Tesla T4 single-device FP16 compute environment on Kaggle.

## The Problem We Target

Standard machine unlearning methods pass full-precision or FP16 evaluation audits but degrade when the deployed model is quantised to INT4/NF4 precision (the Quantisation Recovery Attack, or QRA). We evaluate our proposed method at Llama-3.2-1B scale against this vulnerability.

## Installation & Setup

### Generic Local Development
Local installation requires an active virtual environment to prevent accidental global Python environment mutations:

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install package with local development dependency groups
python3 -m pip install -e ".[quantisation,attribution,evaluation,dev]"

# Or using Makefile (enforces active VIRTUAL_ENV)
make setup-dev
```

### Kaggle Execution Overlay (OG-001)
For canonical Kaggle T4 execution, the environment uses a protected decoupled architecture. See `requirements/README.md` and `handbook/KAGGLE_ENVIRONMENT_ARCHITECTURE.md`.

```bash
# Preview dry-run report validation (non-mutating, fail-closed)
CUDA_VISIBLE_DEVICES=0 python3 scripts/kaggle_setup.py --preview

# Apply overlay installation (requires explicit --apply and CUDA_VISIBLE_DEVICES=0)
CUDA_VISIBLE_DEVICES=0 python3 scripts/kaggle_setup.py --apply
```

## Research Gate Execution

Research gates are executed directly via script:

```bash
python scripts/research_gate.py
```

## Unlearning Durability Toolkit (UDT)

The UDT CLI generates a structured JSON **Empirical Durability Report**:

```bash
unlearn-toolkit certify \
  --model checkpoints/c6_fq_ssiuu/seed42 \
  --base-model meta-llama/Llama-3.2-1B-Instruct \
  --forget-set data/processed/tofu_forget05.jsonl \
  --output certificates/my_durability_report.json
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
