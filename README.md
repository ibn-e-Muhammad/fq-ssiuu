# FQ-SSIUU: Verifiable LLM Unlearning Against Quantisation Recovery Attacks

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![Status: In Development](https://img.shields.io/badge/status-in%20development-orange.svg)]()

## Overview

FQ-SSIUU is a machine unlearning method for LLMs that combines Negative Preference Optimisation (NPO), SSIUU-style attribution-guided regularisation, and Fake-Quantisation QAT (FQ-QAT) with Straight-Through Estimators to produce unlearning results that are verifiable and durable at INT4 deployment precision.

This repository accompanies our paper:

> **FQ-SSIUU: Verifiable LLM Unlearning Durable Under INT4 Quantisation**
> [Authors], [Year]
> [arXiv link — to be added]

## The Problem We Solve

Standard machine unlearning methods pass BF16 evaluation audits but fail when the deployed model is quantised to INT4 (the Quantisation Recovery Attack, or QRA). We empirically characterise this at Llama-3.2-1B scale and evaluate our method against this attack.

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/fq-ssiuu.git
cd fq-ssiuu
pip install -e .
unlearn-toolkit --help
```

## Unlearning Durability Toolkit (UDT)

The UDT CLI generates a structured JSON durability certificate:

```bash
unlearn-toolkit certify \
  --model checkpoints/c6_fq_ssiuu/seed42 \
  --base-model meta-llama/Llama-3.2-1B-Instruct \
  --forget-set data/processed/tofu_forget05.jsonl \
  --output certificates/my_certificate.json
```

## Repository Structure

See [handbook/stage1_foundation.md](handbook/stage1_foundation.md) for the complete project architecture.

## Reproducing Results

See [docs/reproducibility.md](docs/reproducibility.md).

## References

- DurableUn (arXiv:2605.02196)
- SSIUU (arXiv:2509.22263)
- QUAIL (arXiv:2601.15538)
- OpenUnlearning (arXiv:2506.12618)
- TOFU (Maini et al., COLM 2024)

## Licence

MIT. See [LICENSE](LICENSE).
