# Pilot P1: QRA Existence Test

## Status: PENDING EXECUTION

**Objective**: Determine whether the Quantisation Recovery Attack (QRA) manifests
at 1B parameter scale using bitsandbytes NF4 quantisation.

**Checkpoint**: Xafire/GATE1-NPO-Baseline (HuggingFace Hub, private)
**Base Model**: meta-llama/Llama-3.2-1B-Instruct
**Unlearning Method**: NPO (C2), forget05, seed 42

**BF16 FQ (known from GATE-1)**: 0.3937
**BF16 MU (known from GATE-1)**: 0.5545

## Results

| Precision | FQ | MU | Timestamp |
|-----------|----|----|----------|
| FP16 (baseline) | 0.3937 | 0.5545 | 2026-08-08 (GATE-1) |
| INT8 | PENDING | PENDING | — |
| INT4 (NF4) | PENDING | PENDING | — |

## QRA Score

**Q-INT4** = |FQ_INT4 - FQ_BF16| / (FQ_BF16 + ε) = **PENDING**

## Verdict

**PENDING** — awaiting Kaggle execution.

| Q-INT4 Range | Verdict | Action |
|-------------|---------|--------|
| > 0.05 | PASS — QRA confirmed | Proceed to P3+P4 |
| 0.02–0.05 | BORDERLINE | Run 2 more seeds |
| < 0.02 | FAIL — QRA absent | Restructure project (EVB §10 F2) |

## Protocol

- Quantisation: bitsandbytes NF4 (NOT GPTQ/AWQ)
- Evaluation: OpenUnlearning TOFU suite with retain_logs_path
- Single GPU only (DDP incompatible with evaluator)
- Each precision loaded FRESH from disk (never re-quantised in memory)

## Files

- `pilot_p1_results.json`: Complete evaluation data (after execution)
- Experiment script: `experiments/pilot_p1_qra_check.py`
- Kaggle notebook: `experiments/notebooks/kaggle/PILOT_P1_P2.ipynb`
