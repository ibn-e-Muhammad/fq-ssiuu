# Pilot P4: QUAIL Baseline

## Status: PENDING EXECUTION

**Objective**: Establish the Q-INT4 score for the QUAIL logit-hinge unlearning baseline (Condition C8). This will be compared against the NPO baseline (Condition C2, Q-INT4 = 0.2149).

**Configuration**:
- Model: Llama-3.2-1B-Instruct at NF4 (base) + FRESH LoRA adapters
- Reference Model: open-unlearning/tofu_Llama-3.2-1B-Instruct_full (frozen)
- Unlearning Method: QUAIL (C8), forget05, seed 42
- Margin: 4.2e-4 (half of NF4 bin width 8.4e-4)
- Training: 10 epochs, LR 1e-5, gradient clip 1.0, batch size 1 (accum 32)
- QRA Evaluation: fp16, int8, nf4

## Results

| Precision | FQ | MU | Timestamp |
|-----------|----|----|----------|
| FP16 (baseline) | PENDING | PENDING | — |
| INT8 | PENDING | PENDING | — |
| INT4 (NF4) | PENDING | PENDING | — |

## QRA Score

**QUAIL Q-INT4** = |FQ_INT4 - FQ_BF16| / (FQ_BF16 + ε) = **PENDING**

**NPO Q-INT4** = 0.2149 (from Pilot P1)

## Verdict

**PENDING** — awaiting Kaggle execution.

| QUAIL Q-INT4 vs NPO | Verdict | Action |
|---------------------|---------|--------|
| QUAIL < NPO | PASS — QUAIL more robust | Adopt QUAIL if FQ-QAT fails |
| QUAIL ≈ NPO | TIE — No significant diff | Prefer NPO as simpler baseline |
| QUAIL > NPO | FAIL — QUAIL less robust | Discard QUAIL as QRA defence |

## Protocol

- Train on fine-tuned base (not NPO checkpoint) for fair comparison.
- QUAIL loss operates on logits, so no FQ-QAT wrapping is needed.
- Merge LoRA adapter into base model before QRA evaluation.
- Run QRA evaluation at three precisions.

## Files

- `pilot_p4_metadata.json`: Experiment metadata
- `pilot_p4_results.json`: QRA evaluation results
- `training_log.csv`: Training metrics per step
- Experiment script: `experiments/pilot_p4_quail_baseline.py`
- Kaggle notebook: `experiments/notebooks/kaggle/PILOT_P3_P4.ipynb`
