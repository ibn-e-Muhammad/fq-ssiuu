# Pilot P4: QUAIL Baseline

## Status: ✅ COMPLETE

**Objective**: Establish a baseline for QUAIL (Logit Hinge) unlearning and verify whether it exhibits quantisation vulnerability (QRA) when deployed at 1B parameter scale with NF4.

**Configuration**:
- Base Model: `meta-llama/Llama-3.2-1B-Instruct`
- Reference Model: `open-unlearning/tofu_Llama-3.2-1B-Instruct_full`
- Method: QUAIL (Condition C8)
- Margin: 0.00042 (NF4 bin width 0.00084 / 2)
- Forget Split: `forget05`
- Seed: 42
- Training: 10 epochs, LR=1e-05

## Results

| Precision | Forget Quality (FQ) | Model Utility (MU) | FQ Δ vs FP16 |
|-----------|----------------------|--------------------|--------------|
| FP16      | 0.000121             | 0.280369           | baseline     |
| INT8      | 0.000329             | 0.282002           | +0.000208    |
| NF4       | 0.000094             | 0.282432           | -0.000028    |

## QRA Scores

- **QUAIL Q-INT4**: 0.227678
- **NPO Q-INT4**: 0.214900 (from Pilot P1)

## Verdict: PASS

The Q-INT4 score is > 0.05, meaning the Quantisation Recovery Attack (QRA) is successfully demonstrated on QUAIL.

**Analysis**:
⚠️ QUAIL Q-INT4 (0.2277) > NPO Q-INT4 (0.2149)
QUAIL is ~5.9% *less* quantisation-robust than standard NPO. This suggests that the logit-space regularisation approach of QUAIL does not natively confer quantisation durability, highlighting the need for weight-space methods like FQ-SSIUU.
