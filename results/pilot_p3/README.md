# Pilot P3: FQ-QAT + NPO Stability Test

## Status: PENDING EXECUTION

**Objective**: Verify that training with NPO + FQ-QAT on forget01 is numerically
stable before committing to multi-seed full experiments.

**Configuration**:
- Model: Llama-3.2-1B-Instruct at NF4 (base) + LoRA adapters
- Starting from: Xafire/GATE1-NPO-Baseline (NPO-unlearned checkpoint)
- FQ-QAT: FQLoRALinear subclass (fake quantisation on adapter output)
- Training: NPO loss on forget01 (40 examples), 5 epochs, seed 42
- LR: 1e-5, gradient clip: 1.0, batch size: 1

**Why forget01 (not forget05)**:
This is a stability test, not a quality test. forget01 has only 40 examples,
making each epoch fast. We need to observe 5 full epochs of loss/gradient
behaviour, not achieve good unlearning metrics.

## Results

| Epoch | Mean Loss | Max Grad Norm | Coherent? | Timestamp |
|-------|-----------|---------------|-----------|-----------|
| 1     | PENDING   | PENDING       | PENDING   | —         |
| 2     | PENDING   | PENDING       | PENDING   | —         |
| 3     | PENDING   | PENDING       | PENDING   | —         |
| 4     | PENDING   | PENDING       | PENDING   | —         |
| 5     | PENDING   | PENDING       | PENDING   | —         |

## Stability Verdict

| Metric | Value | Threshold |
|--------|-------|-----------|
| Loss trajectory | PENDING | Monotonically decreasing per epoch |
| Max epoch-mean gradient norm | PENDING | < 1.0 |
| Coherence failures | PENDING | ≤ 2 out of 5 |

**VERDICT: PENDING** — awaiting Kaggle execution.

## Accept / Fail Criteria (EVB Table 11)

| Condition | Verdict |
|-----------|---------|
| Loss decreases monotonically per epoch mean, max step-level grad norm may spike but epoch-mean < 1.0, no coherence failures | **PASS** |
| Any epoch shows loss increase > 3× previous epoch's final loss, OR grad norm consistently > 5.0, OR coherence fails > 2 epochs | **FAIL** |

## If FAIL

If P3 fails at LR=1e-5:
1. Retry at LR=1e-6 (lower learning rate reduces gradient instability)
2. If still fails: reduce FQ-QAT to apply on q_proj only (not all 4 attention projections)
3. If still fails: abandon FQ-QAT and use QUAIL (C8) as the primary QRA defence

## Files

- Experiment script: `experiments/pilot_p3_fqat_stability.py`
- Training log: `pilot_p3_checkpoints/training_log.csv` (after execution)
- Kaggle notebook: `experiments/notebooks/kaggle/PILOT_P3_P4.ipynb` (Cell 5)
