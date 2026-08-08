# Pilot P2: SSIUU Attribution Feasibility Test

## Status: PENDING EXECUTION

**Objective**: Determine whether captum's LayerIntegratedGradients can run on
Llama-3.2-1B within 16 GB VRAM (single T4 GPU).

**Configuration**:
- Model: Llama-3.2-1B-Instruct at FP16 (BF16 not supported on T4)
- Attribution method: captum.attr.LayerIntegratedGradients
- Target layer: model.model.layers[14] (middle of 28-layer model)
- IG steps: 20 (fast approximation)
- Batch size: 1 (memory constraint)
- Baseline: zero-embedding tensor

**Why FP16 + full precision (not NF4)**:
Integrated Gradients requires gradient computation on model weights.
bitsandbytes NF4 quantised weights are not differentiable. Attribution
must run on the full-precision model.

## Results

| Metric | Value |
|--------|-------|
| VRAM before IG | PENDING |
| VRAM peak during IG | PENDING |
| VRAM margin (16 GB - peak) | PENDING |
| Time per batch | PENDING |
| OOM occurred | PENDING |
| Attribution tensor shape | PENDING |

## Verdict

**PENDING** — awaiting Kaggle execution.

| Peak VRAM | Verdict | Attribution Method |
|-----------|---------|-------------------|
| < 14 GB | PASS | captum IG (n_steps=20) |
| 14–16 GB | MARGINAL | captum IG (n_steps=10) |
| OOM | FAIL | GradientNormAttribution (fallback in ssiuu_reg.py) |

## Fallback

The gradient-norm fallback is already implemented in
`src/fq_ssiuu/attribution/ssiuu_reg.py` (GradientNormAttribution class).
If P2 fails (OOM), the training loop uses GradientNormAttribution
with no other code changes needed.

## Files

- Experiment script: `experiments/pilot_p2_attribution.py`
- Kaggle notebook: `experiments/notebooks/kaggle/PILOT_P1_P2.ipynb` (Cell 6)
