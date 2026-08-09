# Pilot P2: SSIUU Attribution Feasibility Test

## Status: ✅ PASS — CAPTUM IG FEASIBLE

**Objective**: Determine whether captum's LayerIntegratedGradients can run on
Llama-3.2-1B within 16 GB VRAM (single T4 GPU).

**Configuration**:
- Model: Llama-3.2-1B-Instruct at FP16 (BF16 not supported on T4, CC 7.5)
- Attribution method: `captum.attr.LayerIntegratedGradients`
- Target layer: `model.model.layers[14]` (middle of 28-layer model)
- IG steps: 20
- Batch size: 1 (set for memory safety; captum auto-adjusted to 2 due to input count)
- Baseline: zero-embedding tensor
- Input: 2 forget examples, padded/truncated to 40 tokens

**Why FP16 + full precision (not NF4)**:
Integrated Gradients requires gradient computation through model weights
(∂output/∂input via autograd). bitsandbytes NF4 quantised weights are stored
as integers and are NOT differentiable — `torch.autograd` cannot backpropagate
through them. Attribution must therefore run on the full-precision model.

## Results

| Metric | Value |
|--------|-------|
| VRAM before IG | 2.302 GB |
| VRAM peak during IG | 2.511 GB |
| VRAM margin (16 GB − peak) | **12.489 GB** |
| Time for IG (2 examples) | 1.1 seconds |
| OOM occurred | **NO** ✅ |
| Attribution tensor shape | [2, 40, 2048] |
| Attribution mean | 5.023 × 10⁻³ |
| Attribution max | 186.686 |
| Non-zero fraction | 97.49% |

## Verdict

**✅ PASS — Peak VRAM 2.511 GB << 14 GB threshold**

Recommended attribution method: **captum IG (n_steps=20)**

| Peak VRAM | Verdict | Attribution Method |
|-----------|---------|-------------------|
| **< 14 GB** | **PASS** ← (**this result: 2.5 GB**) | captum IG (n_steps=20) |
| 14–16 GB | MARGINAL | captum IG (n_steps=10) |
| OOM | FAIL | GradientNormAttribution (fallback in ssiuu_reg.py) |

## Interpretation

The Llama-3.2-1B model at FP16 occupies only ~2.3 GB of VRAM, and Integrated
Gradients with 20 interpolation steps adds a mere ~0.2 GB overhead. The 12.5 GB
remaining margin means we can comfortably:

1. **Increase IG steps** to 50 or 100 for higher-fidelity attribution if needed
2. **Process larger batches** — potentially batch_size=4 or more
3. **Use longer sequences** — well beyond the 40-token test here (max_seq_len=128)
4. **Run IG during SSIUU training** alongside the model and optimizer in memory

This massive VRAM margin confirms that captum IG is the right attribution method
for the SSIUU regulariser, and the GradientNormAttribution fallback is not needed.

## Forward Function Design

The captum-compatible forward function uses `inputs_embeds` (not `input_ids`)
to enable gradient flow through the embedding layer:

```python
def embed_forward_func(embeddings):
    outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    target_ids = input_ids[:, 1:]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    target_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
    return target_log_probs.sum(dim=1)  # Shape: (batch_size,)
```

Key design decisions:
- Returns shape `(batch_size,)` — required by captum for per-example attribution
- Uses `sum(dim=1)` over sequence tokens, NOT `sum()` over entire batch
- Log-probability of true next tokens as the attribution target

## Fallback

The gradient-norm fallback is already implemented in
`src/fq_ssiuu/attribution/ssiuu_reg.py` (GradientNormAttribution class).
Since P2 passed with enormous margin, the fallback is **not needed** but remains
available as a safety net.

## Files

- `pilot_p2_results.json`: Complete measurement data
- Experiment script: `experiments/pilot_p2_attribution.py`
- Kaggle notebook: `experiments/notebooks/kaggle/PILOT_P1_P2.ipynb` (Cell 5)
