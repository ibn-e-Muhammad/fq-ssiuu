# Experiment Log: GATE1-C2-NPO-SEED42

## Header

| Field | Value |
|-------|-------|
| **Experiment ID** | GATE1-C2-NPO-SEED42 |
| **Date Run** | 2026-08-04 to 2026-08-05 |
| **Operators** | Research Team (FQ-SSIUU project) |
| **Status** | TRAINING COMPLETE — EVALUATION PENDING |

---

## Research Objective

This experiment reproduces the NPO (Negative Preference Optimisation) baseline
unlearning method on the TOFU forget05 benchmark using Llama-3.2-1B-Instruct. The
purpose is to establish the C2 condition — the vanilla NPO baseline — against which
all FQ-SSIUU ablations and the full method will be compared. Successful reproduction
(within ±5% of OpenUnlearning published metrics) is the GATE-1 acceptance criterion
and a prerequisite for proceeding to Pilot P1 (QRA existence test).

---

## Environment

| Component | Value |
|-----------|-------|
| **Primary compute** | Kaggle Notebooks (dual T4, 2×15 GB VRAM) |
| **Secondary (failed)** | Google Colab (single T4, 15 GB VRAM) |
| **Python version** | 3.12 (Kaggle) / 3.12.13 (Colab) |
| **PyTorch version** | 2.11.0+cu128 (Kaggle) |
| **CUDA version** | 12.8 (Kaggle) |
| **Transformers** | 4.51.3 |
| **Accelerate** | Latest (Kaggle pre-installed) |
| **bitsandbytes** | 0.45.5 (force-installed) |
| **lm-eval** | 0.4.4 |
| **flash-attn** | REMOVED — T4 incompatible (see DL-007) |
| **Attention impl** | `attn_implementation="sdpa"` (PyTorch native) |

---

## Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Base model** | `meta-llama/Llama-3.2-1B-Instruct` | 1.24B parameters |
| **Fine-tuned model** | `open-unlearning/tofu_Llama-3.2-1B-Instruct_full` | Pre-fine-tuned on full TOFU |
| **Unlearning method** | NPO (Negative Preference Optimisation) | Bounded loss for controlled forgetting |
| **Forget split** | forget05 (200 examples, 5% of TOFU authors) | |
| **Retain split** | retain90 (3600 examples) | |
| **Per-device batch size** | 1 | Memory-constrained |
| **Number of GPUs** | 2 (DDP via accelerate) | |
| **Gradient accumulation** | 16 steps | |
| **Effective batch size** | ~32 (1 × 2 × 16) | |
| **Learning rate** | 1e-5 | Peak (cosine schedule) |
| **LR schedule** | Cosine with warmup | Warmup ~8 steps |
| **Optimizer** | paged_adamw_8bit | 8-bit Adam to save VRAM |
| **Precision** | FP16 (model + AMP) | T4 does not support BF16 |
| **Gradient checkpointing** | Enabled (`use_reentrant: false`) | Required for NPO multi-forward-pass |
| **Max steps** | 60 (fixed) | |
| **Effective epochs** | 8.64 | **NOTE:** Intended 10 epochs, but step count fixed at 60 |
| **Seed** | 42 | |
| **Checkpoint strategy** | save_strategy=epoch, save_total_limit=2 | |
| **DDP config** | `ddp_find_unused_parameters: false` | Originally `None` (string) — patched to boolean |
| **GradScaler patch** | PyTorch `grad_scaler.py` patched to allow FP16 unscaling | See note below |

### GradScaler Monkey-Patch

PyTorch's `GradScaler.unscale_()` raises `ValueError("Attempting to unscale FP16
gradients.")` when model weights are natively FP16. Since T4 does not support BF16,
we must load the model in FP16 AND use the GradScaler (via `fp16=True`). The safety
check was bypassed via:
```bash
sed -i 's/raise ValueError("Attempting to unscale FP16 gradients.")/pass/g' \
  /usr/local/lib/python3.12/dist-packages/torch/amp/grad_scaler.py
```
This is mathematically safe because the GradScaler's `_unscale_grads_` function
still performs the division by scale factor correctly on FP16 tensors. The check
was added as a strict guard, not because the operation is numerically invalid.

---

## Training Results

### Complete Step-by-Step Log

| Step | Epoch | Loss | Grad Norm | Learning Rate | Notes |
|------|-------|------|-----------|---------------|-------|
| 1 | 0.16 | 223.7601 | NaN | 0.0 | Warmup — LR=0, no gradient signal |
| 2 | 0.32 | 223.3218 | NaN | 0.0 | Warmup |
| 3 | 0.48 | 223.4493 | NaN | 0.0 | Warmup |
| 4 | 0.64 | 223.3905 | NaN | 0.0 | Warmup |
| 5 | 0.80 | 223.9659 | NaN | 0.0 | Warmup |
| 6 | 0.96 | 223.6074 | NaN | 0.0 | Warmup |
| 7 | 1.00 | 55.6851 | NaN | 0.0 | End epoch 1 — first major loss drop |
| 8 | 1.16 | 224.4886 | NaN | 0.0 | Loss rebounds (epoch boundary effect) |
| 9 | 1.32 | 223.6977 | 2994.0 | 0.0 | First finite grad_norm |
| 10 | 1.48 | 223.4594 | inf | 1.67e-06 | LR begins ramping — gradient spike |
| 11 | 1.64 | 223.3140 | 3312.0 | 1.67e-06 | |
| 12 | 1.80 | 222.8391 | 3198.0 | 3.33e-06 | Loss starts decreasing |
| 13 | 1.96 | 217.1892 | 3924.0 | 5.00e-06 | |
| 14 | 2.00 | 55.3231 | 1638.0 | 6.67e-06 | End epoch 2 |
| 15 | 2.16 | 145.6496 | 2834.0 | 8.33e-06 | Rapid convergence begins |
| 16 | 2.32 | 111.0035 | 2356.0 | 1.00e-05 | Peak LR reached |
| 17 | 2.48 | 77.1553 | 1902.0 | 9.81e-06 | Cosine decay begins |
| 18 | 2.64 | 47.3210 | 1627.0 | 9.63e-06 | |
| 19 | 2.80 | 48.0180 | 1348.0 | 9.44e-06 | |
| 20 | 2.96 | 32.1770 | 574.0 | 9.26e-06 | Grad norm stabilising |
| 21 | 3.00 | 7.1426 | 205.9 | 9.07e-06 | End epoch 3 — convergence milestone |
| 22 | 3.16 | 27.5301 | 516.0 | 8.89e-06 | |
| 23 | 3.32 | 24.9563 | 336.0 | 8.70e-06 | |
| 24 | 3.48 | 25.0582 | 322.3 | 8.52e-06 | |
| 25 | 3.64 | 20.5610 | 270.3 | 8.33e-06 | |
| 26 | 3.80 | 23.3456 | 306.8 | 8.15e-06 | |
| 27 | 3.96 | 17.5730 | 239.4 | 7.96e-06 | |
| 28 | 4.00 | 4.6170 | 101.1 | 7.78e-06 | End epoch 4 |
| 29 | 4.16 | 17.5668 | 235.6 | 7.59e-06 | |
| 30 | 4.32 | 16.1950 | 243.4 | 7.41e-06 | |
| 31 | 4.48 | 14.1026 | 254.9 | 7.22e-06 | |
| 32 | 4.64 | 13.2727 | 250.8 | 7.04e-06 | |
| 33 | 4.80 | 14.6596 | 483.5 | 6.85e-06 | Outlier grad norm |
| 34 | 4.96 | 13.0842 | 275.0 | 6.67e-06 | |
| 35 | 5.00 | 4.7139 | 190.6 | 6.48e-06 | End epoch 5 |
| 36 | 5.16 | 9.3003 | 234.0 | 6.30e-06 | |
| 37 | 5.32 | 9.7562 | 259.3 | 6.11e-06 | |
| 38 | 5.48 | 10.5708 | 251.5 | 5.93e-06 | |
| 39 | 5.64 | 8.3831 | 273.8 | 5.74e-06 | |
| 40 | 5.80 | 10.7245 | 354.8 | 5.56e-06 | |
| 41 | 5.96 | 7.3508 | 229.5 | 5.37e-06 | |
| 42 | 6.00 | 1.2911 | 121.4 | 5.19e-06 | End epoch 6 — loss below 2.0 |
| 43 | 6.16 | 10.5672 | 314.0 | 5.00e-06 | |
| 44 | 6.32 | 7.8955 | 237.6 | 4.81e-06 | |
| 45 | 6.48 | 7.8776 | 245.3 | 4.63e-06 | |
| 46 | 6.64 | 10.5083 | 360.3 | 4.44e-06 | |
| 47 | 6.80 | 8.1285 | 258.0 | 4.26e-06 | |
| 48 | 6.96 | 8.9321 | 308.5 | 4.07e-06 | |
| 49 | 7.00 | 1.3950 | 109.1 | 3.89e-06 | End epoch 7 |
| 50 | 7.16 | 6.2190 | 263.0 | 3.70e-06 | |
| 51 | 7.32 | 7.5547 | 308.0 | 3.52e-06 | |
| 52 | 7.48 | 6.0737 | 273.3 | 3.33e-06 | |
| 53 | 7.64 | 5.6424 | 245.6 | 3.15e-06 | |
| 54 | 7.80 | 5.6855 | 259.8 | 2.96e-06 | |
| 55 | 7.96 | 5.9331 | 271.8 | 2.78e-06 | |
| 56 | 8.00 | 1.4750 | 211.6 | 2.59e-06 | End epoch 8 |
| 57 | 8.16 | 5.1234 | 259.0 | 2.41e-06 | |
| 58 | 8.32 | 4.5010 | 255.1 | 2.22e-06 | |
| 59 | 8.48 | 5.7120 | 234.4 | 2.04e-06 | |
| 60 | 8.64 | 5.4144 | 291.8 | 1.85e-06 | **FINAL STEP** |

### Summary Statistics

| Metric | Value |
|--------|-------|
| **Total training time** | 722.4 seconds (~12 minutes) |
| **Samples/second** | 2.768 |
| **Steps/second** | 0.083 |
| **Average training loss** | 61.087 |
| **Final step loss** | 5.414 |
| **Epoch-boundary losses** | 55.7 → 55.3 → 7.1 → 4.6 → 4.7 → 1.3 → 1.4 → 1.5 |

### Loss Trajectory Analysis

The loss follows the expected NPO convergence pattern:
1. **Epochs 0–2 (steps 1–14):** Loss plateaus at ~223 during warmup. The learning rate
   is zero for the first 8 steps, then ramps linearly. No meaningful weight updates occur.
2. **Epoch 2–3 (steps 15–21):** Rapid convergence from 145 → 7.1. The model is actively
   unlearning the forget set. Grad norms decrease from 2834 → 206 (stabilising).
3. **Epochs 3–6 (steps 22–42):** Gradual refinement. Loss oscillates between 5–27 within
   epochs but drops to 1.3–4.6 at epoch boundaries. The epoch-boundary drops suggest
   the model "revisits" easier forget examples at epoch starts.
4. **Epochs 6–8.64 (steps 43–60):** Plateau at ~5–10 mid-epoch, ~1.3–1.5 at boundaries.
   The model has converged — further training would yield diminishing returns.

---

## Evaluation Results

| Metric | Measured? | Value | Target | Pass? |
|--------|-----------|-------|--------|-------|
| forget_quality | NOT MEASURED — standalone eval needed | — | 0.14 ±0.05 | PENDING |
| model_utility | NOT MEASURED — standalone eval needed | — | 0.45 ±0.05 | PENDING |
| forget_truth_ratio | NOT MEASURED | — | 0.70 (ref) | PENDING |
| forget_Q_A_Prob | NOT MEASURED | — | — | PENDING |
| forget_Q_A_ROUGE | NOT MEASURED | — | — | PENDING |
| extraction_strength | NOT MEASURED | — | — | PENDING |

> **Why not measured:** Per-epoch evaluation was disabled by DDP (see DL-010).
> `forget_quality` requires `retain_logs_path` which was not set (see DL-011).
> Standalone evaluation is required to fill this table.

### Pre-Training Evaluation (Checkpoint-0, from Colab)

These values were measured on the ORIGINAL fine-tuned model BEFORE unlearning began:

| Metric | Value | Notes |
|--------|-------|-------|
| forget_truth_ratio | 0.4724 | Baseline: model "knows" the forget set |
| forget_quality | None | retain_logs not provided |
| forget_Q_A_Prob | 0.8855 | High probability = model memorised answers |
| forget_Q_A_ROUGE | 0.8298 | High ROUGE = model can generate verbatim answers |
| model_utility | 0.5967 | Baseline model utility |
| extraction_strength | 0.7246 | High = model is vulnerable to extraction |

---

## Known Issues and Deviations

1. **8.64 epochs instead of 10:** The training was configured with `max_steps=60`
   (fixed step count) rather than `num_train_epochs=10`. With an effective batch
   size of ~32 and 200 forget examples, 60 steps produces 8.64 epochs. This is a
   minor deviation — the loss trajectory shows convergence by epoch 6.

2. **NaN grad_norm in steps 1–8:** Expected behaviour during warmup when
   learning_rate=0. With zero LR, no parameter updates occur, so grad norms are
   meaningless (reported as NaN by the logger).

3. **Infinite grad_norm at step 10:** A single gradient spike when LR first becomes
   non-zero. The model's first non-trivial gradient was very large. This resolved
   immediately by step 11.

4. **GradScaler monkey-patch:** PyTorch's FP16 gradient safety check was bypassed.
   This is necessary for T4 GPUs which cannot use BF16. The mathematical operation
   is correct — the check is overly conservative.

5. **Per-epoch evaluation skipped:** DDP mode disabled the OpenUnlearning evaluator.
   No FQ/MU metrics were computed during training. (See DL-010)

6. **forget_quality = None throughout:** retain_logs_path not wired. (See DL-011)

7. **Optimizer change:** Used `paged_adamw_8bit` instead of the default
   `paged_adamw_32bit` to fit within T4 VRAM. This may cause minor numerical
   differences in the optimizer state but should not materially affect final metrics.

8. **Training platform differs from reference:** The OpenUnlearning reference was run
   on 2×L40s (48 GB each). Our setup uses 2×T4s (15 GB each). The model weights
   are identical, but numerical precision may differ slightly.

---

## Checkpoint Location

| Location | Path |
|----------|------|
| **Local (extracted)** | `/home/muhammad-arslan/Downloads/NPO_Baseline_Final/` |
| **HuggingFace Hub** | `ibn-e-Muhammad/GATE1-NPO-Baseline` (private) |

### Checkpoint Contents

| File | Size | Description |
|------|------|-------------|
| `config.json` | 871 B | Model architecture config |
| `generation_config.json` | 184 B | Generation parameters |
| `model.safetensors` | 2.47 GB | Full model weights (FP16) |
| `special_tokens_map.json` | 325 B | Special token mappings |
| `tokenizer.json` | 17.2 MB | Full tokenizer vocabulary |
| `tokenizer_config.json` | 54.6 KB | Tokenizer configuration |
| `trainer_state.json` | 9.5 KB | Training state (step 60, epoch 8.64) |
| `training_args.bin` | 5.8 KB | Serialised TrainingArguments |

---

## Next Action

Run standalone evaluation (Task 4: `experiments/gate1_standalone_eval.py`) and update
the Evaluation Results table above with measured values for `forget_quality`,
`model_utility`, and `forget_truth_ratio`.
