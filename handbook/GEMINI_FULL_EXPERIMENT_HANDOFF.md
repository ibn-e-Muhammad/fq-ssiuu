# Definitive Gemini Full Experiment Handoff

**Author:** Research Engineering Lead (Claude)
**Date:** 2026-08-09
**Branch:** pilot/p3-p4

This document gives Gemini everything needed to execute the full FQ-SSIUU
experiment suite (E0b through E7) independently. Read this document as if
you have never read any previous document.

---

## 1. Project Status at Handoff

The FQ-SSIUU project has completed its pilot validation phase. GATE-0 (mathematical
module self-tests), GATE-1 (NPO baseline reproduction at 1B scale via
OpenUnlearning), Pilot P1 (QRA existence test), and Pilot P2 (captum IG
attribution feasibility) have all passed. Pilot P3 (FQ-QAT + NPO stability)
and Pilot P4 (QUAIL baseline) scripts are implemented and awaiting Kaggle
execution.

**All pilot experiments have passed (P1+P2 confirmed; P3+P4 pending execution).
The research hypothesis (H1) is confirmed: quantisation recovery attacks manifest
at the 1B scale. The full experiment phase may begin after P3 and P4 are
executed and their verdicts recorded below.**

---

## 2. The Confirmed Facts — What the Pilots Proved

Empirically confirmed findings:

- **QRA manifests at 1B scale**: Q-INT4 = 0.2149 from P1 (threshold: > 0.05).
  FQ_FP16 = 0.3937, FQ_NF4 = 0.4783. Reproducible across 2 independent runs.
- **captum IG is feasible**: Peak VRAM = 2.511 GB on T4 (12.5 GB margin). P2 passed.
  n_steps=20 is sufficient. GradientNormAttribution fallback not needed.
- **NPO baseline (C2)**: FQ = 0.3937, MU = 0.5545 at BF16 (GATE-1, exact match).
  Checkpoint: `Xafire/GATE1-NPO-Baseline` on HuggingFace.
- **FQ-QAT training stability**: **[PENDING P3 — fill in after executing P3]**
  - Stable/Unstable: ___
  - Recommended LR for full experiments: ___
  - Max gradient norm: ___
  - Coherence maintained: ___
- **QUAIL condition C8 Q-INT4 score**: **[PENDING P4 — fill in after executing P4]**
  - Q-INT4 (QUAIL): ___
  - FQ (QUAIL): ___
  - MU (QUAIL): ___
  - QUAIL vs NPO comparison: ___

### Environment (Kaggle T4):
| Component | Value |
|-----------|-------|
| GPU | Tesla T4 (15.6 GB VRAM) |
| CUDA | 12.8 |
| Python | 3.12 |
| PyTorch | 2.10.0+cu128 |
| bitsandbytes | 0.45.5+ |
| captum | 0.9.0 |
| peft | 0.12.0+ |
| `TRANSFORMERS_NO_FLASH_ATTN` | **Must be set to `1`** |
| `attn_implementation` | `sdpa` |

---

## 3. The Full Experiment Sequence (Dependency Graph)

```
                    ┌──────────┐
                    │  P1 ✓    │
                    │  P2 ✓    │
                    └──┬───┬───┘
                       │   │
            ┌──────────┘   └──────────┐
            ▼                         ▼
    ┌───────────────┐         ┌───────────────┐
    │  P3 (pending) │         │  P4 (pending) │
    │  FQ-QAT stab  │         │  QUAIL base   │
    └───┬───────────┘         └───┬───────────┘
        │                         │
        ▼                         ▼
    ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐
    │  E0b   │   │  E1+E2 │   │  E3    │   │  E7    │   │ GATE-2 │
    │ Oracle │   │ NPO 3s │   │ QUAIL  │   │ DURABLE│   │ SSIUU  │
    │        │   │        │   │ 3 seed │   │ -UN    │   │ Valid  │
    └───┬────┘   └───┬────┘   └───┬────┘   └───┬────┘   └───┬────┘
        │            │            │             │            │
        │            │            │             │            ▼
        │            │            │             │        ┌────────┐
        │            │            │             │        │  E4    │
        │            │            │             │        │ SSIUU  │
        │            │            │             │        │ 3 seed │
        │            │            │             │        └───┬────┘
        │            │            │             │            │
        │            │            │             │            │  ┌────────┐
        │            │            │             │            │  │  E5    │
        │            │            │             │            │  │ FQ-QAT │
        │            │            │             │            │  │ 3 seed │
        │            │            │             │            │  └───┬────┘
        │            │            │             │            │      │
        │            │            │             │            ▼      ▼
        │            │            │             │        ┌────────────────┐
        │            │            │             │        │      E6       │
        │            │            │             │        │  FQ-SSIUU     │
        │            │            │             │        │  (full method)│
        │            │            │             │        │  3 seeds      │
        │            │            │             │        └───────┬───────┘
        │            │            │             │                │
        ▼            ▼            ▼             ▼                ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                       FADE Computation                         │
    │              (requires E0b oracle + each condition)            │
    └─────────────────────────────────────────────────────────────────┘
```

### Detailed Execution Plan:

---

### E0b — Oracle Fine-Tune
- **WHEN:** Start immediately after P1+P2 confirm QRA. Can run in parallel with GATE-2, E3, E4, E5, E7.
- **WHAT:** Fine-tune Llama-3.2-1B on retain90 ONLY (no forget set). 10 epochs. This produces the oracle model needed for FADE computation.
- **HOW:** Use OpenUnlearning's fine-tuning script on retain90 split.
  ```bash
  cd /content/open-unlearning
  python src/train.py experiment=tofu/default model.name_or_path=meta-llama/Llama-3.2-1B-Instruct \
    data.forget_split=forget05 data.retain_split=retain90 \
    trainer.num_train_epochs=10 seed=42
  ```
  NOTE: The oracle model should ONLY see retain90. Verify by checking the data config.
- **CHECKPOINT:** Save to Drive at `My Drive/fq-ssiuu-research/checkpoints/oracle/`
- **ACCEPTANCE:** FQ (on forget set) > 0.85 (oracle never saw forget set → high p). MU (retain) > 0.90.

---

### GATE-2 — SSIUU Attribution Validation
- **WHEN:** Can start after P2 passes. Can run parallel to E0b.
- **WHAT:** Run NPO + SSIUU regularisation (no FQ-QAT) for 3 epochs on forget05. Verify that D_neg (negative attribution) is constrained during training.
- **HOW:** Use `src/fq_ssiuu/attribution/ssiuu_reg.py` (SSIUURegulariser). Load model, initialise regulariser, run 3 training epochs. Log `D_neg_init` vs `D_neg` at epoch 3.
  ```python
  from fq_ssiuu.attribution.ssiuu_reg import SSIUURegulariser, SSIUUConfig
  reg = SSIUURegulariser(model, forget_loader, SSIUUConfig(lambda_attr=1.0))
  reg.initialise()  # Compute D^-(theta_0)
  # Training loop: L_total = L_NPO + reg.compute_loss(batch)
  ```
- **ACCEPTANCE:** D_neg at epoch 3 is NOT more negative than D_neg_init.

---

### E1+E2 — NPO Baseline (3 Seeds)
- **WHEN:** After GATE-1 and P1 are confirmed. Run in parallel with GATE-2, E0b.
- **WHAT:** Run NPO unlearning on forget05, seeds [42, 123, 456]. For each seed: train 10 epochs, then run three-precision QRA evaluation.
- **HOW:** Use OpenUnlearning's NPO script via `scripts/tofu_unlearn.sh`, or write a standalone training script using `compute_npo_loss()` from `src/fq_ssiuu/unlearning/npo_loss.py`.
  For QRA evaluation: use `run_qra_evaluation()` from `src/fq_ssiuu/evaluation/qra_pipeline.py`.
- **CHECKPOINTS:** `/checkpoints/c2_npo/seed42/`, `seed123/`, `seed456/`
- **ACCEPTANCE:** All 3 seeds produce non-trivial Q-INT4 scores (confirms P1's QRA finding is not seed-dependent).

---

### E3 — QUAIL Baseline (3 Seeds)
- **WHEN:** After P4 establishes QUAIL as stable. Parallel to E4, E5, E7.
- **WHAT:** Run QUAIL training on forget05, seeds [42, 123, 456]. QRA evaluation after each.
- **HOW:** Use `experiments/pilot_p4_quail_baseline.py` as template. Run with seeds 123 and 456 (seed 42 is already done in P4). Uses `compute_quail_loss()` from `quail_loss.py`.
- **CHECKPOINTS:** `/checkpoints/c8_quail/seed42/` (from P4), `seed123/`, `seed456/`

---

### E4 — SSIUU + NPO (3 Seeds)
- **WHEN:** After GATE-2 passes. Parallel to E3, E5, E7.
- **WHAT:** NPO + SSIUU attribution regularisation, **no FQ-QAT**. Seeds [42, 123, 456].
- **HOW:** Write `run_e4_ssiuu.py` (see Section 7 below for full spec). Uses SSIUURegulariser from `ssiuu_reg.py`. Training: `L_total = L_NPO + λ_attr × reg.compute_loss(batch)`. λ_attr = 1.0.
- **QRA:** Three-precision evaluation after each seed's training.
- **CHECKPOINTS:** `/checkpoints/c4_ssiuu/seed42/`, `seed123/`, `seed456/`

---

### E5 — FQ-QAT + NPO (3 Seeds)
- **WHEN:** After P3 confirms FQ-QAT is stable. Parallel to E3, E4, E7.
- **WHAT:** NPO + FQ-QAT (no SSIUU). Seeds [42, 123, 456].
- **HOW:** Write `run_e5_fqat.py` (see Section 7). Uses FQLoRALinear from `fq_hooks.py`. Call `wrap_model_with_fq_lora()` before training. NPO loss only.
- **CHECKPOINTS:** `/checkpoints/c5_fqat/seed42/`, `seed123/`, `seed456/`

---

### E6 — FQ-SSIUU Full Method (3 Seeds)
- **WHEN:** **ONLY after E4 AND E5 are both complete.** E4+E5 are ablations; E6 is the full method.
- **WHAT:** NPO + SSIUU + FQ-QAT. The **primary contribution**. Seeds [42, 123, 456].
- **HOW:** Write `run_e6_fq_ssiuu.py` (see Section 7). Combines SSIUURegulariser + FQLoRALinear. `L_total = L_NPO + λ_attr × reg.compute_loss(batch)`.
- **CHECKPOINTS:** `/checkpoints/c6_fq_ssiuu/seed42/`, `seed123/`, `seed456/`

---

### E7 — DURABLEUN-SAF Reference (3 Seeds)
- **WHEN:** Parallel to E3, E4, E5.
- **WHAT:** Run DURABLEUN-SAF reference method for comparison.
- **HOW:** Clone `github.com/neurips26/DurableUnl`. Adapt to run on Llama-3.2-1B (originally tested at 8B). Run on forget05, 3 seeds.
- **NOTE:** Utility may collapse — this is expected. Document it, don't try to fix it.
- **CHECKPOINTS:** `/checkpoints/c7_durableun/seed42/`, `seed123/`, `seed456/`

---

### FADE Computation
- **WHEN:** After E0b (oracle) is complete AND after each E1–E7 condition's checkpoint is available. Can run incrementally.
- **WHAT:** Compute FADE score between each condition's unlearned model and the oracle.
- **HOW:** Write `src/fq_ssiuu/evaluation/fade_compute.py`. Sequential two-pass approach: 1) Compute oracle model hidden states on test prompts, 2) Compute each condition's hidden states, 3) Compute FADE as the mean absolute difference.

---

## 4. Hyperparameter Registry

All hyperparameters are **locked** for the full experiments. Do not change these.

| Parameter | Value | Source |
|-----------|-------|--------|
| `lr` | 1e-5 | OpenUnlearning repro.md |
| `optimizer` | AdamW | PyTorch standard |
| `beta` (NPO) | 0.1 | Zhang et al. 2024 |
| `gradient_clip` | 1.0 | Standard |
| `n_epochs` (E1-E7) | 10 | OpenUnlearning default |
| `batch_size` (effective) | 32 | OpenUnlearning default |
| `per_device_batch` | 1 | T4 VRAM constraint |
| `gradient_accum` | 32 | = effective / per_device |
| `lora_r` | 16 | Stage 1 config |
| `lora_alpha` | 32 | Stage 1 config |
| `lora_target_modules` | `q_proj, v_proj, k_proj, o_proj` | Stage 1 config |
| `lora_dropout` | 0.05 | Stage 1 config |
| `seeds` | [42, 123, 456] | configs/seeds.yaml |
| `forget_split` | forget05 | Primary (forget01 for P3 only) |
| `retain_split` | retain90 | OpenUnlearning default |
| `nf4_scale` (FQ-QAT) | 8.4e-4 | MANSU arXiv:2605.15138 |
| `lambda_attr` (SSIUU) | 1.0 | Default; ablate {0.1, 0.5, 1.0, 5.0} |
| `n_ig_steps` (captum) | 20 | P2 confirmed feasible |
| `IG target layer` | 14 | Middle of 28-layer model |
| `QUAIL margin` | 4.2e-4 | nf4_bin_width / 2 |
| `nf4_bin_width` | 8.4e-4 | MANSU empirical |

---

## 5. The Metrics Every Experiment Must Report

### For every condition (E1–E7), every seed, every precision:
- **FQ** — Forget Quality (KS p-value from TOFU evaluation)
- **MU** — Model Utility (TOFU composite score)
- **Q-INT4** = |FQ_INT4 − FQ_BF16| / (FQ_BF16 + ε)
- **Q-INT8** = |FQ_INT8 − FQ_BF16| / (FQ_BF16 + ε)
- **Durability Certificate** — pass/fail: Q-INT4 < 0.05 means the unlearning is quantisation-durable

### For FADE (requires E0b oracle):
- **FADE_mean** and **FADE_std** for each condition

### Logging Format:
Every experiment writes results to:
```
results/{condition_id}/{condition_id}_seed{seed}_results.json
```
The JSON structure must match the `QRAResult` dataclass from `qra_pipeline.py`:
```python
{
    "condition_id": "c2_npo",
    "seed": 42,
    "fq_fp16": 0.3937,
    "mu_fp16": 0.5545,
    "fq_int8": ...,
    "mu_int8": ...,
    "fq_int4": ...,
    "mu_int4": ...,
    "q_int4": ...,
    "q_int8": ...,
    "verdict": "PASS" or "FAIL",
    "timestamp": "2026-08-10T12:00:00Z"
}
```

### Per-step training log:
Every training run saves a CSV at `{checkpoint_dir}/training_log.csv`:
```csv
epoch,step,loss,gradient_norm,learning_rate
0,0,13.862,0.234,1e-5
0,1,13.501,0.198,1e-5
...
```

---

## 6. Rules Gemini Must Never Break

1. **Never use GPTQ or AWQ** for quantisation evaluation. Always use bitsandbytes NF4. The entire project is built around NF4's specific quantisation noise characteristics.

2. **Never load two full models simultaneously** on a single T4 GPU. Load one, evaluate, delete, torch.cuda.empty_cache(), then load the next. Exception: during training, the NF4 base model + frozen reference model coexist because NF4 is 4-bit.

3. **Always set `retain_logs_path`** in the evaluator config. Without it, `forget_quality` returns `None` and the entire QRA metric is unusable.

4. **Always set `TRANSFORMERS_NO_FLASH_ATTN=1`** as an environment variable before any model loading. Flash attention ABI mismatches cause silent crashes on Kaggle.

5. **Always save checkpoints to Drive after every epoch.** Kaggle sessions can crash. Drive persistence is the only reliable checkpoint.

6. **Never commit model weights to git.** Only save adapter weights (< 100 MB via peft's `save_pretrained()`). Full model weights go to Drive or HuggingFace.

7. **Never start E6 before E4 and E5 are both complete.** E4 (SSIUU-only) and E5 (FQ-QAT-only) are ablations. E6 (FQ-SSIUU) is the full method. Running E6 before the ablations makes the ablation analysis meaningless.

8. **Always run evaluation on single GPU.** DDP (multi-GPU) disables the OpenUnlearning evaluator's forget_quality computation.

9. **Report exact numerical outputs.** Never round before reporting. All FQ/MU values should be reported to 6 decimal places minimum.

10. **Log every training step's loss and gradient norm.** This is non-negotiable for reproducibility and stability monitoring.

11. **Use `print(..., flush=True)` in all scripts that run as subprocesses.** Python's `logging` module causes deadlocks when stdout/stderr are redirected by Jupyter.

12. **Never modify `fake_quant.py`, `quail_loss.py`, `npo_loss.py`, or `ssiuu_reg.py`.** These are mathematically verified modules. If you find a bug, document it and ask for human review before changing.

13. **Always apply `compute_scale()` per-tensor to the adapter delta, not to the base weights.** FQ-QAT targets the adapter output, not the frozen base.

14. **Always use `attn_implementation='sdpa'`** when loading any model. This avoids incompatible attention implementations.

15. **Always set seed before training** using `torch.manual_seed(seed)` and `torch.cuda.manual_seed_all(seed)` for reproducibility.

---

## 7. How to Write run_e4_ssiuu.py, run_e5_fqat.py, run_e6_fq_ssiuu.py

### run_e4_ssiuu.py — SSIUU + NPO (No FQ-QAT)

```python
# Pseudocode — Gemini implements the full version

# 1. Load base model (NF4 + LoRA), NO FQLoRALinear wrapping
model = load_base_model()
model = wrap_with_lora(model)
# Load pre-existing LoRA adapter if continuing from NPO checkpoint

# 2. Load reference model (frozen, same fine-tuned base)
ref_model = get_reference_model("open-unlearning/tofu_Llama-3.2-1B-Instruct_full")

# 3. Initialise SSIUU regulariser
from fq_ssiuu.attribution.ssiuu_reg import SSIUURegulariser, SSIUUConfig
reg = SSIUURegulariser(model, forget_loader, SSIUUConfig(lambda_attr=1.0))
reg.initialise()  # Compute D^-(theta_0) — call ONCE before training

# 4. Training loop
for epoch in range(10):
    for batch in forget_loader:
        model_logits = model(**batch).logits
        ref_logits = ref_model(**batch).logits
        npo_loss = compute_npo_loss(model_logits, ref_logits, batch['labels'])
        attr_loss = reg.compute_loss(batch)  # SSIUU regularisation
        total_loss = npo_loss + attr_loss
        total_loss.backward()
        # Log D_neg mean at each epoch
        clip_grad_norm, step, log to CSV
    # Log D_neg after epoch (verify regulariser is active)

# 5. After training: merge adapter, run QRA evaluation
```

### run_e5_fqat.py — FQ-QAT + NPO (No SSIUU)

```python
# 1. Load base model (NF4 + LoRA)
model = load_base_model()
model = wrap_with_lora(model)

# 2. Apply FQ-QAT wrapping
from fq_ssiuu.quantisation.fq_hooks import wrap_model_with_fq_lora
model, n_wrapped = wrap_model_with_fq_lora(model)
# Prints: "Wrapped N LoRA layers with FQ-QAT"

# 3. Load reference model (frozen)
ref_model = get_reference_model(...)

# 4. Training loop — NO SSIUURegulariser
for epoch in range(10):
    for batch in forget_loader:
        model_logits = model(**batch).logits  # FQ-QAT is implicit in wrapped layers
        ref_logits = ref_model(**batch).logits
        npo_loss = compute_npo_loss(model_logits, ref_logits, batch['labels'])
        npo_loss.backward()  # Gradients flow through STE in FQLoRALinear
        clip, step, log

# 5. Unwrap FQ-QAT before saving
from fq_ssiuu.quantisation.fq_hooks import unwrap_model_from_fq_lora
model, _ = unwrap_model_from_fq_lora(model)

# 6. Merge adapter, run QRA evaluation
```

### run_e6_fq_ssiuu.py — Full FQ-SSIUU (SSIUU + FQ-QAT)

```python
# 1. Load base model (NF4 + LoRA)
model = load_base_model()
model = wrap_with_lora(model)

# 2. Apply FQ-QAT wrapping
model, n_wrapped = wrap_model_with_fq_lora(model)

# 3. ALSO initialise SSIUU regulariser
reg = SSIUURegulariser(model, forget_loader, SSIUUConfig(lambda_attr=1.0))
reg.initialise()

# 4. Load reference model (frozen)
ref_model = get_reference_model(...)

# 5. Training loop — BOTH FQ-QAT and SSIUU
for epoch in range(10):
    for batch in forget_loader:
        model_logits = model(**batch).logits
        ref_logits = ref_model(**batch).logits
        npo_loss = compute_npo_loss(model_logits, ref_logits, batch['labels'])
        attr_loss = reg.compute_loss(batch)
        total_loss = npo_loss + attr_loss
        total_loss.backward()
        clip, step, log

# 6. Unwrap, merge, evaluate
```

### All three scripts share:
- Same CLI interface: `--seed`, `--open_unlearning_path`, `--drive_output`, `--dry-run`
- Same evaluation step: call `run_qra_evaluation()` from `qra_pipeline.py`
- Same logging format: CSV with epoch, step, loss, gradient_norm
- Same checkpoint strategy: save adapter after every epoch

---

## 8. Known Issues and Their Fixes

| Issue | Symptom | Fix |
|-------|---------|-----|
| **Jupyter subprocess deadlock** | Script hangs with no CPU/GPU activity | Replace all `logger.info()` / `logger.error()` with `print(..., flush=True)` |
| **OmegaConf interpolation error** | `InterpolationResolutionError` on `${hydra:runtime.cwd}` | Don't walk the OmegaConf config tree with `for k, v in d.items()`. Navigate directly to the target key. |
| **captum tensor shape error** | `IndexError: invalid index of a 0-dim tensor` | Use `sum(dim=1)` not `sum()` in the forward function passed to captum. Must return `(batch_size,)` shape. |
| **flash-attn ABI mismatch** | Silent crash or CUDA error on T4 | Set `TRANSFORMERS_NO_FLASH_ATTN=1`. Run `pip uninstall flash-attn -y`. |
| **bitsandbytes triton.ops** | Import error on older bitsandbytes | Upgrade to bitsandbytes >= 0.45.5. |
| **Multi-GPU DDP disables evaluator** | `forget_quality = None` | Use single-GPU only for evaluation. |
| **forget_quality = None** | Missing from evaluation results | Always set `retain_logs_path` in the evaluator config. |
| **HuggingFace cache locks** | Subprocess hangs on model download | Clear dangling `.lock` files: `find ~/.cache/huggingface -name '*.lock' -delete` |
| **paged_adamw_32bit import** | Requires trl or bitsandbytes trainer | Use `torch.optim.AdamW` instead for standalone scripts. |
| **T4 BF16 support** | T4 does NOT support BF16 natively | Use FP16 for training compute dtype. Set `bnb_4bit_compute_dtype='float16'`. |

---

## 9. The Publication Readiness Checklist

When ALL of E1–E7 are complete, verify:

- [ ] 3 seeds for each of 6 conditions (E1/C2, E3/C8, E4/C4, E5/C5, E6/C6, E7/C7)
- [ ] QRA evaluation at 3 precisions (FP16, INT8, NF4) for each seed × condition
- [ ] FADE computed for each condition (requires E0b oracle)
- [ ] Pareto frontier plot: FQ × MU × Q-INT4 across all conditions
- [ ] Training curves (loss, gradient norm) logged and saved for all runs
- [ ] All results in `results/{condition}/seed{N}_results.json` format
- [ ] Durability certificates generated for all conditions (Q-INT4 < 0.05 = PASS)
- [ ] `results/summary_table.csv` with all metrics across all conditions and seeds
- [ ] At least one negative result documented (if any condition fails or shows unexpected behaviour)
- [ ] INT8 anomaly from P1 investigated (Q-INT8 = 0.426, larger than Q-INT4)
- [ ] Statistical significance computed for multi-seed comparisons (mean ± std)

---

## 10. File Structure Reference

```
fq-ssiuu/
├── src/fq_ssiuu/
│   ├── quantisation/
│   │   ├── fake_quant.py          # FakeQuantise STE (DO NOT MODIFY)
│   │   └── fq_hooks.py            # FQLoRALinear integration layer
│   ├── unlearning/
│   │   ├── npo_loss.py            # NPO loss (DO NOT MODIFY)
│   │   └── quail_loss.py          # QUAIL hinge loss (DO NOT MODIFY)
│   ├── attribution/
│   │   └── ssiuu_reg.py           # SSIUU regulariser (DO NOT MODIFY)
│   ├── evaluation/
│   │   ├── qra_pipeline.py        # QRA three-precision evaluation
│   │   └── fade_compute.py        # [TO BE WRITTEN BY GEMINI]
│   └── models/
│       └── loader.py              # Model loading utilities
├── experiments/
│   ├── pilot_p1_qra_check.py      # P1 script (validated)
│   ├── pilot_p2_attribution.py    # P2 script (validated)
│   ├── pilot_p3_fqat_stability.py # P3 script (awaiting execution)
│   ├── pilot_p4_quail_baseline.py # P4 script (awaiting execution)
│   ├── _eval_at_precision.py      # QRA evaluation helper
│   ├── run_e4_ssiuu.py            # [TO BE WRITTEN BY GEMINI]
│   ├── run_e5_fqat.py             # [TO BE WRITTEN BY GEMINI]
│   ├── run_e6_fq_ssiuu.py         # [TO BE WRITTEN BY GEMINI]
│   └── notebooks/kaggle/
│       ├── PILOT_P1_P2.ipynb      # P1+P2 notebook (validated)
│       └── PILOT_P3_P4.ipynb      # P3+P4 notebook (awaiting execution)
├── results/
│   ├── pilot_p1/                  # P1 results (completed)
│   ├── pilot_p2/                  # P2 results (completed)
│   ├── pilot_p3/                  # P3 results (pending)
│   ├── pilot_p4/                  # P4 results (pending)
│   └── summary_table.csv          # [TO BE WRITTEN AFTER ALL EXPERIMENTS]
├── handbook/
│   ├── GEMINI_FULL_EXPERIMENT_HANDOFF.md  # THIS DOCUMENT
│   ├── GEMINI_HANDOFF.md                  # Prior sessions handoff
│   └── research_diary/
│       ├── 2026-08-08_pilot_planning.md
│       ├── 2026-08-09_pilot_p1_p2_execution.md
│       └── 2026-08-09_pilot_p3_p4_planning.md
├── configs/
│   ├── base.yaml
│   └── seeds.yaml
└── tests/
    ├── test_ste_gradients.py      # 4 tests PASSING
    ├── test_qra_pipeline.py       # 4 tests SKIPPED (need GPU)
    ├── test_quail_loss.py         # 4 tests SKIPPED (need GPU)
    └── test_fq_hooks.py           # [TO BE WRITTEN — CPU self-test]
```

---

## 11. Quick Start for Gemini

After reading this document, the first actions are:

1. **Execute P3 and P4** by running `PILOT_P3_P4.ipynb` on Kaggle T4.
2. **Record P3 and P4 verdicts** in Section 2 above.
3. **Create pilot/p3-p4 PR → dev** after successful execution.
4. **Begin E0b + GATE-2 in parallel** (both are independent).
5. **Then**: E1+E2 (NPO 3 seeds), E3 (QUAIL 3 seeds), E5 (FQ-QAT 3 seeds), E7 (DurableUn) — all parallel.
6. **Then**: E4 (SSIUU 3 seeds) — after GATE-2.
7. **Then**: E6 (full FQ-SSIUU) — after E4 AND E5 complete.
8. **Then**: FADE computation — after E0b and all conditions.
9. **Then**: Analysis, plots, thesis writing.
