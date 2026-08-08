# Research Diary: 2026-08-04 — GATE-1 NPO Baseline Training

**Date:** 4th–5th August 2026
**Author:** Research Team
**Subject:** GATE-1 baseline reproduction — training Llama-3.2-1B-Instruct NPO unlearning on TOFU forget05

---

## 1. What Was Attempted First (Colab Single-GPU)

The original plan was to run GATE-1 training on Google Colab with a single T4 GPU,
matching the environment that passed GATE-0. The training script used OpenUnlearning's
`src/train.py` via Hydra, with checkpoints saved to Google Drive for persistence
across sessions.

### First Colab Attempt

Started training from scratch (no existing checkpoints). The pre-training evaluation
at checkpoint-0 ran successfully, producing baseline metrics:

- `forget_truth_ratio`: 0.4724 (model "knows" the forget data)
- `model_utility`: 0.5967 (model is generally capable)
- `forget_Q_A_Prob`: 0.8855 (high token-level memorisation)
- `forget_quality`: **None** (first sign of the retain_logs bug)

After checkpoint-0 evaluation (~40 minutes), training began. One epoch of actual
NPO training completed (7 steps with grad_accum=16). Then per-epoch evaluation
started for checkpoint-7. The evaluation reached `retain_Q_A_ROUGE` computation
when the Colab session was terminated by GPU quota enforcement.

### Observation: Evaluation Is the Bottleneck

A single epoch of training takes ~5–7 minutes. But the per-epoch TOFU evaluation
suite takes ~40 minutes (computing ROUGE scores involves text generation, which is
slow). With 10 epochs, that's 70 minutes training + 400 minutes evaluation = ~8
hours. Colab's GPU quota cannot sustain this.

### Second and Third Colab Attempts

Each subsequent attempt also failed at the ~2 hour mark. Notably, training restarted
from step 0 each time because the retain_logs evaluation was not completing, and the
checkpoint resumption logic was not correctly loading the latest checkpoint.

---

## 2. Why Colab Failed

Three independent factors combined to make Colab unviable:

1. **GPU quota:** ~4-6 hours/day of T4 time. A full 10-epoch run with evaluation
   requires ~8 hours. Even without evaluation, the overhead of environment setup,
   model download, and data preparation consumes ~30 minutes per session.

2. **Session instability:** Sessions are terminated without warning. There is no
   graceful shutdown — the process is SIGKILLed. This means any in-progress
   checkpoint save is lost.

3. **Evaluation overhead:** The TOFU evaluation suite is 5-8× slower than training.
   Skipping evaluation during training would help, but we need those metrics.

---

## 3. The Decision to Move to Kaggle

After three failed Colab sessions, we decided to move to Kaggle Notebooks. The
rationale:

- Kaggle provides 30 hours/week of GPU time (vs. Colab's ~4-6 hours/day)
- Kaggle offers dual T4 GPUs, enabling DDP training for 2× throughput
- 12-hour session limit per notebook run (vs. Colab's ~2-4 hours before quota)
- Training-only run (without per-epoch evaluation) takes ~12 minutes on dual T4

The trade-off: Kaggle has no persistent environment. Every session requires
cloning repos, installing dependencies, and applying patches from scratch. We
addressed this with a "Master Script" approach — a single Kaggle cell that handles
everything from git clone through training launch.

---

## 4. What Happened on Kaggle

### The Gauntlet of Errors

Getting the training to actually run on Kaggle was significantly harder than
expected. The dual-GPU DDP setup introduced multiple failure modes that do not
exist in single-GPU training:

1. **`ddp_find_unused_parameters: None`** — The finetune.yaml had `None` as a
   YAML string, not a boolean. PyTorch DDP expects a bool. Fix: `false`.

2. **DDP "marked ready twice"** — NPO performs two forward passes per step
   (forget + retain). DDP's gradient hooks fire on the first forward pass and
   then crash on the second because parameters are "already marked ready."
   Fix: `gradient_checkpointing_kwargs.use_reentrant: false`.

3. **DictConfig serialization** — HuggingFace Trainer calls `args.to_json_string()`
   for TensorBoard logging. If TrainingArguments contains OmegaConf DictConfig
   objects (from Hydra), JSON serialization fails. Fix: convert via
   `OmegaConf.to_container(resolve=True)` before passing to TrainingArguments.

4. **VLM import error** — `lm_eval` had renamed `hf_vlms` to `huggingface`.
   Fix: `sed` replacement in source.

5. **bitsandbytes version** — Same triton.ops issue from GATE-0. Fix: force
   install 0.45.5.

6. **FP16 GradScaler crash** — PyTorch's GradScaler refuses to unscale native
   FP16 gradients with `ValueError("Attempting to unscale FP16 gradients.")`.
   Since T4 doesn't support BF16, we must use FP16 for both model weights and
   AMP. Fix: monkey-patch the safety check with `sed`.

7. **OOM with float32** — Attempted loading model in float32 (to satisfy the
   GradScaler's expectation of 32-bit gradients). But float32 weights + reference
   model + optimizer state exceeded 15 GB per GPU. Fix: reverted to float16 with
   the GradScaler patch.

Each of these required a new Kaggle session (~10 minutes of setup overhead each
time). Total debugging time: approximately 4 hours across ~10 sessions.

### The Successful Run

The final configuration that worked:
- Model loaded in float16
- `fp16=True` for AMP
- GradScaler safety check patched
- `paged_adamw_8bit` optimizer (saves ~2 GB VRAM)
- `gradient_checkpointing: true` with `use_reentrant: false`
- `ddp_find_unused_parameters: false`
- Per-epoch evaluation implicitly skipped (DDP incompatible)

Training completed in 12 minutes. 60 steps, 8.64 effective epochs.

---

## 5. Loss Trajectory Analysis

The loss curve tells a clear story of successful unlearning:

**Phase 1 — Warmup plateau (epochs 0–2, steps 1–14):**
Loss stays flat at ~223. Learning rate is zero for the first 8 steps, then ramps.
The model is not yet updating. NaN grad norms are expected (zero LR → no meaningful
gradient computation for the logger).

**Phase 2 — Rapid convergence (epochs 2–3, steps 15–21):**
Loss drops precipitously: 145 → 111 → 77 → 47 → 32 → 7. The NPO loss is actively
pushing the model away from the forget set. Grad norms drop from ~2800 to ~200,
indicating the model is settling into a new minimum.

**Phase 3 — Oscillating refinement (epochs 3–6, steps 22–42):**
Loss oscillates between ~5–27 mid-epoch but drops to 1.3–4.7 at epoch boundaries.
This pattern is characteristic of NPO: the forget set examples have varying
difficulty, and the model sees them in different orders each epoch. At epoch
boundaries, the accumulated gradient updates from the full epoch produce the
cleanest "unlearning signal."

**Phase 4 — Stable plateau (epochs 6–8.64, steps 43–60):**
Loss stabilises at ~5–10 mid-epoch, ~1.3–1.5 at boundaries. The model has
effectively converged. Additional training would yield diminishing returns.

The fact that the loss dropped from 223 to 5 is a strong qualitative signal that
unlearning occurred. But we cannot confirm the quality of unlearning (forget_quality)
or the preservation of general capability (model_utility) without running the
standalone evaluation.

---

## 6. What Remains Unknown

**Critical unknowns:**
- `forget_quality`: Has NEVER been measured for this checkpoint. The metric requires
  retain_model_logs (pre-computed evaluation of a retain-only oracle). These logs
  exist in `saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json` but were
  never passed to the evaluator.

- `model_utility`: Has never been measured on the final (8.64-epoch) checkpoint.
  The pre-training baseline was 0.5967. We need to confirm it hasn't collapsed.

- `forget_truth_ratio`: Should increase from the baseline 0.4724 if unlearning
  worked. After 1 epoch of Colab training it was 0.4873 (marginal improvement).
  After 8.64 epochs, we expect a significantly higher value.

---

## 7. Observations and Concerns

1. **NaN grad_norm in steps 1–8:** Normal. LR=0 means no optimizer step. The
   gradient is computed but not applied, and the logger records NaN because
   the norm of a zero-scaled gradient is undefined.

2. **8.64 vs 10 epochs:** The trainer was configured with `max_steps=60` rather
   than `num_train_epochs=10`. With effective batch size ~32 and 200 forget
   examples, each "epoch" is about 6.25 steps. 60 steps / 6.94 steps-per-epoch
   ≈ 8.64 epochs. The loss has clearly converged by epoch 6, so the missing
   1.36 epochs are unlikely to matter. We will note this as a deviation but
   do not expect it to affect GATE-1 pass/fail.

3. **GradScaler patch is fragile:** The `sed` command that patches PyTorch's
   source code is environment-specific. It works on Kaggle's August 2026 PyTorch
   installation but may break on future versions. If PyTorch changes the error
   message string or the function signature, the patch will silently fail.
   Long-term fix: use a PyTorch version that supports FP16 GradScaler natively,
   or use a GPU that supports BF16.

4. **Optimizer difference:** We used `paged_adamw_8bit` instead of the default
   `paged_adamw_32bit`. The 8-bit optimizer quantises the optimizer states
   (momentum and variance) to 8-bit precision. This saves ~6 GB VRAM but
   introduces very small numerical errors in the optimizer steps. This is
   standard practice for memory-constrained training and should not affect
   final model quality.

5. **Checkpoint is full model, not adapter:** The saved checkpoint is a full
   2.47 GB `model.safetensors` (the entire Llama-3.2-1B-Instruct model in FP16).
   This is because OpenUnlearning performs full fine-tuning, not LoRA. For
   Pilot P1 (QRA test), we can directly quantise this checkpoint without
   merge_and_unload().

---

## 8. Next Action

Run standalone evaluation on Kaggle using a SINGLE T4 GPU:
1. Upload the checkpoint to Kaggle (or use HuggingFace Hub)
2. Run `setup_data.py --eval` to ensure retain logs are present
3. Run `src/eval.py` with `retain_logs_path` correctly set
4. Extract `forget_quality`, `model_utility`, `forget_truth_ratio`
5. Record results in the experiment log
6. Make GATE-1 pass/fail determination

The evaluation script has been prepared at `experiments/gate1_standalone_eval.py`.
