# GATE-1 Pre-Brief: OpenUnlearning Baseline Reproduction

**Objective:** Reproduce the TOFU forget05 baseline using the OpenUnlearning framework on Llama-3.2-1B-Instruct at Colab T4. This confirms the evaluation pipeline works before any novel FQ-SSIUU experiments.

**Branch:** `gate/1-baseline-repro`
**Fork URL:** `https://github.com/ibn-e-Muhammad/open-unlearning.git` (branch: `fq-ssiuu-integration`)

**Target Model & Split:**
- Model: `meta-llama/Llama-3.2-1B-Instruct`
- Split: `forget05` (TOFU dataset)

**Adapted Training Command (for single T4 15GB VRAM):**
```bash
python3 src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/default \
  forget_split=forget05 retain_split=retain90 trainer=NPO task_name=GATE1_BASELINE \
  trainer.args.per_device_train_batch_size=1 \
  trainer.args.gradient_accumulation_steps=32 \
  trainer.args.gradient_checkpointing=true \
  trainer.args.fp16=false \
  trainer.args.bf16=true \
  trainer.args.save_strategy=epoch \
  trainer.args.save_total_limit=2 \
  trainer.args.output_dir="/content/drive/My Drive/fq-ssiuu-research/checkpoints/c2_npo/seed42"
```

**Training Time Estimate:** 
~10-15 hours total (10 epochs × ~60-90 min/epoch).

**Checkpoint Strategy:** 
Since 10-15 hours exceeds a single Colab session, `trainer.args.save_strategy=epoch` is used. Checkpoints save directly to Google Drive. If the session disconnects, the `--resume_from_checkpoint=true` argument is injected into `trainer.args` to automatically resume from the latest epoch.

**Acceptance Criterion:**
- **forget_quality (FQ):** 0.14 ± 0.05  (Range: 0.09 – 0.19)
- **model_utility (MU):** 0.45 ± 0.05  (Range: 0.40 – 0.50)

**Decision Tree:**
- **If PASS:** Merge `gate/1-baseline-repro` to `dev`, proceed to Pilot P1 (QRA existence test).
- **If FAIL (Fallback):** If FQ is outside tolerance but >0, and MU > 0.35, continue anyway with a note that the baseline deviation is documented, and Pilot P1 still runs. If completely broken (e.g. FQ=0, MU=0), stop and debug OpenUnlearning integration.
