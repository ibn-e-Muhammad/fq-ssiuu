# DL-009: Migrate Training from Colab to Kaggle

**Date:** August 2026
**Status:** DECIDED — ACTIVE
**Author:** Research Team (GATE-1 session)

---

## Context

GATE-1 required training the NPO baseline on the TOFU benchmark (10 epochs, ~60 steps
with gradient accumulation). Three attempts on Google Colab free tier (single T4) each
failed to complete: sessions hit GPU quota after approximately 1 epoch of training plus
evaluation (~2 hours of combined compute).

## Decision

Use Kaggle Notebooks with dual T4 GPUs (2×15 GB VRAM) as the primary training
platform for GATE-1 and subsequent experiments.

## Reason

The total compute required for a full GATE-1 training run:
- Training: ~10 epochs × ~70 seconds/epoch = ~12 minutes (with dual-GPU DDP)
- Evaluation: ~40 minutes per epoch × 10 epochs = ~400 minutes (single-GPU only)
- Total: ~7+ hours for a complete training + evaluation run

Google Colab free tier constraints:
- GPU sessions limited to ~2-4 hours before quota enforcement
- Daily GPU quota: approximately 4-6 hours total
- T4 allocation subject to availability — sometimes downgraded to CPU
- Training cannot complete in a single session

Kaggle Notebooks provide:
- 30 hours/week of GPU time per account
- Dual T4 allocation (2×15 GB = 30 GB total VRAM)
- 12-hour session limit per notebook run
- Sufficient for complete training in a single session

## Consequence

1. Training uses `accelerate launch --multi_gpu --num_processes=2` for DDP
2. OpenUnlearning's custom evaluator is INCOMPATIBLE with DDP (see DL-010)
3. Evaluation must be run as a separate single-GPU job
4. Environment patching (requirements, configs) must be applied via shell commands
   in each Kaggle session (no persistent environment across sessions)
5. Checkpoints must be downloaded or uploaded to HuggingFace Hub for persistence

## Trade-offs

| Aspect | Colab | Kaggle |
|--------|-------|--------|
| GPU count | 1× T4 | 2× T4 |
| Session limit | ~2-4h (quota) | 12h |
| Weekly GPU hours | ~4-6h/day | 30h/week |
| Environment persistence | Drive mount | None (re-setup each session) |
| Internet access | Full | Restricted (must enable) |
| DDP support | N/A (single GPU) | Yes, but evaluator incompatible |
