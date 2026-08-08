# DL-010: Per-Epoch Evaluation Skipped on Kaggle (DDP Incompatibility)

**Date:** 4th August 2026
**Status:** DECIDED — ACCEPTED LIMITATION
**Author:** Research Team (GATE-1 session)

---

## Context

The Kaggle training run used `accelerate launch --multi_gpu --num_processes=2` for
dual-T4 DDP training. OpenUnlearning's custom evaluator (`src/evals/`) includes a
guard that prevents execution when multiple accelerator processes are running.

## Decision

Accept that per-epoch TOFU evaluation will be skipped during multi-GPU training.
Evaluation must be performed separately as a standalone single-GPU job after training.

## Reason

The OpenUnlearning evaluator prints this warning at each epoch boundary:
```
[trainer.base][WARNING] - Custom evaluator can be run with this Trainer only
when a single accelerator process is running.
```

This is by design — the evaluator generates text, computes ROUGE scores, and
aggregates metrics across the full forget/retain sets. Running this on multiple
GPUs would produce duplicated or inconsistent results because:
1. Each GPU only sees a shard of the evaluation data
2. Text generation is non-deterministic across DDP ranks
3. The aggregation logic does not handle multi-rank reduction

## Consequence

1. Training logs contain ONLY loss, grad_norm, and learning_rate — NO evaluation metrics
2. `forget_quality` and `model_utility` are NOT available during training
3. Per-epoch convergence monitoring is limited to loss trajectory only
4. A standalone evaluation cell/script must be run separately after training
5. The standalone evaluation MUST use a single GPU (no `accelerate launch`)

## Impact

This means the model's actual unlearning quality is unknown until the standalone
evaluation is run. The loss trajectory (223 → 5.4) suggests convergence, but
loss alone cannot confirm that forget_quality and model_utility meet GATE-1 targets.
This is a known gap that will be closed by the standalone evaluation script
(see `experiments/gate1_standalone_eval.py`).
