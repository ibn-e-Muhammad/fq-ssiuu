# DL-011: forget_quality Reported None — Missing retain_logs_path

**Date:** 4th August 2026
**Status:** DECIDED — FIX REQUIRED IN EVALUATION
**Author:** Research Team (GATE-1 session)

---

## Context

Throughout all training sessions (both Colab and Kaggle), the `forget_quality` metric
consistently reported `None`. The evaluator logged:
```
[metrics][WARNING] - retain_model_logs not provided in reference_logs,
setting forget_quality to None
```

## Decision

Document as a known gap. The standalone evaluation script must explicitly set
`retain_logs_path` to the pre-computed retain model evaluation logs.

## Reason

The TOFU `forget_quality` metric is computed as:
```
forget_quality = 1 - KS_statistic(
    forget_truth_ratios_of_unlearned_model,
    forget_truth_ratios_of_retain_oracle_model
)
```

This requires `retain_model_logs` — the evaluation output of the retain-only oracle
model (a model fine-tuned only on the retain set, never exposed to the forget set).
These logs serve as the "gold standard" for what a model that never learned the
forget data would look like.

The `setup_data.py --eval` script downloads these pre-computed retain logs to:
```
saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json
```

However, the training command did not pass this path to the evaluator. The evaluator
checked for `retain_model_logs` in the reference logs dictionary, found it absent,
and set `forget_quality = None`.

## Fix Required

The standalone evaluation command must include:
```
retain_logs_path=saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json
```

Additionally, `setup_data.py --eval` must be run before evaluation to ensure the
retain logs are present on disk.

## Consequence

1. All training sessions produced `forget_quality: None` — this metric has NEVER
   been measured for the GATE-1 checkpoint
2. The `model_utility` metric WAS computed correctly (it does not depend on retain logs)
3. The standalone evaluation script (`experiments/gate1_standalone_eval.py`) must
   be the definitive source of both metrics
4. GATE-1 pass/fail determination is BLOCKED until standalone evaluation completes

## Historical Values (from Colab pre-training evaluation, checkpoint-0)

These are evaluation results from the ORIGINAL fine-tuned model BEFORE unlearning:
- `forget_truth_ratio`: 0.4724 (checkpoint-0, epoch 0)
- `model_utility`: 0.5967 (checkpoint-0, epoch 0)
- `forget_quality`: None (retain_logs not provided)
- `forget_Q_A_Prob`: 0.8855
- `forget_Q_A_ROUGE`: 0.8298

After 1 epoch of NPO training (checkpoint-7, from Colab):
- `forget_truth_ratio`: 0.4873
- `forget_quality`: None
- `forget_Q_A_Prob`: 0.8236
- `forget_Q_A_ROUGE`: 0.7334
- `model_utility`: NOT REPORTED (session terminated before full evaluation)
