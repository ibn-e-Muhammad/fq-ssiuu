# Local NPO run 2: measured outcome and audit

Updated 2026-11-03. Run ID `20261031_092340_493700`.
Source snapshot `c0bc9cd`, from checkout tip `06419f0` plus supplied local edits.
Raw evidence: `experiments/reproductions/NPO_2nd-test/npo_results/20261031_092340_493700`.
Original code, notebook outputs, helper tests, interruption and completion records
are preserved. Derived corrections are separate, under
`handbook/evidence/npo_run2/20261103_audit/reanalysis`.

## What completed

Full-weight NPO+RT on pinned TOFU Llama-3.2-1B full, Forget05/Retain95;
beta .1, coefficients 1/1, LR1e-5, weight decay .01, effective batch32,
62 updates/6 warmup, BF16 model with recoverable disk-backed FP32 AdamW state.
The frozen full model is the NPO reference; retain95 is the diagnostic oracle.
Seed 42 only. Original code's fixed index pairing covers 195 of 3800 retained rows.

All nine stages completed 717 questions each: 200 forget, 300 retained diagnostics,
100 real-author and 117 world-fact questions. Final generation is greedy with a
96-token cap. INT8/NF4 stages each record 112 converted linear projections and
include full and retain95 controls. This is actual recorded PTQ evaluation.
The supplied eight-test recovery log is engineering evidence; it does not
establish whole-trajectory equivalence to the original NPO trainer.

The archive's 27 recorded per-stage artifact hashes verify byte-for-byte.
Checkpoint completion receipts exist, but original full/retain/NPO tensor files
are absent from this supplied copy; their weight hashes cannot be reverified
here. There is no new 1B inference or retraining in the post-hoc analysis.

## Main recorded measurements

| Forget05 | Full BF16 | Retain95 BF16 | NPO BF16 | NPO INT8 | NPO NF4 |
|---|---:|---:|---:|---:|---:|
| Gold-answer probability | .878020 | .129694 | .113933 | .113840 | .113245 |
| Gold-answer NLL | .134927 | 2.473386 | 2.353808 | 2.352865 | 2.349265 |
| ROUGE-L recall | .790359 | .396208 | .379982 | .382893 | .388815 |
| Token F1 | .806654 | .436495 | .283646 | .287232 | .277975 |
| Exact match | .470 | .005 | .000 | .000 | .000 |
| Arithmetic model utility | .559091 | .553693 | .436541 | .435681 | .424316 |

Before quantization, geometric FQ against the BF16 retain95 oracle is D=.080,
p=.545271. This is a KS non-rejection, not proof of erasure/equivalence. Retain
ROUGE drops from .793619 in retain95 to .443177 in NPO. Aggregate utility is
about 21.2% below retain95. The gold/generation metrics show no clear material
recovery after INT8/NF4; their movement is small and mixed. Full-model NF4 control
probability drops from .878020 to .699686 and ROUGE from .790359 to .545275,
confirming substantial quantization sensitivity without demonstrating recovery.

| Geometric FQ | Fixed BF16 oracle D / p | Same-precision oracle D / p |
|---|---|---|
| NPO BF16 | .080 / .545271 | .080 / .545271 |
| NPO INT8 | .105 / .220541 | .095 / .328115 |
| NPO NF4 | .070 / .712582 | .095 / .328115 |

The two .095/.328115 matched-oracle entries replace the interpretation of the
old arithmetic matched columns, whose original values remain preserved.

The old `matched_precision_KS_*` columns used arithmetic truth ratios. Corrected
geometric comparisons and paired directional intervals are generated from the
existing per-question CSVs and stored separately; the old matrices are not fixed
in place. Consult `reanalysis_forget_quality.csv` and
`reanalysis_directional_qra.csv`, including author-block sensitivity. A tiny
directional shift must not be conflated with a meaningful recovery mechanism.

The paired geometric-preference calculation gives the following signed changes
(quantized minus BF16); these are descriptive, not KS p-value differences:

| Quantization | Mean log-preference change | 95% question bootstrap | 95% author-block bootstrap |
|---|---:|---|---|
| INT8 | +.000350 | [−.004174, +.004880] | [−.004143, +.004837] |
| NF4 | +.002223 | [−.009268, +.013469] | [−.015713, +.019396] |

Both procedures use 2,000 paired resamples of 200 questions; the author procedure
resamples the ten 20-question TOFU author blocks. Both intervals include zero.
The fraction of questions with increased preference is 51% for INT8 and 52% for
NF4. These measurements do not resolve a directional recovery effect; they also
do not establish equivalence or robustness to untested attacks.

## Generation and resource limits

| NPO generation cap-hit rate | BF16 | NF4 |
|---|---:|---:|
| Forget05 | 41.5% | 51.0% |
| Retain diagnostics | 21.67% | 30.33% |
| Real Authors | 82.0% | 76.0% |
| World Facts | 89.74% | 90.60% |

These are 96-token limit rates, not automatically coded gibberish/fabrication
rates. Exported answers remain available for manual inspection; no systematic
semantic annotation was supplied. Long-output auditing is therefore required.

The recorded training log contains 502.56 seconds of forward/backward compute
(about 8.4 minutes), 33,281.99 seconds of optimizer/checkpoint work where present
(61 rows), and 34,277.50 seconds in the `seconds` field (62 rows, about 9.52 hours).
Some resumed historical rows omit post-commit optimizer timing, so these sums
are not a complete wall-clock accounting. Peak logged Torch allocation is
6.0923 GiB; it is not whole-device memory or a prediction of LoRA runtime.
The supplied environment records RTX3060/12GB and Windows11/16GB RAM.

## Scientific conclusion and next run

This run supplies useful single-seed suppression/PTQ evidence with a substantial
retention cost. It establishes neither a successful QRA attack nor robustness to
all quantizers. The audit's broad retain-coverage concern is real, but the fixed
pairing is inherited from the pinned author implementation. Neither 1B scale nor
quantization-bin crossing has been isolated as the cause of the negative result.

Following the researcher's decision, local run3 uses LoRA, a fresh retain stream,
three seeds and more complete evaluation. These are explicit adaptations; the
two training changes require a controlled ablation to separate their effects.
See [DL-024](../decisions/DL-024_project_lora_training.md),
[audit disposition](../reviews/2026-11-03_npo_run2_audit.md),
[CR-013](../corrections/CR-013_npo_local_run2_audit.md), and
[third-run protocol](../methods/NPO_LOCAL_RUN3_LORA.md).
