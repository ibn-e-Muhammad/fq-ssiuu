# DL-024: LoRA for forthcoming FQ-SSIUU training

Date: 2026-11-03. Status: **RESEARCHER DECIDED; GPU VALIDATION PENDING**.
Authority: the researcher's instruction in this chat to use LoRA for the third
local NPO run and throughout forthcoming FQ-SSIUU experiments, because of the
resource and time consumption of conventional full-weight training.

## Decision

Forthcoming project training uses a frozen unquantized backbone with trainable
LoRA adapters. The initial versioned defaults are rank 16, alpha 32, dropout 0,
bias none, and q/k/v/o plus gate/up/down projections. Adapter parameters and AdamW
states are FP32. The local RTX 3060 backbone and deployment baseline use BF16.
`configs/training/lora_v1.json` records the project choice; the self-contained
third-run notebook embeds these defaults and its concrete settings. The original NPO beta, loss coefficients,
learning rate, weight decay, logical batch and stopping schedule are retained.
Defaults are declared before outcomes; no LoRA LR/retention-weight tuning uses
final Forget05 evaluation scores.

This supersedes full-weight training as the default for new local project runs.
It does not rewrite historical full-weight experiments, alter the protected
Kaggle image, or close constituent scientific gates. Historical FP16/T4 and D1
records remain descriptions of their original scope. No new Kaggle submission,
integration merge or PR is authorized by this work item.

## Scientific classification

LoRA changes the trainable parameter subspace. These runs are policy class C
project adaptations of the constituent methods, not exact original full-weight
reproductions. The NPO loss itself remains Eq. 3, separately combined with
positive retain cross-entropy. Full-model gradients and LoRA gradients are not
assumed equivalent. Baseline checkpoint provenance and model adaptation remain
explicit.

The original NPO pairing maps each forget index to a deterministic retain index.
Run 2 used only 195 unique retained rows. Following the supplied audit, run 3
defaults to a reproducible reshuffled retain stream; this is a second deliberate
change. Its 62 x 32 schedule draws 1,984 distinct retained rows, not the full
3,800. `source_fixed` remains available for a controlled LoRA sampling ablation.
Changing both LoRA and sampling prevents attributing outcome differences to just
one of them without that comparison.

## Reference and quantization boundary

During NPO, disabling adapters on the unchanged frozen full backbone supplies
the original reference under eval/no_grad. This avoids a duplicate full model
without caching or changing reference parameters. Backbone hashes are compared
before and after training; only LoRA parameters may have gradients/updates.

Save adapters, evaluate them unmerged, merge with the original unquantized base
in FP32, check FP32 merge probes, then cast/save/reload BF16 deployment weights.
Quantize **the merged weights** independently to INT8/NF4. Report merging drift
separately. A quantized base with an attached higher-precision adapter is a
different deployment experiment and must not silently replace this path.

An unmerged LoRA intervention leaves the original base weights intact: disabling
or removing the adapter exposes the original full model. Its measured forgetting
is conditional suppression while the adapter is enabled. The deployment artifact
under study is therefore the merged checkpoint. Results do not establish that
the original base/checkpoint has been erased or that adapter removal is resisted.

## Evaluation decisions

Use three declared training seeds (42/43/44), a fixed seeded 300-row retain
diagnostic sample, source-style sampled generation and scheduled full TOFU
measurements at updates 6,12,...,60,62. This cadence approximates source epochs;
record the actual processed-data fraction (final 9.92 forget passes). Add legacy
96-token greedy comparisons and a frozen 20-row-per-component 256-token audit.
Primary TOFU utility uses arithmetic perturbation averaging; geometric FQ is
reported against both fixed BF16 and matched-precision retain95 oracles. Keep
paper-arithmetic KS as an explicitly named diagnostic. Add directional log
preference CIs, author-block sensitivity, generation termination/cap rates, KL,
actual answers, model/packed-state hashes and cross-seed tables.

These are descriptive local measurements. RG-005 remains blocked for confirmatory
attack/durability verdicts; decision fields stay null. No protocol thresholds or
best-checkpoint deployment selection are inferred from this pilot.

## Project-wide consequences

NPO run 3 implements this decision. SSIUU attribution/HVP and FQ-QAT must later
define their objectives and gradient paths on the **effective adapted weights**;
neither method is made correct merely by attaching PEFT. QUAIL requires its
complete objective before adapter training. The novel combination remains
unimplemented and blocked until these adaptations have their own conformance
and resource evidence. This decision records the training policy for all future
conditions; it does not claim those other implementations have been completed.

## Acceptance

Require objective/masking tests, frozen-base/reference checks, deterministic
sampling/coverage, device-microbatch equivalence, adapter-only optimizer state,
checkpoint corruption/resume checks, merge/reload tests, metric scalar oracles,
complete tiny-model report generation and actual CUDA smoke checks. CPU fixtures
are engineering evidence; the actual 1B three-seed results and GPU memory/runtime
remain pending until the third run is executed on the researcher's machine.
