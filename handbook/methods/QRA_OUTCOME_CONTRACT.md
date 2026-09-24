# QRA outcome contract — descriptive revision 1

Status: implemented descriptive contract; **RG-005 remains BLOCKED**. No model
experiment is part of scientific reconciliation A. This document does not freeze
metrics, thresholds or the confirmatory analysis.

## Primary evidence

[TOFU v1, 2024-01-11, §2.2.3 and Appendix A](https://arxiv.org/html/2401.06121v1)
defines Forget Quality as a two-sample KS p-value against a retain-only model.
Non-rejection does not establish equivalence; the p-value is not a magnitude of
forgotten or recovered knowledge. Utility must be assessed separately.

[Catastrophic Failure of LLM Unlearning via Quantization v2, 2025-01-23](https://arxiv.org/html/2410.16454v2)
studies recovery using memorisation, knowledge and privacy-related evaluations
alongside utility. It does not validate a ratio of TOFU p-values. This project's
NF4/1B evaluation remains an adaptation requiring explicit measurement rules.

## Implemented record and invariants

`src/fq_ssiuu/evaluation/qra_pipeline.py` accepts scalar observations at FP16,
INT8 and NF4. Every observation records metric definition/version, kind, direction,
run ID, source checkpoint hash, sample-set hash, reference hash and raw evaluation
hash. The common checkpoint is the merged pre-quantisation source; precision
artifacts may have different evaluation hashes. Hashes provide linkage, not proof
that a producing evaluator was scientifically correct.

Pairs require the same metric, definition, direction, source checkpoint, samples
and reference. P-values are retained without differences or ratios. A directional
memorisation score may have the signed change `quantised - FP16`; orientation
reverses its sign when lower means more memorisation. This is descriptive, not a
detected attack. Utility changes are recorded separately without a recovery label.
Missing values remain null; invalid/nonfinite data fail closed.

A complete descriptive report requires at least one matched non-null utility
metric and one matched non-null memorisation metric across all three precisions,
with one report-level source checkpoint. Unrelated metric names cannot fill a
missing precision. Completeness is not scientific acceptance.

`attack_detected` and `durability_pass` are independent nullable fields. Detected
attack requires durability false; durability true requires attack false. An
undetected attack alone never proves durability. The current writer produces only
null decisions and rejects attempts to save an approved verdict. Writes cannot
overwrite existing files. Historical v1/v2 payloads are preserved with their raw
hash and superseded interpretation; `certified` is never imported as acceptance.

## Remaining acceptance before corrected pilots

Define and review the actual directional metric implementations, sampling/prompt
and matching rules, initial forgetting and retained-utility prerequisites,
uncertainty treatment, across-precision aggregation and a versioned decision rule.
Use synthetic fixtures and independent numerical references before spending GPU
time. Any proposed hypotheses or thresholds require researcher approval before
freezing; none is inferred from historical p-values. Corrected P1 must use new
immutable run IDs after this gate and relevant method criteria permit it.

Behavioral tests: `tests/methods/QRA/`. The Mother scientific entry guard blocks
all corrected pilots while RG-005 is unresolved; environment setup remains frozen.

## Current extension — Work Unit C, descriptive revision 2

The revision-1 narrative above records reconciliation A. Current writes use report
v5, MetricObservation v2 and comparison-group v2. Evaluation hashes are recomputed
from the complete observation content. Precision manifests cover actual runtime
state tensor bytes, ordered quantized modules, source weights, library versions
and complete quantizer settings. No saved quantized checkpoint is implied.
Explicit supersession preserves older report v1–v4 payloads without upgrading them.

The fixed FP16 oracle's raw evidence and exact KS D/p computation are verified.
Nine utility components remain separate observations; their harmonic aggregate
has its own derived manifest. Author-cluster and crossed seed/author procedures
are implemented and CPU-tested, but remain unapproved. C1 matched inference passed
on four fixed samples per precision; its single-forget-row intervals are engineering
diagnostics, not confirmatory uncertainty. Protocol v2 is PROPOSED_NOT_APPROVED.
RG-005 and every scientific decision remain blocked/null.
