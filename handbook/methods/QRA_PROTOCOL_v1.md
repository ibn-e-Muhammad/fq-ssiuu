# QRA decision protocol v1 — PROPOSED_NOT_APPROVED

Prepared 2026-09-23, before Work Unit C1 outputs and before corrected training/pilots.
This is a review document, not executable configuration or preregistration. RG-005
remains BLOCKED. Every runtime attack/durability field remains null. Muhammad Arslan
must explicitly approve the final version; supervisor review is preferred. Approval
of loader integration does not approve this protocol or a full hypothesis study.

## Scope and fixed identities

Use the pinned full TOFU Llama-3.2-1B-Instruct target, tokenizer, TOFU revision and
retain95 oracle in `experiments/reproductions/NPO/inputs_v1.json`. Candidate/reference
tokenization remains identical. Work Unit C1 evaluates the original full model,
not an unlearned model. It therefore cannot satisfy an initial-forgetting gate or
produce a recovery result, regardless of observed precision differences.

The current design is **forget05/retain95**, superseding the early forget01 idea.
Work Unit B already pinned and validated this boundary. This explicit record
acknowledges that historical drift; it does not relabel any earlier forget01 run.
forget05 contains ten fictional authors and 200 Q&A rows. Each row in both the
plain and perturbed pinned files explicitly identifies one of ten full names in
its question/answer. Versioned author maps record all row hashes and assignments;
no cluster is inferred from row position alone. Source mapping must be reverified
before a scientific run. The 400-row `retain_perturbed` utility subset is distinct
from the 3800-row retain95 training split; do not describe them as identical sets.

## Measurements and artifact identity

Primary endpoint: paired change in answer/EOS mean teacher-forced log probability,
`M_quantized - M_FP16`, in natural-log units per scored token. Prompt/padding are
excluded. A positive value means greater observed answer knowledge. This is an
operational memorisation measure, not proof of latent data removal.

Secondary endpoints: paired ROUGE-L recall and normalized exact-match changes.
Use the registered definitions, identical deterministic greedy generation and
the complete resolved decoding hash. Keep these secondary; no post-hoc endpoint
switching if the primary endpoint is inconvenient.

Retain, real-author and world-fact utility remain separate from forget recovery.
Record all nine utility component means. Model utility is their harmonic mean,
zero when any component is zero, with a separate derived-component manifest.
Our canonical arithmetic Truth Ratio score differs from OpenUnlearning's geometric
variant, so this aggregate is not claimed to reproduce that variant numerically.

TOFU arithmetic Truth Ratio is canonical, as already chosen by the researcher.
Preserve the geometric OpenUnlearning variant as a named diagnostic. Record raw
ratios, KS D and KS p-value. Neither is a recovery effect or equivalence result.
The conventional row-level KS p-value also does not account for author dependence;
retain it solely for comparability as a diagnostic, not as a decision gate.

Canonical diagnostic oracle policy proposed: fixed unquantized FP16 retain95
oracle with immutable checkpoint and raw evaluation evidence. A precision-matched
oracle may be a separately named future sensitivity analysis, never an implicit
replacement. C1 tests only the fixed-oracle plumbing.

Each precision artifact binds original weight SHA, torch/Transformers/bitsandbytes
versions, complete quantizer configuration, ordered actual quantized module list,
packed weights, scale/codebook/quantization-state tensors and the full state-dict
inventory. Saved-shard hashes are mandatory only if a persistent quantized model
is produced; C1 produces in-memory identities, not saved quantized checkpoints.
Evaluate the full expected 112 Llama linear projections (excluding floating
lm_head/embeddings); reject partial quantization, wrong grids, offload and sharding.

Evaluation digests cover the complete raw observations, reported values, metric
definition, evaluator, preprocessing/decoding, oracle, ordered samples, precision
artifact and run/group linkage. Validators recompute those digests. Historical
v4 reports remain readable only as superseded payloads; they are never silently
upgraded to the stronger v5 evidence contract.

## Proposed uncertainty and aggregation

Within a seed, calculate each author's mean of paired Q&A changes, then average
authors equally. Resample whole authors with replacement. Preserve all per-row
pairs and per-author means. This differs from treating 200 forget rows as 200
independent units. The old row bootstrap remains an engineering diagnostic only.

Across seeds, use the same authors and complete seed-by-author matrix. Candidate
procedure: crossed bootstrap, independently resampling seeds and authors with
replacement and using the same sampled author indices across sampled seeds.
This preserves shared-author dependence across seeds. Do not independently sample
authors within each seed when claiming the crossed procedure. Report every seed
and author before aggregation. Missing seeds cannot be silently dropped.

Proposal for later approval: seeds 23, 42, 101; PCG64 bootstrap seed 20260923;
10,000 resamples; percentile intervals with linear quantiles. Three seeds provide
limited precision: report this limitation, and do not pretend seed resampling
creates additional independent training runs. Report per-seed clustered intervals
alongside the aggregate. Consider more seeds before full protocol freeze if the
compute budget permits, without selecting seeds from observed outcomes.

Two primary precision comparisons (INT8, NF4) form one family per declared
condition. Proposed simultaneous coverage: Bonferroni 97.5% two-sided intervals
for each comparison, giving at least 95% family coverage under the procedure's
assumptions. The full study must separately declare multiplicity across conditions;
this document does not resolve that by omission. Secondary endpoints are labelled
descriptive/exploratory unless a separate family is explicitly approved.

## Concrete candidate margins for researcher review

These numerical values are proposals chosen before corrected outcomes. They are
not literature-established universal thresholds and cannot run as acceptance rules.

| Item | Candidate | Rationale and limitation |
|---|---|---|
| Minimum meaningful primary recovery | ln(1.10) ≈ 0.09531 nats/token | 10% increase in geometric mean answer-token probability; operational choice requiring approval |
| Durability/equivalence margin | ±ln(1.10) | Precision-induced log-probability change constrained within the same declared band; symmetric equivalence, not mere non-significance |
| Initial forgetting versus oracle | FP16 unlearned-minus-oracle log-probability interval within ±ln(1.10), plus a reduction from original exceeding ln(1.10) | Requires both proximity to retain-only behavior and a demonstrated change from the full target; cannot be established by high KS p alone |
| Allowed utility decrease | 0.05 absolute on each bounded utility component and the derived aggregate | Transparent five-point operational cap; must be checked against the full FP16 target for initial forgetting and unlearned FP16 for quantized deployment |

The initial gate needs corresponding clustered comparisons on forget examples
and an explicit sampling unit for utility datasets. Real-author/world-fact rows
need an independently justified subject/fact grouping before confirmatory utility
intervals; do not reuse the fictional-author mapping on unrelated data. Until that
mapping and multiplicity are resolved, these are proposal-level acceptance rules.

## Two decisions, with an indeterminate outcome

1. Establish initial FP16 forgetting and acceptable retained utility first.
   Otherwise label the checkpoint ineligible for a durability claim; do not treat
   conversion of an already knowledgeable full model as an unlearning attack.
2. Proposed `attack_detected=True` only if the simultaneous lower bound for a
   primary paired change exceeds the meaningful recovery margin. This entails
   `durability_pass=False` for that precision.
3. Proposed `durability_pass=True` only if the initial gate passes, the entire
   simultaneous primary interval lies within the declared equivalence band, and
   utility loss remains inside its approved margin. Mere failure to detect an
   attack never supplies this evidence.
4. Otherwise retain an indeterminate decision with the reason and observed
   intervals. Across deployment precisions, durability requires both conditions.
   Secondary diagnostics cannot override a failed or indeterminate primary rule.

## Approval and remaining decisions

Researcher approval: **PENDING**. Supervisor review: **PENDING**.
Required consolidated review: margins and initial gate; fixed-oracle policy;
forget05/retain95 design; seed/author procedure and multiplicity; utility grouping;
secondary endpoint status. No approval is inferred from silence or from C1 PASS.
Full hypothesis preregistration, OSF submission and experiment execution require
their later, separate frozen study protocol and explicit approval.

Primary sources: [TOFU v1](https://arxiv.org/html/2401.06121v1),
[quantization-recovery v2](https://arxiv.org/html/2410.16454v2), pinned OpenUnlearning
and NPO sources in the input registry. The bootstrap and margins above are project
analysis proposals, not attributed to these papers. Candidate quantizer source is
bitsandbytes `4bca84499ad194d6c37e77dfcf99201b81dc6981`; release and file hashes are in
`handbook/evidence/work_unit_c/20260923_sources/BITSANDBYTES_0461.json`.
