# Work Unit D1 — NPO 1B adapted reproduction

**PARTIAL WITH EVIDENCE — researcher pre-run review and CI 66 passed; D1 attempt 1 failed before 1B; source-preserving API repair validated, one retry remains.**
No completed 1B training, trained checkpoint or effectiveness result is yet established. Tiny random CPU optimizer/checkpoint tests have executed. Intended classification is a bounded source-faithful 1B adapted-reproduction
pilot. Completed work currently establishes upstream/CPU conformance only.

## Lineage and governance

PR #10 was approved by the researcher and merged. Fetched dev/base:
`5ca92bfa0863a9ffadcd70bfc5f28c8ee8c6e76a`; merged head
`29d21a51de784190bdabf4e7267db8925c73260e` has the same tree.
Only branch `repro/npo-1b-source-faithful` was created. Code/evidence commits and
exact file hashes are resolved by the D1 evidence manifest and Git history.

Disabled only `require_last_push_approval` in dev ruleset 20175027, as requested.
Passing CI, one fresh approval, resolved threads and history protections remain.
Before/after receipt: `../evidence/work_unit_d1/20260925_preflight/governance.json`.
No main change, history rewrite, branch deletion or DOCX inclusion. No independent
review by the primary agent is claimed. The researcher approved the substantive
pre-run review: “Review passed; proceed once CI passes.” CI 66 passed.

## Sources and deviations

Author NPO: `1207bbe9a1abf502bc2d7b43a0a467d9524b4036` (2024-07-15).
The TOFU code is identical to the publication-near 2024-04-10 commit; the later
commit adds only a synthetic experiment file. File-level hashes and license
limitation are recorded in `experiments/reproductions/NPO/source_v1.json`.

| Referenced author file | SHA-256 |
|---|---|
| TOFU/data_module.py | `8e076f18c20b663dbad9862a5d97c25e91fe4f9475bed77e597dbc2b349abf31` |
| TOFU/dataloader.py | `a692b709d01d05204a81e40c38b1f6e64faac9286a662d9d7cf9e4cc4dcf6c69` |
| TOFU/forget.py | `0f34bdaae7cc4fbe8f3ee97bf568beed3df37a537c672b57c237ba1b1904f7d9` |
| TOFU/config/forget.yaml | `c0e00825cd3157166b4188e0857239e0c4f5f889509874564bef501ed1b98e60` |
| TOFU/config/ds_config.json | `50112c35f92600290d224134840941254e5553d4432717d94e9ab24a10a4d188` |

OpenUnlearning remains pinned to `4ad738aaf60f6a4385f6e2506d01da99e76c31f3`;
four loss source hashes are in `upstream_parity.json`, and adapter hashes remain
in the unchanged accepted input registry. Source code is referenced externally,
not redistributed under the project license.

The [single source/adaptation specification](../methods/NPO_D1_SOURCE_ADAPTATION.md)
maps every method component. Source default is **NPO+RT (`npo_grad_diff`)**, beta
0.1, both coefficients 1, all parameters trained, full frozen reference, LR 1e-5,
weight decay .01, seed 1001, global batch 32, ten nominal epochs and floored
max_steps 62 on forget05. No replacement objective or LoRA was implemented.
Accepted model/API adaptations are 1B Llama-3 chat/EOT and T4 FP16/SDPA.
Proposed resource difference is accumulation 4 → 8 on one GPU, retaining batch
32. The researcher approved this change; it remains non-executable until implementation acceptance; no project-method adaptation exists.

## Acceptance to date

| D1 requirement | Evidence / status |
|---|---|
| Actual pinned upstream objective comparison | PASS on 16 CPU cases: pure NPO and NPO+RT, FP32/FP64, default/nonunit coefficients; maximum gradient error `2.9802322387695312e-08` |
| Data identity and membership | PASS: 200 forget05 and 3,800 retain95 rows; files match accepted SHA-256 registry and author Git blobs byte-for-byte; no cross-split repeated question |
| Real model adapter tensors | PASS for all 4,000 rows against executed hash-pinned OpenUnlearning preprocessing, length 500; no row truncated; ordered row/tensor hashes preserved |
| Shifted answer/EOT masks; frozen reference | Existing positive/negative tests pass; B GPU evidence remains scoped to frozen forwards; no new 1B runtime claim |
| Intended parameter gradients | New tiny random CPU Llama backward gives finite nonzero gradients to every unique trainable parameter, no reference gradients or mutation; no optimizer step |
| Source schedule | Local values checked against executed author assignments; unchanged knobs on one GPU give batch 16 / 125 steps, whereas proposed accumulation 8 preserves batch 32 / 62 steps |
| Save/resume, scheduler/optimizer integration | CPU state and actual DeepSpeed ZeRO-3 save/load pass; subsequent losses, FP32 masters, scheduler and RNG match exactly. Original paged FP16 GPU optimizer/restore remains unvalidated |
| Immutable execution config | Source/input manifests pinned; inventory rejected as a Mother recipe. No accepted training/evaluation recipe exists |
| Failure artifacts | Preflight failures recorded under new files; behavioral test prohibits overwrite. Training partial-artifact handling remains to implement |
| 1B optimization and source evaluation | NOT RUN |
| GPU memory, checkpoint/archive hash agreement | NOT RUN; checkpoint and archive hashes null; zero of two submissions used |

## Offline evidence and repairs

Final checks/return codes and raw stdout/stderr are in
`../evidence/work_unit_d1/20260925_preflight/CHECKS_FINAL.json`.
The first full suite passed **308 tests**, four existing skips, ten GPU/model
exclusions, one historical xfail. Final totals are **310 passed, 4 skipped, 10 deselected, 1 xfailed**, exit 0; see `cpu_suite_final_stdout.log`. Lint/format and all 14 notebooks are
checked without executing a notebook. No test downloads model/data or uses GPU
credentials; the explicit adapter preflight uses already cached pinned assets.

Two local preflight defects were diagnosed and repaired; their records remain:

- `preflight.json`: an overstrict duplicate-question rule rejected source retain95
  rows 100/440. Their answers differ. The repair preserves both by split/index,
  records the repetition, and still rejects cross-split question overlap.
- `preflight_final.json`: PyYAML parsed upstream `1e-5` as text. Numeric comparison
  now preserves its value and rejects wrong/nonfinite values. No LR change.

An initial tiny-gradient test incorrectly treated the loss-terms object as a
tensor; corrected to inspect/backpropagate `.total`. This was a test defect.
No failed scientific run is being relabelled as passed. `preflight_attempt2.json`
records the successful data boundary before the extra source-schedule checks;
later accepted records preserve the strengthened validation separately.

All **174** existing C1 manifest entries rehash correctly. Historical raw evidence
is unchanged. The D1 manifest binds file contents, test outputs and code identity;
it makes no scientific claim from test counts.

## Training-path continuation — 2026-09-26

Evidence: `../evidence/work_unit_d1/20260926_training_conformance/`. Full final
offline suite: **347 passed, 4 skipped, 10 deselected, 1 xfailed**, return code 0;
lint/format and all 14 notebooks pass. Core new tests are not skipped. The earlier
342-test check is retained separately. Actual author functions independently
verify all 200 forget/retain pairings, 496 ordered microbatches, utility and KS
fixtures. After the scored-logit memory reduction, 16 upstream objective cases
still pass with maximum gradient discrepancy about `2.98e-8`.

Native DeepSpeed 0.19.7 CPU checkpoint/resume uses a random tiny NPO+RT model and
CPU AdamW. After reload, 16 subsequent microbatches have zero loss difference;
FP32 master weights are bit-identical and scheduler/RNG/cursor match. This is
not CUDA, FP16 or bitsandbytes optimizer acceptance. Raw tiny checkpoints remain
at the local path recorded in `DEEPSPEED_EXECUTION.json`; their file hashes are
in `CPU_CHECKPOINT_MANIFEST.json`. Failed sandbox Gloo initialization and earlier
CPU diagnostic runs are preserved, and do not consume Kaggle attempts.

Source/API findings incorporated into the one specification: effective gradient
clipping is zero in all checked publication-era DeepSpeed releases, despite the
Trainer argument default of one; the modern default must be overridden to zero.
The source sampler uses 1001, omitted Trainer seed resolves to 42, and generation
resets seed 0 per batch. All 717 evaluation rows / 4,585 answer variants fit the
source length 200 without truncation. These are input/implementation findings,
not observations of forgetting or retained utility.

## Memory and claims

The tied model config contains 1,235,814,400 unique parameters, verified by an
independent meta-model construction without allocating model weights. Each FP16
model's parameter bytes are 2,471,628,800. A fully resident candidate/reference,
master, FP32 Adam moments and FP16-gradient inventory totals 22,244,659,200 bytes.
**This is not a measured GPU peak or proof that the paged optimizer cannot fit.**
Managed optimizer states can migrate; full-device telemetry is required in the
future run, in addition to PyTorch allocator peaks. No training-memory PASS.

Supported: source/config identity, loss/sampler/metric conformance, full pinned
training-data adapter comparison, tiny CPU gradient wiring and native CPU ZeRO-3
checkpoint/resume conformance. Unsupported: NPO training at
1B, reloadable trained checkpoint, source evaluation results, forgetting,
quantization recovery or durability. No source-defined scientific metrics exist
for D1 yet. Attack/durability decisions remain null, Protocol v2 stays deferred
and non-executable. All research/operational gate states remain unchanged.

## Exact blocker and next action

Researcher decision received: category-3 accumulation **8 instead of 4** is approved
for one T4, preserving effective batch 32, full-parameter NPO+RT, full frozen
FP16 reference and the original paged 32-bit AdamW. D1 automatically allows only
categories 1/2; the proposal changes an explicit source hyperparameter and cannot
be silently applied; explicit approval has now been recorded. Continue **this same D1 branch** with
substantive review of the implemented GPU runner/source evaluation/storage path before the bounded run.
Kaggle officially retains at most 20 GB of notebook output; a full checkpoint plus
a second raw/archive copy would exceed it. The runtime must persist one complete
sealed copy and verify every member, without dropping optimizer state or raw records.

## Runtime readiness — 2026-09-27

Implemented the closed Mother D1 route, frozen training/evaluation configs,
full-parameter FP16/ZeRO-3/paged AdamW runner, source-sized evaluator, separate
retain-only oracle evaluation, native checkpoint save/resume, FP16 export/value
comparison/reload, memory telemetry and truthful partial output. All code remains
in the NPO reproduction namespace; nothing is promoted as an accepted combined
method. Existing C1 locks and raw artifacts are preserved.

Acceptance evidence: `../evidence/work_unit_d1/20260927_runtime_readiness/`.
Code commit: `65c454a20de4658851603cbf49eb4796afed95c5`.
Immutable manifest: `../evidence/work_unit_d1/MANIFEST_20260927.json`, SHA-256
`5370210c9f79a4cffd0560ccb2b4f171c36d6b014e0d6147438068fe7074bbe9`.
The final offline suite records **378 passed, 4 skipped, 10 deselected, 1 xfailed**;
no new core test is skipped. Lint/format and all 14 notebooks pass. Actual pinned
source objective, sampler, clipping and metric checks are rerun offline. A tiny
random **real CPU ZeRO-3** loop completes all 496 microbatches/62 steps with finite
nonzero gradients, unchanged reference and source evaluation cadence. Ordinary
CPU AdamW is used only by that test; it does not validate the runtime paged GPU
optimizer. An initial local-loopback sandbox failure and its successful authorized
reruns are preserved, not counted as Kaggle submissions.

Primary-agent implementation review is not independent review. A substantive
review of this D1 PR remains the first unresolved pre-run requirement. Once that
passes, the exact next action is the first bounded private T4 D1 submission.
No trained 1B checkpoint, scientific metric, GPU memory observation or external
archive hash exists yet; those must not be inferred from CPU acceptance. The
final D1 verdict remains PARTIAL and every research/operational gate is unchanged.

## First submission — 2026-09-27

Researcher review and CI 66 authorized attempt 1, private Mother version 5, from
`ebd7077539a8093162fcad1264c3ff7240ef4a66`. Source dataset
`muhammadii/fq-ssiuu-d1-source-ebd7077` is private. Receipts and exact staging
hashes are in `../evidence/work_unit_d1/20260927_kaggle_attempt1/`. A local sandbox
DNS failure did not reach Kaggle; its logs are retained separately from the
successful network submission. One of two runtime submissions is used. Earlier
readiness snapshots above remain historical; the next action is to collect and
verify the runtime and archives. PR #11 stays open.

## Attempt 1 outcome and bounded repair — 2026-09-28

Run `20260927-130645-e8483bf31420`, Mother v5, source `ebd7077`:
**FAIL_OPERATIONAL_BEFORE_1B**. Environment verification passed with
BASELINE_DIRTY_NO_REGRESSION. The real tiny synthetic GPU test reached backward
and optimizer/scheduler control flow, then failed because scheduler and optimizer
group counts differed. No target 1B model was loaded or trained, so there is no
1B checkpoint, source metric or training-memory claim. All 61 original sealed
members rehash correctly; raw evidence was not edited.

Complete archive SHA-256:
`2dbbfb2f98d0ca51aed1b1c2c5c699aef9265f4ebc18478b9d6171094ee3f77b`.
Private Kaggle dataset: `muhammadii/fq-ssiuu-d1-20260927-e8483bf31420`.
Private HF dataset: `Xafire/fq-ssiuu-run-20260927-130645-e8483bf31420-complete`,
revision `f90c9838ed1b07ae1f11b2c6562f7956b100eb08`. HF archive bytes were
downloaded and verified. Kaggle auto-extracted both tar levels; all members were
downloaded and checked, then both canonical archives were reconstructed with
matching original hashes. This is not a claim of direct tar-byte download from
Kaggle. See attempt-1 `VERDICT.json` and both archive receipts.

Repair code `97e005af195cb79e8faae0dedf09aa42b6f318cb` restores the source
LlamaRMSNorm exclusion and constructs the linear scheduler after DeepSpeed group
normalization. The single source/adaptation specification records the additional
pinned primary files. Training/evaluation configs, source NPO+RT, full frozen
reference, source hyperparameters and approved batch-32 accumulation remain
unchanged. Actual native CPU ZeRO-3 reproduces the old failure, then passes the
full 62-step schedule and checkpoint restore for an empty-group fixture and a
tiny Llama. Executed pinned Trainer grouping matches the exact 1B architecture
on meta tensors. Full suite: **381 passed, 4 skipped, 10 deselected, 1 xfailed**,
return code 0; lint/format, 14 notebooks and upstream objective/schedule/metric
conformance pass. Repair evidence is `../evidence/work_unit_d1/20260928_optimizer_repair/`.

The primary agent reviewed the narrow operational repair; no independent review
is claimed. Initial researcher review and authorization for one operational,
code-reviewed retry are preserved. Next: after passing CI, use the remaining
private T4 submission. D1 remains unaccepted until actual 1B optimization,
evaluation, reload, memory and archives pass. No gate or Protocol v2 activation.
