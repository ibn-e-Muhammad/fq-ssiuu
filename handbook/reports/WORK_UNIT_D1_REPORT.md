# Work Unit D1 — NPO 1B adapted reproduction

**PARTIAL WITH EVIDENCE — CPU training-path conformance; accumulation 8 approved; GPU implementation remains.**
No Kaggle submission, 1B training, trained 1B checkpoint or effectiveness
result. Tiny random CPU optimizer/checkpoint tests have executed. Intended classification is a bounded source-faithful 1B adapted-reproduction
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
review of D1 is claimed; D1 training review remains outstanding.

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
GPU runner/source evaluation/storage implementation and review before the bounded run.
Kaggle officially retains at most 20 GB of notebook output; a full checkpoint plus
a second raw/archive copy would exceed it. The runtime must persist one complete
sealed copy and verify every member, without dropping optimizer state or raw records.
