# Work Unit D1 — bounded NPO 1B adapted-reproduction pilot

**PARTIAL WITH EVIDENCE; researcher-approved resource continuation is in pre-run
validation.** Both authorized T4 attempts failed. Attempt 2 passed the
actual tiny GPU paged optimizer/resume test, then exhausted memory during 1B
optimizer initialization, before any 1B backward or update. NPO at 1B is not yet
reproduced. No trained 1B checkpoint or post-unlearning scientific metric exists.

## Lineage, approval and method

Base `dev`: `5ca92bfa0863a9ffadcd70bfc5f28c8ee8c6e76a` (PR #10 merged).
Branch: `repro/npo-1b-source-faithful`; PR #11 remains open, not merged.
Researcher pre-run review: “Review passed; proceed once CI passes.” CI 66 passed
for attempt 1; CI 67 passed for attempt 2. Repair `97e005a` was reviewed by the
primary agent; no independent review is claimed. The last-push approval restriction
was disabled as requested; other existing protections remain.

The intended classification is a **bounded source-faithful 1B adapted-reproduction
pilot**, not an exact replication of the original headline result or full NPO
reproduction. The [single source/adaptation specification](../methods/NPO_D1_SOURCE_ADAPTATION.md)
maps objective, reference, tokenization, splits, retain term, optimizer, schedule,
stopping, metrics and checkpoint selection. Full NPO+RT (`npo_grad_diff`), beta
0.1, coefficients 1/1, LR 1e-5, weight decay .01, paged FP32 AdamW and all unique
parameters are preserved. Source schedule is 62 updates, six warmup updates;
evaluation at 6…60 and 62. No LoRA, changed retain objective, QAT or SSIUU.

Adaptations: Llama-3.2-1B/chat/EOT model adapter; T4 FP16/SDPA and current dependency
API compatibility; source norm exclusions and post-normalization scheduler factory;
explicitly approved accumulation 4 → 8 on one GPU, preserving effective batch 32.
Sampler seed 1001, source Trainer global seed 42 and generation reset seed 0 are
recorded separately rather than silently conflated. No project-method adaptation.

## Pinned sources and inputs

NPO author commit `1207bbe9a1abf502bc2d7b43a0a467d9524b4036`, 2024-07-15;
TOFU experiment code is identical to the publication-near 2024-04-10 version.
The pin has no license file; minimum source is externally referenced/hash-checked,
not redistributed under the project license. Full file inventory is in
`experiments/reproductions/NPO/source_v1.json`.

| Author file | SHA-256 |
|---|---|
| TOFU/data_module.py | `8e076f18c20b663dbad9862a5d97c25e91fe4f9475bed77e597dbc2b349abf31` |
| TOFU/dataloader.py | `a692b709d01d05204a81e40c38b1f6e64faac9286a662d9d7cf9e4cc4dcf6c69` |
| TOFU/forget.py | `0f34bdaae7cc4fbe8f3ee97bf568beed3df37a537c672b57c237ba1b1904f7d9` |
| TOFU/config/forget.yaml | `c0e00825cd3157166b4188e0857239e0c4f5f889509874564bef501ed1b98e60` |
| TOFU/config/ds_config.json | `50112c35f92600290d224134840941254e5553d4432717d94e9ab24a10a4d188` |

OpenUnlearning: `4ad738aaf60f6a4385f6e2506d01da99e76c31f3`; file hashes are
in the unchanged upstream/adapter registries. Publication-era Transformers
`09f9f566de83eef1f13ee83b5a1bbeebde5c80c1` supplies the executed norm/grouping
reference; file-level hashes/licensing are in the source/adaptation specification.

- Full target/tokenizer and intended NPO reference:
  `open-unlearning/tofu_Llama-3.2-1B-Instruct_full@88e31200b97e4c0c04ae0d2f0b591f427046d192`.
  Original weight hash `35b0cd9f7db2735538ff78b4879fab61b23b7ca316df27b059585bf656700a8d`.
- TOFU: `locuslab/TOFU@324592d84ae4f482ac7249b9285c2ecdb53e3a68`,
  forget05 200 rows / retain95 3,800 rows. Membership and all 4,000 ordered adapter
  tensor records are bound to the run. No replacement dataset.
- Diagnostic oracle, distinct from the full NPO reference:
  `open-unlearning/tofu_Llama-3.2-1B-Instruct_retain95@f05bc9fd6c8d59decf7599f5e186153081061fd2`.
  Weight hash `955e430aac277d9ccc309307f8462ce462c81fd4308cd934a6c24e5518077e94`.
- Unchanged training config hash:
  `f32a35dbe9b82fcefffc12709c09a22224bf2dee05c67326a4a1aa6ccfadd9e1`;
  evaluation config hash:
  `0db526da36ac9c60946cb6e4046005061585f870a42ed3ed10099a4ca6084194`.

## Acceptance and exact limitations

| Requirement | Result |
|---|---|
| Pinned upstream value/gradient conformance | PASS, 16 fixed CPU cases; maximum gradient discrepancy approximately 2.98e-8 |
| Dataset, shifted answer/EOT masks, prompt/padding exclusion | PASS offline; all 4,000 rows compared with executed pinned preprocessing; runtime preparation passed |
| Full frozen reference identity | Contract and prior bounded forwards pass; D1 attempt 2 stopped before loading its full NPO reference |
| Full-parameter gradients | Tiny CPU model passes; no 1B gradient/optimization evidence |
| Source schedule, sampler and metrics | Executed upstream CPU comparisons pass; no trained-1B evaluation |
| Native CPU checkpoint/resume | PASS, source schedule and state restored |
| Actual tiny FP16 GPU paged optimizer/resume | PASS on attempt 2: 16 resumed microbatches, max loss error 0, bit-identical masters, equal scheduler/RNG |
| 1B optimization | FAIL before first update: optimizer initialization OOM |
| Reloadable trained 1B checkpoint | Absent; only tiny random-model checkpoints exist |
| Source-defined unlearned-model metrics | Absent; raw retain-only oracle evaluations are preserved separately |
| GPU memory | Whole-attempt telemetry exists; no successful 1B training-memory acceptance |
| Immutable local/Kaggle/HF archives | PASS for both failed attempts |

Final offline checks: **382 passed, 4 skipped, 10 deselected, 1 xfailed**, exit 0.
The skips/exclusions/xfail are existing cases; no new core acceptance test is
skipped. Lint, format, all 14 notebook schema/cell checks, executed upstream
objective and schedule/metric conformance pass. Commands, exit codes and raw logs:
`../evidence/work_unit_d1/20260928_closeout/CHECKS.json`.

A closeout test correction restores tied embeddings in a bare meta fixture and
compares the source Trainer's deduplicated parameter groups. Unique target count
is 1,235,814,400. The earlier untied topology claim is explicitly superseded in
`INVENTORY_CORRECTION.json`, preserving its original record. The first corrected
fixture check also exposed an unused tied-name alias in the expected-name list;
the final real CPU check passes. Neither correction changes the GPU runner or
scientific configs. Old manifests (228 D1 and 174 C1 entries) rehash unchanged.

## Both Kaggle attempts and archives

| Attempt | Code / run / Mother | Outcome |
|---|---|---|
| 1 of 2 | `ebd7077539a8093162fcad1264c3ff7240ef4a66` / `20260927-130645-e8483bf31420` / v5 | Tiny GPU scheduler/group mismatch; no 1B loading |
| 2 of 2 | `485504904d8116797430da7234e1a6bef5dee22c` / `20260928-062524-7118f1ae8d74` / v6 | Tiny GPU resume passes; oracle evaluated; target loaded; 1B optimizer-init OOM |

Both environment checks passed `BASELINE_DIRTY_NO_REGRESSION`; no new package
conflict or protected-stack drift. Selected single T4 only. Attempt 2 failed at
DeepSpeed's temporary FP32 gradient buffer allocation: requested 3.74 GiB, only
3.58 GiB free. No 1B backward/optimizer step followed.

Whole-attempt peak allocated **11,496,737,792 bytes**, reserved **14,245,953,536**,
sampled device-used **15,935,406,080**. NVML sampling interval is 0.1 seconds;
sub-interval transients are not continuously measured. Being below 16e9 in these
samples does not establish feasibility: the attempted allocation failed.

Complete archive SHA-256 values:

- Attempt 1: `2dbbfb2f98d0ca51aed1b1c2c5c699aef9265f4ebc18478b9d6171094ee3f77b`.
  Private Kaggle `muhammadii/fq-ssiuu-d1-20260927-e8483bf31420`;
  HF `Xafire/fq-ssiuu-run-20260927-130645-e8483bf31420-complete` at
  `f90c9838ed1b07ae1f11b2c6562f7956b100eb08`.
- Attempt 2: `6fe611e1fc3b9361e5b71374532198181d03d483e7e53deabdf65f8f2f6d06e5`.
  Private Kaggle `muhammadii/fq-ssiuu-d1-20260928-7118f1ae8d74`;
  HF `Xafire/fq-ssiuu-run-20260928-062524-7118f1ae8d74-complete` at
  `954cd609f88319382a3480d653b708da8df6f28c`.

Attempt 1 Kaggle auto-extracted archives; members and reconstructed canonical tar
hashes were verified. Attempt 2 uses an opaque `.bin` transport suffix and the
actual downloaded tar bytes match. All five outer and 72 inner members verify.
The two tiny `.pt` files remain in complete private archives; 70 extracted raw
members are committed unchanged. Tiny checkpoint hashes (not trained 1B weights):
model state `2f6dba9bc53c0eb77842784cc459e75d3c0d1f94631e37a38b4809bc03c039df`;
optimizer state `6c098f5fbba0ca909ec79c804cf764be85c5c6e9d706b97e96eeb1ad905ba1a4`.

Attempt verdicts, raw traces, manifests, config/source identities and archive
receipts live under `../evidence/work_unit_d1/20260927_kaggle_attempt1/` and
`../evidence/work_unit_d1/20260928_kaggle_attempt2/`. Closeout manifest:
`../evidence/work_unit_d1/MANIFEST_20260928_CLOSEOUT.json`. Its containing Git commit
resolves the final report/test-helper revision; actual runtime code is `4855049`.

## Exact blocker and next action

Both allowed submissions are used. Proposed category-3 change, **not implemented**:
reduce only ZeRO optimizer `sub_group_size` from 1e9 to 1e8. Read-only execution
of the installed grouping code on the tied architecture reduces the largest
temporary buffer from 4,011,851,776 to 1,050,673,152 bytes. This addresses the
observed failed allocation, not proof of complete training fit. Candidate,
masters, FP16 gradients and full reference alone account for 12,358,144,000 bytes,
excluding managed moments, activations, workspace and fragmentation.

Next single action: obtain researcher approval for that resource-only change and
one additional bounded T4 attempt, conditional on offline update/resume conformance
and code review. Preserve full parameters, full GPU reference, original paged FP32
AdamW, objective, batch and schedule. No further submission before approval.

No research gate changes: RG-006 remains limited to the accepted objective/data/
reference boundary; RG-005 blocks project verdicts; RG-001/002/003 stay blocked.
Protocol v2 remains deferred/non-executable; attack and durability decisions are
null. No forgetting, recovery, durability or effectiveness conclusion follows
from D1. Historical evidence and the four DOCX files remain untouched.

## Approved continuation — 2026-09-28

The researcher subsequently answered **“Approve resource change and one extra
attempt.”** This supersedes the approval blocker above without changing either
failed run's verdict. Total allowance is three, with two used. The new v2 train
config/recipe preserves v1, changes only the approved subgroup size and version
identifiers, and keeps the evaluation config unchanged. Actual CPU ZeRO-3 subgroup
comparison passes: 64 microbatches, identical losses/masters/moments; 16 resumed
microbatches match in each layout; reference unchanged. Source method unchanged.
The next action is offline/CI/review acceptance followed by the one newly approved
T4 attempt. Its GPU result remains unknown. Evidence:
`../evidence/work_unit_d1/20260928_subgroup_adaptation/`.
