# Work Unit D1 — bounded NPO 1B adapted-reproduction pilot

**PARTIAL WITH EVIDENCE / FAILED GPU ACCEPTANCE. Paused for researcher audit.**
All three authorized single-T4 attempts failed. The third cleared optimizer
initialization and completed one real 1B NPO+RT forward with the frozen full
reference, but its first backward exhausted memory. It also exceeded the declared
16,000,000,000-byte sampled memory ceiling. Zero completed 1B optimizer updates;
no trained 1B checkpoint or post-unlearning scientific metric. The researcher
requested diagnosis, outcome commit/push and then a stop; no further work/run.

## Lineage, approval and method

Base `dev`: `5ca92bfa0863a9ffadcd70bfc5f28c8ee8c6e76a` (PR #10 merged).
Branch: `repro/npo-1b-source-faithful`; PR #11 remains open, not merged.
Researcher pre-run review: “Review passed; proceed once CI passes.” CI 66 passed
for attempt 1; CI 67 passed for attempt 2; CI 69 passed for attempt 3. Repair `97e005a` was reviewed by the
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
- Historical v1 training config hash (attempts 1/2, preserved):
  `f32a35dbe9b82fcefffc12709c09a22224bf2dee05c67326a4a1aa6ccfadd9e1`.
  Approved v2 training hash (attempt 3):
  `7b5bb1d7c43c68c6cc274ceb68fcdf20773ea2f36d57de1d66e5b8ba73365711`.
  Unchanged evaluation config hash:
  `0db526da36ac9c60946cb6e4046005061585f870a42ed3ed10099a4ca6084194`.

## Acceptance and exact limitations

| Requirement | Result |
|---|---|
| Pinned upstream value/gradient conformance | PASS, 16 fixed CPU cases; maximum gradient discrepancy approximately 2.98e-8 |
| Dataset, shifted answer/EOT masks, prompt/padding exclusion | PASS offline; all 4,000 rows compared with executed pinned preprocessing; runtime preparation passed |
| Full frozen reference identity | PASS for third-attempt full reference loading/validation and initial forward; no learning claim |
| Full-parameter gradients | Tiny CPU model passes; first 1B backward attempted and OOM; no completed 1B gradient/update acceptance |
| Source schedule, sampler and metrics | Executed upstream CPU comparisons pass; no trained-1B evaluation |
| Native CPU checkpoint/resume | PASS, source schedule and state restored |
| Actual tiny FP16 GPU paged optimizer/resume | PASS on attempts 2 and 3: 16 resumed microbatches, max loss error 0, bit-identical masters, equal scheduler/RNG |
| 1B optimization | FAIL: third attempt first backward OOM; zero completed updates |
| Reloadable trained 1B checkpoint | Absent; only tiny random-model checkpoints exist |
| Source-defined unlearned-model metrics | Absent; raw retain-only oracle evaluations are preserved separately |
| GPU memory | FAIL: third-attempt sampled peak 16,086,401,024 bytes exceeds 16e9; first backward OOM |
| Immutable local/Kaggle/HF archives | Verified preservation/receipts for failed attempts; third-attempt receipts are in its evidence directory |

Run-commit offline checks: **389 passed, 4 skipped, 10 deselected, 1 xfailed**, exit 0.
The skips/exclusions/xfail are existing cases; no new core acceptance test is
skipped. Lint, format, all 14 notebook schema/cell checks, executed upstream
objective and schedule/metric conformance pass. Commands, exit codes and raw logs:
`../evidence/work_unit_d1/20260928_subgroup_adaptation/CHECKS.json`.
No implementation changes or additional training tests were made during the
requested third-attempt outcome-only closeout; integrity checks are recorded separately.

Before the third submission, a CPU test correction restored tied embeddings in a bare meta fixture and
compares the source Trainer's deduplicated parameter groups. Unique target count
is 1,235,814,400. The earlier untied topology claim is explicitly superseded in
`../evidence/work_unit_d1/20260928_closeout/INVENTORY_CORRECTION.json`, preserving its original record. The first corrected
fixture check also exposed an unused tied-name alias in the expected-name list;
the final real CPU check passes. Neither correction changes the GPU runner or
scientific configs. All seven prior D1/C1 manifests rehash unchanged; counts are
recorded in the third-attempt `PRESERVATION.json`.

## Kaggle attempts and archives

| Attempt | Code / run / Mother | Outcome |
|---|---|---|
| 1 of 3 | `ebd7077539a8093162fcad1264c3ff7240ef4a66` / `20260927-130645-e8483bf31420` / v5 | Tiny GPU scheduler/group mismatch; no 1B loading |
| 2 of 3 | `485504904d8116797430da7234e1a6bef5dee22c` / `20260928-062524-7118f1ae8d74` / v6 | Tiny GPU resume passes; oracle evaluated; target loaded; 1B optimizer-init OOM |
| 3 of 3 | `1e4a0937cc58843d1dd74b8ebdcfb2b14a5c1080` / `20260928-124613-efd94997a1e8` / v7 | Approved subgroup reduction clears optimizer initialization; full target/reference forward passes; first backward OOM |

All three environment checks passed `BASELINE_DIRTY_NO_REGRESSION`; no new package
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
`../evidence/work_unit_d1/20260928_kaggle_attempt2/`. The earlier second-attempt
closeout manifest is `../evidence/work_unit_d1/MANIFEST_20260928_CLOSEOUT.json`;
its runtime code is `4855049`. The third-attempt source and manifest follow below.

## Third attempt: exact failure and claim boundary

The researcher explicitly approved subgroup size 1e8 and one extra T4 attempt.
Versioned v2 preserved original v1/evaluation bytes. CPU ZeRO subgroup comparisons
produced identical updates/moments and exact native resume; this did not establish
GPU memory feasibility. No other resource or mathematical change was made.

Run `20260928-124613-efd94997a1e8` started at 12:46:20 UTC and failed at
12:55:06 UTC on 2026-09-28. Environment and tiny GPU resume passed. The raw
parameter inventory records **1,235,814,400** unique trainable parameters. The
full frozen reference loaded and the first forward completed on the selected T4.
The first `engine.backward(terms.total)` then failed: another **124 MiB** was
requested with **20.81 MiB** free. This is a backward-memory failure, not the
previous optimizer-initialization failure.

Exactly one `FORWARD_COMPLETE_BEFORE_BACKWARD` record and an empty completed
microbatch log are preserved. Initial NPO loss **13.862943649291992**, retain loss
**0.08412517607212067**, total **13.94706916809082**. Candidate/reference NLLs match
at initialization; these finite values establish initial wiring only. They are
not a training trajectory, forgetting result or scientific-effectiveness metric.
No completed 1B backward/update, gradient norm, trained checkpoint or post-unlearning
evaluation can be reported.

Whole-attempt peak allocated **14,163,470,336 bytes**, reserved **14,526,971,904**,
sampled device-used **16,086,401,024**. The latter exceeds the 16e9 policy limit by
**86,401,024 bytes**, independently of the OOM failure. NVML samples every 0.1 s;
no continuous sub-interval peak is claimed.

Third complete archive SHA-256:
`0b872fee75c953fa0581d8a54ce9919c32a51f66c85b8ebdf830a8257644cf72`.
Raw inner archive SHA-256:
`fcd9a10525c22b6fb14fb0ba6861246b9e38dca01875fd2eaf7444d814bd1249`.
All **77** sealed members verify; **75** extracted non-checkpoint members are
preserved in Git and the two tiny random `.pt` files remain in complete archives.
Private archive destinations and download-verification receipts are recorded in
`../evidence/work_unit_d1/20260928_kaggle_attempt3/`: Kaggle dataset
`muhammadii/fq-ssiuu-d1-20260928-efd94997a1e8`, and HF dataset
`Xafire/fq-ssiuu-run-20260928-124613-efd94997a1e8-complete` at
`1ca20644a8d597700a71cd0c00ab362e61b05826`. Both downloaded archives match.
Third-attempt manifest:
`../evidence/work_unit_d1/MANIFEST_20260928_ATTEMPT3.json`.

## Stop for researcher audit

All **3/3** authorized submissions are used. The researcher requested that this
failure be diagnosed and its outcomes committed/pushed, with **no extra work**.
No new repair, resource proposal, experiment, fourth submission or merge follows.
PR #11 stays open. Exact next action: await researcher audit and instructions.

D1 has not passed. RG-006 remains scoped to the accepted objective/data/reference
boundary; RG-005 and RG-001/002/003 remain blocked. Protocol v2 is deferred and
non-executable; attack/durability decisions remain null. No forgetting, recovery,
durability or effectiveness claim is supported. Raw historical evidence and the
four DOCX files remain unchanged.
