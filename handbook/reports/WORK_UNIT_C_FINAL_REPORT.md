# Work Unit C — matched precision integration and protocol review

Current interpretation, corrected 2026-09-25: **C1 PASS only within matched
inference and measurement-plumbing scope**. C2 is
**DEFERRED_UNTIL_BASE_METHOD_PILOTS / PROPOSED_NOT_APPROVED**. Source-faithful
constituent reproductions do not await activation of Protocol v2. The researcher
subsequently approved and merged PR #10 at `5ca92bfa` (2026-09-25). No training, corrected pilot or
scientific experiment ran. Attack and durability decisions remain null.

## Lineage and review

- Starting remote dev: `2334f321b3ccf08352c29b1842200863b757c9d1` (merged PR #9).
- Branch: `fix/matched-precision-qra-protocol`; [PR #10](https://github.com/ibn-e-Muhammad/fq-ssiuu/pull/10) merged; final head `29d21a51de784190bdabf4e7267db8925c73260e`.
- Implementation: `b884c5ef0a8bfe7812e583f88f73cb5c1eaf9a83`.
- Executed repair: `14669f34cbaefde8c4e3d7d7f89616a240a6a6c7`.
- Original evidence closure: `afb341b5ba1dd0e9c6de6f8a3842c5596afe967d`. The
  present documentation correction preserves that immutable snapshot. Its original file inventory is in
  `../evidence/work_unit_c/20260924_closure/FILE_INDEX.json`.
- Default branch is dev. Required CI, fresh reviewer approval, resolved threads
  and deletion/force-push protection remain enforced. The additional last-push
  approval restriction was removed at the researcher's request when D1 began.
- [Implementation review](../reviews/2026-09-24_work_unit_c_implementation_review.md)
  is explicitly the implementer's review, not an independent scientific approval.

## Acceptance and limits

| Criterion | Result | Evidence / limit |
|---|---|---|
| Exact same pinned source | PASS | Source weight hash and pinned model/tokenizer/dataset revisions; same four examples/tensors/decoding in each precision |
| Real Transformers loaders | PASS | FP16, bitsandbytes LLM.int8, NF4; 0.46.1 with Transformers 5.0.0 and torch 2.10.0+cu128 |
| Actual quantized state | PASS | 112 expected projections for each quantized model; packed parameter and scale/codebook tensor hashes, complete ordered state inventory |
| Forward / generation / measurements | PASS | Twelve precision/example records, 48 registered observations, three separately derived utility summaries |
| Reference / oracle | PASS within scope | Frozen unquantized FP16 retain95 diagnostic oracle; raw evidence bound and KS recomputed; distinct from the original-full NPO reference |
| Environment | PASS under BASELINE_DIRTY_NO_REGRESSION | Same six native diagnostics; no new conflict or protected drift; full manifests, reports and generated overlay lock |
| One selected T4 | PASS | CUDA_VISIBLE_DEVICES=0, one visible device, no sharding/offload; separate sequential workers reaped before the next |
| Memory | PASS for inference only | Largest reserved peak 2,801,795,072 bytes, below 16,000,000,000; no training-memory inference |
| Evaluation identities | PASS | Complete content hash recomputed; raw measurement/file linkage, definitions, oracle, run/group and artifact provenance checked |
| Utility summary | PASS | Nine registered components and separate harmonic-mean manifest; zero only for a measured zero component, missing data rejected |
| Clustered / multi-seed uncertainty | CPU implementation PASS; unapproved | Verified explicit author maps, whole-author resampling and crossed seed/author behavior; C1 has only one forget author and no training seeds |
| Scientific decisions | BLOCKED | PROPOSED_NOT_APPROVED protocol; no initial forgetting, recovery or durability result |

C1 used **one example per declared dataset component** (forget05_perturbed,
retain_perturbed, real_authors_perturbed, world_facts_perturbed). Each precision
and the oracle had one forget example: the one-versus-one KS result is **not
statistically inferential**. The measured `world_facts_truth_ratio_score` component
was zero in this bounded sample at all three precisions, giving a zero harmonic
utility aggregate. That is **not evidence of general utility failure**. Quantized
tensor bytes were hashed at runtime but not persisted. C1 cannot establish
forgetting, recovery or durability. No raw observation has been rewritten.

## Both bounded executions

| Attempt | Code / Mother version | Run | Outcome |
|---|---|---|---|
| 1 | `b884c5e` / v3 | `20260924-060442-26f863842b16` | Environment and FP16 completed; model remained referenced at same-process release check; INT8/NF4 not reached |
| 2 | `14669f34` / v4 | `20260924-120549-b4c4f06e097d` | All C1 criteria passed with sequential process isolation |

The repair addressed our runner's lifetime assumption. The owner of the retained
upstream reference is unknown; no library-defect claim is made. First-attempt raw
artifacts remain unchanged. **2/2 submissions used; no third submission authorized.**

| Phase | Peak allocated bytes | Peak reserved bytes |
|---|---:|---:|
| FP16 | 2,624,986,112 | 2,801,795,072 |
| INT8 | 2,349,203,456 | 2,394,947,584 |
| NF4 | 1,342,232,576 | 1,436,549,120 |
| FP16 retain95 oracle | 2,614,081,536 | 2,701,131,776 |

Values above derive from the immutable `validation_result.json`, not a training
measurement. No optimizer or backward call occurred.

## Sources and validation

The inherited target/tokenizer is
`open-unlearning/tofu_Llama-3.2-1B-Instruct_full@88e31200b97e4c0c04ae0d2f0b591f427046d192`;
weight SHA-256 `35b0cd9f7db2735538ff78b4879fab61b23b7ca316df27b059585bf656700a8d`.
Dataset `locuslab/TOFU@324592d84ae4f482ac7249b9285c2ecdb53e3a68`.
Retain95 oracle revision `f05bc9fd6c8d59decf7599f5e186153081061fd2`;
weight SHA-256 `955e430aac277d9ccc309307f8462ce462c81fd4308cd934a6c24e5518077e94`.
The source/file registry remains `experiments/reproductions/NPO/inputs_v1.json`.

Bitsandbytes 0.46.1 source commit:
`4bca84499ad194d6c37e77dfcf99201b81dc6981`; official source-file and wheel hashes
are in `../evidence/work_unit_c/20260923_sources/BITSANDBYTES_0461.json`.
The older lock remains unchanged. The new
`requirements/kaggle-overlay.validated-0461.lock` is a byte-identical copy of the
successful run's generated overlay, promoted only after independent acceptance.

Complete local CPU suite: **294 passed, 4 skipped, 10 deselected, 1 xfailed**, exit 0.
Lint/format and notebook checks exit 0. All new core C acceptance cases ran.
The skips are unchanged FADE placeholders; deselections require GPU/models; the
xfail is the known historical SSIUU defect. CI 62 at the executed commit also
passed with exactly those totals. Earlier implementation CI 61 passed 292 tests;
the repair added two process-lifecycle regressions. CPU tests download no model
or dataset and use no GPU/credentials. Exact commands/logs are in the CPU evidence
and `CI62.log`; these totals do not establish scientific-method effectiveness.

## Evidence, storage and reproducibility limits

Accepted run evidence:
`../evidence/work_unit_c/20260924_attempt2_14669f3/`.
Run-level `MANIFEST.json` SHA-256:
`d4e12bd6c5968b2e413bacf2a095f749a93a27608aa7d9012d06269f8b88194b`.
It binds 103 evidence files. The sealed runtime contains 84 verified members.
The complete private archive contains 100 verified members, SHA-256
`7d4ab6716496d56eafa31c0308d97becbf69bfd6cebaca6662bd96d43011c755`.
Later storage receipts are bound by the run-level manifest, outside that archive.

- Private Kaggle: `muhammadii/fq-ssiuu-c1-20260924-b4c4f06e097d`.
- Private Hugging Face: `Xafire/fq-ssiuu-run-20260924-120549-b4c4f06e097d-complete`,
  revision `a9f63fe2b89b270af814d0aec08a78a049a11c92`.
- Both archives were downloaded and checked against recorded content hashes.
- Failed first attempt: `../evidence/work_unit_c/20260924_attempt1_b884c5e/`;
  its failure verdict and private archives remain preserved.

Raw tensor bytes were hashed inside the recorded GPU process. This work unit did
not persist quantized checkpoint files, so the local derivation can rehash their
manifests, not the absent packed weights themselves. Source checkpoint downloads
were hash-checked at runtime. No claim of persistent quantized save/reload parity
or repeat-run bitwise equality is supported by a single successful execution.

## Protocol and gates

[Protocol v2](../methods/QRA_PROTOCOL_v2.md) is
**DEFERRED_UNTIL_BASE_METHOD_PILOTS / PROPOSED_NOT_APPROVED**, non-executable.
Its project choices do not govern source-faithful constituent reproductions.
Published equations, pinned source and source-defined metrics, splits, stopping
rules and hyperparameters have independent authority.
It preserves v1's numeric margins and seed choices. It completes the joint
four-comparison interval family and proposes utility acceptance conditional on
the exact finite benchmarks in every seed. V1 remains preserved; v2 was prepared
after inference integration but before any corrected training or pilot outcome.
This is not the full multi-condition preregistration and cannot authorize it.

OG-001 remains PASS with its original evidence. C1 is a separately recorded
accepted precision-loader extension. RG-006 retains only its earlier bounded
objective/data/reference acceptance. **RG-001, RG-002, RG-003 and RG-005 remain
BLOCKED**; RG-004 remains OPEN. RG-005 blocks confirmatory QRA attack/durability
verdicts and the full FQ-SSIUU study, not source-faithful constituent reproduction
or descriptive collection. No method gate is closed by this correction.

Supported: matched precision inference and measurement execution, source/content
linkage, bounded memory, differential environment preservation and private archival.
Unsupported: NPO training, adequate initial forgetting, QRA detection, durability,
confirmatory uncertainty, faithful QUAIL/FQ-QAT/SSIUU, combined FQ-SSIUU efficacy.

**Correction-unit stopping point:** push the documentation/gate-semantics changes
and report; do not merge, rerun C1 or begin training. A substantive human PR review
remains required. Future constituent-method reproduction may follow source-defined
requirements without Protocol v2 approval; its custom choices are deferred until
1B base-method reproductions/pilots precede the final confirmatory protocol. See
[reproduction policy](../methods/METHOD_REPRODUCTION_POLICY.md), including the
unchanged legacy Mother interface limitation.
Thesis consequences are in `../thesis_updates/WORK_UNIT_C.md`; DOCX files untouched.
