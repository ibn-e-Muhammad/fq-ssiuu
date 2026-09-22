# Work Unit B — final report

**DONE within the requested implementation and bounded validation scope.**
No training, corrected pilot or hypothesis experiment was conducted. No recovery
or durability decision is approved. PR [#9](https://github.com/ibn-e-Muhammad/fq-ssiuu/pull/9)
targets dev and remains unmerged.

## Lineage and changes

Starting dev and branch base: `fb9882790082ac4640d04a87c43260c62566831d`.
Starting and ending working branch: `fix/npo-adapter-qra-metrics`.
Implementation and runtime code: `318e523ac302f9f80f2c74f7c712bff08ff96a8e`.
The second commit contains this closure report, raw evidence, gate updates and
legacy token-loader docstring corrections. Resolve its exact SHA with
`git log -1 -- handbook/reports/WORK_UNIT_B_FINAL_REPORT.md`.
Two coherent commits; no merge, amend, rebase, force push or deletion.

Exact per-commit file lists: [COMMIT_FILES.json](../evidence/work_unit_b/20260922_attempt1_318e523a/COMMIT_FILES.json).
Code is separated in NPO/QRA reproduction folders; reusable identities,
statistics and report contracts are in `src/fq_ssiuu`. Historical configs retain
their original bytes under `configs/historical/august_pilots`; live tombstones and
normal loaders reject execution. Four untracked DOCX files remain untouched.

## Immutable sources

| Input | Identity |
|---|---|
| NPO author | `1207bbe9a1abf502bc2d7b43a0a467d9524b4036` |
| OpenUnlearning | `4ad738aaf60f6a4385f6e2506d01da99e76c31f3` |
| Full target and tokenizer | `open-unlearning/tofu_Llama-3.2-1B-Instruct_full@88e31200b97e4c0c04ae0d2f0b591f427046d192` |
| Target weight SHA256 | `35b0cd9f7db2735538ff78b4879fab61b23b7ca316df27b059585bf656700a8d` |
| Retain95 diagnostic oracle | `open-unlearning/tofu_Llama-3.2-1B-Instruct_retain95@f05bc9fd6c8d59decf7599f5e186153081061fd2` |
| Oracle weight SHA256 | `955e430aac277d9ccc309307f8462ce462c81fd4308cd934a6c24e5518077e94` |
| TOFU | `locuslab/TOFU@324592d84ae4f482ac7249b9285c2ecdb53e3a68`; forget05/retain95 |

All upstream code-file and tokenizer/dataset hashes:
[inputs_v1.json](../../experiments/reproductions/NPO/inputs_v1.json).
Actual pinned upstream function bodies executed; they were not reimplemented as
purported parity or redistributed as project-authored code. Primary references,
the paper/upstream discrepancy and explicit adaptations are in the
[conformance specification](../methods/WORK_UNIT_B_CONFORMANCE.md).

## Adapter acceptance

| Criterion | Outcome |
|---|---|
| Immutable real tokenizer/data; deterministic IDs/order | PASS, four CPU rows and four runtime rows |
| Prompt masking, answer/EOS targets, causal-loss boundary | PASS |
| Left/right padding, real EOS distinct from physical padding | PASS |
| Truncation, fully removed/empty answer, zero targets | PASS, negative tests |
| Source/spec/tensor/order hashes | PASS |
| Actual upstream output tensors | PASS for selected OU chat preprocessing; compatibility wrapper preserves its former token-list return |
| Original author preprocessing | Executed and differences preserved; no false parity claim |

## Reference acceptance

| Criterion | Outcome |
|---|---|
| Exact original fine-tuned checkpoint/config bytes | PASS |
| Safe default independent of quantized ModelConfig | PASS; legacy unpinned reference API rejects callers |
| Quantized/trainable/train-mode/wrong-revision rejection | PASS, negative tests |
| FP16, eval, frozen, one selected T4, no sharding | PASS, bounded runtime |
| Candidate/reference byte-identical inputs and NLL | PASS, all four records |
| no_grad, no output graph, no optimizer/backward | PASS |
| Reference cache | Not implemented; no unkeyed cache exists |

## Measurement and provenance acceptance

| Metric | Definition and direction |
|---|---|
| Answer log probability | Negative mean scored answer/EOS NLL; higher means greater answer knowledge |
| Generated ROUGE-L recall | Stemmed rouge-score recall; higher means more recoverable answer text |
| Generated exact match | NFKC/casefold/whitespace normalization, punctuation retained; higher means more recoverability |
| Paper Truth Ratio | Arithmetic mean perturbed probability / paraphrase probability; diagnostic |
| Upstream Truth Ratio | Geometric variant with upstream epsilon; separate diagnostic |
| KS D / p | Raw-ratio two-sample diagnostics; never recovery effects or equivalence evidence |
| Utility | Named probability/ROUGE/TR components for three datasets; nine-component harmonic mean; higher is better |
| Precision differences | Paired target-minus-FP16 differences and seeded paired-bootstrap intervals; descriptive only |

| Provenance field | Enforcement |
|---|---|
| Run/report/comparison group | Must match one hashed manifest |
| Metric definition/version | Registered definition content hash; arbitrary ID rejected |
| Source checkpoint, model/tokenizer revisions | Required immutable identities |
| Quantized artifact, family/version/config/precision | Must match distinct declared precision slots; labels alone rejected |
| Dataset/split, ordered samples, source rows/tensors | Hash-bound; mismatches rejected |
| Preprocessing/decoding/evaluator | Required hashes, decoding only where applicable |
| Diagnostic oracle | Explicit reference and evaluation hashes/policy |
| Legacy records | Preserved historical payload plus supersession; no silent conversion |

Both scientific decision fields remain null. The separate
[PROPOSED_NOT_APPROVED decision document](../methods/QRA_DECISION_RULE_PROPOSED.md)
is non-executable and introduces no thresholds.

## Verification and runtime

- Full CPU suite: **269 passed, 4 skipped, 10 deselected, 1 xfailed**, return code 0.
  Skips are existing FADE placeholders; xfail is historical SSIUU; no core B skip.
- GitHub CI 58: same totals, success on code `318e523a`.
- Real tokenizer/upstream validation: four rows, exact selected tensor agreement;
  positive/negative fixtures and independent expected values pass.
- Lint/format: return code 0. Notebook validation: 11 at code submission, 12 after
  preserving the submitted notebook as evidence; all JSON/nbformat/cells pass.
- Raw logs contain original trailing whitespace; the authored-file whitespace
  check passes. Raw bytes were not normalized to silence that check.
- Kaggle: **1/2 submissions used**, first attempt passed, Mother kernel version 2,
  run `20260922-101703-ce77656bee84`, child return code 0, 122.875 seconds.
- One Tesla T4, CUDA_VISIBLE_DEVICES=0; peak allocation **5,120,751,616 bytes**,
  peak reservation **5,249,171,456 bytes**. These are forward-validation peaks.
- Four fixed rows exercise NPO/reference, generation, metric and oracle execution.
  One forget row exercises KS plumbing only; no statistical conclusion follows.
- FP16 only. INT8/NF4 model evaluations were not performed. No second submission
  was needed or authorized merely to extend the scope of a successful first run.

## Evidence, gates and claim limits

Runtime [manifest](../evidence/work_unit_b/20260922_attempt1_318e523a/MANIFEST.json)
SHA256: `1a268ece1aa52187cb16fccfec5d45af1afdb660c9957250d9a9eb526623ee29`.
[CPU evidence](../evidence/work_unit_b/20260922_cpu/VERDICT.json) and
[derived runtime verdict](../evidence/work_unit_b/20260922_attempt1_318e523a/VERDICT.json)
are separate from raw output. All 59 sealed runtime members were verified and
measurements independently recomputed. Complete archives contain 68 verified members
and are stored privately in Kaggle and Hugging Face; immutable receipts are included.
The first archive-title submission was too long; its error and corrected successful
creation are preserved. It was not a second kernel execution.

RG-006: **RESOLVED**, limited to objective/data/reference conformance.
RG-005 and RG-001/002/003: **BLOCKED**. RG-004 remains OPEN. OG-001 remains PASS
under BASELINE_DIRTY_NO_REGRESSION; the environment was not redesigned.

Supported: tested adapter semantics, frozen reference safety, descriptive metric
calculations/provenance, and four-example FP16 execution. Unsupported: NPO training
reproduction, quantized-model evaluation, recovery/durability conclusions, corrected
pilot feasibility, training memory, scientific effectiveness or hypothesis tests.
The [thesis packet](../thesis_updates/WORK_UNIT_B.md) concerns Chapters 2–3 only;
no new scientific findings enter Results.

First unresolved blocker: matched quantized-model evaluation and RG-005 decision
approval. Transformers 5.0.0 declares bitsandbytes >=0.46.1 for its quantized-model
loader, while the frozen overlay is 0.45.5; low-level NF4 smoke evidence remains
valid but does not prove that loader works. Default-branch administration, working
thesis migration and historical placeholders also remain open and are not hidden
by this bounded acceptance.

**Exactly one next work unit:** directional QRA precision integration and decision-
protocol review—resolve the specific loader mismatch, validate matched FP16/INT8/NF4
descriptive records, and obtain researcher review before any corrected pilot.
