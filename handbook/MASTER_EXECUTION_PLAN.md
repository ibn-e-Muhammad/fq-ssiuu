# Master execution plan

Updated 2026-09-27. FQ-SSIUU: Verifiable Machine Unlearning Against Quantisation
Recovery Attacks in Large Language Models. Muhammad Arslan (23BSAI020), Rohan Ahmed
(23BSAI008), supervisor Dr. Sania Bhatti; BS AI, MUET, 2026–27.

## Current execution contract

The research aims to test empirical durability under a defined protocol at about
1B parameters. Reproduce individual methods first, then combine them. Historical
pilots establish only the behavior of their historical implementations. Full
hypothesis testing has not begun.

Active branch: `repro/npo-1b-source-faithful`, from remote dev
`5ca92bfa0863a9ffadcd70bfc5f28c8ee8c6e76a` (PR #10 researcher-reviewed and merged).
Prior A/B/C branches are superseded and retained. D1 is NPO-only; no runtime
submission yet. Its category-3 accumulation change to 8 is researcher-approved, not yet runtime-validated.
Researcher scope correction: source-faithful constituent reproductions and
descriptive measurements may proceed independently of deferred Protocol v2;
project decisions/full study still require final confirmatory-protocol approval.
Preserve old overlays and evidence. C1 accepted bitsandbytes 0.46.1 for inference.
D1 adds only a pinned DeepSpeed/hjson candidate for the source training path,
subject to protected-stack resolution, differential consistency and GPU acceptance.

Canonical future compute remains one Tesla T4, CUDA_VISIBLE_DEVICES=0, FP16 and
peak selected-device VRAM below 16,000,000,000 bytes; no sharding or second GPU.
Each future Kaggle work unit has at most two submissions unless newly authorised.
Source transport stays private and parentless; credentials never enter source or
artifacts. Keep one Mother notebook and versioned per-method recipes, then seal
and hash-verify outputs in private Kaggle and Hugging Face stores.

## Workstreams and acceptance

For new rows, “containing commit” means the commit containing this plan/evidence;
resolve with Git history instead of inventing a self-referential hash.

| Workstream | Dependency | Acceptance criteria | Status | Supporting commit | Evidence path | Exact next action | Blocking user decision |
|---|---|---|---|---|---|---|---|
| OG-001 | Private source linkage | Imports, FP16/SDPA/NF4, one T4, protected stack, differential pip, manifests/lock | PASS; frozen | `1efe2b32` / notebook `2be46116` | `handbook/evidence/og001_kaggle/20260906_kernel_v4_2be46116/` | Reuse; repair only demonstrated regressions | None |
| Mother integration/archive | OG-001 | Pinned source/recipe, non-model suite, notebook validation, lazy secrets, private downloaded hashes | PASS; frozen, 1/2 submissions used | `23682ea1`, evidence `c0eb92fd` | `handbook/evidence/mother_runner/20260908_kernel_v1_23682ea1/` | Use same runner for accepted methods | None |
| Operational dev integration | Accepted runtime/review/CI | Reviewed diff, preserved raw evidence, no secrets/DOCX/main changes | COMPLETE, PR #7 | `d310815c`; review fixes `291bbda5` | `handbook/evidence/integration/20260908_pr7/` | Preserve merged historical branches | None |
| Method code organisation | Current dev | Separate candidates/historical methods; one first notebook; imports/CLI/tests compile; raw paths unchanged | PASS within layout/CPU scope | Containing commit | `experiments/LAYOUT.json`; `handbook/evidence/scientific_reconciliation/20260910_contract_and_npo/` | Use published layout at `0715877`; PR #8 merged, dev `fb988279`, CI passed | None |
| Work unit B data/reference/metrics | Work unit A objective parity; OG-001 | Real pinned adapter/upstream tensor checks, immutable unquantized reference, directional metric/paired uncertainty behavior, fail-closed provenance, bounded FP16 runtime | PASS within bounded validation scope; 1/2 submissions used | Containing commit | `handbook/methods/WORK_UNIT_B_CONFORMANCE.md`; `handbook/evidence/work_unit_b/20260922_attempt1_318e523a/` | Merged PR #9; preserve bounded claims and evidence | Paper arithmetic Truth Ratio chosen by researcher; final decision rule remains unapproved |
| C1 matched precision | B boundary; OG-001 | Same pinned checkpoint sequential FP16/INT8/NF4; actual packed-state identities, forward/generation/metrics; differential package consistency; <16 GB one T4; no training | PASS within inference scope; 2/2 submissions used; PR #10 merged | `14669f34` | `handbook/evidence/work_unit_c/20260924_attempt2_14669f3/` | Preserve validated 0.46.1 lock and both attempts | None |
| D1 NPO 1B adapted reproduction | A/B NPO boundary; merged C; pinned primary source | Source conformance, full-parameter optimization, nondegenerate gradients, save/resume, source metrics, immutable identities, <16,000,000,000 bytes and matching private archives | IMPLEMENTED FOR REVIEW; full tiny CPU ZeRO-3 loop passes; 0/2 submissions; no 1B acceptance | Containing commit; new evidence manifest resolves exact code SHA | `handbook/methods/NPO_D1_SOURCE_ADAPTATION.md`; `handbook/evidence/work_unit_d1/20260927_runtime_readiness/` | Complete substantive review of the D1 PR, then first bounded T4 submission; no protocol thresholds | No new method decision; the requested pre-run review remains outstanding. Accumulation 8 already approved |
| C2 project protocol | Constituent 1B reproductions and bounded pilots; primary sources | Content-recomputed evaluations, bound oracle, derived nine-component utility, author-cluster/multi-seed uncertainty, versioned ex-ante rules | DEFERRED_UNTIL_BASE_METHOD_PILOTS / PROPOSED_NOT_APPROVED; decisions null | Containing commit | `handbook/methods/QRA_PROTOCOL_v2.md`; C final report | Keep non-executable; revisit only after constituent reproductions/pilots for final confirmatory approval | Final confirmatory-protocol approval later; no Protocol v2 approval needed for source-faithful reproduction |
| RG-005 QRA/report contract | RG-004 interpretation | No p-value recovery score; matched provenance; nullable independent decisions; real metric definitions and approved decision rule before verdicts | BLOCKED for confirmatory QRA verdicts/full study; source-faithful reproduction/descriptive collection allowed | Containing commit | `handbook/methods/QRA_OUTCOME_CONTRACT.md`; `tests/methods/QRA/` | Preserve null project decisions; apply reproduction policy without activating v2 | Researcher approval before thresholds/protocol freeze |
| RG-006 NPO parity | Pinned author and OpenUnlearning | Causal sequence loss/current-retain gradients; answer/EOS adapter; frozen unquantised reference; explicit variants/adaptations | RESOLVED for objective/data/reference boundary; no training reproduction | Containing commit | `handbook/methods/NPO_CONFORMANCE.md`; reconciliation evidence | Preserve accepted boundary; complete source-faithful training reproduction in a separate work unit | No new scientific choice needed for upstream parity |
| RG-003 QUAIL | Individual upstream specification | Complete forget/retain/all-logit hinge; coefficients/margin/masks/gradient parity; corrected P4 | BLOCKED | Historical source retained | `experiments/reproductions/QUAIL/`; upstream register | Reconcile pinned author code with paper equations/update order after work unit A | Any unresolved published ambiguity |
| RG-002 deployment FQ-QAT | Quantiser specification | Merged-weight/blockwise NF4 semantics, STE/LoRA treatment, disabled-path and gradient tests; corrected P3 | BLOCKED | Historical source retained | `experiments/reproductions/FQ_QAT/` | Specify and compare training/deployment transformations after QUAIL | Explicit novel-method adaptation review |
| RG-001 SSIUU | Parameter attribution specification | Current differentiable/previous detached; first step zero; index selection; HVP reference; tiny then 1B resource evidence | BLOCKED | Historical source retained | `experiments/reproductions/SSIUU/` | Derive primary algorithm and tiny reference, no full Hessian | Any unresolved scientific adaptation |
| Constituent reproductions / bounded pilots | Relevant upstream conformance and validated execution contract; Protocol v2 not required | New run IDs; baseline and P1–P4; finite real outputs, gradients, VRAM, resume and storage | WAITING | — | New immutable namespaces required | Use source-defined requirements; record 1B/data adaptations; no project attack/durability verdicts | None within accepted definitions |
| Protocol/preregistration | Corrected feasibility | Frozen hypotheses, conditions, revisions/splits, ≥3 seeds, hyperparameters, checkpoints, metrics/directions, thresholds, exclusions, stopping/statistics/multiplicity/missing-run rules, budget | WAITING | — | OSF draft and supervisor package pending | Prepare and obtain approval, then tag code/config | Muhammad Arslan approval before submission/freeze |
| Full experiments | Approved protocol/tag | Oracle; core baselines; QUAIL; FQ+NPO; SSIUU+NPO; combined method; specified comparator; precision evaluations; immutable configs and complete run manifests | WAITING | — | — | Execute frozen dependency order with checkpoint/resume/synchronization | Protocol approval |
| Analysis | Completeness audit | Every seed/failure accounted for; registered tests, paired comparisons, effect sizes and uncertainty; machine-generated tables/figures | WAITING | — | — | Validate runs before aggregation; distinguish confirmatory/exploratory | Approval for changes to registered analysis |
| Thesis | Consequential milestones | Chapters 1–3 reflect actual methods; 4–5 use linked results; no manual unsupported metrics | ACTIVE ledger; working DOCX untracked | Containing commit | `handbook/THESIS_UPDATE_LEDGER.md` | Apply accepted corrections to a versioned thesis workflow without touching protected DOCX | Resolve authoritative thesis source before editing it |
| Reproducibility/publication | Accepted analysis | Empirical report, licensing, clean public snapshot, artifacts, reproduction guide, manuscript/limitations/ethics | WAITING | — | — | Prepare release package; retain private history during research | Explicit public submission approval |

## Outstanding pre-pilot engineering

After work unit A: verify evaluator sharded-checkpoint handling and raw pilot
retrieval; replace remaining scientific placeholders; validate real PEFT wrappers;
implement training checkpoint/resume and interval archival. Drive flattening/size-only
verification remains historical and is not the canonical Kaggle/HF path. Repository default is dev and CI/reviewer protections are enforced; human
scientific review and version-controlled thesis source remain required. Do not disguise these as environment-installation work.

## Evidence and progression

Raw artifacts → manifests/hashes → recorded code/config → derived verdict → narrative.
Preserve failure, null and anomalous outputs. Every model run needs source/tree,
notebook hash, run/condition/seed, input revisions, environment/lock, GPU/peak VRAM,
config, stdout/stderr, checkpoints/output hashes and termination/claim limits.
Commit/push coherent changes, review the PR and required acceptance before a
non-force dev merge; never modify main. Proceed through safe work, stop for real
missing access/quota, unresolved scientific approval or two diagnosed executions
with an unresolved blocker. Documentation supports decisions, not a new stage per error.
