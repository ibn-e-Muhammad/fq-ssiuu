# Master execution plan

Updated 2026-09-10. FQ-SSIUU: Verifiable Machine Unlearning Against Quantisation
Recovery Attacks in Large Language Models. Muhammad Arslan (23BSAI020), Rohan Ahmed
(23BSAI008), supervisor Dr. Sania Bhatti; BS AI, MUET, 2026–27.

## Current execution contract

The research aims to test empirical durability under a defined protocol at about
1B parameters. Reproduce individual methods first, then combine them. Historical
pilots establish only the behavior of their historical implementations. Full
hypothesis testing has not begun.

Active branch: `fix/upstream-method-reconciliation`, from dev `d310815c` (PR #7).
Remote dev and reconciliation tips were rechecked on 2026-09-10. Main, published
history, raw evidence and the four untracked DOCX files remain untouched.

Environment and Mother installation are frozen at accepted operational evidence.
The current work unit is repository organisation plus scientific reconciliation A:
QRA/report contract and NPO parity, **CPU only; zero Kaggle/model submissions**.
An earlier proposed tiny-NPO GPU run is superseded and was never submitted.

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
| Method code organisation | Current dev | Separate candidates/historical methods; one first notebook; imports/CLI/tests compile; raw paths unchanged | PASS within layout/CPU scope | Containing commit | `experiments/LAYOUT.json`; `handbook/evidence/scientific_reconciliation/20260910_contract_and_npo/` | Use published layout at `0715877`; PR #8 open, CI 55 passed | None |
| RG-005 QRA/report contract | RG-004 interpretation | No p-value recovery score; matched provenance; nullable independent decisions; real metric definitions and approved decision rule before verdicts | BLOCKED; descriptive contract implemented | Containing commit | `handbook/methods/QRA_OUTCOME_CONTRACT.md`; `tests/methods/QRA/` | Validate concrete evaluator metrics and draft directional decision rule | Researcher approval before thresholds/protocol freeze |
| RG-006 NPO parity | Pinned author and OpenUnlearning | Causal sequence loss/current-retain gradients; answer/EOS adapter; frozen unquantised reference; explicit variants/adaptations | BLOCKED; CPU objective parity passes | Containing commit | `handbook/methods/NPO_CONFORMANCE.md`; reconciliation evidence | Implement/review data and reference adapter with deterministic CPU fixtures | No new scientific choice needed for upstream parity |
| RG-003 QUAIL | Individual upstream specification | Complete forget/retain/all-logit hinge; coefficients/margin/masks/gradient parity; corrected P4 | BLOCKED | Historical source retained | `experiments/reproductions/QUAIL/`; upstream register | Reconcile pinned author code with paper equations/update order after work unit A | Any unresolved published ambiguity |
| RG-002 deployment FQ-QAT | Quantiser specification | Merged-weight/blockwise NF4 semantics, STE/LoRA treatment, disabled-path and gradient tests; corrected P3 | BLOCKED | Historical source retained | `experiments/reproductions/FQ_QAT/` | Specify and compare training/deployment transformations after QUAIL | Explicit novel-method adaptation review |
| RG-001 SSIUU | Parameter attribution specification | Current differentiable/previous detached; first step zero; index selection; HVP reference; tiny then 1B resource evidence | BLOCKED | Historical source retained | `experiments/reproductions/SSIUU/` | Derive primary algorithm and tiny reference, no full Hessian | Any unresolved scientific adaptation |
| Corrected pilots | RG-005 and relevant method acceptance | New run IDs; baseline and P1–P4; finite real outputs, gradients, VRAM, resume and storage | WAITING | — | New immutable namespaces required | Execute only after valid method/evaluation entry criteria; pilot evidence completes relevant method gates | None within accepted definitions |
| Protocol/preregistration | Corrected feasibility | Frozen hypotheses, conditions, revisions/splits, ≥3 seeds, hyperparameters, checkpoints, metrics/directions, thresholds, exclusions, stopping/statistics/multiplicity/missing-run rules, budget | WAITING | — | OSF draft and supervisor package pending | Prepare and obtain approval, then tag code/config | Muhammad Arslan approval before submission/freeze |
| Full experiments | Approved protocol/tag | Oracle; core baselines; QUAIL; FQ+NPO; SSIUU+NPO; combined method; specified comparator; precision evaluations; immutable configs and complete run manifests | WAITING | — | — | Execute frozen dependency order with checkpoint/resume/synchronization | Protocol approval |
| Analysis | Completeness audit | Every seed/failure accounted for; registered tests, paired comparisons, effect sizes and uncertainty; machine-generated tables/figures | WAITING | — | — | Validate runs before aggregation; distinguish confirmatory/exploratory | Approval for changes to registered analysis |
| Thesis | Consequential milestones | Chapters 1–3 reflect actual methods; 4–5 use linked results; no manual unsupported metrics | ACTIVE ledger; working DOCX untracked | Containing commit | `handbook/THESIS_UPDATE_LEDGER.md` | Apply accepted corrections to a versioned thesis workflow without touching protected DOCX | Resolve authoritative thesis source before editing it |
| Reproducibility/publication | Accepted analysis | Empirical report, licensing, clean public snapshot, artifacts, reproduction guide, manuscript/limitations/ethics | WAITING | — | — | Prepare release package; retain private history during research | Explicit public submission approval |

## Outstanding pre-pilot engineering

After work unit A: verify evaluator sharded-checkpoint handling and raw pilot
retrieval; replace remaining scientific placeholders; validate real PEFT wrappers;
implement training checkpoint/resume and interval archival. Drive flattening/size-only
verification remains historical and is not the canonical Kaggle/HF path. Repository
default branch/protection hardening and version-controlled thesis source are still
open. Do not disguise these as environment-installation work.

## Evidence and progression

Raw artifacts → manifests/hashes → recorded code/config → derived verdict → narrative.
Preserve failure, null and anomalous outputs. Every model run needs source/tree,
notebook hash, run/condition/seed, input revisions, environment/lock, GPU/peak VRAM,
config, stdout/stderr, checkpoints/output hashes and termination/claim limits.
Commit/push coherent changes, review the PR and required acceptance before a
non-force dev merge; never modify main. Proceed through safe work, stop for real
missing access/quota, unresolved scientific approval or two diagnosed executions
with an unresolved blocker. Documentation supports decisions, not a new stage per error.
