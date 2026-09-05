# Master execution plan

Research: FQ-SSIUU: Verifiable Machine Unlearning Against Quantisation Recovery
Attacks in Large Language Models. Muhammad Arslan (23BSAI020), Rohan Ahmed
(23BSAI008), supervisor Dr. Sania Bhatti; BS AI, MUET, 2026–27.

This is the canonical execution plan. The research establishes only empirical
durability within the frozen protocol. Historical pilots remain feasibility
evidence. Published history and raw results are immutable.

## Active execution contract

- Starting branch: `fix/kaggle-overlay-evidence-integrity`, starting commit
  `0bbeaeb560acd5b9357ddfc25f18246d103b379b`; remote refs retrieved through the
  connected GitHub integration after credential-free HTTPS fetch could not
  authenticate. Local and remote tips agree; tracked state clean; merge base
  `e395f7af264eaaac9473d8b9e9f1c7a46c825d5c`; 20 ahead, zero behind `dev`.
- Canonical compute: one selected Tesla T4, `CUDA_VISIBLE_DEVICES=0`, FP16,
  peak selected-device memory below 16,000,000,000 bytes. No sharding.
- Immediate OG-001 work unit: at most two new private Kaggle submissions.
  The earlier version-2 attempt is historical evidence, not a submission in
  this work unit. Diagnose and preserve each attempt before any second run.
- Private parentless source snapshots only; scan committed trees and verify
  tree/bundle/manifest identity. Exclude credentials, untracked files and DOCX.
- Never install unrelated native-image conflict fixes to make global pip check
  clean. Accept inherited diagnostics only as `BASELINE_DIRTY_NO_REGRESSION`,
  with no introduced or overlay-owned unsatisfied dependencies and every
  substantive runtime and integrity check passing.
- Continue safe unblocked work. Stop for a real external access/quota blocker,
  a researcher/supervisor scientific decision, or two diagnosed executions
  leaving an unresolved blocker. Public submission and protocol approval remain
  researcher decisions.

## Workstreams and acceptance

| Workstream | Dependency | Falsifiable acceptance criteria | Status | Supporting commit | Evidence | Exact next action | Blocking user decision |
|---|---|---|---|---|---|---|---|
| OG-001 environment closure | Verified starting source | Raw pre/post pip checks; no introduced/overlay diagnostic; resolver policy and unchanged protected stack; required imports, CUDA FP16, SDPA, genuine NF4; one T4; peak memory; verified before/after manifests, verifier record and overlay lock; no scientific run | IN PROGRESS | `0bbeaeb5` (prior partial evidence) | `handbook/evidence/og001_kaggle/20260905_kernel_v2_24b25b5f/` | Implement and test shared differential consistency, then bounded Kaggle validation | None |
| Integration into dev | OG-001 PASS | Complete non-model suite in validated Kaggle runtime; all notebook Python cells compile; secret scan; full dev diff independently reviewed; evidence retained; no operational defect, DOCX or credentials; scientific gates honest | WAITING | — | — | Validate integration suite in closure runtime after OG-001 checks pass, then review PR | None if criteria pass |
| RG-003 faithful QUAIL | Environment/integration | Primary-paper conformance specification; independently verified forget/retain/hinge terms, coefficients, margin, token masks and gradients; deterministic integration; corrected P4 | BLOCKED | Existing gate record | `handbook/research_gates.yaml` | Obtain authoritative paper/version and equations on successor branch | Any unresolved scientific adaptation |
| RG-002 deployment-aligned FQ-QAT | Environment/integration | Weight NF4 conformance, codebook, differentiable approximation and LoRA treatment explicit; gradient/location/disabled-path/reference tests; corrected P3 | BLOCKED | Existing gate record | `handbook/research_gates.yaml` | Reconcile primary sources and deployment weight transform after QUAIL work | Any unresolved scientific adaptation |
| RG-001 faithful SSIUU | Environment/integration | Parameter attribution and prior-step state per primary algorithm; current attribution differentiable; previous detached; first-step zero; index selection/HVP reference/gradient tests; tiny GATE-2 then 1B feasibility within memory | BLOCKED | Existing gate record | `handbook/research_gates.yaml` | Obtain algorithm and derive conformance specification after quantisation work | Any unresolved scientific adaptation |
| Corrected pilots | OG-001 and applicable research gates | New namespaces; P1–P4 and baseline reproducibility; finite real metrics, correct gradients, memory, resume/artifact integrity; feasibility claims only | WAITING | — | — | Execute new corrected pilot configs after gate criteria pass | None within accepted methods |
| Protocol and preregistration | Corrected feasibility pilots | Frozen hypotheses, conditions, revisions/splits, ≥3 seeds, hyperparameters, metrics/directions, thresholds, exclusions, stopping/statistics/multiplicity/missing-run rules and compute/storage budget; KS p-value interpretation; supervisor package | WAITING | — | — | Prepare OSF draft and sign-off package from accepted methods/pilots | Muhammad Arslan approval before submission/freeze tag |
| Full multi-seed experiments | Approved frozen protocol/tag | Oracle then baselines, faithful QUAIL, FQ-QAT+NPO, SSIUU+NPO, full method and specified comparator; applicable FP16/INT8/NF4 QRA; immutable per-condition configs, complete run provenance, checkpoint/resume, hashes, VRAM and artifact synchronization | WAITING | — | — | Execute declared dependency order after freeze | Protocol approval |
| Statistical analysis | Run completeness audit | Every seed and failed/anomalous run accounted for; registered exclusions/tests; paired analysis where appropriate; mean/SD/CI/effect sizes; quantisation-specific recovery; machine-generated tables/figures | WAITING | — | — | Validate machine-readable run completeness before aggregation | Only changes to registered analysis |
| Thesis maintenance | Consequential verified milestones | Ledger linked to raw run IDs; Chapters 1–3 reflect actual method; Chapters 4–5 use verified results/limits; no manually copied unsupported metrics | ACTIVE | `0bbeaeb5` | `handbook/THESIS_UPDATE_LEDGER.md` | Update ledger at operational closure or consequential blocker | Four existing DOCX files remain untouched |
| Reproducibility/publication | Accepted analysis and manuscript | Empirical Durability Report, reproduction instructions, artifact manifest, licenses/model/data cards, limitations/ethics, manuscript and release package; no certification/universal/legal claims | WAITING | — | — | Assemble from immutable code/config/data/results provenance | Explicit approval before any public submission |

## Evidence rule

Use raw artifacts first, machine manifests/hashes second, recorded code/config
third, derived verdict fourth and narrative last. Every scientific/pilot run
records origin/tree/notebook identity, run/condition/seed, model/data revisions,
environment and lock, GPU/peak memory, config, logs, checkpoints and output
hashes, termination and supported/unsupported claims. Corrections supersede
records explicitly; they never overwrite raw evidence.
