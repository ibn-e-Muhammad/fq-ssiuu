# FQ-SSIUU project status

Updated 2026-09-22. [Canonical plan](MASTER_EXECUTION_PLAN.md).
Integration tip: `dev` at `fb9882790082ac4640d04a87c43260c62566831d`.
PR #8 is merged; `fix/upstream-method-reconciliation` is superseded and retained.
Active work unit B: `fix/npo-adapter-qra-metrics`, from that dev tip.
Environment and Mother installation remain frozen. No training or corrected pilot
is authorized by this work unit. CPU and bounded forward-only runtime acceptance
are recorded separately; RG-005/006 remain BLOCKED pending their own criteria.

## Work unit B CPU acceptance — 2026-09-22

269 passed; four unchanged FADE skips, ten GPU/model exclusions and one historical
SSIUU expected failure. Four real TOFU rows match hash-pinned upstream preprocessing
with explicit Transformers return-format and EOS-attention adaptations. Eleven
notebooks compile. [Immutable CPU evidence](evidence/work_unit_b/20260922_cpu/VERDICT.json).
Bounded runtime pending; no gate closure or scientific conclusion follows.

## Current outcome

Method code is separated under `experiments/reproductions/NPO`, `QUAIL`, `SSIUU`,
`FQ_QAT` and `QRA`; old surrogates are explicitly historical. The single current
notebook is `notebooks/kaggle/00_MOTHER_RUNNER.ipynb`. Accepted reusable components
belong in `src/fq_ssiuu`; no training method is yet accepted for a corrected 1B pilot.

Scientific reconciliation A implements a descriptive QRA/report contract and
causal sequence-sum NPO candidate. Actual hash-verified functions from the NPO
author repository and pinned OpenUnlearning are compared on CPU for objective
values, current/retain gradients and reference detachment. This establishes
objective behavior only, not data-adapter or model-training parity.

Local acceptance: **203 passed**, four historical FADE skips, ten GPU/model
deselections and one known SSIUU expected failure; 16 dual-upstream parity cases
and 11 notebooks passed. [Evidence and claim limits](evidence/scientific_reconciliation/20260910_contract_and_npo/VERDICT.json).

The old p-value recovery score and inverted certification are retired. Reports
keep missing observations null and cannot issue attack/durability decisions.
[CR-012](corrections/CR-012_qra_outcome_validity.md) supersedes historical P1/P4
headline conclusions without editing raw artifacts. No full hypothesis experiment
or corrected pilot has run. No Kaggle/model submission belongs to work unit A.

## Gates and evidence

| Gate | Status | Remaining acceptance |
|---|---|---|
| OG-001 | PASS | Frozen [kernel v4 closure](evidence/og001_kaggle/20260906_kernel_v4_2be46116/VERDICT.json) |
| Mother integration/archive | PASS | Frozen [Mother v1](evidence/mother_runner/20260908_kernel_v1_23682ea1/VERDICT.json); 1/2 submissions used |
| RG-005 QRA outcomes | BLOCKED | Actual metric implementations and approved directional decision/uncertainty rules; all corrected pilots blocked |
| RG-006 NPO parity | BLOCKED | Data/caller masking, EOS, original unquantised frozen reference and versioned 1B adapter |
| RG-003 QUAIL | BLOCKED | Complete objective/coefficients/margin/all-logit semantics and corrected P4 |
| RG-002 weight NF4 | BLOCKED | Deployment-aligned weight transformation, gradients and corrected P3 |
| RG-001 SSIUU | BLOCKED | Parameter attribution/state/HVP, gradients and training memory, corrected P2 |
| RG-004 TOFU interpretation | OPEN | KS p-value interpretation throughout current claims; historical corrections remain explicit |

The environment is BASELINE_DIRTY_NO_REGRESSION: six native diagnostics, no new
conflict, all substantive runtime checks pass. Mother run
`20260908-041835-2ef818513f23` at source `23682ea1` passed 144 tests with 12 then
placeholder skips, 8 deselections and one known SSIUU expected failure. Ten notebooks
compiled; private Kaggle/HF downloaded archives were hash-verified. This is operational
evidence, not model-level scientific validation or training-memory evidence.

## Exact next action

Complete work unit B: NPO adapter/reference and descriptive directional QRA
metrics, then at most two bounded method-validation submissions.
Proceed through QUAIL, weight NF4 and SSIUU before corrected pilots. Resolve and
approve the final protocol only after feasibility; no outcome-dependent adjustments.
See [method matrix](METHOD_IMPLEMENTATION_MATRIX.md) and [scientific gates](research_gates.yaml).

Tracked thesis prose is still absent; consequences are in the [thesis ledger](THESIS_UPDATE_LEDGER.md).
Four untracked DOCX files, historical branches and raw results remain untouched.
