# FQ-SSIUU project status

Updated 2026-09-24. [Canonical plan](MASTER_EXECUTION_PLAN.md).
Integration tip: `dev` at `2334f321b3ccf08352c29b1842200863b757c9d1` (PR #9 merged).
Work units A and B are superseded and retained. Active work unit C:
`fix/matched-precision-qra-protocol`, branched from that exact remote dev tip.
C1 validates sequential FP16/LLM.int8/NF4 loading and actual descriptive metrics.
C2 prepares clustered uncertainty and a decision protocol for explicit approval.
No training or corrected pilot is authorized. The only environment change is
candidate bitsandbytes 0.46.1; accepted 0.45.5 evidence remains immutable.
RG-006 remains RESOLVED only for objective/data/frozen-reference conformance;
RG-005 and method gates RG-001/002/003 remain BLOCKED.
GitHub default is now dev; its active rule requires passing CI, one fresh approval
from someone other than the last pusher, and resolved review threads. A substantive
human scientific review is required in addition to automated checks.

## Work unit C progress — 2026-09-24

[PR #10](https://github.com/ibn-e-Muhammad/fq-ssiuu/pull/10), implementation `b884c5e`,
CI 61 passed. C1 attempt 1 (Mother v3, `20260924-060442-26f863842b16`) passed the
candidate environment and FP16 inference but failed its same-process model-release
check before INT8/NF4. Raw failure is preserved; no precision-integration pass is
claimed. Repair: one model per sequential process, fully reaped before the next.
CPU repair acceptance: 294 passed, 4 unchanged skips, 10 deselected, 1 historical
xfail. One authorized submission remains. No corrected pilot or training ran.

## Work unit B CPU acceptance — 2026-09-22

269 passed; four unchanged FADE skips, ten GPU/model exclusions and one historical
SSIUU expected failure. Four real TOFU rows match hash-pinned upstream preprocessing
with explicit Transformers return-format and EOS-attention adaptations. Eleven
notebooks compile. [Immutable CPU evidence](evidence/work_unit_b/20260922_cpu/VERDICT.json).
Bounded runtime passed in attempt 1 (Mother v2), run `20260922-101703-ce77656bee84`.
Four frozen FP16 forwards and actual metrics passed on one T4; peak allocated
5,120,751,616 bytes, reserved 5,249,171,456 bytes. No training/backward/optimizer.
Both private complete archives were downloaded and hash-verified (68 members).
[Runtime verdict](evidence/work_unit_b/20260922_attempt1_318e523a/VERDICT.json).

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
| RG-005 QRA outcomes | BLOCKED | Matched precision runtime and reviewed decision/uncertainty rules; all corrected pilots blocked |
| RG-006 NPO parity | RESOLVED | Objective plus pinned data/frozen FP16 reference boundary accepted; no training-reproduction claim |
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

Implement and CPU-test content-derived artifact/evaluation identities and the
bounded matched precision runner, then submit C1 at most twice. Prepare C2 protocol
for researcher/supervisor review; decisions remain null until explicitly approved.
Next scientific work after approval is corrected NPO training reproduction, then
faithful QUAIL, weight FQ-QAT, SSIUU and their combination. No effectiveness claim
follows from loader or forward-only validation.
See [method matrix](METHOD_IMPLEMENTATION_MATRIX.md) and [scientific gates](research_gates.yaml).

Tracked thesis prose is still absent; consequences are in the [thesis ledger](THESIS_UPDATE_LEDGER.md).
Four untracked DOCX files, historical branches and raw results remain untouched.
