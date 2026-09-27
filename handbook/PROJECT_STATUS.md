# FQ-SSIUU project status

Updated 2026-09-27. [Canonical plan](MASTER_EXECUTION_PLAN.md).
Integration tip: `dev` at `5ca92bfa0863a9ffadcd70bfc5f28c8ee8c6e76a` (PR #10 merged).
Work units A, B and C are merged; their branches and evidence are retained.
Active D1: `repro/npo-1b-source-faithful`, from that exact remote dev tip.
CPU source pairing, accumulation, native ZeRO-3 resume and the complete 62-step
tiny CPU loop pass. The D1 runner/evaluator and closed Mother recipe are implemented
for review; accumulation 8 is researcher-approved. No Kaggle submission or 1B training.
C1 validates sequential FP16/LLM.int8/NF4 loading and actual descriptive metrics.
C2 project choices are deferred until constituent 1B reproductions and bounded
pilots are complete. Source-faithful reproductions and descriptive collection do
not require Protocol v2 activation. D1 has run tiny CPU optimizer/resume tests but no 1B training/pilot. The previously validated environment
change is
bitsandbytes 0.46.1, now validated for matched inference; 0.45.5 evidence remains immutable.
RG-006 remains RESOLVED only for objective/data/frozen-reference conformance;
RG-005 and method gates RG-001/002/003 remain BLOCKED.
GitHub default is dev; its active rule requires passing CI, one fresh approval,
and resolved review threads. On the researcher's request, the additional last-push
approval restriction was disabled; other current protections were preserved. A substantive
human scientific review is required in addition to automated checks.

## Work unit C progress — 2026-09-24

[PR #10](https://github.com/ibn-e-Muhammad/fq-ssiuu/pull/10) was researcher-reviewed
and merged at `5ca92bfa`; its feature head was `29d21a51`.
C1 **PASS within matched inference scope**, run `20260924-120549-b4c4f06e097d`,
code `14669f34`, Mother v4. All three actual loaders, forwards, greedy generation,
112 INT8/NF4 projections, 48 observations and the fixed FP16 oracle validated.
84 sealed members verified; peak reserved memory 2,801,795,072 bytes on one T4.
Environment remains BASELINE_DIRTY_NO_REGRESSION; six unchanged native diagnostics.
New `requirements/kaggle-overlay.validated-0461.lock` copies the successful raw
lock; the old lock is unchanged. See [derived verdict](evidence/work_unit_c/20260924_attempt2_14669f3/VERDICT.json).

Both authorized submissions used. First failed its same-process model-release
check after FP16; preserved unchanged. Sequential child-process repair passed
attempt 2. CPU/CI 62: 294 passed, 4 unchanged skips, 10 deselected, 1 historical
xfail. No training or scientific experiment ran. Packed-state bytes were hashed
in the GPU runtime; no quantized checkpoint was saved for later byte rehashing.

C1 used one example per dataset component; its one-versus-one KS result is not
statistically inferential. The zero harmonic utility aggregate comes from a
measured zero world-fact component, not general utility failure.

[Protocol v2](methods/QRA_PROTOCOL_v2.md) is
**DEFERRED_UNTIL_BASE_METHOD_PILOTS / PROPOSED_NOT_APPROVED**, non-executable.
[Reproduction policy](methods/METHOD_REPRODUCTION_POLICY.md) separates upstream
requirements from project proposals. [Work-unit report](reports/WORK_UNIT_C_FINAL_REPORT.md).

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
| RG-005 QRA outcomes | BLOCKED | Blocks confirmatory QRA verdicts/full FQ-SSIUU study; permits source-faithful constituent reproductions and descriptive collection |
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

Review the implemented D1 runner and source evaluator, then perform the first
bounded T4 submission after review passes. The prepared recipe preserves full
NPO+RT, the frozen full reference, paged 32-bit AdamW, effective batch 32 and 62
source steps. Both submissions remain unused. D1 is not accepted from CPU checks
alone; see [D1 report](reports/WORK_UNIT_D1_REPORT.md) and
[source/adaptation specification](methods/NPO_D1_SOURCE_ADAPTATION.md).
No Protocol v2 approval is needed for source-faithful reproduction. No effectiveness
claim follows from C1, and no further C1 execution is authorized.
See [method matrix](METHOD_IMPLEMENTATION_MATRIX.md) and [scientific gates](research_gates.yaml).

Tracked thesis prose is still absent; consequences are in the [thesis ledger](THESIS_UPDATE_LEDGER.md).
Four untracked DOCX files, historical branches and raw results remain untouched.
