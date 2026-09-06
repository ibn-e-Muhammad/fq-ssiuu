# FQ-SSIUU project status

Updated 2026-09-06. Canonical plan: [MASTER_EXECUTION_PLAN.md](MASTER_EXECUTION_PLAN.md).
Active branch: `fix/kaggle-overlay-evidence-integrity`; not merged into `dev`.

## Current decision

OG-001 is **PARTIAL**. One of two authorised differential closure submissions
has been used. Kernel v3 verified private parentless source, then failed before
installation because the inventory collector rejected duplicate native PyGObject
metadata. The repair preserves all metadata and validates changes by precedence.
The exact next action is the final authorised private OG-001 submission, followed
by the complete non-model suite in that runtime if operational checks pass.

| Evidence | What it establishes | Limit |
|---|---|---|
| [Kernel v2](evidence/og001_kaggle/20260905_kernel_v2_24b25b5f/VERDICT.json) | Private transport, one T4, overlay/editable installs, protected packages preserved | Six global pip diagnostics; no runtime smoke result |
| [Kernel v3](evidence/og001_kaggle/20260906_kernel_v3_82e8736f/VERDICT.json) | Source origin/tree/snapshot linkage verified; failure preserved | Inventory failed before baseline pip check or installation |
| [Local preflight](evidence/og001_kaggle/20260906_reviewed_preflight/VERDICT.json) | Infrastructure/CPU regression tests and seven compiled notebooks; lint/format pass | Eight historical placeholder skips and known SSIUU xfail; no GPU evidence |

The environment may be accepted as `BASELINE_DIRTY_NO_REGRESSION` only after
raw pre/post checks establish no introduced or overlay-related unsatisfied
dependency, with protected/platform integrity and every runtime check passing.
This is not global pip cleanliness. No unrelated native package repair is planned.

## Gates

| Gate | Status | Required next evidence |
|---|---|---|
| OG-001 environment | PARTIAL | Differential consistency, imports, FP16, SDPA, genuine NF4, one T4, memory, manifests and verified overlay lock |
| RG-003 QUAIL fidelity | BLOCKED | Primary-paper specification, complete objective and gradient/masking tests, corrected P4 |
| RG-002 weight quantisation | BLOCKED | Deployment-aligned weight NF4 semantics and gradients, corrected P3 |
| RG-001 SSIUU fidelity | BLOCKED | Faithful parameter attribution/state/HVP, gradient and resource evidence, corrected P2 |
| RG-004 TOFU interpretation | OPEN, documentation only | Preserve interpretation as a KS p-value, not forgetting magnitude |

Machine records: [operational gates](operational_gates.yaml),
[scientific gates](research_gates.yaml). Passing surrogate tests cannot close
a scientific gate. Historical August pilots remain feasibility evidence only;
their raw artifacts and correction records remain unchanged.

## Progression requirements

After OG-001 and the complete validated-runtime acceptance suite pass, review the
full `dev..HEAD` diff independently, open/update one PR and merge non-destructively
into `dev` only if no operational defect remains. Retain historical branches and
the four untracked DOCX files. Create one scientific reconciliation branch from
updated `dev`, then repair RG-003, RG-002 and RG-001 using primary sources.

Corrected pilots precede protocol freeze. Full hypothesis tests require a frozen
protocol, researcher-approved preregistration and a code/config tag; they must not
start concurrently with preregistration drafting. Public submission still requires
Muhammad Arslan's approval. Thesis consequences are recorded in the
[thesis ledger](THESIS_UPDATE_LEDGER.md); no new scientific metrics are claimed.
