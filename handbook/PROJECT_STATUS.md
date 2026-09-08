# FQ-SSIUU project status

Updated 2026-09-08. Canonical plan: [MASTER_EXECUTION_PLAN.md](MASTER_EXECUTION_PLAN.md).
Active branch: `fix/kaggle-overlay-evidence-integrity`; not merged into `dev`.

## Current decision

OG-001 is **PASS**, independently revalidated from kernel v4 raw artifacts.
The environment is `BASELINE_DIRTY_NO_REGRESSION`: six native diagnostics remain,
none were introduced, and all runtime/protected-stack checks pass. The kernel's
ERROR status belongs to a later integration test-package collision. The repaired
Mother notebook completed its first private Kaggle run successfully at source
`23682ea1`: 144 passed, 12 historical placeholder skips, 8 deselected and one
known SSIUU expected failure; all ten notebooks compiled. Environment and archive
hashes were independently replayed. No scientific experiment ran.

The researcher-directed execution loop is faithful upstream method → one Mother
notebook → private Kaggle + Hugging Face archives → consistent research records.
The exact next action is the reviewed integration PR into `dev`; then upstream
NPO parity and scientific reconciliation on a successor branch, before corrected
pilots. Mother run `20260908-041835-2ef818513f23` used 1/2 authorised submissions.
See its [verdict and private archive receipts](evidence/mother_runner/20260908_kernel_v1_23682ea1/VERDICT.json).

| Evidence | What it establishes | Limit |
|---|---|---|
| [Kernel v2](evidence/og001_kaggle/20260905_kernel_v2_24b25b5f/VERDICT.json) | Private transport, one T4, overlay/editable installs, protected packages preserved | Six global pip diagnostics; no runtime smoke result |
| [Kernel v3](evidence/og001_kaggle/20260906_kernel_v3_82e8736f/VERDICT.json) | Source origin/tree/snapshot linkage verified; failure preserved | Inventory failed before baseline pip check or installation |
| [Local preflight](evidence/og001_kaggle/20260906_reviewed_preflight/VERDICT.json) | Infrastructure/CPU regression tests and seven compiled notebooks; lint/format pass | Eight historical placeholder skips and known SSIUU xfail; no GPU evidence |

[Kernel v4](evidence/og001_kaggle/20260906_kernel_v4_2be46116/VERDICT.json) passed
source verification, installations, required imports, FP16, SDPA, genuine NF4,
one-device enforcement, protected/native integrity, manifests and overlay lock.
The observation establishes environment readiness, not model-training memory or
scientific validity. No unrelated native package repair is planned.

## Gates

| Gate | Status | Required next evidence |
|---|---|---|
| OG-001 environment | PASS | [Kernel v4 closure](evidence/og001_kaggle/20260906_kernel_v4_2be46116/VERDICT.json); reproduced in Mother v1 with full implemented non-model suite |
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
