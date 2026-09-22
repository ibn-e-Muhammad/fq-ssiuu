# Thesis Update Ledger

**Purpose:** Track which thesis sections are affected by each correction or finding.
**Convention:** Each row maps a correction ID or finding to the thesis sections that must be updated.

---

| Correction | Thesis Sections Affected | Update Required |
|------------|------------------------|-----------------|
| CR-001 (TOFU FQ interpretation) | §Results, §Discussion, all tables reporting FQ | Revise language: FQ is a KS p-value, not a forgetting magnitude |
| CR-002 (Llama layer count) | §Methodology (model description) | Change "28 layers" → "16 layers"; layer 14 is penultimate, not middle |
| CR-003 (NPO loss bound) | §Background (NPO description) | Remove claim of upper bound 2/β; state infimum is 0, unbounded above |
| CR-004 (Pilot overstatements) | §Pilot Results, §Discussion | Downgrade "confirmed" → "pilot-executed"; qualify all claims |
| CR-005 (QUAIL fidelity) | §Related Work, §Experimental Setup (C8), §Results | Document as "incomplete surrogate"; list missing components |
| CR-006 (SSIUU gradient flow) | §Methodology (SSIUU section), §Results | Cannot claim SSIUU contribution until gradient flow is fixed |
| CR-007 (Quantisation location) | §Methodology (FQ-QAT section), §Results | State as "adapter-output INT4 ablation", not "deployment-equivalent NF4 QAT" |
| CR-008 (Certificate terminology) | §Evaluation (certificate section) | Replace "certificate" with "empirical durability report" throughout |
| CR-009 (Kaggle Environment) | §Environment, §Hardware constraints, §Software dependencies | Update canonical Kaggle environment, precise hardware evidence, dependency readiness limits |
| CR-009 (Gate-0 submissions) | §Reproducibility, §Research procedure/audit trail | Document failed 1st submission as orchestration error, successful 2nd, prohibit treating Gate-0 as scientific result |
| DL-020 (Decoupled Overlay Architecture) | §Methodology (Software Environment), §Reproducibility | Document decoupled Kaggle baseline and overlay architecture with dry-run report verification |
| DL-021 (Pip Report Verification Guard) | §Reproducibility, §Infrastructure | Detail non-mutating pip report verification guard preventing native stack modification |
| CR-011 (Overlay Validation Coverage) | §Reproducibility, §Infrastructure | Document behavioral test coverage and single run ID evidence linkage across Kaggle overlay setup failure paths |
| DL-023 (Stage 1I Kaggle Validation Run) | §Reproducibility, §Infrastructure | Document original anonymous-clone failure and subsequent private bundle transport, which requires no GitHub authentication inside Kaggle |
| OG-001 repaired kernel v2 | §Reproducibility, §Infrastructure | Record verified private parentless source transport; distinguish origin, tree, snapshot and notebook commits. Installation succeeded, but native-image pip-check conflicts prevented runtime smoke validation. OG-001 remains PARTIAL; no scientific claim follows. |

| OG-001 differential kernel v3 | Chapter 3 environment/reproducibility | Preserve the failed pre-install inventory attempt and distinguish duplicate native metadata from introduced dependency conflicts. Source `6cb7933d`, notebook `82e8736f`, setup run `0b2980fa-c045-4c1f-8c66-4f8f890fd447`; raw evidence in `evidence/og001_kaggle/20260906_kernel_v3_82e8736f/`. No runtime or scientific success follows. |

| OG-001 environment closure, kernel v4 | Chapter 3 software environment/reproducibility | PASS under BASELINE_DIRTY_NO_REGRESSION; six native diagnostics unchanged, runtime operations verified. Raw run `04396fe9-0195-476c-bb74-29e6a5ca074b` is linked by `evidence/og001_kaggle/20260906_kernel_v4_2be46116/VERDICT.json`. Integration collection failure is separate. Smoke memory must not be presented as training memory. |
| Upstream implementation reconciliation | Chapters 2–3 methods | QUAIL author code located; NPO sequence reduction/alignment needs upstream equivalence before any new pilot claim. See `methods/UPSTREAM_IMPLEMENTATIONS.md`. |

| Mother integration and archival | Chapter 3 execution/reproducibility | Run `20260908-041835-2ef818513f23`, source `23682ea1`, passed environment and implemented non-model integration checks. Link `evidence/mother_runner/20260908_kernel_v1_23682ea1/VERDICT.json` and storage receipts. Twelve placeholders and SSIUU expected failure remain; no scientific or training-memory claim. Dotenv providers are lazy and values are excluded from artifacts. |

---

## Notes

- The thesis itself (LaTeX/Word documents) is NOT tracked in this repository.
- This ledger tracks WHAT needs updating. The actual thesis edits are the author's responsibility.
- Each correction entry in the correction register (`handbook/corrections/REGISTER.md`) is cross-referenced here.

## Scientific reconciliation A — 2026-09-10

| Consequence | Chapters | Required change and evidence |
|---|---|---|
| CR-012 QRA interpretation/direction | 3–5 | Withdraw historical P1/P4 attack/durability conclusions; retain observations with raw linkage. Cite `corrections/CR-012_qra_outcome_validity.md`. No p-value recovery magnitude or certified pass. |
| NPO objective parity | 2–3 | Describe causal sequence sums and pure NPO versus NPO+RT; CPU value/gradient parity is distinct from a faithful 1B reproduction. See `methods/NPO_CONFORMANCE.md`. |
| Method workspace and promotion | 3 reproducibility | Individual candidate/historical folders, one Mother entry point, accepted-code promotion and later protocol tag are distinct milestones. |

Authoritative working thesis remains untracked; the four protected DOCX files were
not modified. This ledger records pending thesis changes, not completed prose edits.

## Work unit B — 2026-09-22

[Structured update packet](thesis_updates/WORK_UNIT_B.md) covers Chapters 2–3: NPO
adapter/reference, metric taxonomy, paper/upstream distinction, paired uncertainty,
provenance and runtime claim limits. CPU evidence is `evidence/work_unit_b/20260922_cpu/`.
No new scientific findings or DOCX changes. The packet is tracked source; it is not
a claim that the working thesis has been converted or updated.
