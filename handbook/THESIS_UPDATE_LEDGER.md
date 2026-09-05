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
| DL-023 (Stage 1I Kaggle Validation Run) | §Reproducibility, §Infrastructure | Document private Kaggle validation run execution result and authentication requirement for private GitHub repository |

---

## Notes

- The thesis itself (LaTeX/Word documents) is NOT tracked in this repository.
- This ledger tracks WHAT needs updating. The actual thesis edits are the author's responsibility.
- Each correction entry in the correction register (`handbook/corrections/REGISTER.md`) is cross-referenced here.
