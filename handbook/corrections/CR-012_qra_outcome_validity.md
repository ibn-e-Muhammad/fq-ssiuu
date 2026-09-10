# CR-012 — QRA score and durability directionality

Discovered 2026-09-09; correction implemented during scientific reconciliation A.
Status: CODE/DOCUMENTATION CORRECTED; scientific acceptance remains RG-005 BLOCKED.
This extends CR-001/004/008; renaming a field did not repair the original semantics.

## Superseded claims

The historical pipeline divided an absolute difference of TOFU KS p-values by
the FP16 p-value. That quantity is not recovered knowledge and has no valid
attack threshold. The same legacy PASS/certified result meant attack detection
in one module and durability acceptance in another, reversing the conclusion.

Historical P1 produced precision-specific evaluations. Its NF4 p-value increase
does not confirm QRA. P4's low p-values do not mean extreme forgetting, and a
hinge-only run without the complete retain objective cannot establish QUAIL's
utility trade-off or comparative robustness. No August hypothesis is confirmed.
See [primary-source contract](../methods/QRA_OUTCOME_CONTRACT.md).

## Correction and preservation

The original implementations are explicitly historical under
`experiments/reproductions/QRA/historical/`. Current p-value scoring and certificate
entry points fail closed. Descriptive records preserve missing values, enforce
matched provenance and separate nullable attack/durability decisions. Behavioral
tests replace the former QRA/report placeholders.

Raw pilot results, downloaded artifacts and historical evidence have not been
edited. Their narrative conclusions are superseded by this record and the current
project status. Corrected pilots require fresh run IDs. Thesis Chapters 3–5 and
any pilot headline must cite this correction before reusing historical observations.
