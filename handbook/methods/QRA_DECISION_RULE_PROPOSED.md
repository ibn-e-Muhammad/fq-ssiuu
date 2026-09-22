# PROPOSED_NOT_APPROVED — future QRA decision rule

This document is non-executable. It supplies no thresholds and grants no gate
closure, pilot or experimental authorization. RG-005 remains BLOCKED.

A future rule could require a prespecified positive increase in forget answer
knowledge after quantization, measured by paired answer log probability and
supported by generated recoverability. It would also need retained-utility
constraints, a valid pre-attack forgetting baseline, author-level dependence
handling, adequate sample sizes, seed replication, precision comparisons and a
multiple-testing policy. The minimum effect and uncertainty criteria must be
chosen before confirmatory results are seen.

KS p-values would remain diagnostics against a separately pinned retain-only
oracle. No p-value ratio, p-value difference, non-rejection-as-equivalence, or
global NF4 bin width could supply a recovery threshold. Fixed-FP16 versus
precision-matched oracle policy must be explicit in the eventual protocol; the
bounded work-unit-B check does not approve that policy for a full experiment.

If an approved future rule detects an attack, durability_pass must be false.
Failure to detect an attack need not establish durability. A positive durability
decision would require separately approved criteria and adequate evidence.
Today both fields remain null.
