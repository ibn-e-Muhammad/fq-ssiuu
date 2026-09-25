# Researcher post-merge audit — Work Unit B

Source: Muhammad Arslan's review in the project task, 2026-09-23. This is a concise
record of the supplied audit and implementation response, not an independent review
by a second agent. PR #9 is accepted without rollback at merged dev `2334f321`.

Accepted scope: isolated NPO objective/data/frozen-reference boundary, deterministic
TOFU preprocessing, descriptive metrics/provenance, historical config rejection.
Not established: NPO training, quantized model loading, recovery attacks, corrected
pilots, QUAIL, deployment FQ-QAT, faithful SSIUU or their combination.

Required response in one Work Unit C branch/PR: candidate bitsandbytes exactly
0.46.1; real sequential matched precision inference; packed-state artifact identity;
recomputed evaluation digests; separate derived utility; clustered uncertainty;
bound FP16 oracle evidence; explicit protocol review before recovery decisions.
At most two Kaggle executions, no training. Accepted 0.45.5 evidence stays intact.

Governance response: PR #9 merge is reflected in live records. Default branch is
dev. Its rule now requires passing CI, fresh approval, last-pusher separation and
resolved threads. Automated checks are not substantive human scientific review.
Before/after rule JSON is preserved under `evidence/work_unit_c/20260923_governance`.

Research order remains NPO reproduction, faithful QUAIL, deployment-aligned FQ-QAT,
faithful SSIUU, combination, corrected pilots, preregistration, multi-seed study.
The work primarily repairs the ability to interpret later experiments; it does not
establish efficacy. DOCX files remain untouched pending canonical thesis selection.
