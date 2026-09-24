# QRA reproduction workspace

Status: RG-005 blocks **confirmatory attack/durability verdicts and the full
FQ-SSIUU study**. Source-faithful constituent reproductions and descriptive metric
collection do not require Protocol v2 activation. See the
[reproduction policy](../../../handbook/methods/METHOD_REPRODUCTION_POLICY.md).

Matched inference and descriptive measurement plumbing passed C1. Protocol v2 is
DEFERRED_UNTIL_BASE_METHOD_PILOTS / PROPOSED_NOT_APPROVED and non-executable.
`historical/` contains invalid p-value-ratio verdicts, the inverted report, P1 and
the old evaluator helper; these remain retired.

Keep method-specific candidate code, upstream verification scripts and versioned
recipes here. Tests belong in `tests/methods/QRA/`; historical regression tests
are under `tests/historical/`. Promote a candidate to `src/fq_ssiuu/` only after
its declared conformance tests, required runtime checks and review pass. Record
the supporting commit and evidence; promotion alone is not a protocol freeze.

Files in `historical/` are research-history reference code. Run historical
experiments only from their original recorded commits; current pilot entry points
are disabled while scientific gates remain blocked. Never reuse their outputs
as corrected-method validation. No raw result was moved or overwritten.
