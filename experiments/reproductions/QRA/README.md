# QRA reproduction workspace

Status: **BLOCKED (RG-005)** for corrected pilots.

Directional outcome contract is being reconciled. historical/ contains invalid p-value-ratio verdicts, the inverted report, P1 and old evaluator helper.

Keep method-specific candidate code, upstream verification scripts and versioned
recipes here. Tests belong in `tests/methods/QRA/`; historical regression tests
are under `tests/historical/`. Promote a candidate to `src/fq_ssiuu/` only after
its declared conformance tests, required runtime checks and review pass. Record
the supporting commit and evidence; promotion alone is not a protocol freeze.

Files in `historical/` are research-history reference code. Run historical
experiments only from their original recorded commits; current pilot entry points
are disabled while scientific gates remain blocked. Never reuse their outputs
as corrected-method validation. No raw result was moved or overwritten.
