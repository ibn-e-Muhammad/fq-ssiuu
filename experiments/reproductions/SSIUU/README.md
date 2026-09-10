# SSIUU reproduction workspace

Status: **BLOCKED (RG-001)** for corrected pilots.

Published parameter attribution, previous-step state and differentiable HVP regularisation remain unimplemented. historical/ contains LayerIG surrogate and P2.

Keep method-specific candidate code, upstream verification scripts and versioned
recipes here. Tests belong in `tests/methods/SSIUU/`; historical regression tests
are under `tests/historical/`. Promote a candidate to `src/fq_ssiuu/` only after
its declared conformance tests, required runtime checks and review pass. Record
the supporting commit and evidence; promotion alone is not a protocol freeze.

Files in `historical/` are research-history reference code. Run historical
experiments only from their original recorded commits; current pilot entry points
are disabled while scientific gates remain blocked. Never reuse their outputs
as corrected-method validation. No raw result was moved or overwritten.
