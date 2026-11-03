# NPO reproduction workspace

Status: **D1 source/data preflight; training not yet validated**.

candidate.py and validate_upstream.py implement/test causal sequence-sum NPO.
RG-006 is resolved for the objective/data/frozen-reference boundary accepted in
Work Unit B. D1 pins source training requirements in `source_v1.json` and checks
all pinned training rows with `preflight.py`. See
[`NPO_D1_SOURCE_ADAPTATION.md`](../../../handbook/methods/NPO_D1_SOURCE_ADAPTATION.md).
No corrected 1B training reproduction is claimed.

Keep method-specific candidate code, upstream verification scripts and versioned
recipes here. Tests belong in `tests/methods/NPO/`; historical regression tests
are under `tests/historical/`. Promote a candidate to `src/fq_ssiuu/` only after
its declared conformance tests, required runtime checks and review pass. Record
the supporting commit and evidence; promotion alone is not a protocol freeze.

Files in `historical/` are research-history reference code. Run historical
experiments only from their original recorded commits; current pilot entry points
are disabled while scientific gates remain blocked. Never reuse their outputs
as corrected-method validation. No raw result was moved or overwritten.

No accepted D1 training recipe exists yet. The B forward-only recipe remains a
separate historical validation contract. `source_v1.json` is an inventory and
proposal, not a Mother recipe. The researcher approved accumulation 8, preserving the source method; implementation and runtime acceptance remain.
