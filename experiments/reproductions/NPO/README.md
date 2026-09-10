# NPO reproduction workspace

Status: **BLOCKED (RG-006)** for corrected pilots.

candidate.py and validate_upstream.py implement/test causal sequence-sum NPO. CPU value/gradient parity passes against both pinned upstreams; caller masking/reference behavior remains to validate. No corrected 1B reproduction is claimed.

Keep method-specific candidate code, upstream verification scripts and versioned
recipes here. Tests belong in `tests/methods/NPO/`; historical regression tests
are under `tests/historical/`. Promote a candidate to `src/fq_ssiuu/` only after
its declared conformance tests, required runtime checks and review pass. Record
the supporting commit and evidence; promotion alone is not a protocol freeze.

Files in `historical/` are research-history reference code. Run historical
experiments only from their original recorded commits; current pilot entry points
are disabled while scientific gates remain blocked. Never reuse their outputs
as corrected-method validation. No raw result was moved or overwritten.

No accepted runtime recipe exists yet. The earlier proposed GPU draft was never
submitted and is excluded from this change. No Kaggle/model submission belongs
to reconciliation A.
