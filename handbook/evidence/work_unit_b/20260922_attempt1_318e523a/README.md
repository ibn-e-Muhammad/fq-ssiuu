# Work Unit B, attempt 1 (Mother version 2)

Origin code `318e523ac302f9f80f2c74f7c712bff08ff96a8e`; run
`20260922-101703-ce77656bee84`. `raw/` contains the original 63 downloaded files,
unchanged. The Mother archive has 59 members; the complete private archive has 68
members including kernel/bootstrap output and staging/source receipts.

`VERDICT.json` is derived, separate from raw output. To repeat the numerical/hash
checks from the repository root, use `.venv/bin/python handbook/evidence/work_unit_b/20260922_attempt1_318e523a/derive_verdict.py` without Python optimization.
The script checks raw/archive hashes, exact source/notebook linkage, environment
acceptance, input tensor equality, NPO values and recomputed descriptive metrics.
All recovery/durability decisions remain null. This is not a corrected pilot.

Private complete archives and verified immutable revisions are identified by
`kaggle_receipt.json` and `huggingface_complete_receipt.json`. An additional run-only
HF archive is retained with its own receipt. No remote or historical artifact was
overwritten. The initial archive-title error and successful retry are preserved.
