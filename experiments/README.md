# Research code layout

| Folder | Purpose | Current acceptance |
|---|---|---|
| `reproductions/NPO/` | Author/OpenUnlearning NPO parity, then a 1B reproduction | RG-006 blocked; CPU objective parity against both pinned upstreams; adapter pending |
| `reproductions/QUAIL/` | Published complete QUAIL reproduction | RG-003 blocked |
| `reproductions/FQ_QAT/` | Deployment-aligned weight NF4 component validation | RG-002 blocked |
| `reproductions/SSIUU/` | Published parameter-attribution/HVP reproduction | RG-001 blocked |
| `reproductions/QRA/` | Directional recovery evaluation contract | RG-005 blocks all corrected pilots/reports |
| `combined/FQ_SSIUU/` | Combined method after individual components pass | Not implemented; no accepted recipe |
| `infrastructure/` | Historical environment probes | Environment closed; no further setup work planned |

Each method owns its candidates, upstream validation scripts, recipes and historical
surrogates. `tests/methods/METHOD/` holds current conformance tests;
`tests/historical/` preserves historical behavior tests without fidelity claims.
`src/fq_ssiuu/` holds reusable code with declared acceptance scope. No scientific
training method is currently frozen or accepted for a corrected 1B pilot.

Start execution at `notebooks/kaggle/00_MOTHER_RUNNER.ipynb`. The same runner
consumes a versioned recipe; it is not duplicated for every method.
`LAYOUT.json` maps relocated files and their pre-move hashes. Historical notebooks
are archived; raw evidence and result directories have not changed.
