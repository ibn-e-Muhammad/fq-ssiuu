# Research code layout

| Folder | Purpose | Current acceptance |
|---|---|---|
| `reproductions/NPO/` | Canonical import-compatible source conformance namespace | RG-006 resolved within bounded objective/data/reference scope |
| `reproductions/NPO_1st-test/` | Researcher's preserved first-run source snapshot | Historical full-weight D1 |
| `reproductions/NPO_2nd-test/` | Completed local full-weight run, code/tests/raw results | Single-seed adapted evidence, preserved unchanged |
| `reproductions/NPO_3rd-test/` | New LoRA NPO+RT, complete evaluation and post-hoc reanalysis | Researcher-authorized class C; actual 1B GPU acceptance pending |
| `reproductions/QUAIL/` | Published complete QUAIL reproduction | RG-003 blocked |
| `reproductions/FQ_QAT/` | Deployment-aligned weight NF4 component validation | RG-002 blocked |
| `reproductions/SSIUU/` | Published parameter-attribution/HVP reproduction | RG-001 blocked |
| `reproductions/QRA/` | Directional recovery evaluation contract | RG-005 blocks confirmatory verdicts; descriptive records are allowed |
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

New local LoRA work follows [DL-024](../handbook/decisions/DL-024_project_lora_training.md) and the [third-run README](reproductions/NPO_3rd-test/README.md). The existing Mother entry point is preserved for its separate historical recipes.
