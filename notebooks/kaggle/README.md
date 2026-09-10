# Start here: 00_MOTHER_RUNNER.ipynb

This is the single current notebook. It runs the recipe selected in its parameter
cell from a verified source snapshot. Environment code and lock are the validated
versions; no dependency redesign accompanies this reorganisation.

1. For this private project, stage the committed source using
   `python -m scripts.stage_mother --source SOURCE_BUNDLE_DIR --dataset-id OWNER/PRIVATE_SOURCE_DATASET --recipe configs/runs/integration_v1.json --output NEW_STAGE_DIR`.
   The helper fills source/notebook pins and writes private T4 metadata. Import
   the generated `00_MOTHER_RUNNER.ipynb` into Kaggle, or submit that directory
   with the Kaggle CLI. Do not run an unfilled `STAGING_REQUIRED` template.
2. Attach the matching private source dataset; enable internet and Tesla T4.
   Source verification checks the exact attached bundle before importing code.
3. For operations that need credentials, enable individual Kaggle Secrets or
   `FQ_SSIUU_DOTENV` (dotenv text kept in memory). Integration requires none.
   Never attach a `.env` file or paste tokens in notebook cells.
4. Run all cells. Outputs use a fresh run ID and are sealed even on failure.
   Follow the [archival guide](../../handbook/MOTHER_NOTEBOOK.md) from the repo
   root after download; exact bytes must be verified in both private stores.

Required private inputs cannot be inferred from an unconfigured copy. See
[the full guide](../../handbook/MOTHER_NOTEBOOK.md) for source packaging and
archival. Method recipes stay blocked until their scientific criteria pass.
No notebook can currently run a valid full FQ-SSIUU experiment automatically.

`archive/` contains historical notebooks for inspection at their recorded commits.
They are not alternative current launchers. Their bytes are preserved by the
layout move; old embedded paths are intentionally historical.
