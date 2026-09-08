# Mother notebook execution guide

The canonical notebook is `experiments/notebooks/kaggle/MOTHER_RUNNER.ipynb`.
It runs code from one verified source snapshot using a versioned JSON recipe.
The first implemented recipe is `configs/runs/integration_v1.json`. Method pilots
require an accepted conformance record and an upstream-specific adapter; none is
silently mapped to historical surrogate code. Full experiments remain disabled
until the researcher approves the frozen protocol.

## Run loop

1. Commit the method/recipe/test change. Run local tests appropriate to that change.
2. Package the commit with `python scripts/package_kaggle_source.py`, create a
   private Kaggle source dataset and verify the downloaded bundle/manifest hashes.
3. Stage the notebook using `python -m scripts.stage_mother --source SOURCE_DIR
   --dataset-id OWNER/DATASET --recipe configs/runs/integration_v1.json
   --output NEW_STAGE_DIR`. The staging receipt records source, notebook and
   metadata hashes; staging refuses existing output/receipts.
4. Submit with `python -m kaggle kernels push -p NEW_STAGE_DIR --accelerator
   NvidiaTeslaT4`. Metadata pins the validated image, private visibility and T4.
5. Download terminal outputs to a fresh directory. Inspect the recorded result,
   including failed/null outcomes. Preserve kernel logs alongside run artifacts.
6. Seal outputs with `python scripts/archive_run.py seal --run-dir RAW_DIR
   --output SEALED_DIR --run-id RUN_ID`. Store the archive and manifest in a new
   private Kaggle dataset and mirror with `python scripts/archive_run.py mirror-hf
   --sealed SEALED_DIR --receipt NEW_RECEIPT.json`. Never mark synchronization
   complete until downloaded content hashes match at both destinations.

Kaggle automatically expands uploaded tar archives. Verify every expanded file
against `ARCHIVE_MANIFEST.json`; Hugging Face also preserves the container hash.
Each run has its own dataset namespace. Failed synchronization retains the local
and notebook output recovery copies; a retry must not replace differing content.

The OG-001 v4 pipeline has already exercised both private stores. See
`evidence/og001_kaggle/20260907_v4_archive_receipts/` for verified receipts.
The Mother integration recipe still needs its first Kaggle execution.

## Credentials

Use `.env.example` as the key schema. On a local checkout, only the project secret
loader reads the ignored `.env` when an authorised operation needs a key. On
Kaggle, enable individual secrets or create one private secret named
`FQ_SSIUU_DOTENV` containing the same dotenv text. The loader parses it in memory
with `python-dotenv`, without interpolation or exporting unrelated values.
Do not attach `.env` as a dataset or paste values into notebook cells.

Lookup order: process environment, individual Kaggle Secret, Kaggle dotenv text,
local project `.env` (local runtime only). `get_secret("HF_TOKEN")` retrieves a
value for model/data access or private storage. `GITHUB_TOKEN`/`GIT_TOKEN` are
available for explicitly configured authenticated repository operations. The
default parentless bundle workflow requires no Git token and does not clone
authenticated GitHub history inside Kaggle. Integration declares no credentials.

## What is reproducible

The pinned image, source tree, exact overlay pins, recipe and input revisions
define execution inputs. Run IDs, timestamps and measured runtime/memory naturally
differ. Scientific adapters must additionally set deterministic seeds, preserve
checkpoint/resume state, log selected-device peak memory and synchronize at their
declared checkpoint intervals. The integration runner alone does not establish
those scientific-training properties.
