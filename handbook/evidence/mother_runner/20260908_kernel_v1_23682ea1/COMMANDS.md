# Execution and verification record

Source/orchestration: `23682ea12ff03ea867f220406423b6f54f77091c`.

1. Package and verify the private parentless source; source/staging manifests
   record exact commits, bundle, notebook and metadata hashes.
2. Stage with `python -m scripts.stage_mother --source /tmp/fq-mother-source-23682ea1 --dataset-id muhammadii/fq-ssiuu-source-23682ea1 --recipe configs/runs/integration_v1.json --output /tmp/fq-mother-kernel-23682ea1`.
3. Submit with `python -m kaggle kernels push -p /tmp/fq-mother-kernel-23682ea1 --accelerator NvidiaTeslaT4`. Metadata sets private visibility.
4. Terminal status COMPLETE; download once to `/tmp/fq-mother-kernel-v1-output`.
5. Replay `validate_setup_artifacts` at snapshot `8cc68b39e39ee61682b1b659d5d2d64a199ed951` against the committed baseline. Verify actual JUnit/notebook results and every runtime archive member hash against raw bytes.
6. Seal the complete download with `python scripts/archive_run.py seal --run-dir /tmp/fq-mother-kernel-v1-output --output /tmp/fq-mother-v1-complete-sealed --run-id 20260908-041835-2ef818513f23`.
7. Private Kaggle dataset creation expanded nested tar files. Preserve that
   dataset and record the representation limitation in `kaggle_expanded_receipt.json`.
   Create a separate private dataset containing the unchanged tar bytes named
   `artifacts.tar.bin`; download and verify its container and all 56 member hashes.
8. Mirror using `python scripts/archive_run.py mirror-hf --sealed /tmp/fq-mother-v1-complete-sealed --receipt /tmp/fq-mother-v1-hf-receipt.json`. The authorised project loader reads HF_TOKEN only for this operation. Initial sandbox ConnectError did not mutate raw files; network retry succeeded with pinned revision verification.

`RAW_OUTPUT_MANIFEST.json` inventories all 56 downloaded files. Git holds exact
raw bytes except the duplicate runtime tar, retained byte-for-byte inside both
private archives. Derived receipts and verdicts are separate from raw output.
No further GPU submission was used for archival.
