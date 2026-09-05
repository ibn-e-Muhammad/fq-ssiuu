# Execution record: repaired OG-001 attempt

This is a derived record of consequential commands and results from the Codex
session, not a byte-for-byte terminal transcript. The raw test output, submitted
notebook, metadata, setup reports and Kaggle logs are saved alongside it.

| Operation | Observed result |
|---|---|
| Git local branch, HEAD, tracked status and merge base | `fix/kaggle-overlay-evidence-integrity`; `3fd39547a3dc7e467e9a6484c421b1238d4357f5`; tracked tree clean; base `e395f7af264eaaac9473d8b9e9f1c7a46c825d5c` |
| GitHub connected branch comparison | Same remote tip; 17 ahead, 0 behind `dev` |
| Existing project Python / Kaggle CLI | Python 3.11.15; Kaggle CLI 2.2.4 |
| Original dependency-policy unittest suite | 20 tests passed; this did not exercise the broken verifier entrypoint |
| Added entrypoint regression tests before repairs | Failed with undefined `get_git_sha` and `report_data`; fixed afterward. The missing verifier `run_pip_check` was also restored |
| Final `./.venv/bin/python -m unittest -v tests.test_kaggle_dependency_policy tests.test_kaggle_source_transport` | Exit 0, 35 tests passed; see `offline_tests.log` |
| Notebook nbformat, code compilation, static undefined-global checks, credential-pattern scan, compileall, `bash -n scripts/setup_env.sh`, `git diff --check` | PASS; see `local_validation.json` |
| Current tracked tree credential scan | Found a legacy credential literal in `GATE1_evaluation.ipynb`; removed before packaging. Sanitized committed tree scan passed |
| Source publication | GitHub Git-data API, exact staged tree `1a6bd790404e39fae90f96b60cf9700a59bc2de5`, commit `91b6aa39e520f9312b098997134c1deabeabfb8f`, branch update `force=false` |
| `./.venv/bin/python scripts/package_kaggle_source.py --source-repo . --source-origin-sha 91b6aa39e520f9312b098997134c1deabeabfb8f --output-dir /tmp/fq-og001-source-91b6aa39` | Exit 0; verified one parentless commit, identical tree, no original history or forbidden tracked files |
| `./.venv/bin/python -m kaggle datasets init -p /tmp/fq-og001-source-91b6aa39` | Exit 0; metadata template created, then set to the actual new source ID and `other`/MIT description |
| New dataset status query before creation | HTTP 403; existence was not inferred from that response. The subsequent create operation succeeded |
| `./.venv/bin/python -m kaggle datasets create -p /tmp/fq-og001-dataset-lcwecnyq` | Exit 0; CLI reported private dataset creation. Staging contained only bundle, manifest and dataset metadata |
| `./.venv/bin/python -m kaggle datasets status muhammadii/fq-ssiuu-source-91b6aa39` | Exit 0, `ready` |
| `./.venv/bin/python -m kaggle datasets download -d muhammadii/fq-ssiuu-source-91b6aa39 -p /tmp/fq-og001-source-download --unzip` | Exit 0; downloaded bundle, manifest hashes and Git topology verified against local pins |
| Notebook publication | GitHub Git-data API, exact staged tree `bf1af1e77be7f0909212ddab283e320f9c5bd033`, commit `24b25b5f3883646d1841e83cdd6a88a54b3fd93d`, branch update `force=false` |
| Staging | Fresh temporary directory; only notebook and kernel metadata. Only the notebook's `ORCHESTRATION_COMMIT` parameter was populated with its containing commit |
| `./.venv/bin/python -m kaggle kernels push -p /tmp/fq-og001-kernel-k3bl1fih --accelerator NvidiaTeslaT4` | Exit 0; version 2 accepted. This was the only repaired submission; see `submission_attempt.json` and `submission_result.json` |
| `./.venv/bin/python -m kaggle kernels status muhammadii/fq-ssiuu-og-001-overlay-validation` | RUNNING, then ERROR |
| `./.venv/bin/python -m kaggle kernels output muhammadii/fq-ssiuu-og-001-overlay-validation -p /tmp/fq-og001-kernel-v2-output` | Exit 0; all output files downloaded in one command, preserved under `raw/` |
| Runtime setup | Overlay install 0; editable install 0; global pip check 1; setup 1. See `raw/og001_output/setup_status.json` |
| Download verification | All 15 setup-manifest entries matched size and SHA-256; all raw files passed credential-pattern scanning |
| Saved notebook readback | Version-specific `/2` read returned HTTP 403. Latest saved source was pulled successfully and matched all staged cell sources; readback confirms private kernel, T4 and source dataset |

## Scope and handling notes

- The old source pin `502871a57b275bffaa65ba6a24f9e834c3721fb6` was superseded explicitly because its verifier could not run and its tree still included the legacy credential literal. No successful execution of that old pin is claimed.
- HTTPS Git access without a credential helper could not authenticate; an SSH read stalled and was interrupted. Publication used the connected GitHub integration without reading or using a GitHub PAT. Exact GitHub commit objects were reconstructed from Git-data metadata and verified by their SHA before adopting the published commits locally. No branch history was rewritten.
- A GitHub diff tool response displayed the removed legacy credential during commit verification. It was not copied into this evidence or the Kaggle source bundle. The prior handoff says the historical credential had been rotated; validity was not checked.
- The local `.env` was never opened, sourced or uploaded. No local package was installed, upgraded or removed. Existing Kaggle CLI authentication was used for Kaggle operations.
- No DOCX was touched. No model, experimental dataset, scientific implementation, metric, seed or threshold changed. The only legacy scientific-notebook edit removed its hardcoded credential assignment.
- The before-freeze data supports an inherited-image conflict interpretation; no pre-install pip-check transcript exists. No runtime import, FP16, SDPA or NF4 PASS is claimed.
- No second repaired kernel submission, merge or force update occurred. The original failed attempt's raw log is unchanged.
- The final unfiltered staged whitespace check reports one extra blank line at EOF in immutable `raw/kaggle_runner_stderr.log`. That downloaded log is preserved byte-for-byte; the scoped check excluding this one raw log passes for every authored/staged file.

Reference specifications: [Git bundles](https://git-scm.com/docs/git-bundle),
[Kaggle kernel CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md),
[Kaggle dataset metadata and licenses](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets_metadata.md).
