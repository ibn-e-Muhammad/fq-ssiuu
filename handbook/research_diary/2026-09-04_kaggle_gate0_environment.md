# Kaggle Gate-0 Environment Audit

This diary entry details the environment audit executions performed on Kaggle to establish reproducible dependency architectures.

## Version 1: Missing Kernelspec

- **Commit**: `c25989b41c9d833f27105f2b2c41e1e7491eb408`
- **Execution Details**: Kernel submitted privately with T4 requested and internet disabled.
- **Outcome**: Execution failed before the first notebook cell.
- **Root Cause**: The Jupyter notebook lacked a standard `kernelspec` metadata block, causing Kaggle/Papermill execution to crash immediately.
- **Scientific Impact**: No hardware conclusion was possible. No scientific test ran.
- **Classification**: Orchestration/packaging failure.

## Version 2: Complete Environment Audit

- **Repair Commit**: `547db67e6bc75f9efe0b17cbbbc44d8128aff23f`
- **Run ID**: `57588774-bdd5-41c8-b56e-7df0c2f5049b`
- **Status**: COMPLETE
- **Python**: 3.12.13
- **PyTorch**: 2.10.0+cu128
- **CUDA Build**: 12.8
- **cuDNN**: 91002
- **Driver**: 580.159.04
- **Maximum CUDA Capability (nvidia-smi)**: 13.0
- **Actual GPUs**: two Tesla T4 devices
- **Memory**: 15,636,037,632 bytes per device (approximately 14.56 GiB). Note that the project’s `<16 GB` constraint applies per device (`cuda:0`); the two T4 devices must not be combined into one larger VRAM budget.
- **Compute Capability**: 7.5
- **Dtype / Precision**: FP16 canonical. `is_bf16_supported()` returns `true` in PyTorch, but T4 (Turing architecture) lacks native hardware BF16 Tensor Core execution.

### Gate-0 Statuses
- **FP16**: PASS
- **SDPA**: PASS
- **NF4**: UNAVAILABLE (because bitsandbytes was absent from the native Kaggle image)
- **Dependencies**: PARTIAL (Script incorrectly allowed critical absent packages to pass)
- **Overall Gate-0**: PARTIAL

### Verification
No package, model, or dataset was installed or downloaded. No scientific test or experiment ran during this audit.

## Correction Note (Stage 1H-A2 Record Corrections)

The documentation commit (`5045cd59964adcab6e089d570bb00e077c2e1fb4`) preserved the Gate-0 evidence artifacts but introduced inaccurate provenance and stale dependency statements. These are corrected below:

* **Timestamps Corrected**: `kernel_submission_timestamp` is set to `null` because `2026-09-04T18:52:35Z` was hardcoded in an untracked local orchestration script (`process_results.py` line 50) rather than a committed primary artifact. Preserved separately as `reported_submission_timestamp`: `2026-09-04T18:52:35Z` with evidence quality `UNVERIFIED — REPORTED IN LOCAL ORCHESTRATION TRANSCRIPT`. `probe_runtime_timestamp`: `2026-09-04T18:52:48.388417+00:00` (evidence: raw Gate-0 JSON artifacts `gate0_environment_manifest.json`), `artifact_retrieval_timestamp`: `null`.
* **Credential Incident Corrected**: Statement claiming rotation of a Kaggle token and WandB key in commit `5045cd59964adcab6e089d570bb00e077c2e1fb4` was incorrect. The actual documented incident was rotation of an exposed GitHub PAT in the Git remote URL and a hardcoded Hugging Face token in a handoff file, with remote URL sanitized and hardcoded tokens removed.
* **Unresolved Packages Corrected**: OmegaConf was incorrectly listed as unresolved in commit `5045cd59964adcab6e089d570bb00e077c2e1fb4`. `omegaconf==2.3.0` is present and importable in the native Kaggle image and is removed from unresolved lists. Unresolved candidates for overlay installation are `bitsandbytes`, `captum`, `hydra-core`, `lm-eval`, `jsonlines`. TRL is classified as `ABSENT — NO CURRENT IMPORT FOUND — REMOVAL DECISION PENDING`. Native versions accurately recorded from `gate0_package_inventory.json` / `pip_freeze.log`.
* **CUDA Semantics Corrected**: Conflation of `nvidia-smi` CUDA 13.0 with PyTorch runtime build corrected. Documented separately: NVIDIA driver `580.159.04`, max CUDA capability `13.0`, PyTorch CUDA build `12.8`, PyTorch build `2.10.0+cu128`, cuDNN `91002`.
* **Single-Device FP16 Enforcement (DL-017)**: Conflation of pooled T4x2 budget corrected. Two Tesla T4 GPUs (14.56 GiB each) exposed; budget evaluated per selected device (`cuda:0`); single-device FP16 resource enforcement. `is_bf16_supported()` is `true` but T4 (compute capability 7.5) lacks native hardware BF16 Tensor Cores.
* **Scientific Impact**: No scientific code, model, dataset, or experimental result was modified or impacted.
