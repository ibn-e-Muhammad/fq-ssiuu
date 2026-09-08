# CR-009: Kaggle Environment and Dependency Readiness Reconciliation

## Corrected Claims
* The legacy Colab environment is historical and is not the canonical current environment.
* The canonical platform is established as: `Kaggle Tesla T4 runtime, with single-device FP16 resource enforcement`.
* Kaggle hardware readiness passed (2x Tesla T4 GPUs detected, each exposing 15,636,037,632 bytes / ~14.56 GiB).
* Single-device `cuda:0` resource enforcement is required. The second T4 must not be used for model sharding, distributed training, or aggregate VRAM pooling unless a separate pre-registered multi-GPU condition permits it.
* PyTorch `is_bf16_supported()` returning `true` must not be interpreted as proof of native T4 BF16 Tensor Core execution; FP16 remains the canonical training and evaluation dtype.
* Full dependency readiness did not pass (status is PARTIAL, not READY).
* `dependency_readiness: READY` in the raw Gate-0 summary is incorrect for the complete FQ-SSIUU pipeline.
* NF4 was not tested because `bitsandbytes` was absent from the native Kaggle image.
* Genuinely missing packages requiring overlay installation: `bitsandbytes`, `captum`, `hydra-core`, `lm-eval`, `jsonlines`.
* TRL classification: `ABSENT — NO CURRENT IMPORT FOUND — REMOVAL DECISION PENDING`.
* OmegaConf (version 2.3.0) is present and importable in the native Kaggle image and is removed from the unresolved package list.
* Native importable versions recorded from source of truth (`gate0_package_inventory.json` / `pip_freeze.log`):
  * torch `2.10.0+cu128`
  * torchvision `0.25.0+cu128`
  * torchaudio `2.10.0+cu128`
  * triton `3.6.0`
  * transformers `5.0.0`
  * datasets `5.0.0`
  * accelerate `1.13.0`
  * peft `0.19.1`
  * huggingface-hub `1.11.0`
  * numpy `2.0.2`
  * scipy `1.16.3`
  * pandas `2.3.3`
  * omegaconf `2.3.0`
  * matplotlib `3.10.0`
  * seaborn `0.13.2`
  * typer `0.24.2`
  * rich `13.9.4`
  * tqdm `4.67.3`
  * pyyaml `6.0.3`
  * pytest `8.4.2`
  * ruff `0.15.11`
  * python-dotenv `1.2.2`
  * google-api-python-client `2.194.0`
  * google-auth `2.47.0`
* CUDA version semantics documented separately:
  * NVIDIA driver: `580.159.04`
  * Maximum CUDA compatibility reported by `nvidia-smi`: `13.0`
  * CUDA version used to build PyTorch: `12.8`
  * PyTorch build: `2.10.0+cu128`
  * cuDNN integer reported by PyTorch: `91002`
* Provenance and Credential Corrections:
  * Placeholder timestamps (12:00:00Z and 13:00:00Z on 2026-09-04) in commit `5045cd59964adcab6e089d570bb00e077c2e1fb4` are replaced with qualified provenance records. Specifically, `kernel_submission_timestamp` is set to `null` because `2026-09-04T18:52:35Z` originated from an untracked helper script (`process_results.py` line 50) rather than a committed primary artifact; it is recorded as `reported_submission_timestamp`: `2026-09-04T18:52:35Z` with evidence quality `UNVERIFIED — REPORTED IN LOCAL ORCHESTRATION TRANSCRIPT`. `probe_runtime_timestamp`: `2026-09-04T18:52:48.388417+00:00` (evidence: raw Gate-0 JSON artifacts), `artifact_retrieval_timestamp`: `null` (evidence: not recoverable from preserved orchestration evidence).
  * Statement claiming rotation of a Kaggle token and WandB key is corrected: the actual documented incident involved a GitHub PAT in the Git remote URL and a hardcoded Hugging Face token in a handoff file, both subsequently rotated with remote sanitized and hardcoded tokens removed.
* Compatibility of proposed overlay versions remains unresolved until Stage 1I validation in Kaggle.

## Status
`DOCUMENTED — IMPLEMENTATION PENDING`

## Severity
`HIGH — blocks reproducible new experiments but does not invalidate historical pilot observations`
