# FQ-SSIUU Dependency Architecture

This directory defines the decoupled dependency architecture for FQ-SSIUU, separating Kaggle-native baseline packages from candidate overlays, development tools, and optional integrations.

## Architecture & Taxonomy

1. **Hard-Protected Compute Stack (`kaggle-native-baseline.json` -> `hard_protected_compute_stack`)**
   - `torch==2.10.0+cu128`
   - `torchvision==0.25.0+cu128`
   - `torchaudio==2.10.0+cu128`
   - `triton==3.6.0`
   - *Policy:* Immutable. Cannot be upgraded, downgraded, replaced, or uninstalled.

2. **Native-Stability Stack (`kaggle-native-baseline.json` -> `native_stability_stack`)**
   - `transformers==5.0.0`, `datasets==5.0.0`, `accelerate==1.13.0`, `peft==0.19.1`, `huggingface-hub==1.11.0`, `numpy==2.0.2`, `scipy==1.16.3`, `pandas==2.3.3`, `omegaconf==2.3.0`, `tokenizers==0.22.2`, `safetensors==0.7.0`
   - *Policy:* Protected baseline. Must remain unchanged unless an explicit compatibility study is conducted.

3. **Kaggle Overlay Candidates (`kaggle-overlay.in`)**
   - `bitsandbytes>=0.45.0,<0.46.0` (Source: https://pypi.org/project/bitsandbytes/ | Retrieved: 2026-09-05 | Reason: NF4 quantisation / QAT pipeline)
   - `captum>=0.7.0,<0.8.0` (Source: https://pypi.org/project/captum/ | Retrieved: 2026-09-05 | Reason: Integrated Gradients attribution calculation)
   - `hydra-core>=1.3.2,<1.4.0` (Source: https://pypi.org/project/hydra-core/ | Retrieved: 2026-09-05 | Reason: Hierarchical experiment condition configuration)
   - `lm-eval>=0.4.7,<0.5.0` (Source: https://pypi.org/project/lm-eval/ | Retrieved: 2026-09-05 | Reason: Benchmark evaluation harness for TOFU/QRA)
   - `jsonlines>=4.0.0,<5.0.0` (Source: https://pypi.org/project/jsonlines/ | Retrieved: 2026-09-05 | Reason: JSONL dataset formatting and eval output reading)
   - *Status:* Candidate ranges pending Stage 1I Kaggle execution and runtime verification.

4. **Development Dependencies (`dev.in`)**
   - `pytest`, `pytest-cov`, `ruff`, `mypy`. Testing and linting tools only.

5. **Optional Drive Dependencies (`optional-drive.in`)**
   - `google-api-python-client`, `google-auth`, `google-auth-oauthlib`, `google-auth-httplib2`. Drive export integration only.

6. **Historical / Obsolete Dependencies (`legacy/colab-aug-2026-constraints.txt`)**
   - Preserved August 2026 Colab environment constraints for historical audit reference.
   - `trl` decision: Removed from active overlay requirements because static import audit confirmed zero executable imports across `src/`, `scripts/`, `experiments/`, and `tests/`.

7. **Locks Directory (`locks/`)**
   - Contains lock strategy guidelines. Overlay lock generated in Stage 1I.
