# Kaggle Environment & Dependency Architecture

**Version:** 1.0
**Date:** 2026-09-05
**Status:** Differential consistency implemented; OG-001 runtime closure pending.

Global `pip check` is captured before installation, after installation, and again
by the runtime verifier. Exact stdout/stderr and return codes remain immutable
run artifacts. The shared policy parses pip's documented diagnostics and rejects
unknown output, malformed records, introduced diagnostics, and diagnostics
involving overlay roots or actual installed transitives. Remaining inherited
diagnostics are represented as `BASELINE_DIRTY_NO_REGRESSION`, never globally
clean. The complete distribution inventory covers direct-reference installs
that lack `==` pins in freeze output. Protected and unreported package changes
and native platform changes fail the gate.

After successful installation, consistency failure does not suppress runtime
smokes or after-state evidence. Locks are generated only after consistency,
required imports, one-T4 FP16/SDPA/genuine NF4 operations, peak memory and
protected/platform validation pass. The notebook independently verifies every
artifact hash and recomputes consistency, package transitions and lock deltas.

Policy grammar: [pip check documentation](https://pip.pypa.io/en/stable/cli/pip_check/)
and [pip CheckCommand](https://github.com/pypa/pip/blob/main/src/pip/_internal/commands/check.py),
consulted 2026-09-05. No native diagnostic text is hardcoded as an exemption.
**Canonical Hardware Target:** Kaggle Tesla T4 runtime, with single-device FP16 resource enforcement (`cuda:0`)

---

## 1. Executive Summary

FQ-SSIUU establishes Kaggle Tesla T4 as its canonical execution environment. To ensure strict scientific reproducibility without breaking Kaggle's highly optimized CUDA/PyTorch compute stack, we decouple the environment into two distinct tiers:

1. **Kaggle-Native Baseline (Immutable Compute Stack)**
   - Managed directly by Kaggle image updates.
   - Defined in `requirements/kaggle-native-baseline.json`.
   - Cannot be upgraded, downgraded, uninstalled, or replaced by package managers.

2. **Kaggle Scientific Overlay (Candidate Overlay Packages)**
   - Defined in `requirements/kaggle-overlay.in`.
   - Contains only essential scientific packages absent from the native Kaggle image.
   - Validated against a non-mutating pip dry-run report guard (`scripts/verify_pip_report.py`) before installation.

---

## 2. Dependency Taxonomy & Classification

| Taxonomy Category | Packages & Versions | Enforcement & Policy |
|-------------------|---------------------|----------------------|
| **Hard-Protected Compute Stack** | `torch==2.10.0+cu128`, `torchvision==0.25.0+cu128`, `torchaudio==2.10.0+cu128`, `triton==3.6.0` | **Hard Block:** Any proposed change triggers immediate setup abort. |
| **Native-Stability Stack** | `transformers==5.0.0`, `datasets==5.0.0`, `accelerate==1.13.0`, `peft==0.19.1`, `huggingface-hub==1.11.0`, `numpy==2.0.2`, `scipy==1.16.3`, `pandas==2.3.3`, `omegaconf==2.3.0`, `tokenizers==0.22.2`, `safetensors==0.7.0` | **Protected Baseline:** Must remain unchanged; changes require explicit compatibility review. |
| **Kaggle Overlay Candidates** | `bitsandbytes>=0.45.0,<0.46.0`, `captum>=0.7.0,<0.8.0`, `hydra-core>=1.3.2,<1.4.0`, `lm-eval>=0.4.7,<0.5.0`, `jsonlines>=4.0.0,<5.0.0` | **Candidate Ranges:** Installed with `--no-deps` for editable project core. Validated in Stage 1I. |
| **Development Dependencies** | `pytest>=8.0.0`, `pytest-cov>=4.1.0`, `ruff>=0.15.0`, `mypy>=1.8.0` (`requirements/dev.in`) | **Dev Only:** Excluded from production Kaggle scientific overlay. |
| **Optional Drive Integration** | `google-api-python-client`, `google-auth`, `google-auth-oauthlib`, `google-auth-httplib2` (`requirements/optional-drive.in`) | **Optional:** Excluded from Kaggle overlay unless explicit export requested. |
| **Obsolete / Removed Dependencies** | `trl` (Removed: zero executable imports found across codebase) | **Removed:** Preserved in historical Colab constraints file `requirements/legacy/colab-aug-2026-constraints.txt`. |

---

## 3. Non-Mutating Verification Guard Architecture

```
                      +-----------------------------------+
                      |   Kaggle T4 Environment Baseline  |
                      | (requirements/kaggle-native-json) |
                      +-----------------------------------+
                                        |
                                        v
+-----------------------------+     +-------------------------------+     +---------------------------+
|  requirements/              | --> |  pip install --dry-run        | --> |  scripts/                 |
|  kaggle-overlay.in          |     |  --report report.json         |     |  verify_pip_report.py     |
+-----------------------------+     +-------------------------------+     +---------------------------+
                                                                                    |
                                                          +-------------------------+-------------------------+
                                                          |                                                   |
                                                          v                                                   v
                                                 [Violation Detected]                                 [Validation Passed]
                                                          |                                                   |
                                                          v                                                   v
                                                   ABORT SETUP IMMEDIATELY                             Proceed to Overlay Install
                                                   (Exit Code != 0)                                    (Only if --apply specified)
```

---

## 4. TRL Dependency Removal Decision

Static inspection of the entire codebase (`src/`, `scripts/`, `experiments/`, `tests/`, and notebooks) revealed zero executable imports (`import trl` or `from trl import ...`). TRL was previously listed in historical Colab requirements but has no current functional role in FQ-SSIUU.

**Decision:** Remove `trl` from active project dependencies (`pyproject.toml` and `kaggle-overlay.in`).

---

## 5. Lockfile Strategy

- **No Full Environment Lockfile:** Blind `pip freeze` lockfiles are prohibited across heterogeneous systems because the Kaggle-native compute stack is separately fingerprinted.
- **Stage 1I Overlay Lock:** Upon successful installation and runtime verification in Kaggle during Stage 1I, an overlay-only lockfile (`kaggle-overlay.lock`) will be generated inside the run-specific output directory (`results/kaggle_setup_<run_id>/`).

---

## 6. Execution Workflow & Operational Gate Status

- **Default Mode:** `CUDA_VISIBLE_DEVICES=0 python3 scripts/kaggle_setup.py --preview` defaults to `PREVIEW` mode and does not alter the Python environment.
- **Explicit Apply:** Installation requires `CUDA_VISIBLE_DEVICES=0 python3 scripts/kaggle_setup.py --apply`.
- **Operational Gate Status:** OG-001 remains **`PARTIAL`** until Stage 1I execution on Kaggle validates runtime `bitsandbytes` NF4 operations and overlay compatibility.
- **Scientific Confirmation:** No scientific experiment, model training, QRA evaluation, or attribution calculation ran during Stage 1H-B.
