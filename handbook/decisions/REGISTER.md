# FQ-SSIUU Decision Register

**Purpose:** Record all design and methodology decisions with rationale.

---

## DL-013: Rename Q-INT4 to Relative Precision Sensitivity

| Field | Value |
|-------|-------|
| Date | 2026-09-03 |
| Decision | Rename the metric from 'Q-INT4' to 'relative_precision_sensitivity' |
| Rationale | 'Q-INT4' implies a specific attack metric. The actual computation is |FQ_NF4 - FQ_BF16| / FQ_BF16, which measures unsigned relative change in Forget Quality across precisions. This is a sensitivity measurement, not a directional recovery metric. |
| Impact | Schema v2 in certificate.py; correction register CR-008 |

## DL-014: Certificate → Empirical Durability Report

| Field | Value |
|-------|-------|
| Date | 2026-09-03 |
| Decision | Rename 'Unlearning Durability Certificate' to 'Empirical Durability Report' |
| Rationale | 'Certificate' implies formal mathematical or legal guarantees. The output is an empirical measurement report with no such guarantees. |
| Impact | Display text in certificate.py; all documentation; v2 schema field 'threshold_met' replaces 'certified' |

## DL-015: Versioned Report Schema

| Field | Value |
|-------|-------|
| Date | 2026-09-03 |
| Decision | Introduce schema_version field ('v1' legacy, 'v2' current) |
| Rationale | Pilot artifacts P1-P4 use v1 field names. Abruptly removing them would break loading. v2 adds accurate field names; v1 aliases preserved as deprecated properties. |
| Impact | certificate.py load_certificate() migrates v1 → v2 on load |

## DL-016: QUAIL Margin Discrepancy Documented

| Field | Value |
|-------|-------|
| Date | 2026-09-03 |
| Decision | Document that implemented QUAIL margin (4.2e-4) differs from published (0.5) |
| Rationale | Published QUAIL (arXiv:2601.15538) uses Δq=1.0, margin=Δq/2=0.5. Implementation uses NF4 bin width / 2 = 4.2e-4. This is a 1190× difference. P4 results are from an incomplete surrogate. |
| Impact | C8 condition blocked until resolved; correction CR-005 |

## DL-017: Kaggle Tesla T4 Runtime, with Single-Device FP16 Resource Enforcement

| Field | Value |
|-------|-------|
| Date | 2026-09-05 |
| Decision | Kaggle Tesla T4 runtime, with single-device FP16 resource enforcement becomes the canonical environment for new experiments. |
| Rationale | Two Tesla T4 GPUs were visible in Gate-0 (each exposing 15,636,037,632 bytes, ~14.56 GiB). The research budget (<16 GB VRAM) is evaluated per selected device. New experiments must default to single-device `cuda:0` only. The second T4 GPU must not be used for model sharding, distributed training, or aggregate VRAM pooling unless a separate pre-registered multi-GPU condition explicitly permits it. Although PyTorch `is_bf16_supported()` returns `true`, Tesla T4 (Turing compute capability 7.5) lacks native hardware BF16 Tensor Core execution, so PyTorch's `is_bf16_supported()` result must not be interpreted as proof of native T4 BF16 execution. FP16 remains the canonical training and evaluation dtype. |
| Impact | All new experiments require Kaggle evidence, single-device `cuda:0` enforcement, and FP16 dtype. |

## DL-018: Protect the Kaggle-Native Compute Stack

| Field | Value |
|-------|-------|
| Date | 2026-09-05 |
| Decision | Torch, torchvision, torchaudio, Triton and CUDA components supplied by Kaggle must not be replaced without an explicit compatibility study. |
| Rationale | Kaggle environments are highly optimized. Resolver plans must be inspected before installation, and installation must abort if protected packages would change. |
| Impact | Scripts must check dry-run pip reports before proceeding. |

## DL-019: Research Artifact Storage Policy

| Field | Value |
|-------|-------|
| Date | 2026-09-05 |
| Decision | Small sanitized audit/evidence artifacts may be committed. Large checkpoints, datasets and experiment outputs remain external. |
| Rationale | External artifacts require immutable hashes and storage references. Chat messages are not the canonical research record. Every experiment must map to commit SHA, run ID, configuration and artifact manifest. |
| Impact | Large artifacts must use references, evidence JSONs/Manifests are tracked. |

## DL-020: Decoupled Kaggle Overlay Dependency Architecture

| Field | Value |
|-------|-------|
| Date | 2026-09-05 |
| Decision | Decouple Kaggle-native baseline packages from candidate overlays, development tools, and optional integrations. |
| Rationale | Preserves native PyTorch/Triton/CUDA compute stack while enabling essential scientific dependencies (bitsandbytes, captum, hydra-core, lm-eval, jsonlines). |
| Impact | Defined in requirements/ tree. Dry-run pip reports are verified against baseline before any installation. |

## DL-021: Non-Mutating Pip Report Verification Guard

| Field | Value |
|-------|-------|
| Date | 2026-09-05 |
| Decision | Mandate scripts/verify_pip_report.py validation of dry-run pip reports before any overlay installation. |
| Rationale | Automatically rejects any proposed version change, replacement, or modification to hard-protected or native-stability packages. |
| Impact | Prevents silent environment mutation or package downgrades. Setup scripts exit non-zero on violation. |

## DL-022: Removal of TRL Dependency

| Field | Value |
|-------|-------|
| Date | 2026-09-05 |
| Decision | Remove trl from active project dependencies. |
| Rationale | Static AST import audit across src/, scripts/, experiments/, and tests/ confirmed zero executable imports or function calls for trl. |
| Impact | trl is excluded from requirements/kaggle-overlay.in and pyproject.toml. |
