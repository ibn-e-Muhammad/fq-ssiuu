# Changelog

## 2026-09-10 — scientific reconciliation A

- Separate NPO, QUAIL, SSIUU, FQ_QAT and QRA workspaces; preserve historical
  surrogates, notebooks and raw results. One current Kaggle Mother entry point.
- Add causal NPO candidate with CPU parity against both pinned upstreams.
- Retire p-value recovery scores and inverted certification; descriptive report
  decisions stay null. Register blocking RG-005/006.
- Default loading to FP16/device 0, remove nonexistent CLI, prevent Colab setup on import.

The historical entries below record what was claimed at the time, **not accepted
current scientific conclusions**. CR-012 supersedes P1/P4 attack and robustness
claims. Gate 1 is a historical baseline attempt; P2 is offline hidden-layer IG; P3
is an activation-INT4/token-mean surrogate; P4 is hinge-only. No faithful corrected
pilot or full hypothesis test is claimed. The old Stage 2 names do not establish
published SSIUU, QUAIL or weight-NF4 fidelity. Their implementations now live in
method `historical/` folders; see the current method matrix.

## Historical entries (superseded interpretation)

## [Unreleased — Stage 1]

### Added
- Complete repository scaffold and project infrastructure
- Folder structure matching Master Research Engineering Handbook Stage 1
- requirements.txt and pyproject.toml
- Configuration files for all experimental conditions (C2, C4, C5, C6, C8)
- scripts/setup_env.sh, colab_mount.py, sync_to_drive.sh
- Makefile with common development targets
- .gitignore with comprehensive exclusions
- README.md with project overview and quick start
- GitHub Actions CI workflow (see .github/workflows/ci.yml)

## [Unreleased — Stage 2]

### Added
- src/fq_ssiuu/models/loader.py — model loading with NF4, BF16, LoRA, merge support
- src/fq_ssiuu/unlearning/npo_loss.py — NPO loss + KL retain loss + self-test
- src/fq_ssiuu/unlearning/quail_loss.py — QUAIL logit hinge loss + self-test
- src/fq_ssiuu/quantisation/fake_quant.py — FakeQuantiseNF4 STE implementation + 5-check self-test
- src/fq_ssiuu/attribution/ssiuu_reg.py — SSIUU regulariser + gradient-norm fallback + self-test
- src/fq_ssiuu/evaluation/qra_pipeline.py — Three-precision QRA evaluation pipeline
- src/fq_ssiuu/evaluation/certificate.py — JSON durability certificate generation
- tests/test_ste_gradients.py — Real CPU unit tests for STE correctness (4 tests)
- tests/test_npo_loss.py — Real CPU unit tests for NPO loss (3 tests)
- tests/test_attribution.py — Real CPU test for SSIUU zero-at-init property (3 tests)
- handbook/GEMINI_HANDOFF.md — Complete handoff document for Gemini 3.1 Pro

### Mathematical Self-Tests Passing
- NPO loss: closed-form value at ratio=0 confirmed ((2/β)·log(2) = 13.8629)
- QUAIL loss: hinge behaviour at margin confirmed (loss=0 at margin, loss=margin at zero)
- STE: identity backward pass confirmed (all 5 sub-checks)
- SSIUU: zero-loss at initialisation confirmed

### Pending (Gemini's Responsibility)
- All experimental conditions (E0b, E4, E5, E6, E7, GATE-2)

## [Unreleased — Pilots]

### Completed
- **GATE-0 / GATE-1**: OpenUnlearning baseline reproduced.
- **Pilot P1**: QRA Vulnerability Baseline established (Q-INT4 = 0.2149).
- **Pilot P2**: Integrated Gradients (IG) attribution feasible on Llama-3.2-1B.
- **Pilot P3**: FQ-QAT + NPO training confirmed numerically stable.
- **Pilot P4**: QUAIL Baseline established (Q-INT4 = 0.2276). QUAIL shown to be less quantisation-robust than NPO.
