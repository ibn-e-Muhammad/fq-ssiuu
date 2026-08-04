# Changelog

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
- GATE-0: environment verification on Colab L4
- GATE-1: OpenUnlearning baseline reproduction
- Pilot P1–P4: pilot experiments
- All experimental conditions C2–C8
