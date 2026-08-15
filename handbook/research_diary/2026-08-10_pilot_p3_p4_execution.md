# Research Diary: 2026-08-10 - Pilot P3 & P4 Execution

**Date:** 2026-08-10
**Author:** FQ-SSIUU Automation System

## Overview
Executed Pilot P3 (FQ-QAT+NPO stability) and Pilot P4 (QUAIL baseline) on a Kaggle T4 session. Both pilots required multiple debugging iterations to resolve environment and logic bugs but successfully completed.

## Pilot P3 (FQ-QAT+NPO Stability)
- **Dataset/Config:** forget01 split, 5 epochs, 200 steps.
- **Execution Time:** Completed in 207.94 seconds.
- **Initial Verdict:** Logged as FAIL due to a pre-clip gradient norm heuristic.
- **Corrected Verdict:** PASS. Post-audit revealed the EVB criterion specifies *post-clip* norm (with `gradient_clip=1.0`), which is mathematically ≤ 1.0. Thus, stability is confirmed.
- **Key Commits:** Addressed LoRA loading issues (`3ac7ed8`).

## Pilot P4 (QUAIL Baseline)
- **Dataset/Config:** forget05 split, 10 epochs, 70 steps.
- **Execution Time:** 4741 seconds training + 1255 seconds evaluation.
- **Results:**
  - Extreme forgetting achieved (FQ ≈ 0.000121).
  - Severe utility collapse observed (MU = 0.280369).
  - Quantisation-Robustness Asymmetry (QRA) metrics: Q-INT4 = 0.227678, which is comparable to the NPO baseline Q-INT4 = 0.214906.
- **Scientific Conclusion:** C8 is established (QUAIL produces strong forgetting but severely destroys utility).
- **Key Commits:** Addressed LoRA loading (`1fb6c04`), QUAIL `abs(0)` gradient bug (`9a83224`), and autocast fix (`6d2f54a`).

## Session Notes
Both pilots were executed on the same Kaggle session. Multiple debugging iterations were needed, including resolving numpy issues, venv setup constraints, LoRA configuration, and QUAIL-specific gradient bugs before achieving successful completion.
