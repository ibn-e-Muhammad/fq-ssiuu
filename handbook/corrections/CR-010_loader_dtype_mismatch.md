# CR-010: Model Loader Dtype Mismatch and Configuration Discrepancy

| Field | Value |
|-------|-------|
| Status | OPEN — DEFERRED RUNTIME-CODE BLOCKER |
| Date discovered | 2026-09-05 |
| Affected files | `src/fq_ssiuu/models/loader.py`, `configs/base.yaml` |
| Severity | HIGH — configuration mismatch blocks model training until resolved |

## Problem Summary

Static audit of the model loading infrastructure revealed multiple configuration and documentation discrepancies:

1. **Precision Dtype Mismatch:** `configs/base.yaml` specifies FP16 (`torch_dtype: float16`), which is aligned with the canonical Kaggle Tesla T4 single-device environment. However, `ModelConfig` dataclass in `src/fq_ssiuu/models/loader.py` defaults to `bfloat16` (`torch_dtype: str = "bfloat16"`).
2. **Deprecated Script Reference:** The module docstring in `src/fq_ssiuu/models/loader.py` still references the deprecated script `scripts/setup_env.sh`.
3. **Device Placement Enforcement:** `device_map="auto"` in `loader.py` is acceptable only when `CUDA_VISIBLE_DEVICES=0` has already hidden the second GPU.

## Binding Resolution Directive

- `src/fq_ssiuu/models/loader.py` MUST NOT be modified during Stage 1H-C.
- This mismatch is registered as an explicit blocker for new experiments.
- It must be repaired and verified with dedicated unit tests in a future stage before model training or evaluation is initiated.
