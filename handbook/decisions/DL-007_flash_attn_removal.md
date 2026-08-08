# DL-007: Remove flash-attn and Adopt SDPA

**Date:** August 2026
**Status:** DECIDED — PERMANENT
**Author:** Research Team (GATE-0 session)

---

## Context

During GATE-0 environment verification on Google Colab T4, the `flash-attn` package
failed to load with an `undefined symbol` error from `flash_attn_2_cuda.so`.

## Decision

Permanently remove `flash-attn` from the project. Use `attn_implementation="sdpa"`
(PyTorch native Scaled Dot-Product Attention) for all model loading calls.

## Reason

The Tesla T4 GPU is Turing architecture (Compute Capability 7.5). Flash Attention 2.x
requires Ampere architecture (CC 8.0+) at minimum. The precompiled CUDA kernels in
`flash-attn` reference symbols that do not exist on the Turing SM75 ISA.

SDPA (introduced in PyTorch 2.0) provides automatic kernel selection — it will use
the most efficient kernel available for the hardware (memory-efficient attention on T4,
FlashAttention-2 on Ampere+ when available). This makes the code portable across GPU
architectures with zero configuration.

## Evidence

GATE-0 failure log (Colab T4, August 2026):
```
ImportError: /usr/local/lib/.../flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so:
undefined symbol: ...
```

## Consequence

1. `flash-attn` removed from `requirements.txt`
2. `TRANSFORMERS_NO_FLASH_ATTN=1` set in all training/eval environments
3. All `from_pretrained()` calls must pass `attn_implementation="sdpa"`
4. `src/fq_ssiuu/models/loader.py` updated to use SDPA by default
5. `experiments/gate0_env_check.py` updated to verify SDPA instead of flash-attn
6. `scripts/setup_env.sh` no longer attempts flash-attn installation

## Impact on Results

SDPA on T4 uses the memory-efficient attention kernel, which is mathematically
equivalent to flash-attn but approximately 10-30% slower in wall-clock time.
This affects training speed but NOT model quality or evaluation metrics.
