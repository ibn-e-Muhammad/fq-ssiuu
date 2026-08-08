# DL-008: Upgrade bitsandbytes from 0.42.0 to 0.45.5+

**Date:** August 2026
**Status:** DECIDED — PERMANENT
**Author:** Research Team (GATE-0 session)

---

## Context

During GATE-0 environment verification, bitsandbytes 0.42.0 (pinned in the original
`requirements.txt`) failed to import with a `ModuleNotFoundError` for `triton.ops`.

## Decision

Upgrade bitsandbytes from 0.42.0 to ≥0.45.5. In practice, pip resolves this to
0.50.0 on the August 2026 Colab/Kaggle environments.

## Reason

bitsandbytes 0.42.0 imports `triton.ops`, a module that was removed in triton 3.x.
Google Colab and Kaggle now ship with triton 3.x as part of the PyTorch 2.11+
distribution. The internal API change in triton is not backwards-compatible.

bitsandbytes 0.45.5 is the first version that:
1. Removes the `triton.ops` dependency entirely
2. Adds CUDA 13.0 support (required by Kaggle's PyTorch 2.11.0+cu128)

## Evidence

GATE-0 failure log (Colab T4, August 2026):
```
ModuleNotFoundError: No module named 'triton.ops'
```

Traceback originated from `bitsandbytes/autograd/_functions.py` which attempted
`from triton.ops import _matmul as triton_matmul`.

## Consequence

1. `requirements.txt` updated: `bitsandbytes>=0.45.5` (replaces `bitsandbytes==0.42.0`)
2. Version pins for `torch`, `torchvision`, `torchaudio` REMOVED to prevent conflicts
   with pre-installed PyTorch on Colab/Kaggle
3. All Kaggle/Colab setup scripts explicitly install `bitsandbytes==0.45.5` after
   the main `pip install -e .` to override any resolution to older versions

## Impact on Results

bitsandbytes 0.50.0 uses the same NF4 quantisation algorithm as 0.42.0. The INT4
codebook values are identical. No impact on model quality or quantisation fidelity.
