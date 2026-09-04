"""
src/fq_ssiuu/quantisation/fake_quant.py

Fake Quantisation with Straight-Through Estimator (STE) for FQ-QAT.

This is the most mathematically critical module in the project.
A wrong STE implementation produces silently incorrect gradients.

Reference: Bengio et al. 2013, "Estimating or Propagating Gradients Through
Stochastic Neurons for Conditional Computation", arXiv:1308.3432.

FORWARD PASS (Fake Quantisation / Quantise-Dequantise / QDQ):
    x_fq = round(x / scale) * scale
    where scale calibrates the weight range to an INT4-like grid.
    For symmetric INT4: scale = max(|x|) / 7
    This maps the value range +/-max(|x|) into the integer range [-8, 7].
    x_fq is what x will look like after INT4 quantisation and dequantisation.

BACKWARD PASS (Straight-Through Estimator):
    dL/dx = dL/dx_fq  (gradient passes through unchanged)
    The gradient of round() is zero almost everywhere (and undefined at integers).
    The STE approximates d(round(x))/dx ~ 1 (identity).
    This is valid when the quantisation error is small relative to the signal.

ASCII DIAGRAM:
    Forward:   x --> quantise --> x_quant --> dequantise --> x_fq --> loss
                     (round)                    (*scale)
    Backward:  dL/dx <-- dL/dx_fq  (STE: identity pass-through, skip round())

WHY THIS WORKS FOR UNLEARNING:
    Without FQ-QAT: the optimiser updates x by dx. If dx < scale (bin width),
    then round((x + dx)/scale) == round(x/scale), so the quantised model is
    UNCHANGED. This is the Quantisation Recovery Attack (QRA) mechanism:
    unlearning updates are too small to survive quantisation.

    With FQ-QAT: the loss is computed on x_fq (the quantised-then-dequantised x).
    The optimiser receives gradients through the STE that account for the
    quantisation rounding. It must find x such that round(x/scale) differs
    from round(x_original/scale) -- i.e., dx must cross a quantisation boundary.
    This forces the unlearning to make changes that survive INT4 rounding.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass
class FQConfig:
    """Configuration for fake quantisation during training.

    Attributes:
        nf4_scale: Approximate NF4 bin width. Used as a fixed scale in
            some training modes. Value 8.4e-4 is from MANSU (arXiv:2605.15138),
            measured as the mean bin width of the NF4 codebook.
        clip_threshold: STE clips gradients for values outside
            [-clip_threshold, clip_threshold]. Default 1.0 means no clipping
            beyond the quantisation range.
    """

    nf4_scale: float = 8.4e-4
    clip_threshold: float = 1.0


# Structured metadata for the quantisation-location research gate (CR-007).
# This documents what this quantiser does vs. deployment NF4 quantisation.
QUANTISER_META = {
    "target_type": "activation",  # QDQ applied to adapter output activations
    "codebook": "symmetric_int4",  # Uniform symmetric grid [-8, 7]
    "granularity": "per_tensor",  # Single scale for entire tensor
    "bit_width": 4,
    "deployment_equivalent": False,  # NOT equivalent to bitsandbytes NF4
    "ste_enabled": True,  # STE backward pass (identity)
    "scale_computation": "max_abs_div_7",  # scale = max(|x|) / 7
    "mismatch_with_deployment": (
        "Deployment uses bitsandbytes NF4: non-uniform codebook, per-block "
        "quantisation (block size configurable, typically 64), applied to "
        "merged model WEIGHTS (not activations). This module applies "
        "symmetric INT4 QDQ per-tensor to adapter OUTPUT activations. "
        "These are fundamentally different quantisation operations."
    ),
}


class _FakeQuantiseAutograd(torch.autograd.Function):
    """Custom autograd function implementing QDQ forward with STE backward.

    Forward: x_fq = clamp(round(x / scale), -8, 7) * scale
    Backward: dL/dx = dL/dx_fq (identity, STE)

    The STE approximation d(round(x))/dx ~ 1 is the standard approach
    from Bengio et al. 2013.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """QDQ forward pass.

        Args:
            ctx: Autograd context (nothing saved -- STE needs no state).
            x: Input tensor.
            scale: Quantisation scale (scalar tensor).

        Returns:
            x_fq: Fake-quantised tensor, same shape/dtype/device as x.
        """
        # Quantise: scale to integer range, round, clamp to INT4
        x_scaled = x / scale
        x_quant = torch.round(x_scaled)
        x_quant = torch.clamp(x_quant, -8, 7)  # INT4 range
        # Dequantise: scale back to original range
        x_fq = x_quant * scale
        return x_fq

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """STE backward pass: identity.

        Returns:
            Tuple[Tensor, None]: (grad_x, grad_scale).
            grad_x = grad_output (STE identity).
            grad_scale = None (scale has no gradient).
        """
        # STE: gradient passes through unchanged. This is the critical line.
        # If this returned anything other than grad_output, every FQ-QAT
        # experiment would produce silently wrong results.
        return grad_output, None


def compute_scale(x: torch.Tensor, symmetric: bool = True) -> torch.Tensor:
    """Compute per-tensor quantisation scale for INT4-like quantisation.

    For symmetric quantisation:
        scale = max(|x|) / 7

    This maps the weight range [-max(|x|), +max(|x|)] to the INT4
    integer range [-8, 7] (using 7 as the positive bound since INT4
    is asymmetric: [-8, 7]).

    Args:
        x: Input tensor (typically model weights).
        symmetric: If True, use symmetric quantisation. Only True is
            implemented (asymmetric NF4 requires the codebook, which
            is handled by bitsandbytes at deployment time).

    Returns:
        torch.Tensor: Scalar scale on the same device as x.

    Raises:
        ValueError: If symmetric is False (not implemented).
    """
    if not symmetric:
        raise ValueError(
            "Asymmetric quantisation requires the NF4 codebook. "
            "Use bitsandbytes for deployment-time NF4 quantisation. "
            "This function implements symmetric INT4-like quantisation "
            "for the FQ-QAT training approximation."
        )

    abs_max = torch.max(torch.abs(x))

    # Edge case: all-zero tensor -> scale = 1.0 to avoid division by zero
    if abs_max == 0:
        return torch.ones(1, device=x.device, dtype=x.dtype)

    scale = abs_max / 7.0
    return scale


def fake_quantise(x: torch.Tensor, scale: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Fake-quantise a tensor with STE backward pass.

    This is the main function used by the FQ-QAT training loop. It:
    1. Computes the quantisation scale (if not provided)
    2. Applies quantise-dequantise (QDQ) in the forward pass
    3. Passes gradients through unchanged (STE) in the backward pass

    Args:
        x: Input tensor. Must have requires_grad=True for gradient flow.
        scale: Optional pre-computed scale. If None, computed from x.

    Returns:
        torch.Tensor: Fake-quantised tensor, same shape as x.
            In the forward direction: x_fq = round(x/scale) * scale
            In the backward direction: grad flows through unchanged (STE)
    """
    if scale is None:
        scale = compute_scale(x)

    return _FakeQuantiseAutograd.apply(x, scale)


def test_ste_correctness():
    """Comprehensive self-test for the STE implementation.

    Tests ALL of the following properties:
    (a) Forward shape: fake_quantise(x) returns same shape as x
    (b) Forward QDQ error: |x_fq - x| <= scale/2 for all elements
    (c) Backward STE identity: grad is exactly torch.ones for sum loss
    (d) No NaN: fake_quantise(zeros) does not produce NaN
    (e) Device consistency: output device equals input device

    All 5 sub-checks must pass. Any failure indicates a critical bug.
    """
    # (a) Forward shape preservation
    x = torch.randn(10, 20)
    x_fq = fake_quantise(x)
    assert x_fq.shape == x.shape, (
        f"(a) Shape mismatch: input {x.shape}, output {x_fq.shape}"
    )
    print("  (a) Forward shape: PASS")

    # (b) QDQ error bound: |x_fq - x| <= scale/2 + epsilon
    # The maximum error of rounding to the nearest integer scaled by `scale`
    # is scale/2 (half a bin width).
    scale = compute_scale(x)
    error = torch.abs(x_fq - x)
    max_error = error.max().item()
    bound = (scale.item() / 2) + 1e-6
    assert max_error <= bound, (
        f"(b) QDQ error {max_error:.6f} exceeds bound {bound:.6f}"
    )
    print(f"  (b) QDQ error bound: PASS (max error={max_error:.6f}, bound={bound:.6f})")

    # (c) STE backward identity: gradient must be EXACTLY ones
    x_test = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    loss = fake_quantise(x_test).sum()
    loss.backward()
    expected_grad = torch.ones(3)
    assert torch.equal(x_test.grad, expected_grad), (
        f"(c) STE backward not identity: expected {expected_grad}, got {x_test.grad}"
    )
    print("  (c) STE backward identity: PASS")

    # (d) No NaN on zero input
    x_zero = torch.zeros(4, requires_grad=True)
    out = fake_quantise(x_zero)
    assert not torch.isnan(out).any(), "(d) NaN in forward output for zero input"
    out.sum().backward()
    assert not torch.isnan(x_zero.grad).any(), "(d) NaN in gradient for zero input"
    print("  (d) No NaN on zeros: PASS")

    # (e) Device consistency (CPU test -- GPU tested in GATE-0)
    x_cpu = torch.randn(5)
    x_fq_cpu = fake_quantise(x_cpu)
    assert x_fq_cpu.device == x_cpu.device, (
        f"(e) Device mismatch: input {x_cpu.device}, output {x_fq_cpu.device}"
    )
    print("  (e) Device consistency: PASS")

    print("  All 5 STE sub-checks PASSED.")
