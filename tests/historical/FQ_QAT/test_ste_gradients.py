"""
tests/test_ste_gradients.py
CRITICAL: Validates STE backward pass correctness analytically.
Must pass before any FQ-QAT experiment begins.
These tests run on CPU — no GPU required.
"""

import pytest
import torch

from experiments.reproductions.FQ_QAT.historical.activation_int4 import compute_scale, fake_quantise

pytestmark = pytest.mark.cpu  # These run in CI


class TestSTEGradients:
    """STE gradient correctness — CRITICAL tests."""

    def test_fake_quantise_forward_equals_qdq(self):
        """FakeQuantiseNF4 forward pass must equal round(x/scale)*scale."""
        x = torch.randn(50)
        scale = compute_scale(x)
        x_fq = fake_quantise(x, scale)

        # Manual QDQ computation
        expected = torch.clamp(torch.round(x / scale), -8, 7) * scale
        assert torch.allclose(x_fq, expected, atol=1e-6), (
            f"FQ output does not match manual QDQ. Max diff: {(x_fq - expected).abs().max()}"
        )

    def test_ste_backward_is_identity(self):
        """STE backward must return grad_output unchanged (identity pass-through)."""
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        loss = fake_quantise(x).sum()
        loss.backward()
        assert torch.equal(x.grad, torch.ones(2)), (
            f"STE backward not identity: expected ones, got {x.grad}"
        )

    def test_ste_does_not_create_nan_gradients(self):
        """STE backward must not produce NaN under normal input ranges."""
        x = torch.zeros(4, requires_grad=True)
        out = fake_quantise(x)
        out.sum().backward()
        assert not torch.isnan(x.grad).any(), f"NaN gradients detected: {x.grad}"

    def test_qdq_error_is_below_bin_width(self):
        """Quantisation error per parameter must be < scale/2 (half a bin)."""
        x = torch.randn(100)
        scale = compute_scale(x)
        x_fq = fake_quantise(x, scale)
        error = torch.abs(x_fq - x)
        bound = scale / 2 + 1e-6
        assert (error <= bound).all(), (
            f"QDQ error exceeds bound. Max error: {error.max()}, bound: {bound}"
        )
