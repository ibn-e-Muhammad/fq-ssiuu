"""
tests/test_ste_gradients.py
CRITICAL: Validates STE backward pass correctness analytically.
Must pass before any FQ-QAT experiment begins.
These tests run on CPU — no GPU required.
"""
import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = [
    pytest.mark.cpu,  # These run in CI
    pytest.mark.skipif(not HAS_TORCH, reason="torch not installed"),
]


class TestSTEGradients:
    """STE gradient correctness — CRITICAL tests."""

    def test_fake_quantise_forward_equals_qdq(self):
        """FakeQuantiseNF4 forward pass must equal round(x/scale)*scale."""
        pytest.skip("Implementation required — Task 13 of Stage 1; test stubs only")

    def test_ste_backward_is_identity(self):
        """STE backward must return grad_output unchanged (identity pass-through)."""
        pytest.skip("Implementation required")

    def test_ste_does_not_create_nan_gradients(self):
        """STE backward must not produce NaN under normal input ranges."""
        pytest.skip("Implementation required")

    def test_qdq_error_is_below_bin_width(self):
        """Quantisation error per parameter must be < NF4 bin width (8.4e-4)."""
        pytest.skip("Implementation required")
