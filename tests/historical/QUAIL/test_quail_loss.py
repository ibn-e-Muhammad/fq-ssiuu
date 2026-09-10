"""
tests/test_quail_loss.py
Unit tests for QUAIL loss re-implementation.
Validates hinge loss computation and margin correctness.
Reference: Mishra & Mehreen 2026, arXiv:2601.15538
"""

import pytest

try:
    import torch  # noqa: F401 — optional dependency availability probe

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = [
    pytest.mark.requires_model,  # Skipped in CI
    pytest.mark.skipif(not HAS_TORCH, reason="torch not installed"),
]


class TestQUAILLoss:
    """QUAIL hinge loss correctness tests."""

    def test_quail_hinge_loss_zero_when_margin_satisfied(self):
        """When logit gap exceeds margin, hinge loss should be zero."""
        pytest.skip("Implementation required in pilot phase")

    def test_quail_hinge_loss_positive_when_margin_violated(self):
        """When logit gap is below margin, hinge loss should be positive."""
        pytest.skip("Implementation required in pilot phase")

    def test_quail_margin_equals_half_nf4_step(self):
        """QUAIL margin must equal nf4_step / 2 per paper specification."""
        pytest.skip("Implementation required in pilot phase")

    def test_quail_loss_is_non_negative(self):
        """QUAIL loss must always be >= 0."""
        pytest.skip("Implementation required in pilot phase")
