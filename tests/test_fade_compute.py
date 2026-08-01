"""
tests/test_fade_compute.py
Unit tests for FADE (Forget Accuracy Distance Evaluation) metric computation.
Validates statistical correctness of the KS-test-based FADE metric.
"""
import pytest

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

pytestmark = [
    pytest.mark.cpu,  # These run in CI
    pytest.mark.skipif(not HAS_NUMPY, reason="numpy not installed"),
]


class TestFADECompute:
    """FADE metric correctness tests."""

    def test_fade_identical_distributions_yields_one(self):
        """FADE between identical distributions should be 1.0 (perfect unlearning)."""
        pytest.skip("Implementation required in pilot phase")

    def test_fade_disjoint_distributions_yields_zero(self):
        """FADE between completely disjoint distributions should approach 0.0."""
        pytest.skip("Implementation required in pilot phase")

    def test_fade_is_symmetric(self):
        """FADE(A, B) == FADE(B, A)."""
        pytest.skip("Implementation required in pilot phase")

    def test_fade_bounded_zero_one(self):
        """FADE metric must always be in [0, 1]."""
        pytest.skip("Implementation required in pilot phase")
