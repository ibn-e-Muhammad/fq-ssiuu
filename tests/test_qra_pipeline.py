"""
tests/test_qra_pipeline.py
Unit tests for the Quantisation Recovery Attack (QRA) pipeline.
Validates the BF16 → INT4 quantisation and metric comparison pipeline.
"""
import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = [
    pytest.mark.requires_gpu,  # Requires GPU for quantisation
    pytest.mark.skipif(not HAS_TORCH, reason="torch not installed"),
]


class TestQRAPipeline:
    """QRA pipeline correctness tests."""

    def test_quantise_dequantise_roundtrip(self):
        """Quantise-dequantise should produce values within NF4 bin width of originals."""
        pytest.skip("Implementation required in pilot phase")

    def test_qra_metric_delta_positive_for_unlearned_model(self):
        """Q-INT4 metric should show positive delta for a model that has been unlearned."""
        pytest.skip("Implementation required in pilot phase")

    def test_qra_metric_near_zero_for_oracle(self):
        """Oracle model (never saw forget set) should have near-zero QRA metric."""
        pytest.skip("Implementation required in pilot phase")

    def test_nf4_bin_width_matches_config(self):
        """The NF4 bin width used in QRA should match configs/base.yaml."""
        pytest.skip("Implementation required in pilot phase")
