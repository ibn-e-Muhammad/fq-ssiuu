"""
tests/test_quantisation_location.py

Quantisation location audit test (RG-002, CR-007).

Verifies and documents what tensor the current FQ-QAT implementation
quantises. This test does NOT require a GPU.
"""


class TestQuantiserMetadata:
    """Tests for quantiser metadata documentation."""

    def test_fake_quant_metadata_exists(self):
        """QUANTISER_META should exist in fake_quant module."""
        from experiments.reproductions.FQ_QAT.historical.activation_int4 import QUANTISER_META

        assert isinstance(QUANTISER_META, dict)

    def test_quantiser_targets_activation(self):
        """Current quantiser should document that it targets activations."""
        from experiments.reproductions.FQ_QAT.historical.activation_int4 import QUANTISER_META

        assert QUANTISER_META["target_type"] == "activation"

    def test_quantiser_not_deployment_equivalent(self):
        """Current quantiser should document it is NOT deployment-equivalent."""
        from experiments.reproductions.FQ_QAT.historical.activation_int4 import QUANTISER_META

        assert QUANTISER_META["deployment_equivalent"] is False

    def test_quantiser_codebook(self):
        """Current quantiser uses symmetric INT4, not NF4."""
        from experiments.reproductions.FQ_QAT.historical.activation_int4 import QUANTISER_META

        assert QUANTISER_META["codebook"] == "symmetric_int4"

    def test_quantiser_bit_width(self):
        """Current quantiser is 4-bit."""
        from experiments.reproductions.FQ_QAT.historical.activation_int4 import QUANTISER_META

        assert QUANTISER_META["bit_width"] == 4

    def test_fq_hooks_metadata_exists(self):
        """FQ_HOOKS_META should exist in fq_hooks module."""
        from experiments.reproductions.FQ_QAT.historical.activation_lora import FQ_HOOKS_META

        assert isinstance(FQ_HOOKS_META, dict)
        assert FQ_HOOKS_META["deployment_equivalent"] is False

    def test_mismatch_documented(self):
        """Deployment mismatch description should be non-empty."""
        from experiments.reproductions.FQ_QAT.historical.activation_int4 import QUANTISER_META

        assert "mismatch_with_deployment" in QUANTISER_META
        assert len(QUANTISER_META["mismatch_with_deployment"]) > 50
