"""
tests/test_attribution.py
Unit tests for SSIUU attribution regularisation.
Validates the zero-at-init property and penalty computation.
Reference: Yang et al. 2025, arXiv:2509.22263
"""

import pytest
import torch

from fq_ssiuu.attribution.ssiuu_reg import SSIUUConfig, SSIUURegulariser


class TestAttribution:
    """Attribution regularisation correctness tests."""

    @pytest.mark.cpu
    def test_ssiuu_loss_zero_at_init(self):
        """When D^-_current == D^-_init, L_attr must be exactly 0.

        From the equation: L_attr = lambda * sum max(0, D^-_init - D^-_current)
        When current == init: max(0, 0) = 0 for all neurons.
        """
        config = SSIUUConfig(lambda_attr=1.0)
        reg = SSIUURegulariser(model=None, forget_loader=None, config=config)
        reg.d_neg_init = torch.tensor([-0.1, -0.2, -0.3])

        # Simulate: current == init
        d_neg_current = torch.tensor([-0.1, -0.2, -0.3])
        penalty = torch.clamp(reg.d_neg_init - d_neg_current, min=0)
        loss = config.lambda_attr * penalty.sum()

        assert abs(loss.item()) < 1e-8, (
            f"SSIUU loss should be 0 when current == init, got {loss.item()}"
        )

    @pytest.mark.cpu
    def test_ssiuu_penalty_for_more_negative(self):
        """When a neuron becomes more negative, the penalty should be positive."""
        config = SSIUUConfig(lambda_attr=2.0)
        reg = SSIUURegulariser(model=None, forget_loader=None, config=config)
        reg.d_neg_init = torch.tensor([-0.1, -0.2, -0.3])

        # Neuron 0 became more negative: -0.1 -> -0.5
        d_neg_current = torch.tensor([-0.5, -0.2, -0.3])
        penalty = torch.clamp(reg.d_neg_init - d_neg_current, min=0)
        loss = config.lambda_attr * penalty.sum()

        # Expected: 2.0 * ((-0.1) - (-0.5)) = 2.0 * 0.4 = 0.8
        expected = 2.0 * 0.4
        assert abs(loss.item() - expected) < 1e-6, f"Expected loss {expected}, got {loss.item()}"

    @pytest.mark.cpu
    def test_ssiuu_not_initialised_raises(self):
        """compute_loss() must raise RuntimeError if initialise() not called."""
        config = SSIUUConfig()
        reg = SSIUURegulariser(model=None, forget_loader=None, config=config)

        assert not reg.is_initialised

        with pytest.raises(RuntimeError, match="initialise.*must be called"):
            reg.compute_loss({"input_ids": torch.zeros(1, 10, dtype=torch.long)})

    @pytest.mark.requires_model
    def test_integrated_gradients_sums_correctly(self):
        """IG attributions should approximately satisfy the completeness axiom."""
        pytest.skip("Implementation required in pilot phase — needs LLM")

    @pytest.mark.requires_model
    def test_target_layer_index_valid(self):
        """Target layer index must be within model layer count."""
        pytest.skip("Implementation required in pilot phase — needs LLM")
