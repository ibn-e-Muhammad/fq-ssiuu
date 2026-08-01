"""
tests/test_npo_loss.py
Unit tests for NPO loss implementation.
Tests that can run on CPU use dummy logits — no model loading required.
Reference: Zhang et al. 2024, arXiv:2404.05868
"""
import math

import pytest
import torch

from fq_ssiuu.unlearning.npo_loss import compute_npo_loss, compute_retain_kl_loss


class TestNPOLoss:
    """NPO loss mathematical correctness tests."""

    @pytest.mark.cpu
    def test_npo_loss_identical_models_is_zero(self):
        """When current model == reference model, NPO loss equals (2/beta)*log(2).

        At ratio=0: softplus(0) = log(2), so L_NPO = (2/beta) * log(2).
        For beta=0.1: L_NPO = 20 * 0.6931 = 13.8629.
        """
        batch, seq, vocab = 2, 4, 100
        logits = torch.randn(batch, seq, vocab)
        labels = torch.randint(0, vocab, (batch, seq))
        labels[0, -1] = -100  # Test masking

        beta = 0.1
        loss = compute_npo_loss(logits, logits, labels, beta=beta)
        expected = (2.0 / beta) * math.log(2)

        assert abs(loss.item() - expected) < 1e-4, (
            f"NPO loss with identical logits: expected {expected:.4f}, got {loss.item():.4f}"
        )

    @pytest.mark.cpu
    def test_npo_loss_is_non_negative(self):
        """NPO loss should always be non-negative.

        The loss is defined as (2/beta) * softplus(beta * ratio), which is always > 0.
        """
        beta = 0.1
        batch, seq, vocab = 3, 5, 50

        # Test with various logit magnitudes
        for scale in [0.1, 1.0, 10.0]:
            model_logits = torch.randn(batch, seq, vocab) * scale
            ref_logits = torch.randn(batch, seq, vocab) * scale
            labels = torch.randint(0, vocab, (batch, seq))

            loss = compute_npo_loss(model_logits, ref_logits, labels, beta=beta)
            assert loss.item() >= 0.0, (
                f"NPO loss {loss.item():.4f} is negative at scale {scale}"
            )

    @pytest.mark.cpu
    def test_retain_kl_zero_when_equal(self):
        """KL divergence should be zero when model and reference logits are identical."""
        batch, seq, vocab = 2, 4, 100
        logits = torch.randn(batch, seq, vocab)
        kl = compute_retain_kl_loss(logits, logits)
        assert abs(kl.item()) < 1e-5, f"KL with identical logits should be ~0, got {kl.item()}"

    @pytest.mark.requires_model
    def test_npo_loss_decreases_as_probability_drops(self):
        """As model probability on forget set decreases, NPO loss should decrease."""
        pytest.skip("Implementation required in pilot phase — needs training loop")

    @pytest.mark.requires_model
    def test_npo_loss_no_gradient_explosion(self):
        """NPO gradient should not exceed max_grad_norm under normal conditions."""
        pytest.skip("Implementation required in pilot phase — needs training loop")
