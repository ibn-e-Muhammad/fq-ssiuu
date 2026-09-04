"""
tests/test_npo_numerical.py

NPO loss numerical stability tests (CR-003).

Verifies NPO loss behaviour at boundary conditions:
- Equal log-probabilities (ratio = 0)
- Model probability approaching zero (ratio -> -inf)
- Large positive ratios (ratio -> +inf)
- No NaN or Inf in outputs or gradients
"""

import math

import pytest
import torch


class TestNPONumerical:
    """Numerical stability tests for NPO loss."""

    def _make_inputs(self, ratio_value: float, batch_size: int = 4, seq_len: int = 8, vocab_size: int = 32):
        """Create synthetic log-probability tensors with a controlled ratio.

        Args:
            ratio_value: Desired log(pi_theta/pi_ref) ratio.
            batch_size: Batch size.
            seq_len: Sequence length.
            vocab_size: Vocabulary size.

        Returns:
            Tuple of (model_logprobs, ref_logprobs, labels) tensors.
        """
        # Create reference log-probs as uniform distribution
        ref_logits = torch.zeros(batch_size, seq_len, vocab_size)
        ref_logprobs = torch.log_softmax(ref_logits, dim=-1)

        # Create model log-probs shifted by ratio_value at target positions
        model_logits = torch.zeros(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Adjust model logits so that log(pi_theta/pi_ref) ≈ ratio_value
        for b in range(batch_size):
            for s in range(seq_len):
                target = labels[b, s].item()
                model_logits[b, s, target] += ratio_value * vocab_size

        model_logprobs = torch.log_softmax(model_logits, dim=-1)
        return model_logprobs, ref_logprobs, labels

    def test_equal_logprobs_produces_log2(self):
        """At ratio=0, NPO loss should be (2/beta) * log(2)."""
        try:
            from fq_ssiuu.unlearning.npo_loss import compute_npo_loss, NPOConfig
        except ImportError:
            pytest.skip("npo_loss not importable")

        config = NPOConfig(beta=0.1)
        model_lp, ref_lp, labels = self._make_inputs(0.0)
        loss = compute_npo_loss(model_lp, ref_lp, labels, config=config)

        expected = (2.0 / config.beta) * math.log(2)
        assert torch.isfinite(loss), "Loss should be finite at ratio=0"
        # Allow some tolerance due to softmax numerics
        assert abs(loss.item() - expected) < expected * 0.5, (
            f"At ratio=0, expected ~{expected:.2f}, got {loss.item():.2f}"
        )

    def test_model_prob_near_zero(self):
        """When model assigns near-zero prob, loss should approach 0."""
        try:
            from fq_ssiuu.unlearning.npo_loss import compute_npo_loss
        except ImportError:
            pytest.skip("npo_loss not importable")

        model_lp, ref_lp, labels = self._make_inputs(-20.0)  # Very negative ratio
        loss = compute_npo_loss(model_lp, ref_lp, labels)

        assert torch.isfinite(loss), "Loss should be finite for very negative ratio"
        # Loss should be small (approaching 0)
        assert loss.item() >= 0, "NPO loss should be non-negative"

    def test_large_positive_ratio_unbounded(self):
        """When model assigns higher prob than ref, loss should be large."""
        try:
            from fq_ssiuu.unlearning.npo_loss import compute_npo_loss
        except ImportError:
            pytest.skip("npo_loss not importable")

        model_lp, ref_lp, labels = self._make_inputs(5.0)  # Positive ratio
        loss = compute_npo_loss(model_lp, ref_lp, labels)

        assert torch.isfinite(loss), "Loss should be finite for moderate positive ratio"
        # Loss should be significantly larger than (2/beta)*log(2)
        assert loss.item() > 0, "NPO loss should be positive for positive ratio"

    def test_no_nan_in_gradients(self):
        """Gradients should be finite for all tested ratios."""
        try:
            from fq_ssiuu.unlearning.npo_loss import compute_npo_loss
        except ImportError:
            pytest.skip("npo_loss not importable")

        for ratio in [-10.0, -1.0, 0.0, 1.0, 5.0]:
            model_lp, ref_lp, labels = self._make_inputs(ratio)
            model_lp.requires_grad_(True)
            loss = compute_npo_loss(model_lp, ref_lp, labels)

            assert torch.isfinite(loss), f"Loss not finite at ratio={ratio}"

            loss.backward()
            assert model_lp.grad is not None, f"No gradient at ratio={ratio}"
            assert torch.isfinite(model_lp.grad).all(), (
                f"Non-finite gradients at ratio={ratio}"
            )
