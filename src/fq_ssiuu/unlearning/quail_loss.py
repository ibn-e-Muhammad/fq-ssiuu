"""
src/fq_ssiuu/unlearning/quail_loss.py

QUAIL logit-space hinge loss for quantisation-aware unlearning.

Reference: Mishra & Mehreen 2026, "QUAIL: Quantization-Aware Unlearning
with Invariant Logits", arXiv:2601.15538.

NOTE: This is a re-implementation from the paper description. No official
code repository is available as of this writing.

Equation:
    L_QUAIL = E_{(x,y) in D_f} [max(0, delta/2 - |logit_theta(y|x) - logit_ref(y|x)|)]

Where:
    - delta is the NF4 quantisation bin width (~8.4e-4)
    - delta/2 is the hinge margin
    - logit_theta is the logit from the current (unlearning) model at the target token
    - logit_ref is the logit from the frozen reference model at the target token

Intuition:
    A logit deviation larger than half a quantisation bin cannot be erased by
    INT4 rounding. The sign of the deviation is preserved across the bin boundary.
    By enforcing |logit_theta - logit_ref| > delta/2, QUAIL ensures that the
    unlearning signal survives quantisation.

IMPORTANT DISTINCTION:
    This operates in OUTPUT space (logits). It is fundamentally different from
    FQ-SSIUU which operates in WEIGHT space (adapter parameters). Both target
    QRA robustness but via different mechanisms:
    - QUAIL: forces output divergence beyond quantisation resolution
    - FQ-SSIUU: forces weight updates beyond quantisation bin boundaries
    This distinction must be clearly stated when comparing the two in any paper.
"""

import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class QUAILConfig:
    """Configuration for QUAIL hinge loss.

    Attributes:
        nf4_bin_width: NF4 quantisation bin width. Value 8.4e-4 is the
            empirical mean bin width from MANSU (arXiv:2605.15138), measured
            across the 16 NF4 codebook bins for typical LLM weight distributions.
        margin: Hinge margin, set to nf4_bin_width / 2. A logit deviation
            exceeding this margin survives INT4 rounding.
    """

    nf4_bin_width: float = 8.4e-4
    margin: float = 4.2e-4  # nf4_bin_width / 2


def compute_quail_loss(
    model_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    labels: torch.Tensor,
    config: QUAILConfig | None = None,
) -> torch.Tensor:
    """Compute the QUAIL logit-space hinge loss.

    For each non-masked token position, extracts the logit at the target
    token index, computes the absolute deviation from the reference, and
    applies the hinge: max(0, margin - |deviation|).

    Equation:
        L_QUAIL = mean_{valid tokens} max(0, margin - |logit_theta_i - logit_ref_i|)

    When |logit_theta - logit_ref| >= margin: loss = 0 (deviation is large enough)
    When |logit_theta - logit_ref| < margin: loss > 0 (pushes deviation wider)

    Args:
        model_logits: Raw logits from the current model.
            Shape: [batch_size, seq_len, vocab_size]
        ref_logits: Raw logits from the frozen reference model.
            Shape: [batch_size, seq_len, vocab_size]
        labels: Token IDs. Positions with -100 are masked.
            Shape: [batch_size, seq_len]
        config: QUAIL configuration. Uses defaults if None.

    Returns:
        torch.Tensor: Scalar QUAIL hinge loss averaged over valid tokens.

    Raises:
        ValueError: If tensor shapes are incompatible.
    """
    if config is None:
        config = QUAILConfig()

    # --- Input validation ---
    if model_logits.dim() != 3:
        raise ValueError(
            f"model_logits must be 3D [B, T, V], got shape {model_logits.shape}"
        )
    if ref_logits.shape != model_logits.shape:
        raise ValueError(
            f"ref_logits shape {ref_logits.shape} must match "
            f"model_logits shape {model_logits.shape}"
        )
    if labels.dim() != 2:
        raise ValueError(f"labels must be 2D [B, T], got shape {labels.shape}")
    if labels.shape[0] != model_logits.shape[0] or labels.shape[1] != model_logits.shape[1]:
        raise ValueError(
            f"labels shape {labels.shape} must match "
            f"model_logits[:2] shape {model_logits.shape[:2]}"
        )

    # Step 1: Create mask for valid (non-padding) positions
    mask = labels != -100  # [B, T]
    n_valid = mask.float().sum()

    if n_valid == 0:
        logger.warning("All tokens are masked (-100). Returning zero loss.")
        return torch.tensor(0.0, device=model_logits.device, requires_grad=True)

    # Step 2: Gather logits at target token positions
    safe_labels = labels.clamp(min=0)  # Replace -100 with 0 for gather
    # Shape: [B, T, 1] -> [B, T]
    model_target_logits = model_logits.gather(
        dim=-1, index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    ref_target_logits = ref_logits.gather(
        dim=-1, index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)

    # Step 3: Compute absolute deviation at target positions
    diff = model_target_logits - ref_target_logits  # [B, T]
    # Break symmetry at exact 0 to ensure gradient flow (PyTorch abs(0) backward is 0)
    diff = diff + 1e-6 * torch.randn_like(diff)
    deviation = torch.abs(diff)

    # Step 4: Apply hinge: max(0, margin - |deviation|)
    hinge = torch.clamp(config.margin - deviation, min=0)  # [B, T]

    # Step 5: Mask and average
    hinge = hinge * mask.float()
    loss = hinge.sum() / n_valid

    return loss


def test_quail_loss_at_margin():
    """Self-test verifying hinge behaviour at and below the margin.

    Test 1: When |logit_theta - logit_ref| == margin exactly, loss == 0.
    Test 2: When |logit_theta - logit_ref| == 0, loss == margin.
    """
    config = QUAILConfig()
    B, T, V = 2, 3, 10

    # Test 1: Deviation exactly at margin -> loss should be 0
    ref_logits = torch.zeros(B, T, V)
    model_logits = torch.zeros(B, T, V)
    labels = torch.zeros(B, T, dtype=torch.long)

    # Set logit at token 0 to have deviation = margin
    model_logits[:, :, 0] = config.margin
    ref_logits[:, :, 0] = 0.0

    loss_at_margin = compute_quail_loss(model_logits, ref_logits, labels, config)
    assert abs(loss_at_margin.item()) < 1e-7, (
        f"Test 1 FAILED: loss at margin should be 0, got {loss_at_margin.item()}"
    )
    print(f"  Test 1: loss at margin = {loss_at_margin.item():.8f} (expected 0) — PASS")

    # Test 2: Zero deviation -> loss should equal margin
    model_logits_zero = torch.zeros(B, T, V)
    ref_logits_zero = torch.zeros(B, T, V)

    loss_zero_dev = compute_quail_loss(model_logits_zero, ref_logits_zero, labels, config)
    assert abs(loss_zero_dev.item() - config.margin) < 1e-5, (
        f"Test 2 FAILED: loss at zero deviation should be {config.margin}, "
        f"got {loss_zero_dev.item()}"
    )
    print(
        f"  Test 2: loss at zero deviation = {loss_zero_dev.item():.8f} "
        f"(expected {config.margin}) — PASS"
    )

    print("  QUAIL hinge self-test: ALL CHECKS PASSED")
