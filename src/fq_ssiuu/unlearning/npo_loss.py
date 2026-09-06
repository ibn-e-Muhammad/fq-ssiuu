"""
src/fq_ssiuu/unlearning/npo_loss.py

Negative Preference Optimisation (NPO) loss for machine unlearning.

Reference: Zhang et al. 2024, "Negative Preference Optimization: From
Catastrophic Collapse to Effective Unlearning", arXiv:2404.05868.

Equation (log-probability form, numerically stable):
    L_NPO = (2/β) · E_{(x,y)∈D_f} [log(1 + exp(β · (log π_θ(y|x) − log π_ref(y|x))))]

Which simplifies via softplus:
    ratio = β · (log_prob_θ − log_prob_ref)
    L_NPO = (2/β) · softplus(ratio)

Where softplus(x) = log(1 + exp(x)).

Why NPO prevents catastrophic collapse:
    Standard Gradient Ascent (GA) maximises the loss on the forget set, which pushes
    π_θ(y|x) → 0 for forget examples. But GA has no upper bound — it can diverge,
    causing catastrophic collapse on the retain set.

    NPO clips the effective loss contribution: as π_θ → 0, the ratio
    log(π_θ/π_ref) → -∞, so softplus(β·ratio) → 0. The loss approaches:
        L_NPO → 0 (the global infimum)
    At ratio = 0 (identical models):
        L_NPO = (2/β) · log(2) ≈ (2/β) · 0.693
    As ratio → +∞ (model assigns higher probability than reference):
        L_NPO → 2 · ratio, which is unbounded above.

    In the intended unlearning direction (π_θ → 0), the loss is effectively
    bounded, which prevents catastrophic collapse. In the reverse direction
    (model probability increases), the loss grows without bound.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class NPOTrainingConfig:
    """Configuration for NPO-based unlearning training.

    Attributes:
        beta: NPO temperature parameter. Controls the sharpness of the
            preference boundary. Default 0.1 from Zhang et al. 2024.
        lambda_retain: Weight for the KL-divergence retain loss.
        use_retain_loss: Whether to add KL retain regularisation.
    """

    beta: float = 0.1
    lambda_retain: float = 0.0
    use_retain_loss: bool = False


def compute_npo_loss(model_logits, ref_logits, labels, beta: float = 0.1):
    """Compute the NPO loss between the current model and the reference model.

    Implements the numerically stable log-probability form:
        ratio = β · (log_prob_θ − log_prob_ref)
        L_NPO = (2/β) · mean(softplus(ratio))

    Args:
        model_logits: Raw logits from the current (unlearning) model.
            Shape: [batch_size, seq_len, vocab_size]
        ref_logits: Raw logits from the frozen reference model.
            Shape: [batch_size, seq_len, vocab_size]
        labels: Token IDs for the target sequence. Positions with value -100
            are masked and excluded from the loss computation.
            Shape: [batch_size, seq_len]
        beta: NPO temperature parameter (default 0.1).

    Returns:
        torch.Tensor: Scalar NPO loss averaged over the batch.

    Raises:
        ValueError: If tensor shapes are incompatible or beta <= 0.
    """
    import torch.nn.functional as functional

    # --- Input validation ---
    if model_logits.dim() != 3:
        raise ValueError(f"model_logits must be 3D [B, T, V], got shape {model_logits.shape}")
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
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")

    batch_size, seq_len, vocab_size = model_logits.shape

    # Step 1: Compute per-token log-probabilities via log_softmax
    # Shape: [B, T, V]
    model_log_probs = functional.log_softmax(model_logits, dim=-1)
    ref_log_probs = functional.log_softmax(ref_logits, dim=-1)

    # Step 2: Gather log-probabilities at the label positions
    # Create a mask for valid (non-padding) tokens
    mask = labels != -100  # Shape: [B, T]
    # Replace -100 with 0 for gather (will be masked out later)
    safe_labels = labels.clamp(min=0)  # Shape: [B, T]

    # Gather: select the log-prob of the correct token at each position
    # Shape: [B, T, 1] → [B, T]
    model_token_log_probs = model_log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(
        -1
    )
    ref_token_log_probs = ref_log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)

    # Step 3: Mask out padding positions
    model_token_log_probs = model_token_log_probs * mask.float()
    ref_token_log_probs = ref_token_log_probs * mask.float()

    # Step 4: Compute per-example average log-probability
    # Sum over sequence, divide by number of valid tokens per example
    n_valid = mask.float().sum(dim=-1).clamp(min=1)  # [B], avoid div-by-zero
    model_avg_log_prob = model_token_log_probs.sum(dim=-1) / n_valid  # [B]
    ref_avg_log_prob = ref_token_log_probs.sum(dim=-1) / n_valid  # [B]

    # Step 5: Compute NPO loss
    # ratio = β · (log_prob_θ − log_prob_ref)
    ratio = beta * (model_avg_log_prob - ref_avg_log_prob)  # [B]
    # L_NPO = (2/β) · softplus(ratio)
    npo_loss = (2.0 / beta) * functional.softplus(ratio)  # [B]

    # Step 6: Average over batch
    loss = npo_loss.mean()

    return loss


def compute_retain_kl_loss(model_logits, ref_logits):
    """Compute KL(π_θ ‖ π_ref) for the retain set.

    This regularisation term prevents the unlearning process from degrading
    the model's performance on the retain set.

    Property: when model_logits == ref_logits, KL = 0 exactly.

    IMPORTANT: functional.kl_div argument order matters:
        functional.kl_div(input=log_probs, target=probs, reduction='batchmean', log_target=False)
    The 'input' is LOG-probabilities from the current model.
    The 'target' is PROBABILITIES from the reference model.
    Getting this wrong produces incorrect gradients.

    Args:
        model_logits: Raw logits from the current model.
            Shape: [batch_size, seq_len, vocab_size]
        ref_logits: Raw logits from the frozen reference model.
            Shape: [batch_size, seq_len, vocab_size]

    Returns:
        torch.Tensor: Scalar KL divergence loss.

    Raises:
        ValueError: If shapes don't match.
    """
    import torch.nn.functional as functional

    if model_logits.shape != ref_logits.shape:
        raise ValueError(
            f"Shape mismatch: model_logits {model_logits.shape} vs ref_logits {ref_logits.shape}"
        )

    # KL(π_θ ‖ π_ref) = Σ π_ref(x) · [log π_ref(x) − log π_θ(x)]
    # functional.kl_div computes: Σ target · (log(target) − input)
    # which equals KL(target ‖ input) when target=π_ref and input=log(π_θ)
    # Wait — functional.kl_div(input, target) = target * (log(target) - input)
    # This gives KL(target ‖ exp(input)), which is what we want: KL(π_ref ‖ π_θ).
    # Actually, we want KL(π_θ ‖ π_ref), so we need:
    # functional.kl_div(input=log_ref, target=probs_model) — but convention in unlearning
    # is to use KL(π_θ ‖ π_ref), keeping the model close to the reference.
    #
    # For retain regularisation, we use KL(π_θ ‖ π_ref):
    # = Σ π_θ(x) · [log π_θ(x) − log π_ref(x)]
    # functional.kl_div(input=log_ref, target=π_θ) gives Σ π_θ · (log π_θ − log_ref) ✓
    ref_log_probs = functional.log_softmax(ref_logits, dim=-1)
    model_probs = functional.softmax(model_logits, dim=-1)

    kl_loss = functional.kl_div(
        input=ref_log_probs,
        target=model_probs,
        reduction="batchmean",
        log_target=False,
    )

    return kl_loss


def test_npo_loss_zero_when_equal():
    """Self-test: verify NPO loss equals the closed-form value when logits are identical.

    When model_logits == ref_logits:
        log_prob_θ == log_prob_ref
        ratio = β · 0 = 0
        L_NPO = (2/β) · softplus(0) = (2/β) · log(2)

    For β = 0.1: L_NPO = 20 · log(2) ≈ 13.8629

    This test also verifies that KL = 0 when logits are identical.
    """
    import math

    import torch

    batch, seq, vocab = 2, 4, 100
    logits = torch.randn(batch, seq, vocab)
    labels = torch.randint(0, vocab, (batch, seq))
    # Set one position to -100 (padding) to test masking
    labels[0, -1] = -100

    beta = 0.1
    loss = compute_npo_loss(logits, logits, labels, beta=beta)

    expected = (2.0 / beta) * math.log(2)
    assert abs(loss.item() - expected) < 1e-4, (
        f"NPO loss with identical logits: expected {expected:.4f}, got {loss.item():.4f}"
    )

    # Also verify KL = 0 with identical logits
    kl = compute_retain_kl_loss(logits, logits)
    assert abs(kl.item()) < 1e-5, f"KL with identical logits should be 0, got {kl.item()}"

    logger.info(
        "Self-test PASSED: NPO=%.4f (expected %.4f), KL=%.6f", loss.item(), expected, kl.item()
    )
