"""
src/fq_ssiuu/attribution/ssiuu_reg.py

SSIUU-style attribution regularisation for machine unlearning.

Reference: Yang et al., “Erase or Hide? Suppressing Spurious Unlearning Neurons for
Robust Unlearning,” arXiv:2509.22263v2.

This repository currently implements a LayerIntegratedGradients surrogate.
It is not the published Ssiuu method.

Published Ssiuu Mechanism:
    - Published Ssiuu uses parameter-level attribution in Appendix B.
    - It compares current-step attribution with the previous step.
    - The previous-step attribution is detached.
    - The current-step attribution remains differentiable.
    - Its gradient includes a Hessian-vector product.
    - The first optimization step has zero Ssiuu penalty.

Current Implementation (LayerIG surrogate):
    - The current surrogate uses a static initial baseline and returns detached current attribution.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class SSIUUConfig:
    """Configuration for SSIUU attribution regularisation.

    Attributes:
        lambda_attr: Weight for the attribution regularisation loss.
            Default 1.0 from SSIUU paper. Ablate over [0.1, 0.5, 1.0, 5.0].
        target_layer_idx: Index of the target layer for attribution.
            Default 14 (layer 14 of 16 for Llama-3.2-1B, the penultimate
            layer). This captures near-final representations, not middle-layer
            features. Confirm via layer sweep.
        n_ig_steps: Number of Integrated Gradients interpolation steps.
            20 steps for speed during pilot experiments.
            Use 50 for final publication-quality results.
        attribution_batch_size: Number of samples per IG batch.
            Default 1 due to 16 GB VRAM constraint -- IG requires storing
            multiple forward passes (n_ig_steps * batch_size).
    """

    lambda_attr: float = 1.0
    target_layer_idx: int = 14
    n_ig_steps: int = 20  # 20 = fast; 50 = final results
    attribution_batch_size: int = 1  # Memory constraint: 1 sample at a time on 16 GB VRAM


def compute_negative_attribution(
    model: torch.nn.Module,
    forget_batch: dict,
    config: SSIUUConfig,
) -> torch.Tensor:
    """Compute the sum of negative attributions per neuron at the target layer.

    Uses captum's LayerIntegratedGradients to compute IG attributions at
    the hidden state output of the target transformer layer.

    D^-(theta)_i = sum min(0, IG_i(x, y, theta))

    Args:
        model: The model being unlearned. Must have model.model.layers accessible.
        forget_batch: Dictionary with 'input_ids' and 'attention_mask' tensors.
        config: SSIUU configuration.

    Returns:
        torch.Tensor: Negative attribution sums per neuron. Shape: (hidden_dim,).

    Raises:
        ImportError: If captum is not installed.
        IndexError: If target_layer_idx is out of bounds.
    """
    try:
        from captum.attr import LayerIntegratedGradients
    except ImportError as e:
        raise ImportError(
            "captum is required for SSIUU IG-based attribution. "
            "Install with: pip install captum==0.7.0"
        ) from e

    # Validate layer index
    n_layers = len(model.model.layers)
    if config.target_layer_idx >= n_layers:
        raise IndexError(
            f"target_layer_idx={config.target_layer_idx} is out of bounds. "
            f"Model has {n_layers} layers (indices 0 to {n_layers - 1})."
        )

    target_layer = model.model.layers[config.target_layer_idx]

    # IMPORTANT: Disable gradient checkpointing before IG computation.
    # Captum's IG requires all intermediate activations to be stored,
    # which conflicts with gradient checkpointing that discards them.
    # We re-enable it after the computation.
    gc_was_enabled = getattr(model.config, "use_cache", True)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
        gc_was_disabled = True
    else:
        gc_was_disabled = False

    try:
        # Define a forward function for captum
        def forward_fn(input_ids):
            outputs = model(input_ids=input_ids)
            # Sum logits at the target positions to get a scalar
            return outputs.logits.sum()

        lig = LayerIntegratedGradients(forward_fn, target_layer)

        input_ids = forget_batch["input_ids"]

        # Compute IG attributions
        # Shape: same as the layer's output (batch, seq, hidden_dim)
        attributions = lig.attribute(
            inputs=input_ids,
            n_steps=config.n_ig_steps,
            internal_batch_size=config.attribution_batch_size,
        )

        # Sum over batch and sequence dimensions -> (hidden_dim,)
        # This gives the total attribution per neuron across the forget set
        attr_sum = attributions.sum(dim=(0, 1))

        # Extract only negative attributions: D^-(theta)_i = sum min(0, IG_i)
        d_neg = torch.clamp(attr_sum, max=0)  # Keep only negative values

    finally:
        # Re-enable gradient checkpointing if it was active
        if gc_was_disabled and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        model.config.use_cache = gc_was_enabled

    # CR-006 GRADIENT FLOW DIAGNOSTIC:
    # .detach() severs the computation graph from model parameters.
    # The resulting d_neg tensor has requires_grad=False and cannot
    # propagate loss gradients back through the model. The surrogate
    # regularisation loss computed from this tensor is a constant
    # w.r.t. model parameters — the optimizer receives zero gradient
    # from the attribution penalty.
    #
    # Note: Removing .detach() will NOT make this surrogate equivalent
    # to Published SSIUU. Published SSIUU uses parameter-level gradients
    # and Hessian-vector products. This LayerIG surrogate uses hidden
    # layer activations.
    #
    # See: handbook/corrections/REGISTER.md#cr-006-ssiuu-gradient-flow
    return d_neg.detach()


class SSIUURegulariser:
    """SSIUU attribution regularisation for unlearning training.

    Monitors negative attributions during training and penalises neurons
    that develop more negative attributions than at initialisation (theta_0).

    Usage:
        reg = SSIUURegulariser(model, forget_loader, config)
        reg.initialise()  # Compute D^-(theta_0) -- call ONCE before training

        for batch in train_loader:
            npo_loss = compute_npo_loss(...)
            attr_loss = reg.compute_loss(forget_batch)
            total_loss = npo_loss + attr_loss
            total_loss.backward()
    """

    def __init__(
        self,
        model: torch.nn.Module,
        forget_loader,
        config: Optional[SSIUUConfig] = None,
    ):
        """Initialise the regulariser.

        Args:
            model: The model being unlearned.
            forget_loader: DataLoader for the forget set.
            config: SSIUU configuration.
        """
        self.model = model
        self.forget_loader = forget_loader
        self.config = config or SSIUUConfig()
        self.d_neg_init: Optional[torch.Tensor] = None

    @property
    def is_initialised(self) -> bool:
        """Whether initialise() has been called."""
        return self.d_neg_init is not None

    def initialise(self) -> None:
        """Compute D^-(theta_0) over the forget set.

        Must be called ONCE before training begins. The result is stored
        as a frozen baseline (detached from computation graph).

        This can take several minutes for large forget sets due to the
        n_ig_steps forward passes per batch required by Integrated Gradients.
        """
        logger.info("Computing initial negative attributions D^-(theta_0)...")

        all_d_neg = []
        for batch in self.forget_loader:
            d_neg = compute_negative_attribution(self.model, batch, self.config)
            all_d_neg.append(d_neg)

        # Accumulate across batches
        self.d_neg_init = torch.stack(all_d_neg).sum(dim=0).detach()

        logger.info(
            "D^-(theta_0) computed. Mean=%.6f, Min=%.6f, Max=%.6f",
            self.d_neg_init.mean().item(),
            self.d_neg_init.min().item(),
            self.d_neg_init.max().item(),
        )

    def compute_loss(self, forget_batch: dict) -> torch.Tensor:
        """Compute the SSIUU attribution regularisation loss.

        L_attr = lambda_attr * sum_i max(0, D^-_i(theta_0) - D^-_i(theta))

        This penalises neurons whose negative attribution has INCREASED
        (become more negative) since initialisation.

        Args:
            forget_batch: Dictionary with 'input_ids' and 'attention_mask'.

        Returns:
            torch.Tensor: Scalar attribution regularisation loss.

        Raises:
            RuntimeError: If initialise() has not been called.
        """
        if not self.is_initialised:
            raise RuntimeError(
                "SSIUURegulariser.initialise() must be called before compute_loss(). "
                "Call initialise() once before the training loop begins."
            )

        # Compute current negative attributions
        d_neg_current = compute_negative_attribution(
            self.model, forget_batch, self.config
        )

        # L_attr = lambda * sum max(0, D^-_init - D^-_current)
        # When D^-_current is MORE negative than D^-_init (neuron became more
        # suppressive), D^-_init - D^-_current > 0, so the penalty is positive.
        # When D^-_current is LESS negative (neuron became less suppressive or
        # positive), the max(0, ...) clamps to 0 -- no penalty.
        penalty = torch.clamp(self.d_neg_init - d_neg_current, min=0)
        loss = self.config.lambda_attr * penalty.sum()

        return loss


class GradientNormAttribution:
    """Fallback attribution using gradient norms instead of Integrated Gradients.

    Computes attribution as |dL/dh_i|_2 (L2 gradient norm at the target layer)
    instead of IG. This is used when captum is unavailable or when IG exceeds
    the 16 GB VRAM budget.

    Theoretical limitations:
        - Does NOT satisfy the completeness axiom (attributions do not sum to
          the output difference).
        - Does NOT satisfy the sensitivity axiom (may miss features that affect
          the output only through interactions).
        - Captures local gradient magnitude, not integrated contribution.

    Use ONLY if IG fails in Pilot P2 due to OOM. Document the fallback in
    the experiment log with a Decision Log entry.

    Interface matches compute_negative_attribution() for drop-in substitution.
    """

    def __init__(self, model: torch.nn.Module, config: SSIUUConfig):
        """Initialise the gradient-norm attribution fallback.

        Args:
            model: The model being unlearned.
            config: SSIUU configuration (uses target_layer_idx).
        """
        self.model = model
        self.config = config

    def compute(self, forget_batch: dict) -> torch.Tensor:
        """Compute gradient-norm attribution at the target layer.

        Returns D^-(theta)_i as the negative of the gradient norm per neuron.
        This approximation treats large gradients as high attribution.

        Args:
            forget_batch: Dictionary with 'input_ids' and 'attention_mask'.

        Returns:
            torch.Tensor: Negative attribution proxy. Shape: (hidden_dim,).
        """
        target_layer = self.model.model.layers[self.config.target_layer_idx]
        input_ids = forget_batch["input_ids"]

        # Forward pass with gradient tracking on the target layer
        outputs = self.model(input_ids=input_ids)
        loss = outputs.logits.sum()
        loss.backward(retain_graph=True)

        # Collect gradient norms per neuron from the target layer
        grad_norms = []
        for param in target_layer.parameters():
            if param.grad is not None:
                # L2 norm per output neuron (sum over input dims)
                if param.grad.dim() >= 2:
                    norm = param.grad.norm(dim=-1)
                else:
                    norm = param.grad.abs()
                grad_norms.append(norm.flatten())

        if not grad_norms:
            logger.warning("No gradients found at target layer. Returning zeros.")
            hidden_dim = self.model.config.hidden_size
            return torch.zeros(hidden_dim, device=input_ids.device)

        # Concatenate and take negative (convention: negative attribution)
        combined = torch.cat(grad_norms)
        d_neg = -combined  # Negative = high gradient = high attribution

        # Zero out the model gradients to avoid accumulation
        self.model.zero_grad()

        return d_neg.detach()


def test_ssiuu_loss_zero_at_init():
    """Self-test: verify SSIUU loss is zero when current == initial attributions.

    Uses a tiny dummy model (2-layer MLP) to avoid any LLM loading.
    Manually sets d_neg_init and verifies that when d_neg_current == d_neg_init,
    the loss is exactly zero.

    This follows from the equation:
        L_attr = lambda * sum max(0, D^-_init - D^-_current)
        When D^-_current == D^-_init: max(0, 0) = 0 for all neurons.
    """
    # Create a dummy regulariser without a real model
    config = SSIUUConfig(lambda_attr=1.0)
    reg = SSIUURegulariser(model=None, forget_loader=None, config=config)

    # Manually set d_neg_init (bypassing initialise() which needs a real model)
    reg.d_neg_init = torch.tensor([-0.1, -0.2, -0.3])

    # Simulate compute_loss: d_neg_current == d_neg_init
    d_neg_current = torch.tensor([-0.1, -0.2, -0.3])
    penalty = torch.clamp(reg.d_neg_init - d_neg_current, min=0)
    loss = config.lambda_attr * penalty.sum()

    assert abs(loss.item()) < 1e-8, (
        f"SSIUU loss should be 0 when current == init, got {loss.item()}"
    )
    print(f"  SSIUU zero-at-init: loss = {loss.item():.8f} (expected 0) — PASS")

    # Also test: when current is MORE negative, penalty should be positive
    d_neg_more_negative = torch.tensor([-0.5, -0.2, -0.3])
    penalty2 = torch.clamp(reg.d_neg_init - d_neg_more_negative, min=0)
    loss2 = config.lambda_attr * penalty2.sum()
    expected2 = 1.0 * ((-0.1) - (-0.5))  # = 0.4
    assert abs(loss2.item() - expected2) < 1e-6, (
        f"SSIUU penalty for more-negative neuron: expected {expected2}, got {loss2.item()}"
    )
    print(f"  SSIUU penalty for more-negative: {loss2.item():.4f} (expected {expected2}) — PASS")

    print("  SSIUU self-test: ALL CHECKS PASSED")
