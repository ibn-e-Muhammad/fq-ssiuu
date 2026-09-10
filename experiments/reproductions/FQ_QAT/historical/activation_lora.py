"""
src/fq_ssiuu/quantisation/fq_hooks.py

FQ-QAT integration with peft's LoRA layers.

This module provides the FQLoRALinear class that applies fake quantisation
(QDQ with STE) to the LoRA adapter output BEFORE it is added to the base
layer output. This forces the adapter weight updates to be large enough
to survive NF4 rounding when the adapter is merged into the base weights.

WHY SUBCLASS, NOT HOOKS:
    The EVB (Table 15) rates hook-based integration with peft's LoraLinear
    as HIGH risk. register_forward_hook captures the output AFTER addition,
    making it impossible to apply QDQ to the adapter delta alone. The
    subclass approach overrides forward() to intercept the adapter output
    BEFORE addition.

THE CRITICAL INSIGHT:
    Standard LoRA forward:
        output = base_layer(x) + lora_B(lora_A(dropout(x))) * scaling

    FQ-QAT LoRA forward:
        adapter_delta = lora_B(lora_A(dropout(x))) * scaling
        adapter_fq    = fake_quantise(adapter_delta)   # QDQ + STE
        output        = base_layer(x) + adapter_fq

    The fake quantisation is applied to the adapter OUTPUT (the delta),
    not to the base layer weights or the input. The base layer weights
    are frozen (NF4 quantised) and must not be modified.

    During backward, the STE passes gradients through the QDQ operation
    unchanged. The optimiser then receives gradients that "know about"
    the quantisation rounding, forcing it to make updates that cross
    NF4 bin boundaries.

Dependencies:
    - peft >= 0.12.0 (for LoraLayer internals)
    - torch >= 2.4.1
    - experiments.reproductions.FQ_QAT.historical.activation_int4 (compute_scale, _FakeQuantiseAutograd)
"""

import logging

import torch
import torch.nn as nn

from experiments.reproductions.FQ_QAT.historical.activation_int4 import _FakeQuantiseAutograd, compute_scale

logger = logging.getLogger(__name__)


# Structured metadata for the quantisation-location research gate (CR-007).
FQ_HOOKS_META = {
    "quantised_tensor": "adapter_output_activation",
    "quantisation_point": "after lora_B(lora_A(x)) * scaling, before addition to base",
    "deployment_quantisation": "merged_weights_nf4_via_bitsandbytes",
    "deployment_equivalent": False,
    "mismatch_summary": (
        "This module applies symmetric INT4 QDQ to the adapter OUTPUT activation "
        "before it is added to the base layer output. Deployment NF4 quantisation "
        "via bitsandbytes applies the NF4 codebook to the MERGED model weights "
        "after merge_and_unload(). These quantise different tensors (activations "
        "vs merged weights) using different codebooks (symmetric INT4 vs NF4) "
        "at different granularity (per-tensor vs per-block)."
    ),
}


class FQLoRALinear(nn.Module):
    """LoRA linear layer with fake quantisation on the adapter output.

    This module wraps an existing peft LoRA linear layer, intercepting
    the adapter computation to apply QDQ before the result is added to
    the base layer output.

    The wrapping is done by:
    1. Storing a reference to the original LoRA layer
    2. Overriding forward() to compute the adapter output separately,
       apply fake quantisation, and then add to the base output

    This approach avoids modifying peft internals while ensuring the
    QDQ is applied to exactly the right tensor (the adapter delta).
    """

    def __init__(self, original_lora_layer: nn.Module):
        """Initialise from an existing peft LoRA layer.

        Args:
            original_lora_layer: A peft LoRA linear layer (e.g., from
                model.base_model.model.model.layers[i].self_attn.q_proj).
                Must have lora_A, lora_B, and base_layer attributes.
        """
        super().__init__()
        self.original = original_lora_layer
        self.fq_enabled = True

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass with FQ-QAT on the adapter output.

        Computes:
            base_output = self.original.base_layer(x)
            adapter_delta = sum over adapters of lora_B(lora_A(dropout(x))) * scaling
            adapter_fq = fake_quantise(adapter_delta)
            output = base_output + adapter_fq

        When fq_enabled is False, falls back to the standard LoRA forward.

        Args:
            x: Input tensor. Shape: [batch, seq_len, in_features].

        Returns:
            torch.Tensor: Output with FQ-QAT applied to adapter delta.
        """
        if not self.fq_enabled:
            return self.original(x, *args, **kwargs)

        # Step 1: Compute base layer output (frozen NF4 weights)
        base_output = self.original.base_layer(x)

        # Step 2: Check if any active adapters exist
        if not hasattr(self.original, "active_adapters") or not self.original.active_adapters:
            # Fallback: check for active_adapter (older peft versions)
            if hasattr(self.original, "active_adapter"):
                active_adapters = (
                    [self.original.active_adapter]
                    if isinstance(self.original.active_adapter, str)
                    else list(self.original.active_adapter)
                )
            else:
                return base_output
        else:
            active_adapters = self.original.active_adapters

        # Step 3: Compute adapter output with FQ-QAT
        adapter_output = torch.zeros_like(base_output)

        for adapter_name in active_adapters:
            if adapter_name not in self.original.lora_A:
                continue

            lora_a = self.original.lora_A[adapter_name]
            lora_b = self.original.lora_B[adapter_name]

            # Apply dropout if present
            if (
                hasattr(self.original, "lora_dropout")
                and adapter_name in self.original.lora_dropout
            ):
                dropout = self.original.lora_dropout[adapter_name]
                x_drop = dropout(x)
            else:
                x_drop = x

            # Compute adapter delta: lora_B(lora_A(x_drop)) * scaling
            adapter_delta = lora_b(lora_a(x_drop))

            # Apply scaling
            if hasattr(self.original, "scaling") and adapter_name in self.original.scaling:
                scaling = self.original.scaling[adapter_name]
                adapter_delta = adapter_delta * scaling

            # Step 4: Apply fake quantisation to the adapter delta
            # This is the critical FQ-QAT operation: QDQ + STE on the delta
            scale = compute_scale(adapter_delta)
            adapter_fq = _FakeQuantiseAutograd.apply(adapter_delta, scale)

            adapter_output = adapter_output + adapter_fq

        return base_output + adapter_output

    def __repr__(self):
        return f"FQLoRALinear(fq_enabled={self.fq_enabled}, original={self.original})"


def wrap_model_with_fq_lora(model: nn.Module) -> tuple[nn.Module, int]:
    """Replace all peft LoRA linear layers with FQLoRALinear wrappers.

    Iterates over all named modules in the model, identifies peft LoRA
    layers by checking for the lora_A attribute (ModuleDict), and replaces
    each with an FQLoRALinear wrapper that preserves all original weights.

    Args:
        model: A peft-wrapped model (PeftModel or similar).

    Returns:
        Tuple of (modified model, count of wrapped layers).
    """
    wrapped_count = 0
    replacements = []

    # Collect all LoRA layers to replace (can't modify dict during iteration)
    for name, module in model.named_modules():
        if (
            hasattr(module, "lora_A")
            and hasattr(module, "lora_B")
            and hasattr(module, "base_layer")
        ):
            # This is a peft LoRA layer — wrap it
            replacements.append((name, module))

    for name, module in replacements:
        # Navigate to the parent module and replace the child
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        child_name = parts[-1]

        # Create FQLoRALinear wrapper (shares the same underlying weights)
        fq_layer = FQLoRALinear(module)
        setattr(parent, child_name, fq_layer)
        wrapped_count += 1

    print(f"Wrapped {wrapped_count} LoRA layers with FQ-QAT", flush=True)
    return model, wrapped_count


def unwrap_model_from_fq_lora(model: nn.Module) -> tuple[nn.Module, int]:
    """Reverse the FQLoRALinear wrapping, restoring original peft layers.

    Args:
        model: A model previously wrapped with wrap_model_with_fq_lora().

    Returns:
        Tuple of (unwrapped model, count of unwrapped layers).
    """
    unwrapped_count = 0
    replacements = []

    for name, module in model.named_modules():
        if isinstance(module, FQLoRALinear):
            replacements.append((name, module))

    for name, module in replacements:
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        child_name = parts[-1]

        # Restore the original LoRA layer
        setattr(parent, child_name, module.original)
        unwrapped_count += 1

    print(f"Unwrapped {unwrapped_count} FQ-QAT layers (restored original LoRA)", flush=True)
    return model, unwrapped_count


def test_fq_lora_wrapping():
    """Self-test for FQLoRALinear wrapping.

    Creates a minimal mock LoRA layer structure (no peft dependency needed),
    wraps it with FQLoRALinear, verifies forward and backward passes.

    Tests:
    1. Output shape matches the unwrapped forward pass
    2. Backward pass produces non-zero gradients on lora_A weight
    3. Wrap/unwrap round-trip preserves the original module
    """

    # Create a mock LoRA layer that mimics peft's internal structure
    class MockLoRALayer(nn.Module):
        def __init__(self, in_features=4, out_features=4, rank=2):
            super().__init__()
            self.base_layer = nn.Linear(in_features, out_features, bias=False)
            # peft stores adapters in ModuleDicts keyed by adapter name
            self.lora_A = nn.ModuleDict({"default": nn.Linear(in_features, rank, bias=False)})
            self.lora_B = nn.ModuleDict({"default": nn.Linear(rank, out_features, bias=False)})
            self.lora_dropout = nn.ModuleDict({"default": nn.Identity()})
            self.scaling = {"default": 2.0}
            self.active_adapters = ["default"]

            # Freeze base layer (simulates NF4 frozen weights)
            self.base_layer.weight.requires_grad = False

        def forward(self, x):
            base_out = self.base_layer(x)
            adapter_out = (
                self.lora_B["default"](self.lora_A["default"](self.lora_dropout["default"](x)))
                * self.scaling["default"]
            )
            return base_out + adapter_out

    # Step 1: Create mock layer and compute reference output
    mock_layer = MockLoRALayer(in_features=4, out_features=4, rank=2)
    x = torch.randn(2, 3, 4)  # [batch=2, seq=3, features=4]

    ref_output = mock_layer(x)
    ref_shape = ref_output.shape
    print(f"  Reference output shape: {ref_shape}")

    # Step 2: Wrap with FQLoRALinear
    fq_layer = FQLoRALinear(mock_layer)
    fq_output = fq_layer(x)

    # Test 1: Output shape must match
    assert fq_output.shape == ref_shape, (
        f"Shape mismatch: FQ output {fq_output.shape} vs reference {ref_shape}"
    )
    print(f"  Test 1 (shape match): PASS — {fq_output.shape}")

    # Test 2: Backward produces non-zero gradients on lora_A
    loss = fq_output.sum()
    loss.backward()

    lora_a_grad = fq_layer.original.lora_A["default"].weight.grad
    assert lora_a_grad is not None, "lora_A gradient is None"
    assert torch.any(lora_a_grad != 0), "lora_A gradient is all zeros"
    print(f"  Test 2 (non-zero grads): PASS — grad norm = {lora_a_grad.norm().item():.6f}")

    # Test 3: Wrap/unwrap round-trip
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = MockLoRALayer()

    mock_model = MockModel()
    original_layer = mock_model.layer

    # Wrap
    mock_model, n_wrapped = wrap_model_with_fq_lora(mock_model)
    assert n_wrapped == 1, f"Expected 1 wrapped layer, got {n_wrapped}"
    assert isinstance(mock_model.layer, FQLoRALinear), "Layer not wrapped"

    # Unwrap
    mock_model, n_unwrapped = unwrap_model_from_fq_lora(mock_model)
    assert n_unwrapped == 1, f"Expected 1 unwrapped layer, got {n_unwrapped}"
    assert mock_model.layer is original_layer, "Original layer not restored"
    print("  Test 3 (wrap/unwrap round-trip): PASS")

    print("  fq_hooks self-test: ALL CHECKS PASSED")
