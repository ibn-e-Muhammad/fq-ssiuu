"""Compatibility adapter for the author's paged 32-bit AdamW, without new math."""

import torch


def restore_paged_moments(optimizer):
    """Reallocate loaded FP32 moments using the same allocator as fresh bnb state.

    bitsandbytes 0.46.1's generic load_state_dict copies saved tensors; it does not
    recreate CUDA managed allocations. Values and step counters are unchanged.
    Call only after its load with move_to_device=False, before any update.
    """
    if not optimizer.is_paged or optimizer.args.optim_bits != 32:
        raise ValueError("Only source paged 32-bit optimizer is supported")
    count = 0
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            state = optimizer.state.get(parameter, {})
            if not state:
                continue
            if parameter.dtype != torch.float32:
                raise ValueError("Full-precision master parameters required")
            for name in ("state1", "state2"):
                saved = state.get(name)
                if (
                    not isinstance(saved, torch.Tensor)
                    or saved.dtype != torch.float32
                    or saved.shape != parameter.shape
                ):
                    raise ValueError("Malformed FP32 optimizer moment")
                if not torch.isfinite(saved).all():
                    raise ValueError("Nonfinite saved optimizer moment")
            for name in ("state1", "state2"):
                buffer = optimizer.get_state_buffer(parameter, dtype=torch.float32)
                buffer.copy_(state[name])
                state[name] = buffer
                count += 1
    return count


def paged_adamw32(parameter_groups, *, lr, betas, eps):
    import bitsandbytes as bnb

    class ResumablePagedAdamW32bit(bnb.optim.PagedAdamW32bit):
        def load_state_dict(self, state_dict, move_to_device=True):
            # No moment is first restored as a large ordinary GPU allocation.
            super().load_state_dict(state_dict, move_to_device=False)
            restore_paged_moments(self)

    return ResumablePagedAdamW32bit(parameter_groups, lr=lr, betas=tuple(betas), eps=eps)


def parameter_groups(model, weight_decay):
    """Publication-era Trainer: exclude registered layer norms and all bias names."""
    from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
    from transformers.trainer_pt_utils import get_parameter_names

    named = list(model.named_parameters())
    if not named or any(not parameter.requires_grad for _, parameter in named):
        raise ValueError("D1 requires every unique model parameter trainable")
    if len({id(p) for _, p in named}) != len(named):
        raise ValueError("Tied weights must be counted only once")
    decay = {
        name for name in get_parameter_names(model, ALL_LAYERNORM_LAYERS) if "bias" not in name
    }
    groups = [
        {"params": [p for n, p in named if n in decay], "weight_decay": weight_decay},
        {"params": [p for n, p in named if n not in decay], "weight_decay": 0.0},
    ]
    inventory = [
        {
            "name": n,
            "shape": list(p.shape),
            "numel": p.numel(),
            "weight_decay": weight_decay if n in decay else 0.0,
        }
        for n, p in named
    ]
    return groups, inventory
