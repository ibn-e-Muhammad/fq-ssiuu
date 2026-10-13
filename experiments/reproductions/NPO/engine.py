"""D1 uses the original full-parameter optimizer through native DeepSpeed ZeRO-3."""

import torch

from experiments.reproductions.NPO.optimizer import paged_adamw32, parameter_groups
from experiments.reproductions.NPO.training_contract import deepspeed_config, linear_schedule


def source_scheduler(optimizer):
    """Build after DeepSpeed has removed empty groups, preserving the source LR."""
    return torch.optim.lr_scheduler.LambdaLR(optimizer, linear_schedule)


def create_engine(model, *, sub_group_size=1000000000):
    import deepspeed

    model.requires_grad_(True)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    groups, inventory = parameter_groups(model, 0.01)
    optimizer = paged_adamw32(groups, lr=1e-5, betas=(0.9, 0.999), eps=1e-8)
    config = deepspeed_config(model.config.hidden_size, sub_group_size=sub_group_size)
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=source_scheduler,
        config=config,
        dist_init_required=False,
    )
    if engine.zero_optimization_stage() != 3 or not engine.fp16_enabled():
        raise ValueError("D1 requires native FP16 ZeRO-3")
    if any(p.dtype != torch.float32 for p in engine.optimizer.fp32_partitioned_groups_flat):
        raise ValueError("D1 requires FP32 master parameters")
    underlying = engine.optimizer.optimizer
    if not underlying.is_paged or underlying.args.optim_bits != 32:
        raise ValueError("D1 requires original paged 32-bit optimizer")
    if any(p.device.type != "cuda" for p in engine.optimizer.fp32_partitioned_groups_flat):
        raise ValueError("Unapproved master-parameter offload")
    return engine, inventory, config


def destroy_engine(engine):
    """Release only this optimizer's managed buffers after its state is saved.

    bnb's process-global page manager retains strong references. Leaving those
    entries alive would retain old moments while validating a fresh reload.
    """
    underlying = engine.optimizer.optimizer
    owned = {
        id(value)
        for state in underlying.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor) and getattr(value, "is_paged", False)
    }
    underlying.page_mng.paged_tensors[:] = [
        value for value in underlying.page_mng.paged_tensors if id(value) not in owned
    ]
    engine.destroy()
