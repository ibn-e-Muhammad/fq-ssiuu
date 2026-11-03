"""NPO+RT with only LoRA parameters trainable and the frozen base as reference."""

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
from peft import LoraConfig, TaskType, get_peft_model

from .checkpoint import restore_checkpoint, save_checkpoint
from .data import build_plan
from .evaluation import tensor_fingerprint
from .metrics import digest, sha256, write_json
from .objective import npo_loss, sequence_nll


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def attach_lora(base, config, seed):
    seed_everything(seed)
    base.requires_grad_(False)
    adapter = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_targets,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base, adapter, autocast_adapter_dtype=True)
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in n for n, _ in trainable):
        raise ValueError("Only LoRA parameters may be trainable")
    # Preserve small AdamW updates even when the frozen backbone uses BF16.
    for _, parameter in trainable:
        parameter.data = parameter.data.float()
    return model


def collate(examples, tokenizer, config):
    width = max(len(e["input_ids"]) for e in examples)
    ids = torch.full(
        (len(examples), width), tokenizer.pad_token_id, dtype=torch.long, device=config.device
    )
    labels = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(examples):
        size = len(row["input_ids"])
        ids[i, :size] = torch.tensor(row["input_ids"], device=config.device)
        labels[i, :size] = torch.tensor(row["labels"], device=config.device)
        mask[i, :size] = 1
    return {"input_ids": ids, "attention_mask": mask}, labels


def forward_sequence(model, examples, tokenizer, config):
    inputs, labels = collate(examples, tokenizer, config)
    logits = model(**inputs, use_cache=False).logits
    return sequence_nll(logits, labels), int((labels[:, 1:] != -100).sum())


def frozen_reference_nll(model, examples, tokenizer, config):
    """Adapters off + all base weights frozen = the original unquantized full model."""
    previous = model.training
    try:
        model.eval()
        with model.disable_adapter(), torch.no_grad():
            result, _ = forward_sequence(model, examples, tokenizer, config)
        if result.requires_grad:
            raise ValueError("Reference has a gradient graph")
        return result.detach()
    finally:
        model.train(previous)


def learning_rate(step, total, config):
    if step < config.warmup_steps:
        return config.learning_rate * step / max(1, config.warmup_steps)
    return config.learning_rate * max(0.0, (total - step) / max(1, total - config.warmup_steps))


def train(
    model,
    tokenizer,
    forget,
    retain,
    config,
    seed,
    folder,
    binding,
    evaluate_callback=None,
    resume=False,
    stop_after=None,
):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    plan = build_plan(len(forget), len(retain), config, seed)
    identity = digest(
        {
            "binding": binding,
            "config": config.to_dict(),
            "plan": plan,
            "training_code": sha256(Path(__file__)),
            "objective_code": sha256(Path(__file__).with_name("objective.py")),
            "forget": forget,
            "retain": retain,
        }
    )
    write_json(folder / "training_plan.json", plan)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if any("lora_" not in name for name, _ in named) or not named:
        raise ValueError("Full-weight training is forbidden in this LoRA experiment")
    groups = [
        {
            "params": [p for n, p in named if p.ndim >= 2 and not n.endswith("bias")],
            "weight_decay": config.weight_decay,
        },
        {"params": [p for n, p in named if p.ndim < 2 or n.endswith("bias")], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        groups, lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-8, foreach=False
    )
    base_before = tensor_fingerprint(model, skip_adapters=True)["sha256"]
    history, first_step = [], 0
    checkpoint_root = folder / "checkpoints"
    if resume:
        first_step, history, rejected = restore_checkpoint(
            checkpoint_root, model, optimizer, identity
        )
        write_json(
            folder / "resume_receipt.json",
            {"next_update": first_step + 1, "rejected": rejected, "identity": identity},
        )
    elif list(checkpoint_root.glob("step-*")):
        raise ValueError("Training checkpoints exist; explicitly resume this run")
    write_json(
        folder / "training_identity.json",
        {
            "identity": identity,
            "base_tensor_sha256": base_before,
            "trainable_names": [name for name, _ in named],
            "trainable_parameters": sum(p.numel() for _, p in named),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "lora": model.peft_config["default"].to_dict(),
            "reference": "same frozen unquantized full backbone with adapters disabled, eval, no_grad",
        },
    )
    model.config.use_cache = False
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    optimizer.zero_grad(set_to_none=True)
    try:
        if evaluate_callback and first_step and first_step in plan["evaluation_steps"]:
            # A power cut may have followed the checkpoint but preceded its evaluation.
            evaluate_callback(model, first_step, identity)
        for step in range(first_step, plan["steps"]):
            started = time.perf_counter()
            if str(config.device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(config.device)
            forget_loss, retain_loss = 0.0, 0.0
            for index in range(step * config.accumulation, (step + 1) * config.accumulation):
                forget_ids, retain_ids = (
                    plan["forget_batches"][index],
                    plan["retain_batches"][index],
                )
                retain_tokens = sum(
                    sum(x != -100 for x in retain[i]["labels"][1:]) for i in retain_ids
                )
                for start in range(0, len(forget_ids), config.device_micro_batch):
                    subset = [
                        forget[i] for i in forget_ids[start : start + config.device_micro_batch]
                    ]
                    reference = frozen_reference_nll(model, subset, tokenizer, config)
                    current, _ = forward_sequence(model, subset, tokenizer, config)
                    term = npo_loss(current, reference, config.beta)
                    weight = len(subset) / (len(forget_ids) * config.accumulation)
                    (config.npo_coefficient * term * weight).backward()
                    forget_loss += float(term.detach()) * weight
                    del current, reference, term
                if config.retain_coefficient:
                    for start in range(0, len(retain_ids), config.device_micro_batch):
                        subset = [
                            retain[i] for i in retain_ids[start : start + config.device_micro_batch]
                        ]
                        nll, _ = forward_sequence(model, subset, tokenizer, config)
                        term = nll.sum() / retain_tokens
                        (config.retain_coefficient * term / config.accumulation).backward()
                        retain_loss += float(term.detach()) / config.accumulation
                        del nll, term
            for name, parameter in named:
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"Missing or nonfinite LoRA gradient: {name}")
            lr = learning_rate(step, plan["steps"], config)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()  # No clipping, matching source effective clipping=0.
            optimizer.zero_grad(set_to_none=True)
            if str(config.device).startswith("cuda"):
                torch.cuda.synchronize(config.device)
            record = {
                "update": step + 1,
                "lr": lr,
                "npo_loss": forget_loss,
                "retain_ce": retain_loss,
                "total_loss": config.npo_coefficient * forget_loss
                + config.retain_coefficient * retain_loss,
                "effective_forget_passes": (step + 1) * config.effective_batch / len(forget),
                "compute_seconds": time.perf_counter() - started,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(config.device)
                if str(config.device).startswith("cuda")
                else 0,
                "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(config.device)
                if str(config.device).startswith("cuda")
                else 0,
                "process_ram_bytes": psutil.Process().memory_info().rss,
            }
            if not all(math.isfinite(value) for value in record.values()):
                raise FloatingPointError("Nonfinite training record")
            history.append(record)
            checkpoint = save_checkpoint(
                checkpoint_root,
                model,
                optimizer,
                step + 1,
                identity,
                history,
                config.keep_recent_checkpoints,
            )
            pd.DataFrame(history).to_csv(folder / "training_log.csv", index=False)
            print(json.dumps({"seed": seed, **record, "checkpoint": str(checkpoint)}), flush=True)
            if evaluate_callback and step + 1 in plan["evaluation_steps"]:
                evaluate_callback(model, step + 1, identity)
            if stop_after is not None and step + 1 >= stop_after:
                return {"status": "interrupted-fixture", "step": step + 1, "identity": identity}
        base_after = tensor_fingerprint(model, skip_adapters=True)["sha256"]
        if base_before != base_after:
            raise ValueError("Frozen backbone changed during LoRA training")
        model.eval()
        model.config.use_cache = True
        if config.gradient_checkpointing:
            model.gradient_checkpointing_disable()
        adapter_dir = folder / "adapter"
        model.save_pretrained(adapter_dir, safe_serialization=True)
        tokenizer.save_pretrained(adapter_dir)
        receipt = {
            "status": "complete",
            "identity": identity,
            "updates": plan["steps"],
            "base_unchanged": True,
            "base_tensor_sha256": base_after,
            "seed": seed,
            "adapter_hashes": {p.name: sha256(p) for p in adapter_dir.glob("*") if p.is_file()},
        }
        write_json(folder / "training_complete.json", receipt)
        return receipt
    except BaseException as exc:
        write_json(
            folder / "training_failure.json",
            {
                "error": repr(exc),
                "identity": identity,
                "completed_updates": len(history),
                "instruction": "Resume this run; the latest complete adapter/optimizer/RNG checkpoint is authoritative.",
            },
        )
        raise
