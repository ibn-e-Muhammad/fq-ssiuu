"""Bounded source loop shared by tiny CPU acceptance and the Kaggle runtime."""

import json
import math
from pathlib import Path

import torch

from experiments.reproductions.common import digest
from experiments.reproductions.NPO.training_data import fixed_batch
from experiments.reproductions.NPO.training_forward import microbatch_objective
from experiments.reproductions.NPO.training_state import validate_cursor


def run_training(
    engine,
    reference,
    pair_dataset,
    plan,
    reference_config,
    output,
    *,
    pad_token_id,
    evaluate,
    observe_memory,
    resume_cursor=None,
    runtime=True,
):
    """DeepSpeed alone scales by accumulation and updates at its own boundary.

    Evaluation runs on source steps 6,12,...60. The caller saves and evaluates
    the final step 62 separately. No outcome-based stop or best-checkpoint rule.
    """
    body = {k: v for k, v in plan.items() if k not in {"sha256", "generator_state"}}
    if digest(body) != plan["sha256"] or plan["accumulation"] != 8 or plan["max_steps"] != 62:
        raise ValueError("Changed source batch plan")
    start = 0
    if resume_cursor is not None:
        validate_cursor(resume_cursor)
        start = resume_cursor["optimizer_step"] * 8
        if resume_cursor["micro_step"] != 0 or engine.global_steps != start // 8:
            raise ValueError("Resume cursor and engine disagree")
        if start > len(plan["batches"]):
            raise ValueError("Resume beyond source stopping rule")
        if not torch.equal(resume_cursor["sampler_state"], plan["generator_state"]):
            raise ValueError("Resume sampler differs from the complete source plan")
        if start and (resume_cursor["epoch"], resume_cursor["batch_in_epoch"]) != (
            plan["batches"][start - 1]["epoch"],
            plan["batches"][start - 1]["batch_in_epoch"] + 1,
        ):
            raise ValueError("Resume epoch/batch cursor differs from source plan")
    elif engine.global_steps:
        raise ValueError("Non-pristine engine without an explicit resume cursor")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    completed = []
    engine.train()
    with (
        (output / "microbatches.jsonl").open("x") as log,
        (output / "forward_observations.jsonl").open("x") as forward_log,
    ):
        for position in range(start, len(plan["batches"])):
            batch_plan = plan["batches"][position]
            paired = [pair_dataset[i] for i in batch_plan["forget_indices"]]
            forget = fixed_batch(
                [x["forget"] for x in paired], max_length=500, pad_token_id=pad_token_id
            )
            retain = fixed_batch(
                [x["retain"] for x in paired], max_length=500, pad_token_id=pad_token_id
            )
            terms, forward = microbatch_objective(
                engine, reference, forget, retain, reference_config, runtime=runtime
            )
            record = {
                "microbatch": position,
                **batch_plan,
                "retain_indices": [x["retain_index"] for x in paired],
                "forget_sample_ids": forget["sample_ids"],
                "retain_sample_ids": retain["sample_ids"],
                "npo_loss": float(terms.forget.detach()),
                "retain_loss": float(terms.retain.detach()),
                "total_loss": float(terms.total.detach()),
                "lr_before_step": [group["lr"] for group in engine.optimizer.param_groups],
                **forward,
            }
            boundary = engine.is_gradient_accumulation_boundary()
            if boundary != ((position + 1) % 8 == 0):
                raise ValueError("DeepSpeed and source accumulation boundaries differ")
            forward_log.write(
                json.dumps({**record, "phase": "FORWARD_COMPLETE_BEFORE_BACKWARD"}, allow_nan=False)
                + "\n"
            )
            forward_log.flush()
            engine.backward(terms.total)
            del terms
            engine.step()
            record["optimizer_step"] = engine.global_steps
            record["optimizer_boundary"] = boundary
            record["skipped_optimizer_steps"] = engine.skipped_steps
            if boundary:
                overflow = bool(getattr(engine.optimizer, "overflow", False))
                norm = float(engine.get_global_grad_norm()) if not overflow else float("nan")
                # A dynamic FP16 overflow is recorded explicitly, never an
                # invented zero gradient or silently successful update.
                record["gradient_norm"] = norm if math.isfinite(norm) else None
                record["gradient_norm_nonfinite"] = not math.isfinite(norm)
                record["fp16_overflow"] = overflow
                record["loss_scale"] = float(engine.optimizer.loss_scale)
                if engine.global_steps != (position + 1) // 8:
                    raise ValueError("Engine steps differ from source stopping schedule")
                completed.append(record)
            record["memory"] = observe_memory()
            log.write(json.dumps(record, allow_nan=False) + "\n")
            log.flush()
            if boundary and engine.global_steps % 6 == 0:
                engine.eval()
                evaluate(engine.global_steps)
                engine.train()
    if engine.global_steps != 62:
        raise ValueError("Source stopping rule not reached")
    last = plan["batches"][-1]
    cursor = {
        "optimizer_step": 62,
        "micro_step": 0,
        "epoch": last["epoch"],
        "batch_in_epoch": last["batch_in_epoch"] + 1,
        "sampler_state": plan["generator_state"],
    }
    return cursor, completed
