"""Tiny random FP16/ZeRO-3/paged-AdamW resume acceptance, inside D1's one submission."""

import copy
import gc
from pathlib import Path

import torch

from experiments.reproductions.common import digest, file_digest, tensors_digest
from experiments.reproductions.NPO.candidate import compute_causal_npo_rt_loss
from experiments.reproductions.NPO.engine import create_engine, destroy_engine
from experiments.reproductions.NPO.evaluation_runtime import write_new
from experiments.reproductions.NPO.training_state import (
    capture_rng,
    load_deepspeed_state,
    save_deepspeed_state,
)


def validate(directory, identities, *, sub_group_size=1000000000):
    from transformers import LlamaConfig, LlamaForCausalLM

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(1001)
    cfg = LlamaConfig(
        vocab_size=1024,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        attention_dropout=0.0,
        max_position_embeddings=64,
        tie_word_embeddings=True,
        attn_implementation="sdpa",
    )
    model = LlamaForCausalLM(cfg).half()
    model.tie_weights()
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    parent_identities = identities
    identities = {
        **identities,
        "initial_checkpoint_sha256": tensors_digest(initial),
        "reference_checkpoint_sha256": tensors_digest(initial),
        "train_config_sha256": digest(
            {
                "model": cfg.to_dict(),
                "warmup_limit_updates": 16,
                "resume_comparison_microbatches": 16,
                "zero_sub_group_size": sub_group_size,
            }
        ),
        "eval_config_sha256": digest({"evaluation": "none"}),
        "tokenizer_revision": identities["source_commit"],
        "dataset_revision": identities["source_commit"],
        "preprocessing_sha256": file_digest(__file__),
        "sample_membership_sha256": digest(
            {
                "generator": "torch.randint",
                "shape": [4, 8],
                "vocabulary": 1024,
                "source": file_digest(__file__),
            }
        ),
    }
    reference = copy.deepcopy(model).cuda().eval().requires_grad_(False)
    del model

    def construct():
        candidate = LlamaForCausalLM(cfg).half()
        candidate.tie_weights()
        candidate.load_state_dict(initial)
        return create_engine(candidate.cuda(), sub_group_size=sub_group_size)[0]

    def micro(engine):
        ids = torch.randint(0, 1024, (4, 8), device="cuda:0")
        retained = torch.randint(0, 1024, (4, 8), device="cuda:0")
        labels, retain_labels = ids.clone(), retained.clone()
        labels[:, :3] = retain_labels[:, :3] = -100
        mask = torch.ones_like(ids)
        with torch.no_grad():
            ref = reference(input_ids=ids, attention_mask=mask).logits
        loss = compute_causal_npo_rt_loss(
            engine(input_ids=ids, attention_mask=mask).logits,
            ref,
            labels,
            engine(input_ids=retained, attention_mask=mask).logits,
            retain_labels,
        ).total
        value = float(loss.detach())
        engine.backward(loss)
        engine.step()
        return value

    engine = construct()
    # Bounded dynamic-loss-scale warmup on a random tiny model, never TOFU.
    for _ in range(16):
        for _ in range(8):
            micro(engine)
        if engine.global_steps - engine.skipped_steps >= 2:
            break
    if engine.global_steps - engine.skipped_steps < 2:
        raise ValueError("Tiny paged optimizer did not perform two finite updates")

    def paged_inventory(current):
        records = []
        for state in current.optimizer.optimizer.state.values():
            for key in ("state1", "state2"):
                if key in state:
                    tensor = state[key]
                    records.append(
                        {
                            "name": key,
                            "numel": tensor.numel(),
                            "dtype": str(tensor.dtype),
                            "is_paged": bool(getattr(tensor, "is_paged", False)),
                        }
                    )
        if not any(x["is_paged"] for x in records):
            raise ValueError("Tiny validation never exercised actual managed moment allocation")
        if any(x["dtype"] != "torch.float32" for x in records):
            raise ValueError("Wrong paged optimizer state precision")
        return records

    before = paged_inventory(engine)
    cursor = {
        "optimizer_step": engine.global_steps,
        "micro_step": 0,
        "epoch": 0,
        "batch_in_epoch": engine.global_steps * 8,
        "sampler_state": torch.Generator().manual_seed(1001).get_state(),
    }
    manifest = save_deepspeed_state(directory / "checkpoint", engine, cursor, identities)
    expected = [micro(engine) for _ in range(16)]
    expected_rng = capture_rng()
    masters = [p.detach().cpu().clone() for p in engine.optimizer.fp32_partitioned_groups_flat]
    scheduler = copy.deepcopy(engine.lr_scheduler.state_dict())
    final_steps = engine.global_steps
    destroy_engine(engine)
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    engine = construct()
    restored = load_deepspeed_state(directory / "checkpoint", engine, identities)
    after = paged_inventory(engine)
    actual = [micro(engine) for _ in range(16)]
    torch.testing.assert_close(torch.tensor(expected), torch.tensor(actual), rtol=0, atol=0)
    for expected_master, actual_master in zip(
        masters, engine.optimizer.fp32_partitioned_groups_flat, strict=True
    ):
        torch.testing.assert_close(expected_master, actual_master.cpu(), rtol=0, atol=0)
    rng = capture_rng()
    if not torch.equal(expected_rng["torch"], rng["torch"]) or any(
        not torch.equal(a, b) for a, b in zip(expected_rng["cuda"], rng["cuda"], strict=True)
    ):
        raise ValueError("GPU resume changed random stream")
    if engine.lr_scheduler.state_dict() != scheduler or engine.global_steps != final_steps:
        raise ValueError("GPU resume changed scheduler/step state")
    if restored["optimizer_step"] != cursor["optimizer_step"]:
        raise ValueError("GPU resume changed data cursor")
    result = {
        "status": "PASS_TINY_RANDOM_GPU_RESUME_ONLY",
        "checkpoint_manifest": manifest,
        "before_moments": before,
        "restored_moments": after,
        "compared_microbatches": 16,
        "max_loss_error": max(abs(a - b) for a, b in zip(expected, actual, strict=True)),
        "masters_bit_identical": True,
        "scheduler_equal": True,
        "rng_equal": True,
        "not_a_1b_training_result": True,
        "zero_sub_group_size": sub_group_size,
        "parent_run_identities": parent_identities,
        "synthetic_identity_semantics": "Random initial tensor hash; code revision identifies synthetic integer-token generator. No TOFU/model checkpoint identity is claimed for this test.",
    }
    write_new(directory / "RESULT.json", result)
    destroy_engine(engine)
    del engine
    reference = None
    gc.collect()
    torch.cuda.empty_cache()
    return result
