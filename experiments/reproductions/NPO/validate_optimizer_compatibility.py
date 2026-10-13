"""Offline source and real CPU ZeRO-3 checks for the D1 attempt-1 repair.

No downloads or real model weights. CPU AdamW is a scheduler-test fixture only;
the runtime continues to use the unchanged paged FP32 AdamW implementation.
"""

import argparse
import ast
import copy
import json
import os
from pathlib import Path

import torch

from experiments.reproductions.common import file_digest
from experiments.reproductions.NPO.engine import source_scheduler
from experiments.reproductions.NPO.optimizer import parameter_groups
from experiments.reproductions.NPO.validate_training_source import extract

PINNED_FILES = {
    "modeling_llama.py": "d837afb31f42c167c43e095923a2a89642424f54d603948e20d38ed4076fb478",
    "trainer_pt_utils.py": "b77d4dee27208c4cd97254d43ba249d446f86436c344ab7ddba646fd546a3833",
}


def meta_model_for_config(config):
    """Match from_pretrained's embedding aliases without allocating weights."""
    from transformers import LlamaForCausalLM

    with torch.device("meta"):
        model = LlamaForCausalLM(config)
    model.tie_weights()
    if config.tie_word_embeddings and model.lm_head.weight is not model.model.embed_tokens.weight:
        raise ValueError("Meta fixture did not reproduce tied input/output weights")
    return model


def source_groups(sources, trainer_path, target_config):
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    sources, trainer_path = Path(sources), Path(trainer_path)
    for name, sha in PINNED_FILES.items():
        if file_digest(sources / name) != sha:
            raise ValueError("Optimizer source hash mismatch")
    if (
        file_digest(trainer_path)
        != "fb3579779306c512c0b2ea0d3ce658f60cd572b4b207e80031d08ad06e45d45d"
    ):
        raise ValueError("Historical Trainer hash mismatch")
    if (
        file_digest(target_config)
        != "f851a2874b759ee56099c11786b9de7f3e6051677615d4a88259ef0e206ba8a9"
    ):
        raise ValueError("Pinned target architecture hash mismatch")
    ns = {"ALL_LAYERNORM_LAYERS": [torch.nn.LayerNorm], "LlamaRMSNorm": LlamaRMSNorm, "List": list}
    tree = ast.parse((sources / "modeling_llama.py").read_bytes())
    registration = next(
        node
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value) == "ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)"
    )
    # Execute the pinned registration and grouping functions. The modern
    # equivalent norm class is the sole model/API binding in this namespace.
    exec(compile(ast.Module(body=[registration], type_ignores=[]), "pinned_llama", "exec"), ns)
    extract(sources / "trainer_pt_utils.py", "get_parameter_names", ns)
    upstream = extract(trainer_path, "get_decay_parameter_names", ns, "Trainer")
    configs = [
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
        ),
        LlamaConfig.from_json_file(str(target_config)),
    ]
    results = []
    for config in configs:
        # Bare meta construction in Transformers 5 does not apply the tied
        # embedding alias used by from_pretrained. Match the runtime topology.
        model = meta_model_for_config(config)
        decay_names = set(upstream(None, model))
        # Trainer creates groups from named_parameters(), which deduplicates
        # tied aliases; get_decay_parameter_names alone also lists lm_head.
        expected = {name for name, _ in model.named_parameters() if name in decay_names}
        _, inventory = parameter_groups(model, 0.01)
        actual = {x["name"] for x in inventory if x["weight_decay"] == 0.01}
        if expected != actual:
            raise ValueError("Decay assignment differs from executed pinned Trainer")
        results.append(
            {
                "parameters": sum(x["numel"] for x in inventory),
                "tensor_count": len(inventory),
                "no_decay": [x for x in inventory if x["name"] not in actual],
            }
        )
    return {
        "status": "PASS_PINNED_SOURCE_GROUPS",
        "source_commit": "09f9f566de83eef1f13ee83b5a1bbeebde5c80c1",
        "files": PINNED_FILES,
        "source_registration_executed": True,
        "cases": results,
    }


def native_scheduler(output):
    import deepspeed
    from transformers import LlamaConfig, LlamaForCausalLM

    if torch.cuda.is_available():
        raise ValueError("CPU regression validation only")
    output = Path(output).resolve()
    torch.set_num_threads(1)
    deepspeed.ops.__compatible_ops__["deepspeed_shm_comm"] = False
    os.environ.update(LOCAL_RANK="0", RANK="0", WORLD_SIZE="1")
    torch.distributed.init_process_group(
        "gloo", init_method="file://" + str(output / "rendezvous"), rank=0, world_size=1
    )
    config = {
        "train_batch_size": 1,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "zero_optimization": {"stage": 3},
        "zero_allow_untested_optimizer": True,
        "gradient_clipping": 0.0,
    }

    def construct(early, llama):
        model = (
            LlamaForCausalLM(
                LlamaConfig(
                    vocab_size=32,
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                )
            )
            if llama
            else torch.nn.Linear(2, 1, bias=False)
        )
        groups = parameter_groups(model, 0.01)[0]
        optimizer = torch.optim.AdamW(groups, lr=1e-5)
        scheduler = source_scheduler(optimizer) if early else source_scheduler
        return deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            config=config,
            dist_init_required=False,
        )[0]

    def micro(engine, llama=False):
        if llama:
            ids = torch.tensor([[1, 2, 3, 4]])
            loss = engine(input_ids=ids, labels=ids).loss
        else:
            loss = engine(torch.ones(1, 2)).square().sum()
        engine.backward(loss)
        engine.step()

    try:
        old = construct(True, False)
        try:
            micro(old)
        except ValueError as error:
            if "zip() argument 2 is longer than argument 1" not in str(error):
                raise
            reproduced = str(error)
        else:
            raise ValueError("Old initialization did not reproduce the recorded failure")
        finally:
            old.destroy()
        curves = []
        for llama in (False, True):
            engine = construct(False, llama)
            curve = []
            for step in range(63):
                expected = 1e-5 * (step / 6 if step < 6 else max(0, (62 - step) / 56))
                actual = engine.lr_scheduler.get_last_lr()
                if any(abs(x - expected) > 1e-16 for x in actual):
                    raise ValueError("Source learning-rate trajectory changed")
                curve.append(actual)
                if step < 62:
                    micro(engine, llama)
            curves.append(
                {
                    "llama": llama,
                    "steps": engine.global_steps,
                    "groups": len(engine.optimizer.param_groups),
                    "lr_curve": curve,
                }
            )
            # Native checkpoint restore retains both the final group inventory
            # and scheduler state, even after DeepSpeed cleared parameter lists.
            checkpoint = output / ("llama_checkpoint" if llama else "empty_group_checkpoint")
            saved = copy.deepcopy(engine.lr_scheduler.state_dict())
            engine.save_checkpoint(str(checkpoint), tag="step62")
            engine.destroy()
            del engine
            engine = construct(False, llama)
            restored, _ = engine.load_checkpoint(str(checkpoint), tag="step62")
            if (
                not restored
                or engine.lr_scheduler.state_dict() != saved
                or engine.global_steps != 62
            ):
                raise ValueError("Scheduler restore mismatch")
            engine.destroy()
        return {
            "status": "PASS_CPU_NATIVE_ZERO3_SCHEDULER_ONLY",
            "old_failure_reproduced": reproduced,
            "cases": curves,
            "native_scheduler_restore": True,
            "gpu_or_paged_optimizer_validated": False,
        }
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--trainer", required=True)
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "source": source_groups(args.sources, args.trainer, args.target_config),
        "native_cpu": native_scheduler(output),
    }
    (output / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {"source": result["source"]["status"], "native_cpu": result["native_cpu"]["status"]}
        )
    )
