"""Offline D1 source/data/resource inspection. Never a training entry point."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

from experiments.reproductions.common import digest, file_digest

SOURCE = Path(__file__).with_name("source_v1.json")


def same_numeric_setting(value, expected):
    """PyYAML reads the author's `1e-5` spelling as text; compare its numeric value."""
    if isinstance(value, bool) or isinstance(expected, bool):
        return False
    try:
        return float(value) == float(expected)
    except (ValueError, TypeError, OverflowError):
        return False


def positive_integer(value):
    if type(value) is not int or value <= 0:
        raise ValueError("Positive integer required")
    return value


def source_schedule(rows, epochs, micro_batch, accumulation, world_size):
    """Author forget.py uses integer floor, not ceil(epoch batches)."""
    for value in (rows, epochs, micro_batch, accumulation, world_size):
        positive_integer(value)
    effective_batch = micro_batch * accumulation * world_size
    interval = rows // effective_batch
    if interval == 0:
        raise ValueError("Source schedule would have zero warmup/evaluation interval")
    steps = epochs * rows // effective_batch
    return {
        "effective_batch": effective_batch,
        "max_steps": steps,
        "warmup_steps": interval,
        "eval_steps": list(range(interval, steps + 1, interval)),
    }


def llama_parameter_count(config):
    """Unique bias-free GQA Llama weights; reject other architectures explicitly."""
    if (
        config.get("model_type") != "llama"
        or config.get("attention_bias") is not False
        or config.get("mlp_bias") is not False
        or type(config.get("tie_word_embeddings")) is not bool
    ):
        raise ValueError("Expected explicit bias-free Llama configuration")
    h, layers, intermediate, vocab, heads, kv_heads, head_dim = (
        positive_integer(config[key])
        for key in (
            "hidden_size",
            "num_hidden_layers",
            "intermediate_size",
            "vocab_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
        )
    )
    if h != heads * head_dim or heads % kv_heads:
        raise ValueError("Unsupported attention dimensions")
    per_layer = 2 * h * heads * head_dim + 2 * h * kv_heads * head_dim
    per_layer += 3 * h * intermediate + 2 * h
    embedding_copies = 1 if config["tie_word_embeddings"] else 2
    return embedding_copies * vocab * h + layers * per_layer + h


def state_inventory(parameters):
    """Logical state sizes, NOT a prediction of resident paged-optimizer VRAM."""
    positive_integer(parameters)
    components = {
        "candidate_fp16": 2 * parameters,
        "reference_fp16": 2 * parameters,
        "master_fp32": 4 * parameters,
        "adam_first_moment_fp32": 4 * parameters,
        "adam_second_moment_fp32": 4 * parameters,
        "gradients_fp16": 2 * parameters,
    }
    return {
        "parameters": parameters,
        "components_bytes": components,
        "all_resident_bytes_with_fp16_gradients": sum(components.values()),
        "all_resident_bytes_with_fp32_gradients": sum(components.values()) + 2 * parameters,
        "excludes": ["activations", "allocator", "communication", "temporary buffers"],
        "paged_optimizer_resident_peak_bytes": None,
        "runtime_memory_acceptance": "NOT_MEASURED",
        "qualification": "Author selects paged 32-bit AdamW; managed states can migrate to CPU. Logical state bytes do not prove OOM or fit.",
    }


def verify_files(root, records):
    root = Path(root).resolve()
    result = {}
    for name, record in records.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Missing or unsafe pinned input path")
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if sha != record["sha256"] or ("git_blob" in record and blob != record["git_blob"]):
            raise ValueError("Pinned input content mismatch: " + name)
        result[name] = {"sha256": sha, "git_blob": blob, "bytes": len(data)}
    return result


def membership(rows_by_split):
    records = {}
    all_questions = {}
    for split, rows in rows_by_split.items():
        if not rows:
            raise ValueError("Empty source split")
        repeated = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or any(
                not isinstance(row.get(key), str) or not row[key].strip()
                for key in ("question", "answer")
            ):
                raise ValueError("Malformed TOFU question/answer")
            question = row["question"]
            if question in all_questions:
                old_split, old_index = all_questions[question]
                if old_split != split:
                    raise ValueError("Overlapping forget/retain questions require review")
                # Pinned retain95 contains a repeated question with different
                # answers. Preserve both source rows, identified by split/index.
                repeated.append([old_index, index])
            all_questions[question] = (split, index)
        records[split] = {
            "count": len(rows),
            "ordered_source_records_sha256": digest(rows),
            "ordered_question_sha256": digest([row["question"] for row in rows]),
            "repeated_question_indices_preserved": repeated,
        }
    return records


def validate(upstream, assets):
    import torch
    import transformers
    import yaml

    from experiments.reproductions.NPO.adapter import encode_example
    from experiments.reproductions.NPO.assets import PINS
    from experiments.reproductions.NPO.validate_adapter import (
        UpstreamTokenizerCompatibility,
        specification,
        upstream_nodes,
    )

    source = json.loads(SOURCE.read_text())
    verified_source = verify_files(upstream, source["files"])
    source_config = yaml.safe_load((Path(upstream) / "TOFU/config/forget.yaml").read_text())
    args = source["source_training"]
    for name, local_name in (
        ("forget_loss", "variant"),
        ("npo_coeff", "npo_coeff"),
        ("grad_diff_coeff", "grad_diff_coeff"),
        ("beta", "beta"),
        ("ref_policy", "ref_policy"),
        ("lr", "learning_rate"),
        ("weight_decay", "weight_decay"),
        ("seed", "seed"),
        ("batch_size", "micro_batch"),
        ("gradient_accumulation_steps", "gradient_accumulation"),
        ("num_epochs", "num_epochs"),
    ):
        actual, expected = source_config[name], args[local_name]
        equal = same_numeric_setting(actual, expected) if name == "lr" else actual == expected
        if not equal:
            raise ValueError("Source specification disagrees with pinned config: " + name)
    if source_config["LoRA"]["r"] != args["lora_rank"]:
        raise ValueError("Source trainable-parameter configuration changed")
    if file_digest(PINS) != source["inputs_registry_sha256"]:
        raise ValueError("Previously accepted input registry changed")
    pins = json.loads(PINS.read_text())
    assets = Path(assets)
    for item in pins["files"]:
        directory = "dataset" if item["repo"] == pins["dataset"]["id"] else "tokenizer"
        verify_files(assets / directory, {item["path"]: item})
    upstream_records = pins["upstreams"]["openunlearning"]["files"]
    verify_files(assets / "openunlearning", {k: {"sha256": v} for k, v in upstream_records.items()})
    cfg = json.loads((assets / "tokenizer/config.json").read_text())
    count = llama_parameter_count(cfg)
    # A second count uses the real library structure without allocating weights.
    with torch.device("meta"):
        model = transformers.LlamaForCausalLM(transformers.LlamaConfig(**cfg))
        # Direct construction on meta does not automatically tie in Transformers 5.
        model.tie_weights()
    if sum(p.numel() for p in model.parameters()) != count:
        raise ValueError("Analytic and tied meta-model parameter counts disagree")
    del model
    rows = {}
    for split in (source["selected_split"]["forget"], source["selected_split"]["retain"]):
        path = assets / "dataset" / (split + ".json")
        data = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if blob != source["selected_split"]["upstream_git_blobs"][path.name]:
            raise ValueError("Accepted dataset differs from author's published membership")
        rows[split] = [json.loads(line) for line in data.splitlines()]
    data_membership = membership(rows)
    for split, count_expected in (("forget05", 200), ("retain95", 3800)):
        if len(rows[split]) != count_expected:
            raise ValueError("Unexpected pinned split count")

    from typing import Any, Union

    ns = {
        "torch": torch,
        "Dict": dict,
        "List": list,
        "Union": Union,
        "Any": Any,
        "IGNORE_INDEX": -100,
    }
    upstream_nodes(
        assets / "openunlearning/src/data/utils.py",
        upstream_records["src/data/utils.py"],
        ["preprocess_chat_instance"],
        ns,
    )
    template = yaml.safe_load(
        (assets / "openunlearning/configs/model/Llama-3.2-1B-Instruct.yaml").read_text()
    )["template_args"]
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        assets / "tokenizer", local_files_only=True
    )
    spec = specification(pins, max_length=source["source_training"]["max_length"])
    boundary = []
    for split, split_rows in rows.items():
        for index, row in enumerate(split_rows):
            encoded = encode_example(row, index, split, tokenizer, spec)
            other = ns["preprocess_chat_instance"](
                UpstreamTokenizerCompatibility(tokenizer),
                template,
                [row["question"]],
                [row["answer"]],
                spec.max_length,
            )
            # Any real truncation demands a reviewed source/model adaptation.
            if encoded["truncation"]["truncated"]:
                raise ValueError("Real row truncated at source length; review token boundary")
            for key in ("input_ids", "attention_mask", "labels"):
                torch.testing.assert_close(encoded[key], other[key], rtol=0, atol=0)
            boundary.append(
                {
                    key: encoded[key]
                    for key in (
                        "sample_id",
                        "source_record_sha256",
                        "tensors_sha256",
                        "prompt_token_count",
                        "scored_token_count",
                        "truncation",
                    )
                }
            )
    schedules = {
        "source_two_gpu": source_schedule(200, 10, 4, 4, 2),
        "unchanged_knobs_one_gpu": source_schedule(200, 10, 4, 4, 1),
        "proposed_preserved_global_batch": source_schedule(200, 10, 4, 8, 1),
    }
    # Execute the actual two upstream assignments, not a second local formula.
    from types import SimpleNamespace

    parsed = ast.parse((Path(upstream) / "TOFU/forget.py").read_bytes())
    main_node = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    assignments = [
        n
        for n in main_node.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id in {"steps_per_epoch", "max_steps"} for t in n.targets
        )
    ]
    if len(assignments) != 2:
        raise ValueError("Expected source schedule definitions missing")
    for name, accumulation, devices in (
        ("source_two_gpu", 4, 2),
        ("unchanged_knobs_one_gpu", 4, 1),
        ("proposed_preserved_global_batch", 8, 1),
    ):
        namespace = dict(
            cfg=SimpleNamespace(num_epochs=10),
            torch_format_dataset=range(200),
            batch_size=4,
            gradient_accumulation_steps=accumulation,
            num_devices=devices,
        )
        exec(
            compile(
                ast.Module(body=assignments, type_ignores=[]), "verified_author_schedule", "exec"
            ),
            namespace,
        )
        if (namespace["max_steps"], namespace["steps_per_epoch"]) != (
            schedules[name]["max_steps"],
            schedules[name]["warmup_steps"],
        ):
            raise ValueError("Local schedule differs from executed upstream")
    return {
        "status": "PREFLIGHT_ONLY_RESOURCE_ADAPTER_PENDING",
        "source_manifest_sha256": file_digest(SOURCE),
        "source_files": verified_source,
        "inputs_registry_sha256": file_digest(PINS),
        "source_variant": args["variant"],
        "identities": {k: pins[k] for k in ("target", "dataset")},
        "membership": data_membership,
        "preprocessing_sha256": spec.sha256,
        "adapter_records": boundary,
        "adapter_records_sha256": digest(boundary),
        "parameter_state_inventory": state_inventory(count),
        "schedules": schedules,
        "schedule_checked_against_executed_author_assignments": True,
        "candidate_reference_gpu_validation": "NOT_RUN",
        "training_checkpoint_resume_acceptance": "NOT_IMPLEMENTED",
        "kaggle_submissions": 0,
        "attack_detected": None,
        "durability_pass": None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    # Reserve output before work: neither success nor failure may overwrite evidence.
    with Path(args.output).open("x") as output:
        try:
            result = validate(args.upstream, args.assets)
        except Exception as error:
            result = {"status": "FAIL", "error_type": type(error).__name__, "kaggle_submissions": 0}
        json.dump(result, output, indent=2)
        output.write("\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k in {"status", "error_type", "kaggle_submissions"}}
        )
    )
    return 1 if result["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
