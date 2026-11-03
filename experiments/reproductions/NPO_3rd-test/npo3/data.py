"""Pinned TOFU inputs, explicit chat masking, and replayable sampling plans."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics import sha256

PINS = json.loads((Path(__file__).parent.parent / "inputs_v1.json").read_text())


def resolve_inputs(config):
    from huggingface_hub import hf_hub_download, snapshot_download
    from transformers import AutoTokenizer

    paths = {}
    for kind, setting, pin_key in (
        ("full", "full_path", "target"),
        ("retain95", "retain_path", "retain_only_oracle"),
    ):
        pin = PINS[pin_key]
        value = getattr(config, setting)
        folder = (
            Path(value)
            if value
            else Path(
                snapshot_download(
                    pin["model_id"],
                    revision=pin["revision"],
                    local_files_only=not config.allow_model_download,
                    allow_patterns=["*.safetensors", "*.json"],
                )
            )
        )
        weight = folder / "model.safetensors"
        if not weight.is_file():
            raise FileNotFoundError(f"Original downloaded weights required: {weight}")
        if sha256(weight) != pin["checkpoint_sha256"]:
            raise ValueError(f"Wrong {kind} model bytes")
        settings = json.loads((folder / "config.json").read_text())
        if settings.get("quantization_config") or settings.get("model_type") != "llama":
            raise ValueError("Original unquantized Llama checkpoint required")
        paths[kind] = folder.resolve()
    if sha256(paths["full"] / "config.json") != PINS["target"]["config_sha256"]:
        raise ValueError("Full-model configuration changed")
    generation_path = paths["full"] / "generation_config.json"
    if (
        sha256(generation_path)
        != "f571455b7715ad274df4ffe832ab20e374e2ea1d6fde467a80e04da04e038f71"
    ):
        raise ValueError("Original full-model generation configuration changed")
    source_generation = json.loads(generation_path.read_text())
    eos = source_generation["eos_token_id"]
    config.generation_eos_ids = eos if isinstance(eos, list) else [eos]
    paths["tokenizer"] = Path(config.tokenizer_path) if config.tokenizer_path else paths["full"]
    tokenizer_files = [
        f
        for f in PINS["files"]
        if f["repo"] == PINS["target"]["model_id"]
        and f["path"].startswith(("tokenizer", "special_tokens"))
    ]
    for item in tokenizer_files:
        if sha256(paths["tokenizer"] / item["path"]) != item["sha256"]:
            raise ValueError(f"Tokenizer differs: {item['path']}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(paths["tokenizer"]), local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    data, receipts = {}, {}
    for item in PINS["files"]:
        if item["repo"] != PINS["dataset"]["id"]:
            continue
        path = (
            Path(config.data_path) / item["path"] if config.data_path else Path("__not_supplied__")
        )
        if not path.is_file():
            path = Path(
                hf_hub_download(
                    PINS["dataset"]["id"],
                    filename=item["path"],
                    repo_type="dataset",
                    revision=PINS["dataset"]["revision"],
                    local_files_only=not config.allow_data_download,
                )
            )
        if sha256(path) != item["sha256"]:
            raise ValueError(f"Dataset bytes changed: {path}")
        name = Path(item["path"]).stem
        data[name] = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        receipts[name] = {"sha256": item["sha256"], "path": str(path), "rows": len(data[name])}
    if len(data["forget05"]) != 200 or len(data["retain95"]) != 3800:
        raise ValueError("Expected complete forget05 and retain95 splits")
    return paths, tokenizer, data, receipts


def prompt_ids(tokenizer, question, config):
    return list(
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": question},
            ],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            date_string=config.template_date,
        )
    )


def encode_qa(tokenizer, question, answer, config, max_length=None):
    if (
        not isinstance(question, str)
        or not question.strip()
        or not isinstance(answer, str)
        or not answer.strip()
    ):
        raise ValueError("Nonempty question and answer required")
    prompt = prompt_ids(tokenizer, question, config)
    ids = list(
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=False,
            date_string=config.template_date,
        )
    )
    if ids[: len(prompt)] != prompt:
        raise ValueError("Prompt is not an exact token prefix")
    if ids[-1] != tokenizer.eos_token_id:
        ids.append(tokenizer.eos_token_id)
    limit = max_length or config.training_max_length
    if len(ids) > limit:
        raise ValueError(
            f"Unexpected truncation would be required ({len(ids)} > {limit}): {question}"
        )
    if len(ids) <= len(prompt) + 1:
        raise ValueError("No answer content")
    return {
        "input_ids": ids,
        "labels": [-100] * len(prompt) + ids[len(prompt) :],
        "prompt_length": len(prompt),
    }


def one_batch(example, device, pad_id, fixed_length=None):
    ids, labels = list(example["input_ids"]), list(example["labels"])
    length = len(ids)
    target_length = fixed_length or length
    if target_length < length:
        raise ValueError("Truncation forbidden")
    mask = [1] * length + [0] * (target_length - length)
    ids += [pad_id] * (target_length - length)
    labels += [-100] * (target_length - length)
    return {
        "input_ids": torch.tensor([ids], device=device),
        "attention_mask": torch.tensor([mask], device=device),
    }, torch.tensor([labels], device=device)


def prepare_examples(data, tokenizer, config):
    forget_rows, retain_rows = data["forget05"], data["retain95"]
    forget_q = {r["question"] for r in forget_rows}
    retain_q = {r["question"] for r in retain_rows}
    if forget_q & retain_q:
        raise ValueError("Forget/retain leakage")
    forget = [encode_qa(tokenizer, r["question"], r["answer"], config) for r in forget_rows]
    retain = [encode_qa(tokenizer, r["question"], r["answer"], config) for r in retain_rows]
    if {r["question"] for r in data["forget05_perturbed"]} != forget_q:
        raise ValueError("Forget diagnostic differs from training split")
    eligible = [(i, r) for i, r in enumerate(data["retain_perturbed"]) if r["question"] in retain_q]
    rng = np.random.default_rng(config.evaluation_seed)
    chosen = sorted(
        rng.choice(len(eligible), config.evaluation_retain_count, replace=False).tolist()
    )
    selected = {
        "forget05": list(enumerate(data["forget05_perturbed"])),
        "retain": [eligible[i] for i in chosen],
        "real_authors": list(enumerate(data["real_authors_perturbed"])),
        "world_facts": list(enumerate(data["world_facts_perturbed"])),
    }
    source_indices = {row["question"]: i for i, row in enumerate(forget_rows)}
    packs = {}
    for split, rows in selected.items():
        packs[split] = []
        for source_index, row in rows:
            base = row["paraphrased_answer"] if split in ("forget05", "retain") else row["answer"]
            pack = {
                "id": f"{split}:{source_index:04d}",
                "split": split,
                "source_index": source_index,
                "question": row["question"],
                "ground_truth": row["answer"],
                "gold": encode_qa(
                    tokenizer, row["question"], row["answer"], config, config.evaluation_max_length
                ),
                "base": encode_qa(
                    tokenizer, row["question"], base, config, config.evaluation_max_length
                ),
                "wrong": [
                    encode_qa(tokenizer, row["question"], x, config, config.evaluation_max_length)
                    for x in row["perturbed_answer"]
                ],
            }
            if split == "forget05":
                # The pinned TOFU split stores 20 contiguous QAs per fictitious author.
                pack["author_block"] = source_indices[row["question"]] // 20
            packs[split].append(pack)
    selection = {
        split: [
            {"id": p["id"], "source_index": p["source_index"], "question": p["question"]}
            for p in rows
        ]
        for split, rows in packs.items()
    }
    return forget, retain, packs, selection


def build_plan(forget_count, retain_count, config, seed):
    steps = forget_count * config.nominal_epochs // config.effective_batch
    if steps <= config.warmup_steps or forget_count % config.logical_micro_batch:
        raise ValueError("Unsupported schedule or incomplete logical microbatch")
    generator = torch.Generator().manual_seed(1001 + seed - config.seeds[0])
    loader = DataLoader(
        list(range(forget_count)),
        batch_size=config.logical_micro_batch,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )
    batches = []
    while len(batches) < steps * config.accumulation:
        for batch in loader:
            batches.append(batch.tolist())
            if len(batches) == steps * config.accumulation:
                break
    retain_generator = torch.Generator().manual_seed(seed + 2001)
    pool, cursor, retain_batches = [], 0, []
    for batch in batches:
        paired = []
        for index in batch:
            if config.retain_sampling == "source_fixed":
                draw = torch.randint(
                    retain_count, (1,), generator=torch.Generator().manual_seed(index)
                ).item()
                paired.append((index + draw) % retain_count)
            else:
                if cursor == len(pool):
                    pool = torch.randperm(retain_count, generator=retain_generator).tolist()
                    cursor = 0
                paired.append(pool[cursor])
                cursor += 1
        retain_batches.append(paired)
    return {
        "steps": steps,
        "forget_batches": batches,
        "retain_batches": retain_batches,
        "evaluation_steps": sorted(
            set(range(config.evaluation_every_updates, steps + 1, config.evaluation_every_updates))
            | {steps}
        ),
        "unique_retain_rows": len({i for batch in retain_batches for i in batch}),
        "retain_sampling": config.retain_sampling,
        "effective_data_passes": steps * config.effective_batch / forget_count,
        "seed": seed,
    }
