"""Exploratory RTX-friendly LoRA pilot; not the source-faithful D1 runner."""

import argparse
import gc
import json
import random
from pathlib import Path

import numpy as np
import torch

from experiments.reproductions.NPO.adapter import AdapterSpec, collate_examples, encode_rows
from experiments.reproductions.NPO.assets import PINS, materialize
from experiments.reproductions.NPO.candidate import causal_sequence_nll, npo_from_sequence_nll
from experiments.reproductions.NPO.training_data import SourcePairDataset, source_batch_plan
from fq_ssiuu.evaluation.identity import file_digest

ROOT = Path(__file__).resolve().parents[3]
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def validate_settings(max_steps, max_length, accumulation, rank, eval_samples):
    if type(max_steps) is not int or not 1 <= max_steps <= 62:
        raise ValueError("max_steps must be between 1 and 62")
    if type(max_length) is not int or not 3 <= max_length <= 500:
        raise ValueError("max_length must be between 3 and 500")
    if type(accumulation) is not int or not 1 <= accumulation <= 8:
        raise ValueError("accumulation must be between 1 and 8")
    if type(rank) is not int or not 1 <= rank <= 16:
        raise ValueError("LoRA rank must be between 1 and 16")
    if type(eval_samples) is not int or not 1 <= eval_samples <= 32:
        raise ValueError("eval_samples must be between 1 and 32")


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def load_rows(dataset_dir):
    return {
        split: [
            json.loads(line)
            for line in (dataset_dir / f"{split}.json").read_text(encoding="utf-8").splitlines()
        ]
        for split in ("forget05", "retain95")
    }


def checked_model_directory(target):
    from huggingface_hub import hf_hub_download

    config = Path(
        hf_hub_download(
            target["model_id"], "config.json", revision=target["revision"], token=False
        )
    )
    weights = Path(
        hf_hub_download(
            target["model_id"], "model.safetensors", revision=target["revision"], token=False
        )
    )
    if file_digest(config) != target["config_sha256"]:
        raise ValueError("Downloaded model config differs from the pinned SHA-256")
    if file_digest(weights) != target["checkpoint_sha256"]:
        raise ValueError("Downloaded model weights differ from the pinned SHA-256")
    if config.parent != weights.parent:
        raise ValueError("Pinned model files resolved to different Hugging Face snapshots")
    return config.parent, {
        "config.json": {"sha256": file_digest(config), "bytes": config.stat().st_size},
        "model.safetensors": {"sha256": file_digest(weights), "bytes": weights.stat().st_size},
    }


def load_model(model_dir):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )
    model.to("cuda:0")
    model.config.use_cache = False
    return model


def prepare_batch(examples, tokenizer, spec):
    batch = collate_examples(examples, tokenizer, spec)
    return {name: value.to("cuda:0") for name, value in batch["tensors"].items()}


def sequence_nll(model, batch):
    with torch.no_grad():
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        values = causal_sequence_nll(logits, batch["labels"])
    del logits
    return values


def evaluate(model, examples, tokenizer, spec):
    forget_values = []
    retain_values = []
    for example in examples["forget05"]:
        batch = prepare_batch([example], tokenizer, spec)
        forget_values.append(float(sequence_nll(model, batch).item()))
    for example in examples["retain95"]:
        batch = prepare_batch([example], tokenizer, spec)
        values = sequence_nll(model, batch)
        scored_tokens = int((batch["labels"][:, 1:] != -100).sum().item())
        retain_values.append(float((values.sum() / scored_tokens).item()))
    return {
        "forget_sequence_nll_mean": float(np.mean(forget_values)),
        "retain_token_nll_mean": float(np.mean(retain_values)),
        "forget_count": len(forget_values),
        "retain_count": len(retain_values),
    }


def run(args):
    validate_settings(args.max_steps, args.max_length, args.accumulation, args.rank, args.eval_samples)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("The local pilot requires exactly one visible CUDA GPU")

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    cache = Path(args.cache_dir).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    target = pins["target"]
    spec = AdapterSpec(
        target["model_id"],
        target["revision"],
        target["tokenizer_id"],
        target["tokenizer_revision"],
        pins["dataset"]["id"],
        pins["dataset"]["revision"],
        max_length=args.max_length,
    )
    spec.validate()
    run_config = {
        "classification": "exploratory-local-lora-pilot",
        "source_faithful_d1": False,
        "model_id": target["model_id"],
        "model_revision": target["revision"],
        "dataset_id": pins["dataset"]["id"],
        "dataset_revision": pins["dataset"]["revision"],
        "objective": "NPO+retain; beta=0.1; coefficients=1.0/1.0",
        "optimizer": "torch.optim.AdamW",
        "optimizer_betas": [0.9, 0.999],
        "optimizer_epsilon": 1e-8,
        "weight_decay": 0.01,
        "scheduler": "constant",
        "gradient_clipping": 0.0,
        "trainable_parameters": "LoRA adapters",
        "lora": {"rank": args.rank, "alpha": args.rank * 2, "dropout": 0.05},
        "target_modules": TARGET_MODULES,
        "micro_batch": 1,
        "gradient_accumulation": args.accumulation,
        "optimizer_steps": args.max_steps,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "precision": "fp16 base model; LoRA adapter default PEFT dtype",
        "seed": args.seed,
        "evaluation_samples_per_split": args.eval_samples,
        "limitations": [
            "LoRA replaces full-parameter training.",
            "Single-example GPU passes and local settings are not source-faithful D1.",
            "Exploratory losses do not establish unlearning effectiveness or quantisation durability.",
        ],
    }
    write_new(output / "run_config.json", run_config)
    write_new(
        output / "environment.json",
        {
            "python": __import__("sys").version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "total_gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        },
    )

    asset_records = materialize(cache, upstream=False)
    tokenizer_dir = cache / "tokenizer"
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
    if tokenizer.pad_token_id is None:
        raise ValueError("Pinned tokenizer has no pad token")
    rows = load_rows(cache / "dataset")
    if len(rows["forget05"]) != 200 or len(rows["retain95"]) != 3800:
        raise ValueError("Pinned TOFU split sizes differ from the expected membership")
    encoded = {
        split: encode_rows(values, split, tokenizer, spec) for split, values in rows.items()
    }
    truncated = {
        split: sum(int(row["truncation"]["truncated"]) for row in values)
        for split, values in encoded.items()
    }
    write_new(
        output / "input_summary.json",
        {
            "forget_count": len(encoded["forget05"]),
            "retain_count": len(encoded["retain95"]),
            "truncated_examples": truncated,
            "adapter_spec": spec.__dict__,
        },
    )

    model_dir, model_files = checked_model_directory(target)
    write_new(output / "input_assets.json", {"files": asset_records, "model_files": model_files})
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.reset_peak_memory_stats(0)

    reference = load_model(model_dir)
    reference.eval().requires_grad_(False)
    reference_nll = []
    with torch.inference_mode():
        for example in encoded["forget05"]:
            batch = prepare_batch([example], tokenizer, spec)
            logits = reference(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            ).logits
            reference_nll.append(float(causal_sequence_nll(logits, batch["labels"]).item()))
            del logits, batch
    del reference
    gc.collect()
    torch.cuda.empty_cache()
    write_new(
        output / "reference_nll.json",
        {
            "model_revision": target["revision"],
            "values": reference_nll,
            "note": "FP16 pinned-reference sequence NLLs, precomputed and stored on CPU.",
        },
    )

    from peft import LoraConfig, get_peft_model

    candidate = load_model(model_dir)
    candidate.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    candidate.enable_input_require_grads()
    candidate = get_peft_model(
        candidate,
        LoraConfig(
            r=args.rank,
            lora_alpha=args.rank * 2,
            lora_dropout=0.05,
            target_modules=TARGET_MODULES,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    candidate.train()
    trainable = sum(parameter.numel() for parameter in candidate.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in candidate.parameters())
    if not trainable or trainable >= total:
        raise ValueError("Expected a nonempty LoRA-only trainable parameter set")
    write_new(
        output / "parameter_summary.json",
        {"trainable_parameters": trainable, "total_parameters": total},
    )

    eval_examples = {
        "forget05": encoded["forget05"][: args.eval_samples],
        "retain95": encoded["retain95"][: args.eval_samples],
    }
    candidate.eval()
    before = evaluate(candidate, eval_examples, tokenizer, spec)
    candidate.train()

    optimizer = torch.optim.AdamW(
        [parameter for parameter in candidate.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    dataset = SourcePairDataset(encoded["forget05"], encoded["retain95"])
    plan = source_batch_plan(
        len(encoded["forget05"]), 1, args.accumulation, args.max_steps, args.seed
    )
    ref_nll = torch.tensor(reference_nll, dtype=torch.float32)
    torch.cuda.reset_peak_memory_stats(0)
    with (output / "training.jsonl").open("x", encoding="utf-8") as log:
        for step in range(args.max_steps):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for micro_index in range(args.accumulation):
                position = step * args.accumulation + micro_index
                sample_index = plan["batches"][position]["forget_indices"][0]
                paired = dataset[sample_index]
                forget = prepare_batch([paired["forget"]], tokenizer, spec)
                retain = prepare_batch([paired["retain"]], tokenizer, spec)

                forget_logits = candidate(
                    input_ids=forget["input_ids"], attention_mask=forget["attention_mask"]
                ).logits
                candidate_nll = causal_sequence_nll(forget_logits, forget["labels"])
                del forget_logits
                forget_loss = npo_from_sequence_nll(
                    candidate_nll,
                    ref_nll[sample_index : sample_index + 1].to("cuda:0"),
                    beta=0.1,
                )
                retain_logits = candidate(
                    input_ids=retain["input_ids"], attention_mask=retain["attention_mask"]
                ).logits
                retain_nll = causal_sequence_nll(retain_logits, retain["labels"])
                scored_tokens = (retain["labels"][:, 1:] != -100).sum()
                retain_loss = retain_nll.sum() / scored_tokens
                loss = forget_loss + retain_loss
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite local pilot loss")
                (loss / args.accumulation).backward()
                losses.append(
                    {
                        "sample_index": sample_index,
                        "npo": float(forget_loss.detach().item()),
                        "retain": float(retain_loss.detach().item()),
                        "total": float(loss.detach().item()),
                    }
                )
                del forget, retain, candidate_nll, forget_loss, retain_logits
                del retain_nll, retain_loss, loss

            optimizer.step()
            record = {
                "step": step + 1,
                "microbatches": losses,
                "mean_npo_loss": float(np.mean([item["npo"] for item in losses])),
                "mean_retain_loss": float(np.mean([item["retain"] for item in losses])),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "peak_allocated_memory_bytes": torch.cuda.max_memory_allocated(0),
                "peak_reserved_memory_bytes": torch.cuda.max_memory_reserved(0),
            }
            log.write(json.dumps(record, allow_nan=False) + "\n")
            log.flush()

    candidate.eval()
    after = evaluate(candidate, eval_examples, tokenizer, spec)
    write_new(
        output / "evaluation.json",
        {
            "before": before,
            "after": after,
            "interpretation": "descriptive exploratory NLL only; no pass/fail or durability verdict",
        },
    )
    adapter_dir = output / "lora_adapter"
    candidate.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(output / "tokenizer")
    write_new(
        output / "RESULT.json",
        {
            "status": "COMPLETED_EXPLORATORY_LOCAL_PILOT",
            "optimizer_steps": args.max_steps,
            "source_faithful_d1": False,
            "attack_detected": None,
            "durability_pass": None,
            "adapter_directory": str(adapter_dir),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--cache-dir",
        default=str(Path.home() / ".cache" / "fq-ssiuu" / "local-lora-pilot"),
    )
    parser.add_argument("--max-steps", type=int, default=62)
    parser.add_argument("--max-length", type=int, default=500)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--eval-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1001)
    args = parser.parse_args()
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be a finite positive number")
    run(args)


if __name__ == "__main__":
    main()
