"""Four fixed examples, frozen FP16 forward passes, no training or recovery verdict."""

# ruff: noqa: E402

import argparse
import gc
import importlib.metadata
import json
import os
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch

from experiments.reproductions.NPO.adapter import collate_examples, encode_example
from experiments.reproductions.NPO.assets import PINS, materialize
from experiments.reproductions.NPO.candidate import causal_sequence_nll, compute_causal_npo_loss
from experiments.reproductions.NPO.reference import (
    ModelIdentity,
    ReferenceConfig,
    forward_pair,
    load_reference,
)
from experiments.reproductions.NPO.validate_adapter import specification
from experiments.reproductions.NPO.validation_contract import read_recipe
from experiments.reproductions.QRA.metrics import (
    answer_log_probability,
    decoding_spec,
    forget_quality,
    greedy_answer,
    implementation_hash,
    model_utility,
    text_metrics,
    truth_ratios,
    utility_components,
)
from fq_ssiuu.evaluation.identity import digest, file_digest, tensor_record


def write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def verified_checkpoint(identity, directory):
    from huggingface_hub import hf_hub_download

    directory.mkdir(parents=True, exist_ok=False)
    receipt = {}
    for name in ("config.json", "model.safetensors"):
        source = Path(
            hf_hub_download(identity["model_id"], name, revision=identity["revision"], token=False)
        )
        sha = file_digest(source)
        expected = (
            identity.get("config_sha256")
            if name == "config.json"
            else identity["checkpoint_sha256"]
        )
        if expected is not None and sha != expected:
            raise ValueError("Pinned checkpoint bytes differ")
        # Links avoid copying multi-GB public inputs into run output/archives.
        (directory / name).symlink_to(source.resolve())
        receipt[name] = {"sha256": sha, "bytes": source.stat().st_size}
    return receipt


def mean_nll(model, example):
    inputs = {
        name: example[name].unsqueeze(0).to("cuda:0")
        for name in ("input_ids", "attention_mask", "labels")
    }
    with torch.no_grad():
        logits = model(**inputs).logits
    value = -answer_log_probability(logits, inputs["labels"], inputs["attention_mask"])[0]
    del logits
    return value


def evaluate_row(model, tokenizer, row, index, split, spec, *, generate=True):
    example = encode_example(row, index, split, tokenizer, spec)
    correct = mean_nll(model, example)
    paraphrase = row.get("paraphrased_answer", row["answer"])
    paraphrase_example = encode_example(
        {**row, "answer": paraphrase}, index, split, tokenizer, spec
    )
    para_nll = mean_nll(model, paraphrase_example)
    wrong = []
    variant_records = []
    for answer in row["perturbed_answer"]:
        variant = encode_example({**row, "answer": answer}, index, split, tokenizer, spec)
        wrong.append(mean_nll(model, variant))
        variant_records.append(
            {
                "answer": answer,
                "tensors": tensor_record(
                    {k: variant[k] for k in ("input_ids", "attention_mask", "labels")}
                ),
                "tensors_sha256": variant["tensors_sha256"],
            }
        )
    ratios = truth_ratios(para_nll, wrong)
    record = {
        "sample_id": example["sample_id"],
        "source_split": split,
        "source_index": index,
        "source_record": row,
        "source_record_sha256": example["source_record_sha256"],
        "tensors_sha256": example["tensors_sha256"],
        "tensors": tensor_record(
            {k: example[k] for k in ("input_ids", "attention_mask", "labels")}
        ),
        "adapter": {
            k: v for k, v in example.items() if k not in ("input_ids", "attention_mask", "labels")
        },
        "answer_log_probability": -correct,
        "paraphrase_mean_nll": para_nll,
        "paraphrase_tensors_sha256": paraphrase_example["tensors_sha256"],
        "perturbation_mean_nlls": wrong,
        "perturbation_records": variant_records,
        **ratios,
    }
    if generate:
        generated = greedy_answer(
            model,
            tokenizer,
            example["input_ids"][: example["prompt_token_count"]],
            decoding_spec(tokenizer.eos_token_id, tokenizer.pad_token_id, 64),
        )
        record.update(generation=generated, **text_metrics(row["answer"], generated["text"]))
        dataset = {
            "retain_perturbed": "retain",
            "real_authors_perturbed": "real_authors",
            "world_facts_perturbed": "world_facts",
        }.get(split)
        if dataset:
            record["utility_components"] = utility_components(
                correct,
                wrong,
                record["rouge_l_recall"],
                dataset=dataset,
                paraphrase_mean_nll=para_nll,
            )
    return record


def execute(recipe, output, run_id):
    from transformers import AutoTokenizer

    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != "0"
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        raise ValueError("Exactly one selected visible CUDA device required")
    if "T4" not in torch.cuda.get_device_name(0):
        raise ValueError("Canonical Tesla T4 required")
    torch.manual_seed(recipe["seed"])
    torch.cuda.manual_seed_all(recipe["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats(0)
    # SDPA's math kernel permits deterministic forward execution; no training graph.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    pins = json.loads(PINS.read_text())
    assets = Path("/kaggle/working") / ("fq-validation-inputs-" + run_id)
    receipts = materialize(assets, upstream=False)
    write(output / "input_files.json", receipts)
    tokenizer = AutoTokenizer.from_pretrained(assets / "tokenizer", local_files_only=True)
    spec = specification(pins, max_length=recipe["max_length"])
    write(output / "preprocessing.json", {"spec": asdict(spec), "sha256": spec.sha256})
    target_dir = assets / "target"
    target_receipt = verified_checkpoint(pins["target"], target_dir)
    write(output / "target_checkpoint.json", {"identity": pins["target"], "files": target_receipt})
    identity = ModelIdentity(**pins["target"])
    cfg = ReferenceConfig(identity, identity)
    write(output / "reference_config.json", asdict(cfg))
    reference = load_reference(cfg, target_dir)
    candidate = load_reference(
        cfg, target_dir
    )  # Same pinned pre-unlearning model; frozen for this check.
    selected = []
    for split, indices in recipe["sample_indices"].items():
        rows = [
            json.loads(line)
            for line in (assets / "dataset" / f"{split}.json").read_text().splitlines()
        ]
        selected.extend((split, index, rows[index]) for index in indices)
    npo_records = []
    for split, index, row in selected:
        example = encode_example(row, index, split, tokenizer, spec)
        batch = collate_examples([example], tokenizer, spec)
        pair = forward_pair(candidate, reference, batch, cfg)
        labels = batch["tensors"]["labels"].to("cuda:0")
        loss = compute_causal_npo_loss(
            pair["candidate_logits"], pair["reference_logits"], labels, recipe["beta"]
        )
        torch.testing.assert_close(
            pair["candidate_logits"], pair["reference_logits"], rtol=0, atol=0
        )
        if not torch.isfinite(loss) or loss.requires_grad:
            raise ValueError("Invalid forward-only NPO loss")
        npo_records.append(
            {
                "sample_id": example["sample_id"],
                "loss": float(loss),
                "candidate_sequence_nll": causal_sequence_nll(pair["candidate_logits"], labels)
                .cpu()
                .tolist(),
                "reference_sequence_nll": causal_sequence_nll(pair["reference_logits"], labels)
                .cpu()
                .tolist(),
                "input_hashes": pair["input_hashes"],
                "gradient_graph": pair["gradient_graph"],
            }
        )
        del pair, loss
    if any(p.grad is not None for model in (candidate, reference) for p in model.parameters()):
        raise ValueError("Unexpected parameter gradient")
    write(
        output / "npo_forward.json",
        {
            "records": npo_records,
            "cache": "NOT_IMPLEMENTED",
            "optimizer_steps": 0,
            "backward_calls": 0,
        },
    )
    del reference
    gc.collect()
    torch.cuda.empty_cache()
    records = []
    for split, index, row in selected:
        record = evaluate_row(candidate, tokenizer, row, index, split, spec)
        records.append(record)
        write(output / (split + ".json"), record)
    del candidate
    gc.collect()
    torch.cuda.empty_cache()
    # TOFU diagnostic uses the exact retain95 oracle, never the full NPO reference.
    oracle_dir = assets / "retain95"
    oracle_receipt = verified_checkpoint(pins["retain_only_oracle"], oracle_dir)
    oracle_identity = ModelIdentity(
        **pins["retain_only_oracle"], config_sha256=oracle_receipt["config.json"]["sha256"]
    )
    oracle = load_reference(ReferenceConfig(oracle_identity, oracle_identity), oracle_dir)
    split, index, row = next(x for x in selected if x[0] == "forget05_perturbed")
    oracle_record = evaluate_row(oracle, tokenizer, row, index, split, spec, generate=False)
    write(
        output / "retain95_oracle.json",
        {
            "identity": asdict(oracle_identity),
            "files": oracle_receipt,
            "record": oracle_record,
            "policy": "fixed_fp16_oracle_validation_only",
        },
    )
    forget = next(x for x in records if x["source_split"] == split)
    quality = forget_quality([forget["tofu_truth_ratio"]], [oracle_record["tofu_truth_ratio"]])
    components = {
        key: value
        for record in records
        for key, value in record.get("utility_components", {}).items()
    }
    allocated = torch.cuda.max_memory_allocated(0)
    reserved = torch.cuda.max_memory_reserved(0)
    if max(allocated, reserved) >= 16_000_000_000:
        raise ValueError("Selected-device peak memory exceeded budget")
    result = {
        "schema_version": "work-unit-b-forward-validation-v1",
        "status": "PASS",
        "run_id": run_id,
        "comparison_group_id": run_id + "-method-validation",
        "precision": "fp16",
        "source_checkpoint_sha256": identity.checkpoint_sha256,
        "evaluator_sha256": implementation_hash(),
        "pins_sha256": file_digest(PINS),
        "sample_ids_sha256": digest([r["sample_id"] for r in records]),
        "raw_records_sha256": digest(records),
        "ks_diagnostic": quality,
        "ks_limitation": "One fixed forget example exercises serialization/calculation only; no statistical interpretation",
        "utility": model_utility(components),
        "attack_detected": None,
        "durability_pass": None,
        "decision_status": "BLOCKED_RG005",
        "precision_comparisons": None,
        "precision_comparison_limitation": "No INT8/NF4 model evaluations in this bounded reference-validation run; CPU paired-comparison tests are separate",
        "gpu": torch.cuda.get_device_name(0),
        "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "device_count": torch.cuda.device_count(),
        "peak_allocated_bytes": allocated,
        "peak_reserved_bytes": reserved,
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "transformers",
                "bitsandbytes",
                "numpy",
                "scipy",
                "rouge-score",
                "tokenizers",
            )
        },
        "optimizer_steps": 0,
        "backward_calls": 0,
        "model_training": False,
        "supported_claims": ["Four fixed-example NPO/reference and descriptive metric execution"],
        "unsupported_claims": [
            "NPO training reproduction",
            "Quantization recovery",
            "Durability",
            "Pilot feasibility",
            "Hypothesis test",
        ],
    }
    del oracle
    gc.collect()
    torch.cuda.empty_cache()
    write(output / "validation_result.json", result)
    # Inputs are cached outside the sealed output. Only receipts and derived records enter it.
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    recipe = read_recipe(args.recipe)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        result = execute(recipe, output, args.run_id)
    except Exception as exc:
        # This child uses only public input access, never secrets or auth URLs.
        traceback.print_exc()
        result = {
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "run_id": args.run_id,
            "attack_detected": None,
            "durability_pass": None,
        }
        write(output / "failure.json", result)
    finally:
        paths = sorted(path for path in output.rglob("*") if path.is_file())
        write(
            output / "OUTPUT_MANIFEST.json",
            {
                "run_id": args.run_id,
                "artifacts": [
                    {
                        "path": str(path.relative_to(output)),
                        "sha256": file_digest(path),
                        "bytes": path.stat().st_size,
                    }
                    for path in paths
                ],
            },
        )
    print(json.dumps({"status": result["status"], "run_id": args.run_id}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
