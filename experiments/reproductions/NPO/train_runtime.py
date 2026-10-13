"""D1 bounded full-parameter NPO+RT pilot; no project durability decisions."""

# ruff: noqa: E402

import argparse
import gc
import importlib.metadata
import json
import os
import random
import re
import subprocess
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from experiments.reproductions.common import digest, file_digest, require_hash
from experiments.reproductions.NPO.adapter import encode_rows
from experiments.reproductions.NPO.assets import PINS, fetch_verified, materialize
from experiments.reproductions.NPO.checkpoint_values import compare_values
from experiments.reproductions.NPO.engine import create_engine, destroy_engine
from experiments.reproductions.NPO.evaluation_runtime import (
    evaluate_oracle,
    evaluate_source,
    score_to_file,
    write_new,
)
from experiments.reproductions.NPO.memory import MemoryMonitor
from experiments.reproductions.NPO.preflight import membership
from experiments.reproductions.NPO.reference import ModelIdentity, ReferenceConfig, load_reference
from experiments.reproductions.NPO.training_contract import deepspeed_config, read_configs
from experiments.reproductions.NPO.training_data import SourcePairDataset, source_batch_plan
from experiments.reproductions.NPO.training_loop import run_training
from experiments.reproductions.NPO.training_state import load_deepspeed_state, save_deepspeed_state
from experiments.reproductions.NPO.validate_adapter import specification
from experiments.reproductions.NPO.validate_paged_gpu import validate as validate_paged_gpu
from experiments.reproductions.NPO.validate_runtime import verified_checkpoint


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def files_manifest(root):
    return [
        {"path": str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": file_digest(p)}
        for p in sorted(Path(root).rglob("*"))
        if p.is_file()
    ]


def execute(recipe, output, run_id, provenance, resume_checkpoint=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != "0"
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        raise ValueError("D1 requires CUDA_VISIBLE_DEVICES=0 and one visible GPU")
    if "T4" not in torch.cuda.get_device_name(0):
        raise ValueError("D1 requires Tesla T4")
    if any(os.environ.get(k, "1") != "1" for k in ("WORLD_SIZE",)):
        raise ValueError("D1 forbids distributed GPU sharding")
    versions = {
        k: importlib.metadata.version(k)
        for k in ("torch", "transformers", "bitsandbytes", "deepspeed", "accelerate")
    }
    if versions != {
        "torch": "2.10.0+cu128",
        "transformers": "5.0.0",
        "bitsandbytes": "0.46.1",
        "deepspeed": "0.19.7",
        "accelerate": "1.13.0",
    }:
        raise ValueError("Unvalidated D1 runtime version drift")
    train, evaluation = read_configs(
        ROOT / recipe["train_config"],
        ROOT / recipe["eval_config"],
        recipe["train_config_sha256"],
        recipe["eval_config_sha256"],
    )
    write_new(output / "training_config.json", train)
    write_new(output / "evaluation_config.json", evaluation)
    write_new(output / "runtime_versions.json", versions)
    sub_group_size = train.get("zero_sub_group_size", 1000000000)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    # Deterministic model-adapter path; abort rather than silently substitute a
    # nondeterministic implementation. No source hyperparameters change.
    torch.use_deterministic_algorithms(True)
    seed_all(1001)
    monitor = MemoryMonitor(output / "device_memory.jsonl")
    engine = None
    try:
        pins = json.loads(PINS.read_text())
        runtime_sources = json.loads(
            (Path(__file__).with_name("runtime_sources_v1.json")).read_text()
        )
        assets = Path("/kaggle/tmp/fq-d1-inputs") / run_id
        receipts = materialize(assets, upstream=False)
        write_new(output / "input_files.json", receipts)
        tokenizer = AutoTokenizer.from_pretrained(assets / "tokenizer", local_files_only=True)
        train_spec, eval_spec = (
            specification(pins, max_length=500),
            specification(pins, max_length=200),
        )
        write_new(
            output / "adapter_specs.json",
            {"training": asdict(train_spec), "evaluation": asdict(eval_spec)},
        )
        generation_pin = runtime_sources["files"]["generation_config.json"]
        fetch_verified(
            generation_pin["url"], assets / "generation_config.json", generation_pin["sha256"]
        )
        generation = json.loads((assets / "generation_config.json").read_text())
        if generation != runtime_sources["generation_config"]:
            raise ValueError("Pinned generation settings changed")
        rows = {
            split: [
                json.loads(x)
                for x in (assets / "dataset" / (split + ".json")).read_text().splitlines()
            ]
            for split in ("forget05", "retain95")
        }
        members = membership(rows)
        if [members[s]["count"] for s in ("forget05", "retain95")] != [200, 3800]:
            raise ValueError("Dataset boundary changed")
        write_new(output / "membership.json", members)
        encoded = {s: encode_rows(r, s, tokenizer, train_spec) for s, r in rows.items()}
        boundary = [
            {k: x[k] for k in ("sample_id", "source_record_sha256", "tensors_sha256", "truncation")}
            for items in encoded.values()
            for x in items
        ]
        if any(x["truncation"]["truncated"] for x in boundary):
            raise ValueError("Unexpected source-length truncation")
        write_new(output / "training_adapter_records.json", boundary)
        identities = {
            "source_commit": provenance["source_origin_sha"],
            "source_spec_sha256": file_digest(Path(__file__).with_name("source_v1.json")),
            "train_config_sha256": recipe["train_config_sha256"],
            "eval_config_sha256": recipe["eval_config_sha256"],
            "initial_checkpoint_sha256": pins["target"]["checkpoint_sha256"],
            "reference_checkpoint_sha256": pins["target"]["checkpoint_sha256"],
            "tokenizer_revision": pins["target"]["tokenizer_revision"],
            "dataset_revision": pins["dataset"]["revision"],
            "preprocessing_sha256": train_spec.sha256,
            "sample_membership_sha256": digest(members),
        }
        write_new(output / "identities.json", identities)
        os.environ.update(LOCAL_RANK="0", RANK="0", WORLD_SIZE="1")
        torch.cuda.set_device(0)
        torch.distributed.init_process_group(
            "nccl", init_method="file://" + str(output / "rendezvous"), rank=0, world_size=1
        )
        validate_paged_gpu(output / "tiny_paged_resume", identities, sub_group_size=sub_group_size)
        monitor.observe()

        # Compute the retain-only diagnostic oracle, then unload it before the
        # candidate and ORIGINAL FULL NPO reference are loaded.
        oracle_dir = assets / "oracle"
        oracle_receipt = verified_checkpoint(pins["retain_only_oracle"], oracle_dir)
        write_new(
            output / "oracle_checkpoint.json",
            {"identity": pins["retain_only_oracle"], "files": oracle_receipt},
        )
        oracle = AutoModelForCausalLM.from_pretrained(
            oracle_dir,
            local_files_only=True,
            torch_dtype=torch.float16,
            device_map={"": 0},
            attn_implementation="sdpa",
        )
        oracle.eval().requires_grad_(False)
        oracle_path = evaluate_oracle(
            oracle, tokenizer, eval_spec, assets / "dataset", pins, output / "oracle_evaluation"
        )
        oracle_sha = file_digest(oracle_path)
        del oracle
        gc.collect()
        torch.cuda.empty_cache()
        monitor.observe()

        target_dir = assets / "target"
        target_receipt = verified_checkpoint(pins["target"], target_dir)
        write_new(
            output / "target_checkpoint.json", {"identity": pins["target"], "files": target_receipt}
        )
        candidate = AutoModelForCausalLM.from_pretrained(
            target_dir,
            local_files_only=True,
            torch_dtype=torch.float16,
            device_map={"": 0},
            attn_implementation="sdpa",
        )
        # Preserve the attempted resource settings even if initialization fails.
        write_new(
            output / "deepspeed_config.json",
            deepspeed_config(candidate.config.hidden_size, sub_group_size=sub_group_size),
        )
        engine, inventory, ds_config = create_engine(candidate, sub_group_size=sub_group_size)
        del candidate
        if sum(x["numel"] for x in inventory) != 1235814400:
            raise ValueError("Unexpected trainable parameter set")
        write_new(output / "parameter_inventory.json", inventory)
        identity = ModelIdentity(**pins["target"])
        reference_config = ReferenceConfig(identity, identity)
        reference = load_reference(reference_config, target_dir)
        monitor.observe()
        pairs = SourcePairDataset(encoded["forget05"], encoded["retain95"])
        plan = source_batch_plan(200, 4, 8, 62, 1001)
        write_new(
            output / "batch_plan.json", {k: v for k, v in plan.items() if k != "generator_state"}
        )

        def evaluate(step):
            result = evaluate_source(
                engine,
                reference,
                tokenizer,
                eval_spec,
                assets / "dataset",
                generation,
                output / f"evaluation_step_{step:04d}",
                {
                    "run_id": run_id,
                    "optimizer_step": step,
                    "identities": identities,
                    "source_provenance": provenance,
                    "checkpoint_selection": "source-step-not-best-metric",
                },
                oracle_path,
                oracle_sha,
                pins,
            )
            monitor.observe()
            return result

        # Author Trainer resets global seed to its default 42; custom data
        # loader keeps its separate source seed 1001. Both are recorded.
        seed_all(42)
        restored_cursor = None
        if resume_checkpoint is not None:
            restored_cursor = load_deepspeed_state(resume_checkpoint, engine, identities)
            write_new(
                output / "resume_source.json",
                {
                    "checkpoint_manifest_sha256": file_digest(
                        Path(resume_checkpoint) / "MANIFEST.json"
                    ),
                    "restored_optimizer_step": restored_cursor["optimizer_step"],
                    "identities": identities,
                },
            )
        cursor, updates = run_training(
            engine,
            reference,
            pairs,
            plan,
            reference_config,
            output / "training",
            pad_token_id=tokenizer.pad_token_id,
            evaluate=evaluate,
            observe_memory=monitor.observe,
            resume_cursor=restored_cursor,
        )
        if not any(
            x["gradient_norm"] is not None and x["gradient_norm"] > 0 and not x["fp16_overflow"]
            for x in updates
        ):
            raise ValueError("No finite nonzero gradient update")
        if engine.global_steps - engine.skipped_steps < 2:
            raise ValueError("Insufficient actual source optimizer updates")
        native = save_deepspeed_state(output / "training_checkpoint", engine, cursor, identities)
        engine.eval()
        final = evaluate(62)
        state = engine._zero3_consolidated_16bit_state_dict()
        final_dir = output / "final_model"
        final_dir.mkdir(exist_ok=False)
        engine.module.save_pretrained(
            final_dir, state_dict=state, safe_serialization=True, max_shard_size="3GB"
        )
        tokenizer.save_pretrained(final_dir)
        del state
        exported = files_manifest(final_dir)
        write_new(output / "final_model_manifest.json", exported)
        skipped = engine.skipped_steps
        destroy_engine(engine)
        del engine
        engine = None
        reference = None
        gc.collect()
        torch.cuda.empty_cache()
        changes = compare_values(target_dir / "model.safetensors", final_dir / "model.safetensors")
        write_new(output / "parameter_changes.json", changes)
        reloaded = AutoModelForCausalLM.from_pretrained(
            final_dir,
            local_files_only=True,
            torch_dtype=torch.float16,
            device_map={"": 0},
            attn_implementation="sdpa",
        )
        reloaded.eval().requires_grad_(False)
        # Same primary batch as the final evaluation, not a different padding
        # or batch-size pathway. A full reload is required for acceptance.
        forget_rows = [
            json.loads(x)
            for x in (assets / "dataset/forget05_perturbed.json").read_text().splitlines()
        ][:30]
        check_examples = encode_rows(forget_rows, "forget05_perturbed", tokenizer, eval_spec)
        checked = score_to_file(
            output / "reloaded_primary.jsonl", reloaded, check_examples, tokenizer, eval_spec, 30
        )
        expected = json.loads((output / "evaluation_step_0062/eval_log_forget.json").read_text())[
            "primary"
        ][:30]
        errors = [
            abs(a["sequence_nll"] - b["sequence_nll"])
            for a, b in zip(expected, checked, strict=True)
        ]
        if any(
            a["tensors_sha256"] != b["tensors_sha256"]
            for a, b in zip(expected, checked, strict=True)
        ):
            raise ValueError("Reload validation tensors differ")
        # Identical weights / FP16 / SDPA / shape should be exactly repeatable.
        if max(errors) != 0:
            raise ValueError("Reloaded final model changed source evaluation")
        memory = monitor.observe()
        return {
            "status": "PASS_LOCAL_RUNTIME_CRITERIA_ARCHIVAL_PENDING",
            "classification": train["classification"],
            "run_id": run_id,
            "source_provenance": provenance,
            "identities": identities,
            "attempted_optimizer_updates": 62,
            "skipped_fp16_updates": skipped,
            "actual_optimizer_updates": 62 - skipped,
            "checkpoint_manifest_sha256": file_digest(output / "training_checkpoint/MANIFEST.json"),
            "native_checkpoint_files": native["files"],
            "final_model_files": exported,
            "reload_max_sequence_nll_error": max(errors),
            "final_source_metrics": final,
            "memory": memory,
            "attack_detected": None,
            "durability_pass": None,
            "limitations": [
                "One-seed feasibility pilot; no effectiveness generalization",
                "No INT8/NF4 checkpoint evaluation in D1",
                "Tiny GPU optimizer resume tested; 1B final weights reloaded; full 1B optimizer resume not executed",
                "Private archive verification remains required",
            ],
        }
    finally:
        try:
            write_new(output / "memory_final.json", monitor.snapshot())
        finally:
            monitor.close()
        if engine is not None:
            destroy_engine(engine)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument(
        "--resume-checkpoint",
        help="Own hash-verified native state from the exact same source/config/data identities",
    )
    args = parser.parse_args()
    from experiments.reproductions.NPO.d1_recipe import read_recipe

    recipe = read_recipe(args.recipe)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    provenance = json.loads(Path(args.provenance).read_text())
    write_new(
        output / "STARTED.json",
        {"run_id": args.run_id, "recipe": recipe, "source_provenance": provenance},
    )
    try:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{5,120}", args.run_id):
            raise ValueError("Invalid immutable run namespace")
        for key in ("source_origin_sha", "source_tree_sha", "snapshot_commit_sha"):
            require_hash(provenance[key], 40)
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
        tree = (
            subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT).decode().strip()
        )
        if head != provenance["snapshot_commit_sha"] or tree != provenance["source_tree_sha"]:
            raise ValueError("D1 source provenance differs from verified checkout")
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
            raise ValueError("D1 source checkout changed after snapshot verification")
        result = execute(recipe, output, args.run_id, provenance, args.resume_checkpoint)
    except Exception as error:
        with (output / "traceback.log").open("x") as handle:
            traceback.print_exc(file=handle)
        result = {
            "status": "FAIL_PARTIAL_PRESERVED",
            "run_id": args.run_id,
            "error_type": type(error).__name__,
            "attack_detected": None,
            "durability_pass": None,
        }
    write_new(output / "RESULT.json", result)
    print(json.dumps({k: v for k, v in result.items() if k in {"status", "run_id", "error_type"}}))
    return 0 if result["status"] == "PASS_LOCAL_RUNTIME_CRITERIA_ARCHIVAL_PENDING" else 1


if __name__ == "__main__":
    raise SystemExit(main())
