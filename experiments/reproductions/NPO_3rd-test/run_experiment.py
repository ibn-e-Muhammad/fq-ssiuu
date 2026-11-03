"""Run the declared three-seed LoRA NPO+RT experiment and all evaluation exports."""

import argparse
import importlib.metadata
import json
import platform
import shutil
import time
from pathlib import Path

import pandas as pd
import psutil
import torch
from npo3.config import Config
from npo3.data import PINS, one_batch, prepare_examples, resolve_inputs
from npo3.evaluation import (
    evaluate,
    model_loader,
    model_metadata,
    preserve_rng,
    release_memory,
)
from npo3.metrics import digest, sha256, summarize, write_json
from npo3.reporting import export_cross_seed, export_seed_report
from npo3.training import attach_lora, train


def environment(config):
    names = [
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "numpy",
        "pandas",
        "scipy",
        "rouge-score",
        "safetensors",
    ]
    if config.quantization_modes:
        names.append("bitsandbytes")
    versions = {name: importlib.metadata.version(name) for name in names}
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "cuda": torch.version.cuda,
        "ram_available_bytes": psutil.virtual_memory().available,
        "requested_device": config.device,
        "dtype": config.dtype,
    }
    if str(config.device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError("CUDA-enabled PyTorch and the correct notebook kernel are required")
        if (
            config.dtype != "bfloat16"
            or not torch.cuda.is_bf16_supported()
            or torch.cuda.get_device_capability(config.device)[0] < 8
        ):
            raise ValueError("Native BF16 is required for this declared local configuration")
        free, total = torch.cuda.mem_get_info(config.device)
        if free < 7 * 1024**3:
            raise ValueError(
                "Close other GPU applications; at least 7 GiB free is required to start"
            )
        info.update(
            gpu=torch.cuda.get_device_name(config.device),
            free_vram_bytes=free,
            total_vram_bytes=total,
        )
        existing = Path(config.output_root).resolve()
        while not existing.exists():
            existing = existing.parent
        disk_free = shutil.disk_usage(existing).free
        if disk_free < 12 * 1024**3:
            raise ValueError(
                "Reserve at least 12 GiB on the output SSD for three merged 1B checkpoints and adapter/results state"
            )
        if info["ram_available_bytes"] < 2 * 1024**3:
            raise ValueError(
                "Close other applications; at least 2 GiB free system RAM is required to start"
            )
        info["output_disk_free_bytes"] = disk_free
    torch.set_num_threads(min(8, __import__("os").cpu_count() or 1))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return info


def quantization_preflight(config):
    if not config.quantization_modes:
        return
    import bitsandbytes as bnb

    dtype = getattr(torch, config.dtype)
    with torch.no_grad():
        for mode in config.quantization_modes:
            layer = (
                bnb.nn.Linear8bitLt(32, 32, bias=False, has_fp16_weights=False)
                if mode == "int8"
                else bnb.nn.Linear4bit(
                    32,
                    32,
                    bias=False,
                    compute_dtype=dtype,
                    quant_type="nf4",
                    compress_statistics=True,
                )
            )
            layer.requires_grad_(False)
            layer = layer.to(device=config.device, dtype=dtype)
            output = layer(torch.randn(1, 32, device=config.device, dtype=dtype))
            if not torch.isfinite(output).all():
                raise ValueError(f"Broken {mode} backend")
            del layer, output
    release_memory()


def merge_export(model, tokenizer, forget, config, folder):
    """Merge in FP32 on GPU, avoiding a full FP32 CPU copy, then deploy BF16."""
    folder = Path(folder)
    model.eval()
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.to(dtype=torch.float32)
    probes, before = forget[:3], []
    with torch.no_grad():
        for row in probes:
            inputs, _ = one_batch(row, config.device, tokenizer.pad_token_id)
            before.append(model(**inputs, use_cache=False).logits[:, -3:].float().cpu())
        merged = model.merge_and_unload(safe_merge=True)
        errors = []
        for row, expected in zip(probes, before):
            inputs, _ = one_batch(row, config.device, tokenizer.pad_token_id)
            actual = merged(**inputs, use_cache=False).logits[:, -3:].float().cpu()
            errors.append(float((actual - expected).abs().max()))
            torch.testing.assert_close(actual, expected, atol=3e-4, rtol=1e-4)
    merged.to(dtype=getattr(torch, config.dtype))
    merged.to("cpu")
    merged.save_pretrained(folder, safe_serialization=True, max_shard_size="1GB")
    tokenizer.save_pretrained(folder)
    receipt = {
        "status": "complete",
        "merge_precision": "float32",
        "deployment_dtype": config.dtype,
        "fp32_merge_probe_max_abs_errors": errors,
        "probe_positions": "last three logits including scored answer/EOS positions",
        "quantized_during_training": False,
        "weight_hashes": {p.name: sha256(p) for p in folder.glob("*.safetensors")},
    }
    write_json(folder / "merge_complete.json", receipt)
    return merged, receipt


def run(config, resume_dir=None):
    config.validate()
    env = environment(config)
    quantization_preflight(config)
    paths, tokenizer, data, data_receipts = resolve_inputs(config)
    forget, retain, packs, selection = prepare_examples(data, tokenizer, config)
    code = {
        str(p.relative_to(Path(__file__).parent)): sha256(p)
        for p in sorted(Path(__file__).parent.rglob("*.py"))
    }
    identity = digest(
        {
            "config": config.to_dict(),
            "inputs": data_receipts,
            "selection": selection,
            "code": code,
            "packages": env["packages"],
        }
    )
    if resume_dir:
        root = Path(resume_dir).resolve()
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest["identity"] != identity:
            raise ValueError("Resume config/input/code/package identity changed; create a new run")
    else:
        root = Path(config.output_root) / (
            time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1_000_000:06d}"
        )
        root.mkdir(parents=True, exist_ok=False)
        write_json(
            root / "manifest.json",
            {
                "identity": identity,
                "config": config.to_dict(),
                "environment": env,
                "code": code,
                "inputs": data_receipts,
                "models": {key: str(value) for key, value in paths.items()},
                "model_sha256": {
                    "full": PINS["target"]["checkpoint_sha256"],
                    "retain95": PINS["retain_only_oracle"]["checkpoint_sha256"],
                },
            },
        )
        write_json(root / "evaluation_selection.json", selection)
    print(f"Run directory: {root}", flush=True)
    profiles = ("source_sampled", "legacy_greedy96", "long_greedy")
    controls, matrices = {}, {}

    @preserve_rng()
    def measure(
        stage,
        path,
        folder,
        oracle=None,
        quant=None,
        loaded=None,
        extra=None,
        selected_profiles=profiles,
    ):
        model, reference = loaded, None
        owned = loaded is None
        try:
            if owned:
                model = model_loader(path, config, quant)
            details = model_metadata(model, quant)
            if config.evaluate_kl:
                reference = (
                    model
                    if path == paths["full"] and quant is None and owned
                    else model_loader(paths["full"], config)
                )
            binding = {
                "run_identity": identity,
                "stage": stage,
                "model_path": str(path),
                "model": details,
                "KL_reference": PINS["target"]["checkpoint_sha256"] if config.evaluate_kl else None,
                "KL_positions": "all 200 source positions including prompt/padding",
                **(extra or {}),
            }
            frame = evaluate(
                model,
                tokenizer,
                packs,
                config,
                folder,
                binding,
                reference=reference,
                oracle=oracle,
                profiles=selected_profiles,
            )
            write_json(Path(folder) / "model_identity.json", binding)
            return frame
        finally:
            reference = None
            if owned:
                model = None
            release_memory()

    try:
        controls["full_bf16"] = measure("full_bf16", paths["full"], root / "controls/full_bf16")
        controls["retain95_bf16"] = measure(
            "retain95_bf16", paths["retain95"], root / "controls/retain95_bf16"
        )
        if config.quantized_controls:
            for mode in config.quantization_modes:
                for kind in ("full", "retain95"):
                    name = f"{kind}_{mode}"
                    controls[name] = measure(
                        name,
                        paths[kind],
                        root / "controls" / name,
                        oracle=controls["retain95_bf16"],
                        quant=mode,
                    )
        author_blocks = {p["question"]: p["author_block"] for p in packs["forget05"]}
        for seed in config.seeds:
            seed_folder = root / f"seed_{seed}"
            complete = seed_folder / "seed_complete.json"
            if complete.is_file():
                old = json.loads(complete.read_text())
                for name, expected in old["report_hashes"].items():
                    if sha256(seed_folder / name) != expected:
                        raise ValueError("Completed seed report was modified")
                matrices[seed] = pd.read_csv(seed_folder / "comparison_matrix.csv")
                continue
            seed_folder.mkdir(exist_ok=True)
            base = model_loader(paths["full"], config)
            model = attach_lora(base, config, seed)
            del base

            def during_training(current, step, training_identity):
                total = len(forget) * config.nominal_epochs // config.effective_batch
                if step == total:
                    return  # Final evaluation below includes all three profiles on the final checkpoint.
                location = seed_folder / "epoch_evaluations" / f"update_{step:06d}"
                frame = measure(
                    f"npo_adapter_step_{step}",
                    seed_folder / "adapter",
                    location,
                    oracle=controls["retain95_bf16"],
                    loaded=current,
                    extra={"training_identity": training_identity, "update": step},
                    selected_profiles=("source_sampled",),
                )
                row, _ = summarize(frame, controls["retain95_bf16"])
                write_json(
                    location / "epoch_summary.json",
                    {
                        "seed": seed,
                        "update": step,
                        "effective_forget_passes": step * config.effective_batch / len(forget),
                        **row,
                    },
                )
                milestone = location / "adapter"
                milestone_receipt = location / "adapter_checkpoint.json"
                if milestone_receipt.exists():
                    archived = json.loads(milestone_receipt.read_text())
                    for name, expected in archived["hashes"].items():
                        if sha256(milestone / name) != expected:
                            raise ValueError("Archived epoch adapter was modified")
                else:
                    current.save_pretrained(milestone, safe_serialization=True)
                    write_json(
                        milestone_receipt,
                        {
                            "update": step,
                            "training_identity": training_identity,
                            "hashes": {
                                p.name: sha256(p) for p in milestone.glob("*") if p.is_file()
                            },
                        },
                    )

            training = train(
                model,
                tokenizer,
                forget,
                retain,
                config,
                seed,
                seed_folder,
                {"run_identity": identity, "seed": seed},
                during_training,
                resume=bool(resume_dir and list((seed_folder / "checkpoints").glob("step-*"))),
            )
            stages = dict(controls)
            stages["npo_adapter"] = measure(
                "npo_adapter",
                seed_folder / "adapter",
                seed_folder / "npo_adapter",
                oracle=controls["retain95_bf16"],
                loaded=model,
                extra={"training_identity": training["identity"]},
            )
            merged_path = seed_folder / "merged_model"
            merged, merge = merge_export(model, tokenizer, forget, config, merged_path)
            del model, merged
            release_memory()
            stages["npo_bf16"] = measure(
                "npo_bf16",
                merged_path,
                seed_folder / "npo_bf16",
                oracle=controls["retain95_bf16"],
                extra={"merge": merge},
            )
            for mode in config.quantization_modes:
                stages[f"npo_{mode}"] = measure(
                    f"npo_{mode}",
                    merged_path,
                    seed_folder / f"npo_{mode}",
                    oracle=controls["retain95_bf16"],
                    quant=mode,
                    extra={"merge": merge},
                )
            matrices[seed] = export_seed_report(seed_folder, stages, config, author_blocks)
            epoch_rows = [
                json.loads(p.read_text())
                for p in sorted((seed_folder / "epoch_evaluations").glob("*/epoch_summary.json"))
            ]
            final_summary, _ = summarize(stages["npo_adapter"], controls["retain95_bf16"])
            epoch_rows.append(
                {
                    "seed": seed,
                    "update": training["updates"],
                    "effective_forget_passes": training["updates"]
                    * config.effective_batch
                    / len(forget),
                    **final_summary,
                }
            )
            pd.DataFrame(epoch_rows).to_csv(seed_folder / "epoch_metrics.csv", index=False)
            reports = [
                "comparison_matrix.csv",
                "utility_components.csv",
                "forget_quality_fixed_and_matched.csv",
                "paired_directional_recovery.csv",
                "merge_drift.csv",
                "answers_side_by_side.csv",
                "epoch_metrics.csv",
            ]
            reports += [
                str(p.relative_to(seed_folder))
                for p in seed_folder.glob("npo*/**/*")
                if p.is_file()
                and p.name in ("per_question.csv", "per_question.jsonl", "metadata.json")
            ]
            write_json(
                complete,
                {
                    "status": "complete",
                    "seed": seed,
                    "training_identity": training["identity"],
                    "report_hashes": {name: sha256(seed_folder / name) for name in reports},
                },
            )
            export_cross_seed(root, matrices, config.seeds)
        aggregate = export_cross_seed(root, matrices, config.seeds)
        write_json(
            root / "run_complete.json",
            {
                "status": "complete",
                "identity": identity,
                "seeds": config.seeds,
                "quantization_modes": config.quantization_modes,
                "minimum_three_seeds_met": len(matrices) >= 3,
                "attack_detected": None,
                "durability_pass": None,
            },
        )
        print(aggregate.to_string())
        print(f"All measured results: {root}")
        return root
    except BaseException as exc:
        write_json(
            root / "run_failure.json",
            {
                "error": repr(exc),
                "identity": identity,
                "completed_seeds": list(matrices),
                "instruction": "Use --resume with this run directory and unchanged config/code.",
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--resume", type=Path)
    arguments = parser.parse_args()
    run(Config.load(arguments.config), arguments.resume)
