"""C1: sequential matched precision inference on four fixed rows; no training."""

# ruff: noqa: E402
import argparse
import gc
import importlib.metadata
import json
import os
import sys
import traceback
import weakref
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch

from experiments.reproductions.NPO.assets import PINS, materialize
from experiments.reproductions.NPO.reference import ModelIdentity, ReferenceConfig, load_reference
from experiments.reproductions.NPO.validate_adapter import specification
from experiments.reproductions.NPO.validate_runtime import evaluate_row, verified_checkpoint, write
from experiments.reproductions.QRA.metrics import forget_quality, implementation_hash
from experiments.reproductions.QRA.validation_contract import read_recipe
from fq_ssiuu.evaluation.certificate import generate_report, save_report
from fq_ssiuu.evaluation.content_identity import evaluation_hash, oracle_manifest
from fq_ssiuu.evaluation.derived_summary import utility_summary
from fq_ssiuu.evaluation.identity import digest, file_digest
from fq_ssiuu.evaluation.metric_definitions import definition_for, definition_hash
from fq_ssiuu.evaluation.precision_artifact import build_artifact
from fq_ssiuu.evaluation.qra_pipeline import COMMON, SLOT, MetricObservation


def quantizer(precision):
    from transformers import BitsAndBytesConfig

    if precision == "fp16":
        return None, {"dtype": "float16", "quantization": None}
    if precision == "int8":
        config = BitsAndBytesConfig(
            load_in_8bit=True,
            load_in_4bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            llm_int8_enable_fp32_cpu_offload=False,
        )
    elif precision == "nf4":
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            load_in_8bit=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_storage=torch.uint8,
        )
    else:
        raise ValueError("Unsupported precision")
    return config, config.to_dict()


def expected_modules(config):
    if config["model_type"] != "llama":
        raise ValueError("Pinned Llama architecture required")
    return sorted(
        f"model.layers.{i}.{part}.{proj}"
        for i in range(config["num_hidden_layers"])
        for part, projections in (
            ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
            ("mlp", ("gate_proj", "up_proj", "down_proj")),
        )
        for proj in projections
    )


def evaluator_identity():
    return digest(
        {
            "base_metrics": implementation_hash(),
            "runtime": file_digest(__file__),
            "artifact": file_digest(ROOT / "src/fq_ssiuu/evaluation/precision_artifact.py"),
            "content_identity": file_digest(ROOT / "src/fq_ssiuu/evaluation/content_identity.py"),
        }
    )


def make_observations(records, artifacts, oracle, pins, spec, run_id):
    group = {
        "schema_version": "qra-comparison-group-v2",
        "comparison_group_id": run_id + "-c1",
        "run_id": run_id,
        "report_id": run_id + "-descriptive",
        "checkpoint_sha256": pins["target"]["checkpoint_sha256"],
        "model_id": pins["target"]["model_id"],
        "model_revision": pins["target"]["revision"],
        "tokenizer_id": spec.tokenizer_id,
        "tokenizer_revision": spec.tokenizer_revision,
        "evaluator_sha256": evaluator_identity(),
        "artifacts": {},
        "metrics": {},
        "oracle_evidence": oracle,
    }
    for precision, manifest in artifacts.items():
        group["artifacts"][precision] = {
            "precision": precision,
            "artifact_manifest": manifest,
            "artifact_sha256": digest(manifest),
            "quantizer_name": {
                "fp16": "identity",
                "int8": "bitsandbytes-llm-int8",
                "nf4": "bitsandbytes-nf4",
            }[precision],
            "quantizer_version": manifest["versions"][
                "torch" if precision == "fp16" else "bitsandbytes"
            ],
            "quantizer_config": manifest["quantizer_config"],
            "quantizer_config_sha256": digest(manifest["quantizer_config"]),
        }
    pending = []
    for precision, rows in records.items():
        for row in rows:
            metrics = dict(row.get("utility_components", {}))
            if row["source_split"] == "forget05_perturbed":
                metrics.update(
                    {
                        m: row[m]
                        for m in (
                            "answer_log_probability",
                            "rouge_l_recall",
                            "exact_match",
                            "tofu_truth_ratio",
                            "openunlearning_truth_ratio",
                        )
                    }
                )
                quality = forget_quality(
                    [row["tofu_truth_ratio"]],
                    [r["tofu_truth_ratio"] for r in oracle["raw_records"]],
                )
                metrics.update(
                    ks_statistic=quality["ks_statistic"], ks_p_value=quality["ks_p_value"]
                )
            for metric, value in metrics.items():
                diagnostic = metric in {"ks_statistic", "ks_p_value"}
                generated = metric.endswith("rouge_l_recall") or metric == "exact_match"
                context = {
                    "definition_sha256": definition_hash(metric),
                    "dataset_id": spec.dataset_id,
                    "dataset_revision": spec.dataset_revision,
                    "source_split": row["source_split"],
                    "sample_ids_sha256": digest([row["sample_id"]]),
                    "preprocessing_sha256": spec.sha256,
                    "decoding_sha256": row["generation"]["decoding_sha256"] if generated else None,
                    "reference_sha256": oracle["checkpoint_sha256"] if diagnostic else None,
                    "reference_evaluation_sha256": digest(oracle) if diagnostic else None,
                    "reference_policy": "fixed_fp16_oracle" if diagnostic else "not_applicable",
                }
                key = metric + "@" + row["source_split"]
                if key in group["metrics"] and context != group["metrics"][key]:
                    raise ValueError("Cross-precision preprocessing/decoding context differs")
                group["metrics"][key] = context
                raw = [
                    {
                        "sample_id": row["sample_id"],
                        "source_record_sha256": row["source_record_sha256"],
                        "tensors_sha256": row["tensors_sha256"],
                        "measurement": row,
                        "value": row["tofu_truth_ratio"] if diagnostic else value,
                    }
                ]
                definition = definition_for(metric)
                pending.append(
                    dict(
                        precision=precision,
                        metric=metric,
                        value=value,
                        raw_records=raw,
                        raw_records_sha256=digest(raw),
                        **context,
                        **{k: definition[k] for k in ("definition_id", "kind", "direction")},
                    )
                )
    observations = []
    for args in pending:
        obs = MetricObservation(
            **args,
            **{k: group[k] for k in COMMON},
            **{k: group["artifacts"][args["precision"]][k] for k in SLOT},
            comparison_group_sha256=digest(group),
            evaluation_sha256="0" * 64,
        )
        obs = replace(obs, evaluation_sha256=evaluation_hash(obs))
        obs.validate(group)
        observations.append(obs)
    return group, observations


def release(model):
    # The caller owns the final reference; this helper only asserts no graphs.
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise ValueError("Unexpected trainable parameter or gradient")
    return weakref.ref(model)


def execute(recipe, output, run_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != "0"
        or torch.cuda.device_count() != 1
        or "T4" not in torch.cuda.get_device_name(0)
    ):
        raise ValueError("One visible selected T4 required")
    versions = {k: importlib.metadata.version(k) for k in recipe["expected_versions"]}
    if versions != recipe["expected_versions"]:
        raise ValueError("Candidate runtime versions differ")
    torch.manual_seed(recipe["seed"])
    torch.cuda.manual_seed_all(recipe["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    pins = json.loads(PINS.read_text())
    assets = Path("/kaggle/working") / ("fq-c-inputs-" + run_id)
    write(output / "input_files.json", materialize(assets, upstream=False))
    tokenizer = AutoTokenizer.from_pretrained(assets / "tokenizer", local_files_only=True)
    spec = specification(pins, max_length=recipe["max_length"])
    write(output / "preprocessing.json", {"spec": asdict(spec), "sha256": spec.sha256})
    target_dir = assets / "target"
    write(output / "target_checkpoint.json", verified_checkpoint(pins["target"], target_dir))
    config = json.loads((target_dir / "config.json").read_text())
    selected = []
    for split, indices in recipe["sample_indices"].items():
        rows = [
            json.loads(line)
            for line in (assets / "dataset" / (split + ".json")).read_text().splitlines()
        ]
        selected.extend((split, index, rows[index]) for index in indices)
    records, artifacts, memory = {}, {}, {}
    for precision in recipe["evaluation_precisions"]:
        torch.cuda.reset_peak_memory_stats(0)
        quant, quant_config = quantizer(precision)
        kwargs = {} if quant is None else {"quantization_config": quant}
        model = AutoModelForCausalLM.from_pretrained(
            target_dir,
            local_files_only=True,
            dtype=torch.float16,
            device_map={"": 0},
            attn_implementation="sdpa",
            **kwargs,
        )
        model.eval().requires_grad_(False)
        if any(p.device != torch.device("cuda:0") for p in model.parameters()):
            raise ValueError("Offload/sharding/nonselected device prohibited")
        with torch.no_grad():
            records[precision] = [
                evaluate_row(model, tokenizer, row, index, split, spec)
                for split, index, row in selected
            ]
        write(output / (precision + "_records.json"), records[precision])
        artifacts[precision] = build_artifact(
            model,
            precision=precision,
            checkpoint_sha256=pins["target"]["checkpoint_sha256"],
            versions=versions,
            quantizer_config=quant_config,
            expected_quantized_modules=[] if precision == "fp16" else expected_modules(config),
        )
        write(output / (precision + "_artifact.json"), artifacts[precision])
        memory[precision] = {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
        }
        if max(memory[precision].values()) >= 16_000_000_000:
            raise ValueError("Selected-device memory budget exceeded")
        ref = release(model)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        if ref() is not None:
            raise ValueError("Model not released before next precision")
        memory[precision]["allocated_after_unload"] = torch.cuda.memory_allocated(0)
        if memory[precision]["allocated_after_unload"] != 0:
            raise ValueError("Live CUDA tensors remain after unloading model")
        write(output / (precision + "_memory.json"), memory[precision])
    oracle_dir = assets / "oracle"
    files = verified_checkpoint(pins["retain_only_oracle"], oracle_dir)
    identity = ModelIdentity(
        **pins["retain_only_oracle"], config_sha256=files["config.json"]["sha256"]
    )
    torch.cuda.reset_peak_memory_stats(0)
    oracle_model = load_reference(ReferenceConfig(identity, identity), oracle_dir)
    split, index, row = next(x for x in selected if x[0] == "forget05_perturbed")
    raw_oracle = evaluate_row(oracle_model, tokenizer, row, index, split, spec, generate=False)
    oracle = oracle_manifest(
        checkpoint_sha256=identity.checkpoint_sha256,
        model_revision=identity.revision,
        tokenizer_revision=spec.tokenizer_revision,
        dataset_revision=spec.dataset_revision,
        preprocessing_sha256=spec.sha256,
        evaluator_sha256=evaluator_identity(),
        records=[raw_oracle],
    )
    write(output / "oracle_evidence.json", oracle)
    memory["oracle_fp16"] = {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
    }
    if max(memory["oracle_fp16"].values()) >= 16_000_000_000:
        raise ValueError("Oracle exceeded selected-device memory budget")
    oracle_ref = release(oracle_model)
    del oracle_model
    gc.collect()
    torch.cuda.empty_cache()
    if oracle_ref() is not None or torch.cuda.memory_allocated(0) != 0:
        raise ValueError("Oracle CUDA tensors remain after unload")
    group, observations = make_observations(records, artifacts, oracle, pins, spec, run_id)
    write(output / "comparison_group.json", group)
    save_report(
        generate_report(
            run_id, recipe["condition"], observations, group, bootstrap=recipe["bootstrap"]
        ),
        output / "report.json",
    )
    write(
        output / "utility_summaries.json",
        [
            utility_summary(
                [o for o in observations if o.precision == p and o.kind == "utility_score"], group
            )
            for p in recipe["evaluation_precisions"]
        ],
    )
    result = {
        "schema_version": "work-unit-c-precision-validation-v1",
        "status": "PASS",
        "run_id": run_id,
        "versions": versions,
        "gpu": torch.cuda.get_device_name(0),
        "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "device_count": torch.cuda.device_count(),
        "memory": memory,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "model_training": False,
        "attack_detected": None,
        "durability_pass": None,
        "decision_status": "BLOCKED_RG005",
        "supported_claims": [
            "Sequential matched FP16/INT8/NF4 loader, forward, generation and metric execution"
        ],
        "unsupported_claims": [
            "Initial forgetting",
            "NPO training",
            "Recovery attack",
            "Durability",
            "Confirmatory uncertainty",
        ],
    }
    write(output / "validation_result.json", result)
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
        traceback.print_exc()
        result = {"status": "FAIL", "error_type": type(exc).__name__, "run_id": args.run_id}
        write(output / "failure.json", result)
    finally:
        write(
            output / "OUTPUT_MANIFEST.json",
            {
                "run_id": args.run_id,
                "artifacts": [
                    {
                        "path": str(p.relative_to(output)),
                        "sha256": file_digest(p),
                        "bytes": p.stat().st_size,
                    }
                    for p in sorted(output.rglob("*"))
                    if p.is_file()
                ],
            },
        )
    print(json.dumps(result))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
