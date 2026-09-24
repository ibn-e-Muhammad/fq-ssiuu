from dataclasses import replace

import torch

from fq_ssiuu.evaluation.content_identity import evaluation_hash, oracle_manifest
from fq_ssiuu.evaluation.identity import digest
from fq_ssiuu.evaluation.metric_definitions import definition_for, definition_hash
from fq_ssiuu.evaluation.precision_artifact import tensor_identity
from fq_ssiuu.evaluation.qra_pipeline import MetricObservation

BOOTSTRAP = {"seed": 23, "resamples": 100, "confidence": 0.95}


def context(metric):
    reference = metric in {"ks_p_value", "ks_statistic"}
    return {
        "definition_sha256": definition_hash(metric),
        "dataset_id": "fixture/tofu",
        "dataset_revision": "c" * 40,
        "source_split": "forget05" if not metric.startswith("retain_") else "retain95",
        "sample_ids_sha256": digest(["sample-0", "sample-1"]),
        "preprocessing_sha256": "4" * 64,
        "decoding_sha256": "5" * 64
        if metric.endswith("rouge_l_recall") or metric == "exact_match"
        else None,
        "reference_sha256": "6" * 64 if reference else None,
        "reference_evaluation_sha256": digest(oracle()) if reference else None,
        "reference_policy": "fixed_fp16_oracle" if reference else "not_applicable",
    }


def group(metrics=("answer_log_probability", "retain_probability")):
    common = {
        "schema_version": "qra-comparison-group-v2",
        "comparison_group_id": "group-unit",
        "run_id": "run-unit",
        "report_id": "report-unit",
        "checkpoint_sha256": "a" * 64,
        "model_id": "fixture/model",
        "model_revision": "a" * 40,
        "tokenizer_id": "fixture/tokenizer",
        "tokenizer_revision": "b" * 40,
        "evaluator_sha256": "8" * 64,
    }
    common["artifacts"] = {}
    common["oracle_evidence"] = oracle()
    for precision, code, name in [
        ("fp16", "1", "identity"),
        ("int8", "2", "bitsandbytes-llm-int8"),
        ("nf4", "3", "bitsandbytes-nf4"),
    ]:
        config = (
            {"dtype": "float16", "quantization": None}
            if precision == "fp16"
            else (
                {"load_in_8bit": True, "load_in_4bit": False, "llm_int8_threshold": 6.0}
                if precision == "int8"
                else {
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": "float16",
                    "bnb_4bit_use_double_quant": False,
                }
            )
        )
        artifact = fixture_artifact(precision, config)
        common["artifacts"][precision] = {
            "artifact_manifest": artifact,
            "precision": precision,
            "artifact_sha256": digest(artifact),
            "quantizer_name": name,
            "quantizer_version": "fixture-1",
            "quantizer_config": config,
            "quantizer_config_sha256": digest(config),
        }
    common["metrics"] = {
        metric + "@" + context(metric)["source_split"]: context(metric) for metric in metrics
    }
    return common


def observation(
    metric="answer_log_probability",
    precision="fp16",
    values=(-3.0, -2.0),
    *,
    manifest=None,
    value=None,
):
    manifest = manifest or group((metric,))
    definition = definition_for(metric)
    records = [
        {
            "sample_id": f"sample-{i}",
            "value": x,
            "source_record_sha256": "9" * 64,
            "tensors_sha256": "b" * 64,
        }
        for i, x in enumerate(values)
    ]
    args = {
        key: manifest[key]
        for key in (
            "comparison_group_id",
            "run_id",
            "report_id",
            "checkpoint_sha256",
            "model_id",
            "model_revision",
            "tokenizer_id",
            "tokenizer_revision",
            "evaluator_sha256",
        )
    }
    args.update(
        {
            key: manifest["artifacts"][precision][key]
            for key in (
                "artifact_sha256",
                "quantizer_name",
                "quantizer_version",
                "quantizer_config_sha256",
            )
        }
    )
    obs = MetricObservation(
        precision=precision,
        metric=metric,
        definition_id=definition["definition_id"],
        kind=definition["kind"],
        direction=definition["direction"],
        value=(
            sum(values) / len(values)
            if value is None and all(x is not None for x in values)
            else value
        ),
        comparison_group_sha256=digest(manifest),
        evaluation_sha256="d" * 64,
        raw_records=records,
        raw_records_sha256=digest(records),
        **args,
        **context(metric),
    )
    return replace(obs, evaluation_sha256=evaluation_hash(obs))


def observations(manifest):
    return [
        observation(
            metric=metric.split("@")[0],
            precision=precision,
            manifest=manifest,
            values=(-3, -2) if metric.startswith("answer_") else (0.8, 0.6),
        )
        for metric in manifest["metrics"]
        for precision in ("fp16", "int8", "nf4")
    ]


def oracle():
    return oracle_manifest(
        checkpoint_sha256="6" * 64,
        model_revision="d" * 40,
        tokenizer_revision="b" * 40,
        dataset_revision="c" * 40,
        preprocessing_sha256="4" * 64,
        evaluator_sha256="8" * 64,
        records=[
            {
                "sample_id": f"sample-{i}",
                "tofu_truth_ratio": v,
                "source_record_sha256": "9" * 64,
                "tensors_sha256": "b" * 64,
            }
            for i, v in enumerate([0.2, 0.3])
        ],
    )


def fixture_artifact(precision, config):
    weight = tensor_identity(
        torch.tensor(
            [1, 2], dtype={"fp16": torch.float16, "int8": torch.int8, "nf4": torch.uint8}[precision]
        )
    )
    modules = (
        []
        if precision == "fp16"
        else [
            {
                "name": "model.proj",
                "class": "bitsandbytes.nn.modules."
                + ("Linear4bit" if precision == "nf4" else "Linear8bitLt"),
                "weight": weight,
                "state": {
                    "quant_type": "nf4",
                    "blocksize": 64,
                    "quant_state": {
                        "absmax": {"tensor": tensor_identity(torch.ones(1))},
                        "quant_map": {"tensor": tensor_identity(torch.linspace(-1, 1, 16))},
                    },
                }
                if precision == "nf4"
                else {"SCB": {"tensor": tensor_identity(torch.ones(2))}},
            }
        ]
    )
    return {
        "schema_version": "precision-artifact-v1",
        "precision": precision,
        "checkpoint_sha256": "a" * 64,
        "versions": {
            "torch": "fixture-1",
            "transformers": "fixture-1",
            "bitsandbytes": "fixture-1",
        },
        "quantizer_config": config,
        "expected_quantized_modules": [m["name"] for m in modules],
        "quantized_modules": modules,
        "state_tensors": [{"name": "model.proj.weight", **weight}],
        "saved_shards": [],
        "persistence": "in_memory_only",
    }
