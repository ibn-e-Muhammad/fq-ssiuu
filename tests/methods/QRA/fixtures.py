from fq_ssiuu.evaluation.identity import digest
from fq_ssiuu.evaluation.metric_definitions import definition_for, definition_hash
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
        "reference_evaluation_sha256": "7" * 64 if reference else None,
        "reference_policy": "fixed_fp16_oracle" if reference else "not_applicable",
    }


def group(metrics=("answer_log_probability", "retain_probability")):
    common = {
        "schema_version": "qra-comparison-group-v1",
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
        common["artifacts"][precision] = {
            "precision": precision,
            "artifact_sha256": code * 64,
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
    return MetricObservation(
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
