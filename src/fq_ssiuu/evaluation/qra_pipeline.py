"""Versioned matched QRA observations. Artifact identities come from explicit precision slots."""

import math
from dataclasses import asdict, dataclass

from fq_ssiuu.evaluation.identity import digest, require_hash
from fq_ssiuu.evaluation.metric_definitions import definition_for, definition_hash

COMMON = (
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
METRIC_CONTEXT = (
    "definition_sha256",
    "dataset_id",
    "dataset_revision",
    "source_split",
    "sample_ids_sha256",
    "preprocessing_sha256",
    "decoding_sha256",
    "reference_sha256",
    "reference_policy",
    "reference_evaluation_sha256",
)
SLOT = ("artifact_sha256", "quantizer_name", "quantizer_version", "quantizer_config_sha256")


def metric_key(observation):
    return observation.metric + "@" + observation.source_split


def validate_group(group):
    if group.get("schema_version") != "qra-comparison-group-v1":
        raise ValueError("Unknown comparison group schema")
    if set(group["artifacts"]) != {"fp16", "int8", "nf4"}:
        raise ValueError("All precision artifact slots must be declared")
    for precision, slot in group["artifacts"].items():
        if slot.get("precision") != precision:
            raise ValueError("Artifact precision slot differs")
        require_hash(slot["artifact_sha256"])
        if not slot.get("quantizer_name") or not slot.get("quantizer_version"):
            raise ValueError("Quantizer name/version required")
        if slot["quantizer_config_sha256"] != digest(slot["quantizer_config"]):
            raise ValueError("Quantizer configuration hash differs")
        expected = {"fp16": "identity", "int8": "bitsandbytes-llm-int8", "nf4": "bitsandbytes-nf4"}
        if slot["quantizer_name"] != expected[precision]:
            raise ValueError("Quantizer family incompatible with precision")
        config = slot["quantizer_config"]
        if precision == "nf4" and (
            config.get("load_in_4bit") is not True
            or config.get("bnb_4bit_quant_type") != "nf4"
            or config.get("bnb_4bit_compute_dtype") != "float16"
        ):
            raise ValueError("NF4 requires explicit weight-NF4 configuration")
        if precision == "int8" and (
            config.get("load_in_8bit") is not True or config.get("load_in_4bit", False) is not False
        ):
            raise ValueError("INT8 requires explicit LLM.int8 configuration")
        if precision == "fp16" and config != {"dtype": "float16", "quantization": None}:
            raise ValueError("FP16 slot must be unquantized")
    if not group.get("metrics"):
        raise ValueError("Comparison metric contexts required")


@dataclass(frozen=True, kw_only=True)
class MetricObservation:
    precision: str
    metric: str
    definition_id: str
    definition_sha256: str
    kind: str
    direction: str
    value: float | None
    comparison_group_id: str
    comparison_group_sha256: str
    run_id: str
    report_id: str
    checkpoint_sha256: str
    artifact_sha256: str
    quantizer_name: str
    quantizer_version: str
    quantizer_config_sha256: str
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    dataset_id: str
    dataset_revision: str
    source_split: str
    sample_ids_sha256: str
    preprocessing_sha256: str
    decoding_sha256: str | None
    reference_sha256: str | None
    reference_policy: str
    reference_evaluation_sha256: str | None
    evaluator_sha256: str
    evaluation_sha256: str
    raw_records: list[dict]
    raw_records_sha256: str

    def validate(self, group=None):
        if self.precision not in {"fp16", "int8", "nf4"}:
            raise ValueError("Unknown precision")
        definition = definition_for(self.metric)
        if self.definition_sha256 != definition_hash(self.metric) or any(
            getattr(self, key) != definition[key] for key in ("definition_id", "kind", "direction")
        ):
            raise ValueError("Metric implementation contract differs from registered definition")
        for name in (
            "comparison_group_id",
            "run_id",
            "report_id",
            "model_id",
            "tokenizer_id",
            "dataset_id",
            "source_split",
            "quantizer_name",
            "quantizer_version",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError("Nonempty provenance identities required")
        for name in (
            "comparison_group_sha256",
            "checkpoint_sha256",
            "artifact_sha256",
            "quantizer_config_sha256",
            "sample_ids_sha256",
            "preprocessing_sha256",
            "evaluator_sha256",
            "evaluation_sha256",
            "raw_records_sha256",
        ):
            require_hash(getattr(self, name))
        for name in ("model_revision", "tokenizer_revision", "dataset_revision"):
            require_hash(getattr(self, name), 40)
        generated = self.metric.endswith("rouge_l_recall") or self.metric == "exact_match"
        if generated:
            require_hash(self.decoding_sha256)
        elif self.decoding_sha256 is not None:
            raise ValueError("Non-generation metric cannot borrow decoding provenance")
        if self.reference_policy == "not_applicable":
            if self.reference_sha256 is not None or self.reference_evaluation_sha256 is not None:
                raise ValueError("Unexpected reference provenance")
        elif self.reference_policy == "fixed_fp16_oracle":
            require_hash(self.reference_sha256)
            require_hash(self.reference_evaluation_sha256)
        else:
            raise ValueError(
                "Explicit supported reference policy required; no implicit precision matching"
            )
        if (
            self.metric in {"ks_statistic", "ks_p_value"}
            and self.reference_policy != "fixed_fp16_oracle"
        ):
            raise ValueError("KS requires an explicitly bound oracle evaluation")
        if not self.raw_records or self.raw_records_sha256 != digest(self.raw_records):
            raise ValueError("Raw per-example records or their hash differ")
        ids = [r["sample_id"] for r in self.raw_records]
        if (
            len(set(ids)) != len(ids)
            or any(not isinstance(i, str) or not i for i in ids)
            or digest(ids) != self.sample_ids_sha256
        ):
            raise ValueError("Sample identities/order differ")
        values = []
        for record in self.raw_records:
            require_hash(record["source_record_sha256"])
            require_hash(record["tensors_sha256"])
            value = record["value"]
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                raise ValueError("Raw metrics must be finite or null")
            values.append(value)
        if self.value is not None:
            if type(self.value) not in (int, float) or not math.isfinite(self.value):
                raise ValueError("Metric must be finite or null")
            if self.metric in {"ks_p_value", "ks_statistic"}:
                if not 0 <= self.value <= 1:
                    raise ValueError("KS diagnostic outside [0,1]")
            elif any(v is None for v in values) or not math.isclose(
                self.value, math.fsum(values) / len(values), rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError("Aggregate differs from raw per-example records")
        if group is not None:
            validate_group(group)
            if digest(group) != self.comparison_group_sha256:
                raise ValueError("Comparison group manifest hash differs")
            for name in COMMON:
                if group[name] != getattr(self, name):
                    raise ValueError("Comparison group provenance differs: " + name)
            context = group["metrics"].get(metric_key(self))
            if context is None:
                raise ValueError("Metric context absent from group")
            for name in METRIC_CONTEXT:
                if context[name] != getattr(self, name):
                    raise ValueError("Metric provenance differs: " + name)
            for name in SLOT:
                if group["artifacts"][self.precision][name] != getattr(self, name):
                    raise ValueError("Precision artifact provenance differs: " + name)


def compare_observations(baseline, target, group, *, bootstrap=None):
    baseline.validate(group)
    target.validate(group)
    if baseline.precision != "fp16" or target.precision not in {"int8", "nf4"}:
        raise ValueError("Compare FP16 with INT8 or NF4")
    if metric_key(baseline) != metric_key(target):
        raise ValueError("Comparison metric/split differs")
    for a, b in zip(baseline.raw_records, target.raw_records, strict=True):
        if any(a[key] != b[key] for key in ("sample_id", "source_record_sha256", "tensors_sha256")):
            raise ValueError("Paired source records or input tensors differ")
    result = {
        "baseline": asdict(baseline),
        "target": asdict(target),
        "signed_metric_change": None,
        "oriented_memorisation_change": None,
        "paired_comparison": None,
        "attack_detected": None,
        "durability_pass": None,
        "decision_status": "BLOCKED_RG005",
    }
    if baseline.value is None or target.value is None:
        result["comparison_status"] = "INCOMPLETE"
    elif baseline.kind in {"ks_p_value", "diagnostic"}:
        if bootstrap is not None:
            raise ValueError("Diagnostic metrics cannot produce recovery differences or intervals")
        result.update(
            comparison_status="DIAGNOSTIC_ONLY",
            interpretation="Diagnostics do not measure recovery or establish equivalence",
        )
    else:
        if bootstrap is None:
            raise ValueError("Explicit paired bootstrap specification required")
        from fq_ssiuu.evaluation.statistics import paired_bootstrap

        ids = [r["sample_id"] for r in baseline.raw_records]
        paired = paired_bootstrap(
            ids,
            [r["value"] for r in baseline.raw_records],
            [r["value"] for r in target.raw_records],
            **bootstrap,
        )
        result.update(
            comparison_status="DESCRIPTIVE_ONLY",
            signed_metric_change=paired["mean_change"],
            paired_comparison=paired,
        )
        if baseline.kind == "memorisation_score":
            result["oriented_memorisation_change"] = (
                paired["mean_change"] if baseline.direction == "higher" else -paired["mean_change"]
            )
            result["interpretation"] = (
                "Positive oriented change means greater observed memorisation; no attack decision"
            )
        else:
            result["interpretation"] = (
                "Utility change only; higher utility is better, not a recovery score"
            )
    return result


def validate_decisions(attack_detected, durability_pass):
    if any(
        value is not None and type(value) is not bool
        for value in (attack_detected, durability_pass)
    ):
        raise ValueError("Decisions must be bool or null")
    if attack_detected is True and durability_pass is not False:
        raise ValueError("Detected attack requires durability_pass=False")
    if durability_pass is True and attack_detected is not False:
        raise ValueError("Durability pass requires attack_detected=False")


def compute_q_score(*args, **kwargs):
    raise RuntimeError("RG-005: ratios of KS p-values are invalid recovery scores")


def run_qra_evaluation(*args, **kwargs):
    raise RuntimeError(
        "RG-005: full corrected evaluation requires an approved outcome protocol; use bounded metric functions only"
    )
