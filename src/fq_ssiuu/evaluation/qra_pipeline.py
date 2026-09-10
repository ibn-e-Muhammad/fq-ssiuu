"""Descriptive QRA outcome contract; scientific decisions remain blocked by RG-005.

KS p-values are diagnostics, never recovery magnitudes. No default attack or
durability threshold is defined. Historical execution lives in experiments/.
"""

import math
import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MetricObservation:
    precision: str
    metric: str
    definition_id: str
    kind: str  # ks_p_value, memorisation_score, utility_score
    direction: str  # higher, lower, diagnostic
    value: float | None
    run_id: str
    checkpoint_sha256: str  # same merged, pre-quantisation checkpoint for a comparison
    sample_ids_sha256: str
    reference_sha256: str
    evaluation_sha256: str

    def validate(self):
        if self.precision not in {"fp16", "int8", "nf4"}:
            raise ValueError("Unknown precision")
        if self.kind not in {"ks_p_value", "memorisation_score", "utility_score"}:
            raise ValueError("Unknown metric kind")
        if self.kind != "memorisation_score" and self.direction != "diagnostic":
            raise ValueError("P-values and utility are not directional recovery metrics")
        if self.kind == "memorisation_score" and self.direction not in {"higher", "lower"}:
            raise ValueError("Memorisation metric requires an explicit direction")
        if not all((self.metric, self.definition_id, self.run_id)):
            raise ValueError("Metric definition and run identity required")
        for digest in (
            self.checkpoint_sha256,
            self.sample_ids_sha256,
            self.reference_sha256,
            self.evaluation_sha256,
        ):
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("Immutable checkpoint/sample/reference/evaluation hashes required")
        if self.value is not None:
            if isinstance(self.value, bool) or not math.isfinite(self.value):
                raise ValueError("Metric must be finite or null")
            if self.kind == "ks_p_value" and not 0 <= self.value <= 1:
                raise ValueError("KS p-value must lie in [0,1]")


def compare_observations(baseline: MetricObservation, target: MetricObservation) -> dict:
    """Describe matched observations; an oriented delta is not an attack decision."""
    baseline.validate()
    target.validate()
    if baseline.precision != "fp16" or target.precision not in {"int8", "nf4"}:
        raise ValueError("Compare FP16 with INT8 or NF4")
    for name in (
        "metric",
        "definition_id",
        "kind",
        "direction",
        "checkpoint_sha256",
        "sample_ids_sha256",
        "reference_sha256",
    ):
        if getattr(baseline, name) != getattr(target, name):
            raise ValueError(f"Incompatible comparison provenance: {name}")
    result = {
        "baseline": asdict(baseline),
        "target": asdict(target),
        "signed_metric_change": None,
        "oriented_memorisation_change": None,
        "attack_detected": None,
        "durability_pass": None,
        "decision_status": "BLOCKED_RG005",
    }
    if baseline.value is None or target.value is None:
        result["comparison_status"] = "INCOMPLETE"
    elif baseline.kind == "ks_p_value":
        result["comparison_status"] = "DIAGNOSTIC_ONLY"
        result["interpretation"] = "KS p-values do not measure recovery or establish equivalence"
    else:
        result["comparison_status"] = "DESCRIPTIVE_ONLY"
        delta = target.value - baseline.value
        result["signed_metric_change"] = delta
        if baseline.kind == "memorisation_score":
            result["oriented_memorisation_change"] = (
                delta if baseline.direction == "higher" else -delta
            )
            result["interpretation"] = (
                "Positive oriented change means more observed memorisation; no attack threshold is approved"
            )
        else:
            result["interpretation"] = (
                "Utility change only; utility acceptance requires a separately approved rule"
            )
    return result


def validate_decisions(attack_detected, durability_pass):
    """Logical constraint only: lack of detected attack does not imply durability."""
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
    raise RuntimeError("RG-005: corrected model evaluation requires an accepted outcome protocol")
