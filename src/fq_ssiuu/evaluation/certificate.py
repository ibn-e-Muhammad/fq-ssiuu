"""Non-certifying empirical durability records with explicit unresolved decisions.

Historical v1/v2 payloads are preserved on read, never silently relabelled valid.
"""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from fq_ssiuu.evaluation.qra_pipeline import (
    MetricObservation,
    compare_observations,
    validate_decisions,
)


def generate_report(run_id: str, condition_id: str, observations: list[MetricObservation]) -> dict:
    if not run_id or not condition_id:
        raise ValueError("Run and condition identities required")
    seen = set()
    for observation in observations:
        observation.validate()
        key = observation.precision, observation.metric
        if key in seen:
            raise ValueError("Duplicate precision/metric observation")
        seen.add(key)
    if len({o.checkpoint_sha256 for o in observations}) > 1:
        raise ValueError("Report observations must share one source checkpoint")
    comparisons = []
    for baseline in observations:
        if baseline.precision == "fp16":
            for target in observations:
                if target.metric == baseline.metric and target.precision != "fp16":
                    comparisons.append(compare_observations(baseline, target))
    groups = {}
    for observation in observations:
        key = (
            observation.metric,
            observation.definition_id,
            observation.kind,
            observation.direction,
        )
        if observation.value is not None:
            groups.setdefault(key, set()).add(observation.precision)
    complete = all(
        any(
            key[2] == kind and precisions == {"fp16", "int8", "nf4"}
            for key, precisions in groups.items()
        )
        for kind in ("utility_score", "memorisation_score")
    )
    return {
        "schema_version": "v3",
        "report_type": "Empirical Durability Report",
        "run_id": run_id,
        "condition_id": condition_id,
        "report_status": "VALID_DESCRIPTIVE_RECORD" if complete else "INCOMPLETE",
        "observations": [asdict(o) for o in observations],
        "comparisons": comparisons,
        "attack_detected": None,
        "durability_pass": None,
        "decision_status": "BLOCKED_RG005",
        "unresolved_prerequisites": [
            "Approved versioned directional decision rule",
            "Initial forgetting and retained-utility acceptance",
            "Uncertainty and across-precision aggregation policy",
        ],
        "supported_claims": ["Recorded observations only"],
        "unsupported_claims": [
            "Attack confirmation",
            "Empirical durability pass",
            "Formal certification",
        ],
    }


def save_report(report: dict, output_path) -> None:
    if report.get("schema_version") != "v3":
        raise ValueError("Only current descriptive records can be saved by this writer")
    validate_decisions(report.get("attack_detected"), report.get("durability_pass"))
    if report.get("attack_detected") is not None or report.get("durability_pass") is not None:
        raise ValueError("Scientific decision generation is unavailable until RG-005 closes")
    expected = generate_report(
        report["run_id"],
        report["condition_id"],
        [MetricObservation(**o) for o in report["observations"]],
    )
    if report != expected:
        raise ValueError("Report differs from the validated descriptive contract")
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(encoded)


def load_report(path) -> dict:
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    if data.get("schema_version") != "v3":
        return {
            "schema_version": "historical",
            "report_status": "SUPERSEDED_INTERPRETATION",
            "historical_payload": data,
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "attack_detected": None,
            "durability_pass": None,
            "reason": "Legacy p-value-ratio and certified/threshold_met fields do not establish recovery or durability",
        }
    validate_decisions(data.get("attack_detected"), data.get("durability_pass"))
    expected = generate_report(
        data["run_id"], data["condition_id"], [MetricObservation(**o) for o in data["observations"]]
    )
    if data != expected:
        raise ValueError("Unsupported or inconsistent current report")
    return data


def generate_certificate(*args, **kwargs):
    raise RuntimeError("RG-005: legacy certificate generation is retired; use generate_report")
