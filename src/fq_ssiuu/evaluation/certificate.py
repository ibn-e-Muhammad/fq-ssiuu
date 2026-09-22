"""Descriptive v4 reports bound to a comparison manifest; no scientific verdicts."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from fq_ssiuu.evaluation.qra_pipeline import (
    MetricObservation,
    compare_observations,
    metric_key,
    validate_decisions,
    validate_group,
)


def generate_report(run_id, condition_id, observations, group, *, bootstrap):
    validate_group(group)
    if not condition_id or run_id != group["run_id"]:
        raise ValueError("Report/run linkage differs")
    seen = {}
    for observation in observations:
        observation.validate(group)
        key = (observation.precision, metric_key(observation))
        if key in seen:
            raise ValueError("Duplicate precision/metric observation")
        seen[key] = observation
    comparisons = []
    for (precision, key), baseline in seen.items():
        if precision == "fp16":
            for target_precision in ("int8", "nf4"):
                target = seen.get((target_precision, key))
                if target is not None:
                    config = None if baseline.kind in {"diagnostic", "ks_p_value"} else bootstrap
                    comparisons.append(
                        compare_observations(baseline, target, group, bootstrap=config)
                    )
    complete = all(
        (p, key) in seen and seen[p, key].value is not None
        for key in group["metrics"]
        for p in ("fp16", "int8", "nf4")
    )
    complete = complete and all(
        any(o.kind == kind for o in observations)
        for kind in ("memorisation_score", "utility_score")
    )
    return {
        "schema_version": "v4",
        "report_type": "Empirical Durability Report",
        "run_id": run_id,
        "report_id": group["report_id"],
        "condition_id": condition_id,
        "report_status": "VALID_DESCRIPTIVE_RECORD" if complete else "INCOMPLETE",
        "comparison_group": group,
        "bootstrap_spec": bootstrap,
        "observations": [asdict(o) for o in observations],
        "comparisons": comparisons,
        "attack_detected": None,
        "durability_pass": None,
        "decision_status": "BLOCKED_RG005",
        "supported_claims": ["Recorded paired descriptive measurements only"],
        "unsupported_claims": [
            "Recovery decision",
            "Durability pass",
            "Formal certification",
            "Initial forgetting acceptance",
        ],
    }


def _validate_report(report):
    if report.get("schema_version") != "v4":
        raise ValueError("Only v4 provenance-bound reports can be saved")
    validate_decisions(report.get("attack_detected"), report.get("durability_pass"))
    if report.get("attack_detected") is not None or report.get("durability_pass") is not None:
        raise ValueError("Scientific decisions are unavailable until RG-005 approval")
    expected = generate_report(
        report["run_id"],
        report["condition_id"],
        [MetricObservation(**o) for o in report["observations"]],
        report["comparison_group"],
        bootstrap=report["bootstrap_spec"],
    )
    if report != expected:
        raise ValueError("Report differs from validated descriptive contract")


def save_report(report, output_path):
    _validate_report(report)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")


def load_report(path):
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    if data.get("schema_version") != "v4":
        return {
            "schema_version": "historical",
            "report_status": "SUPERSEDED_INTERPRETATION",
            "historical_payload": data,
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "migration": {
                "from_schema": data.get("schema_version"),
                "to_schema": None,
                "automatic_migration": False,
                "reason": "Legacy provenance/decision contracts cannot establish current paired measurements",
            },
            "attack_detected": None,
            "durability_pass": None,
        }
    _validate_report(data)
    return data


def generate_certificate(*args, **kwargs):
    raise RuntimeError("RG-005: legacy certificate generation is retired; use generate_report")
