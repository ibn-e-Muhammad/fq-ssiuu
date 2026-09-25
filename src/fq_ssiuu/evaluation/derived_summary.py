"""Explicit derived utility summary; never a per-example observation."""

import math

from fq_ssiuu.evaluation.identity import digest
from fq_ssiuu.evaluation.qra_pipeline import COMMON, SLOT

UTILITY_COMPONENTS = tuple(
    f"{split}_{metric}"
    for split in ("retain", "real_authors", "world_facts")
    for metric in ("probability", "rouge_l_recall", "truth_ratio_score")
)
UTILITY_DEFINITION = {
    "metric": "model_utility",
    "version": "derived-v1",
    "direction": "higher",
    "formula": "harmonic mean of the nine split-level arithmetic component means; zero if any component is zero",
    "components": list(UTILITY_COMPONENTS),
    "truth_ratio_variant": "TOFU paper arithmetic canonical; differs from OU geometric variant",
}


def utility_summary(observations, group):
    by_name = {}
    for obs in observations:
        obs.validate(group)
        if obs.metric not in UTILITY_COMPONENTS or obs.metric in by_name:
            raise ValueError("Exactly nine distinct registered utility components required")
        if obs.value is None or not 0 <= obs.value <= 1:
            raise ValueError("Complete utility components in [0,1] required")
        by_name[obs.metric] = obs
    if set(by_name) != set(UTILITY_COMPONENTS):
        raise ValueError("Incomplete utility summary")
    first = next(iter(by_name.values()))
    for obs in by_name.values():
        if any(getattr(obs, k) != getattr(first, k) for k in (*COMMON, *SLOT, "precision")):
            raise ValueError("Utility component provenance differs")
    values = [by_name[k].value for k in UTILITY_COMPONENTS]
    value = 0.0 if 0 in values else len(values) / math.fsum(1 / v for v in values)
    result = {
        "schema_version": "derived-utility-v1",
        "definition": UTILITY_DEFINITION,
        "definition_sha256": digest(UTILITY_DEFINITION),
        "value": value,
        "direction": "higher",
        "precision": first.precision,
        "comparison_group_sha256": digest(group),
        "components": [
            {
                "metric": k,
                "value": by_name[k].value,
                "evaluation_sha256": by_name[k].evaluation_sha256,
            }
            for k in UTILITY_COMPONENTS
        ],
        "attack_detected": None,
        "durability_pass": None,
    }
    result["summary_sha256"] = digest(result)
    return result


def validate_utility_summary(summary, observations, group):
    if summary != utility_summary(observations, group):
        raise ValueError("Derived utility content differs from component observations")
