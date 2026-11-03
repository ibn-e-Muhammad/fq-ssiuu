"""Closed, forward-only Work Unit B contract; not a pilot gate bypass."""

import json
from pathlib import Path

from fq_ssiuu.evaluation.identity import file_digest


def expected_recipe():
    return {
        "schema_version": "1.0",
        "kind": "method_validation",
        "method_status": "VALIDATION_ONLY",
        "condition": "work-unit-b-forward-only-v1",
        "seed": 23,
        "overlay_lock": "requirements/kaggle-overlay.validated.lock",
        "entrypoint": "experiments/reproductions/NPO/validate_runtime.py",
        "pins": "experiments/reproductions/NPO/inputs_v1.json",
        "pins_sha256": file_digest(Path(__file__).with_name("inputs_v1.json")),
        "sample_indices": {
            "forget05_perturbed": [0],
            "retain_perturbed": [0],
            "real_authors_perturbed": [0],
            "world_facts_perturbed": [0],
        },
        "reference_precision": "fp16",
        "evaluation_precisions": ["fp16"],
        "max_length": 512,
        "max_new_tokens": 64,
        "beta": 0.1,
        "backward": False,
        "optimizer_steps": 0,
        "reference_cache": False,
        "oracle_policy": "fixed_fp16_oracle_validation_only",
        "required_secrets": [],
        "timeout_seconds": 2400,
    }


def validate_recipe(recipe):
    if recipe != expected_recipe():
        raise ValueError("Only the exact four-example forward-only validation recipe is allowed")
    return recipe


def read_recipe(path):
    return validate_recipe(json.loads(Path(path).read_text()))
