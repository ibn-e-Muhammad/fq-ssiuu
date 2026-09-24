"""Closed Work Unit C1 loader/measurement validation; no training permission."""

import json
from pathlib import Path

from experiments.reproductions.NPO.validation_contract import expected_recipe as recipe_b
from fq_ssiuu.evaluation.identity import file_digest

ROOT = Path(__file__).resolve().parents[3]


def expected_recipe():
    recipe = recipe_b()
    recipe.update(
        condition="work-unit-c-matched-precision-v1",
        entrypoint="experiments/reproductions/QRA/validate_runtime.py",
        overlay_lock="requirements/kaggle-overlay.candidate-0461.in",
        overlay_sha256=file_digest(ROOT / "requirements/kaggle-overlay.candidate-0461.in"),
        evaluation_precisions=["fp16", "int8", "nf4"],
        expected_versions={
            "bitsandbytes": "0.46.1",
            "transformers": "5.0.0",
            "torch": "2.10.0+cu128",
        },
        bootstrap={"seed": 23, "resamples": 2000, "confidence": 0.95},
        timeout_seconds=3600,
    )
    return recipe


def validate_recipe(recipe):
    if recipe != expected_recipe():
        raise ValueError("Only the exact sequential four-example C1 validation recipe is allowed")
    return recipe


def read_recipe(path):
    return validate_recipe(json.loads(Path(path).read_text()))
