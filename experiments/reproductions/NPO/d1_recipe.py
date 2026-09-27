"""One closed source reproduction route; never an RG-005 confirmatory bypass."""

import json
from pathlib import Path

from experiments.reproductions.common import file_digest
from experiments.reproductions.NPO.training_contract import read_configs

ROOT = Path(__file__).resolve().parents[3]


def expected_recipe():
    files = {
        "train_config": "configs/reproductions/NPO/d1_train_v1.json",
        "eval_config": "configs/reproductions/NPO/d1_eval_v1.json",
        "overlay": "requirements/kaggle-overlay.d1-candidate.in",
        "source_spec": "experiments/reproductions/NPO/source_v1.json",
        "runtime_sources": "experiments/reproductions/NPO/runtime_sources_v1.json",
    }
    return {
        "schema_version": "1.0",
        "kind": "adapted_reproduction",
        "condition": "d1-npo-source-faithful-1b-v1",
        "seed": 1001,
        "entrypoint": "experiments/reproductions/NPO/train_runtime.py",
        "overlay_lock": files["overlay"],
        "no_build_isolation": True,
        "required_secrets": [],
        "timeout_seconds": 39600,
        "storage": "scratch-raw-single-working-archive",
        "attack_detected": None,
        "durability_pass": None,
        "inputs_registry_sha256": "9c24f66e568ce60dba39c61e92e948e48027b1318a0580ff859928d5ec02dbc6",
        **files,
        **{key + "_sha256": file_digest(ROOT / path) for key, path in files.items()},
    }


def validate_recipe(recipe):
    if json.dumps(recipe, sort_keys=True) != json.dumps(expected_recipe(), sort_keys=True):
        raise ValueError("Only the exact approved-scope D1 source reproduction recipe is allowed")
    if file_digest(Path(__file__).with_name("inputs_v1.json")) != recipe["inputs_registry_sha256"]:
        raise ValueError("Accepted model/data identities changed")
    read_configs(
        ROOT / recipe["train_config"],
        ROOT / recipe["eval_config"],
        recipe["train_config_sha256"],
        recipe["eval_config_sha256"],
    )
    return recipe


def read_recipe(path):
    return validate_recipe(json.loads(Path(path).read_text()))
