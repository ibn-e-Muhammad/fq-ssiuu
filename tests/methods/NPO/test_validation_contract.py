import json

import pytest

from experiments.reproductions.NPO.validation_contract import expected_recipe, validate_recipe
from scripts.mother_runner import load_recipe


def test_forward_only_recipe_can_pass_without_opening_pilot_gate(tmp_path):
    recipe = expected_recipe()
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe))
    assert load_recipe(path) == recipe
    recipe["kind"] = "pilot"
    path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError, match="RG-005"):
        load_recipe(path)


@pytest.mark.parametrize(
    "change",
    [
        {"optimizer_steps": 1},
        {"backward": True},
        {"evaluation_precisions": ["nf4"]},
        {"reference_precision": "nf4"},
        {"seed": 42},
        {"sample_indices": {"forget05": [0, 1, 2, 3, 4]}},
        {"entrypoint": "other.py"},
        {"unregistered_threshold": 0.1},
        {"pins_sha256": "a" * 64},
    ],
)
def test_changed_validation_scope_fails_closed(change):
    with pytest.raises(ValueError, match="exact four-example"):
        validate_recipe(expected_recipe() | change)
