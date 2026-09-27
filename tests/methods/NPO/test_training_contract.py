import json
from pathlib import Path

import pytest

from experiments.reproductions.common import file_digest
from experiments.reproductions.NPO.training_contract import (
    deepspeed_config,
    linear_schedule,
    read_configs,
)

ROOT = Path(__file__).resolve().parents[3]


def test_frozen_configs_and_historical_effective_defaults():
    train = ROOT / "configs/reproductions/NPO/d1_train_v1.json"
    evaluation = ROOT / "configs/reproductions/NPO/d1_eval_v1.json"
    a, b = read_configs(train, evaluation, file_digest(train), file_digest(evaluation))
    assert a["micro_batch"] * a["accumulation"] * a["world_size"] == 32
    assert a["max_steps"] == 62 and b["attack_detected"] is None
    ds = deepspeed_config(2048)
    assert ds["gradient_clipping"] == 0  # Publication-era effective value, not new DS default.
    assert ds["zero_optimization"]["reduce_bucket_size"] == 4194304
    assert ds["zero_optimization"]["stage3_prefetch_bucket_size"] == 3774873
    assert ds["zero_optimization"]["offload_optimizer"]["device"] == "none"
    assert [linear_schedule(x) for x in (0, 3, 6, 34, 62, 63)] == [0, 0.5, 1, 0.5, 0, 0]


@pytest.mark.parametrize(
    "change",
    [
        {"accumulation": 4},
        {"trainable_parameters": "lora"},
        {"gradient_clipping": 1},
        {"optimizer": "adamw_8bit"},
        {"npo_coeff": True},
        {"threshold": 0.05},
    ],
)
def test_changed_source_configuration_rejected_even_with_recomputed_file_hash(tmp_path, change):
    source = ROOT / "configs/reproductions/NPO/d1_train_v1.json"
    evaluation = ROOT / "configs/reproductions/NPO/d1_eval_v1.json"
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(json.loads(source.read_text()) | change))
    with pytest.raises(ValueError, match="accepted source"):
        read_configs(path, evaluation, file_digest(path), file_digest(evaluation))
    with pytest.raises(ValueError, match="bytes changed"):
        read_configs(source, evaluation, "0" * 64, file_digest(evaluation))
