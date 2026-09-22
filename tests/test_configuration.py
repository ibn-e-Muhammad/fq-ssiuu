import json
from pathlib import Path

import pytest

from fq_ssiuu.configuration import load_config
from scripts.mother_runner import load_recipe

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "name",
    [
        "base.yaml",
        "conditions/c2_npo.yaml",
        "conditions/c4_ssiuu.yaml",
        "conditions/c5_fqat.yaml",
        "conditions/c6_fq_ssiuu.yaml",
        "conditions/c8_quail.yaml",
    ],
)
def test_standard_paths_reject_historical_configs(name, tmp_path):
    for path in (ROOT / "configs" / name, ROOT / "configs/historical/august_pilots" / name):
        with pytest.raises(ValueError):
            load_config(path)
        with pytest.raises((ValueError, json.JSONDecodeError)):
            load_recipe(path)
    copied = tmp_path / "copied.yaml"
    copied.write_bytes((ROOT / "configs/historical/august_pilots" / name).read_bytes())
    with pytest.raises(ValueError):
        load_config(copied)


def test_only_explicit_accepted_schema_can_load(tmp_path):
    p = tmp_path / "accepted.yaml"
    p.write_text('schema_version: "1.0"\nmethod_status: ACCEPTED_FOR_VALIDATION\n')
    assert load_config(p)["method_status"] == "ACCEPTED_FOR_VALIDATION"
