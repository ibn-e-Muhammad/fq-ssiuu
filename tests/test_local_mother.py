"""Local entry point never claims D1 readiness when its host contract is unmet."""

import json
from unittest.mock import patch

from scripts import local_mother


def test_windows_3060_d1_is_blocked_before_any_training(tmp_path):
    with (
        patch.object(
            local_mother, "source_identity", return_value={"snapshot_commit_sha": "a" * 40}
        ),
        patch.object(local_mother.platform, "system", return_value="Windows"),
        patch.object(local_mother, "system_ram_bytes", return_value=8_000_000_000),
        patch.object(
            local_mother, "gpu_inventory", return_value=[{"name": "RTX 3060", "vram_mib": 12288}]
        ),
        patch.object(local_mother, "run_logged") as run_logged,
    ):
        output, record = local_mother.execute("d1", tmp_path, "local-d1-test")
    assert record["status"] == "BLOCKED"
    assert record["scientific_experiment_executed"] is False
    assert any("NCCL" in reason for reason in record["reasons"])
    assert any("Tesla T4" in reason for reason in record["reasons"])
    assert json.loads((output / "run_result.json").read_text()) == record
    run_logged.assert_not_called()


def test_refuses_output_inside_checkout():
    try:
        local_mother.execute("check", local_mother.ROOT / "tmp", "inside-checkout")
    except ValueError as exc:
        assert "outside" in str(exc)
    else:
        raise AssertionError("Runner accepted output under repository")


def test_recipe_file_tampering_is_detected(tmp_path, monkeypatch):
    recipe = json.loads(local_mother.RECIPE.read_text())
    recipe["source_spec_sha256"] = "0" * 64
    bogus = tmp_path / "recipe.json"
    bogus.write_text(json.dumps(recipe))
    monkeypatch.setattr(local_mother, "RECIPE", bogus)
    try:
        local_mother.validate_recipe()
    except ValueError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("Tampered recipe passed")
