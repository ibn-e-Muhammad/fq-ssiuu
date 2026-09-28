import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from experiments.reproductions.NPO.checkpoint_values import compare_values
from experiments.reproductions.NPO.d1_recipe import expected_recipe, validate_recipe
from experiments.reproductions.NPO.engine import destroy_engine
from scripts import mother_runner


@pytest.mark.parametrize(
    "change",
    [
        {"condition": "fq-ssiuu"},
        {"attack_detected": True},
        {"durability_pass": True},
        {"seed": 42},
        {"kind": "pilot"},
        {"no_build_isolation": False},
        {"entrypoint": "scripts/train.py"},
        {"threshold": 0.05},
    ],
)
def test_closed_recipe_cannot_expand_scope(change):
    recipe = expected_recipe()
    assert validate_recipe(recipe) == recipe
    with pytest.raises(ValueError, match="exact"):
        validate_recipe(recipe | change)


def test_recipe_uses_scratch_without_duplicate_working_checkpoint():
    raw, sealed = mother_runner.output_locations(
        expected_recipe(), "/kaggle/working/runs/d1-test", "d1-test"
    )
    assert raw == Path("/kaggle/tmp/fq-d1-runs/d1-test")
    assert sealed == Path("/kaggle/working/runs/d1-test-sealed")
    with pytest.raises(ValueError):
        mother_runner.output_locations(expected_recipe(), "/tmp/run", "d1-test")


def test_actual_model_values_must_change_and_remain_finite(tmp_path):
    original = {"weight": torch.arange(12, dtype=torch.float16).reshape(3, 4)}
    a, b = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    save_file(original, a)
    save_file(original, b)
    with pytest.raises(ValueError, match="No trained"):
        compare_values(a, b)
    changed = original["weight"].clone()
    changed[1, 2] += 0.5
    save_file({"weight": changed}, b)
    assert compare_values(a, b) == [
        {"parameter": "weight", "numel": 12, "changed_values": 1, "max_absolute_change": 0.5}
    ]
    changed[0, 0] = float("nan")
    save_file({"weight": changed}, b)
    with pytest.raises(ValueError, match="Nonfinite"):
        compare_values(a, b)


def test_cleanup_releases_own_paged_moments_only():
    a, b = torch.ones(2), torch.zeros(2)
    a.is_paged = b.is_paged = True
    page_manager = SimpleNamespace(paged_tensors=[a, b])
    optimizer = SimpleNamespace(state={"own": {"state1": a}}, page_mng=page_manager)
    called = []
    engine = SimpleNamespace(
        optimizer=SimpleNamespace(optimizer=optimizer), destroy=lambda: called.append(True)
    )
    destroy_engine(engine)
    assert len(page_manager.paged_tensors) == 1 and page_manager.paged_tensors[0] is b
    assert called == [True]


def test_failed_d1_child_keeps_partial_output_in_real_sealed_archive(tmp_path, monkeypatch):
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps(expected_recipe()))
    raw, sealed = tmp_path / "scratch/run", tmp_path / "working/sealed"
    monkeypatch.setattr(mother_runner, "output_locations", lambda *a: (raw, sealed))
    monkeypatch.setattr(mother_runner.subprocess, "check_output", lambda *a, **k: b"snapshot\n")
    monkeypatch.setattr(
        mother_runner, "validate_setup_artifacts", lambda *a: {"run_id": "env-fixture"}
    )
    calls = []

    def child(name, argv, output, env, timeout):
        calls.append((name, argv, env))
        if name == "environment":
            return 0
        directory = output / "adapted_reproduction"
        directory.mkdir()
        (directory / "partial.jsonl").write_text('{"optimizer_step":1}\n')
        return 7

    monkeypatch.setattr(mother_runner, "run_command", child)
    result = mother_runner.execute(
        recipe, tmp_path / "requested", {"snapshot_commit_sha": "snapshot"}, "d1-test-failed"
    )
    assert result["status"] == "FAIL" and result["failure_phase"] == "adapted_reproduction"
    assert result["adapted_reproduction_attempted"]
    assert "--no-build-isolation" in calls[0][1]
    assert calls[0][2]["DS_BUILD_OPS"] == "0" and calls[0][2]["CUDA_VISIBLE_DEVICES"] == "0"
    with tarfile.open(sealed / "artifacts.tar") as archive:
        assert (
            archive.extractfile("adapted_reproduction/partial.jsonl").read()
            == b'{"optimizer_step":1}\n'
        )
        assert json.load(archive.extractfile("run_result.json"))["status"] == "FAIL"
    assert (raw / "adapted_reproduction/partial.jsonl").exists()


def test_versioned_resource_recipe_keeps_original_and_rejects_mixed_identity():
    original, resource = expected_recipe(1), expected_recipe(2)
    assert validate_recipe(original) == original
    assert validate_recipe(resource) == resource
    changed = {k for k in original if original[k] != resource[k]}
    assert changed == {"condition", "train_config", "train_config_sha256"}
    for key in changed:
        with pytest.raises(ValueError, match="exact"):
            validate_recipe(resource | {key: original[key]})
    with pytest.raises(ValueError, match="Unknown"):
        expected_recipe(3)
