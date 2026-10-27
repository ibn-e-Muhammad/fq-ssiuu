"""Local checkpoint layouts and exact-identity saved-dataset adaptation."""

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from experiments.reproductions.NPO import local_datasets, local_inputs


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    registry = tmp_path / "registry"
    registry.mkdir()
    models, data = tmp_path / "models/NPO", tmp_path / "data/tofu"
    target = models / "tofu_Llama-3.2-1B-Instruct"
    target.mkdir(parents=True)
    data.mkdir(parents=True)
    pins = json.loads((local_inputs.HERE / "inputs_v1.json").read_text())
    for item in pins["files"]:
        is_data = item["repo"] == pins["dataset"]["id"]
        raw = b'{"question":"a/b","answer":"caf\\u00e9"}\n' if is_data else b"{}"
        item["sha256"] = sha(raw)
        ((data if is_data else target) / item["path"]).write_bytes(raw)
    for key, folder in (("target", target), ("retain_only_oracle", models / "oracle")):
        folder.mkdir(exist_ok=True)
        (folder / "model.safetensors").write_bytes(key.encode())
        (folder / "config.json").write_bytes(b"{}")
        pins[key]["checkpoint_sha256"] = sha(key.encode())
        pins[key]["config_sha256"] = sha(b"{}")
    (target / "generation_config.json").write_bytes(b"{}")
    (registry / "inputs_v1.json").write_text(json.dumps(pins))
    (registry / "runtime_sources_v1.json").write_text(json.dumps({
        "files": {"generation_config.json": {"sha256": sha(b"{}")}}
    }))
    monkeypatch.setattr(local_inputs, "HERE", registry)
    return models, data, target, pins


@pytest.mark.parametrize("direct", [False, True])
def test_downloaded_model_layout_and_colocated_tokenizer(inputs, direct):
    models, data, target, _ = inputs
    # An incomplete nested tokenizer must not hide complete files at model root.
    (target / "tokenizer").mkdir()
    (models / "target").mkdir()  # An unused legacy placeholder is not a checkpoint.
    resolved = local_inputs.prepare(target if direct else models, data)
    assert resolved["target"] == target
    assert resolved["tokenizer"] == target
    assert resolved["oracle"] == models / "oracle"
    assert resolved["generation"] == target / "generation_config.json"
    assert not resolved["arrow_export_pending"]


def test_legacy_layout_is_still_supported(inputs):
    models, data, target, _ = inputs
    legacy = models / "target"
    target.rename(legacy)
    tokenizer = models / "tokenizer"
    tokenizer.mkdir()
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        (tokenizer / name).write_bytes((legacy / name).read_bytes())
        if name != "config.json":
            (legacy / name).unlink()
    (legacy / "generation_config.json").rename(models / "generation_config.json")
    resolved = local_inputs.prepare(models, data)
    assert resolved["target"] == legacy
    assert resolved["tokenizer"] == tokenizer
    assert resolved["generation"] == models / "generation_config.json"


def test_changed_model_is_rejected(inputs):
    models, data, target, _ = inputs
    (target / "model.safetensors").write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="hash mismatch"):
        local_inputs.prepare(models, data)


def test_oracle_is_not_replaced_by_target(inputs):
    models, data, _, _ = inputs
    (models / "oracle/model.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="oracle"):
        local_inputs.prepare(models, data)


@pytest.mark.parametrize("newline", [False, True])
def test_pinned_serializer_preserves_escaping_and_newline(newline):
    raw = b'{"question":"a\\/b","answer":"caf\\u00e9"}' + (b"\n" if newline else b"")
    assert local_datasets.pinned_jsonl([json.loads(raw)], sha(raw)) == raw
    with pytest.raises(ValueError, match="pinned TOFU JSON hash"):
        local_datasets.pinned_jsonl([{"question": "changed", "answer": "café"}], sha(raw))


def saved_split(data, name):
    folder = data / name
    folder.mkdir()
    (folder / "state.json").write_text(json.dumps({
        "_data_files": [{"filename": "data-00000-of-00001.arrow"}]
    }))
    (folder / "dataset_info.json").write_text("{}")
    return folder


def test_missing_arrow_and_evaluation_splits_are_reported_together(tmp_path):
    folder = saved_split(tmp_path, "forget05")
    with pytest.raises(FileNotFoundError) as error:
        local_datasets.inspect_datasets(tmp_path, [
            {"path": "forget05.json"}, {"path": "world_facts_perturbed.json"}
        ])
    assert str(folder / "data-00000-of-00001.arrow") in str(error.value)
    assert "world_facts_perturbed" in str(error.value)


def test_saved_shards_cannot_escape_split(tmp_path):
    folder = saved_split(tmp_path, "forget05")
    (folder / "state.json").write_text('{"_data_files":[{"filename":"../outside.arrow"}]}')
    with pytest.raises(ValueError, match="inside its split"):
        local_datasets.inspect_datasets(tmp_path, [{"path": "forget05.json"}])


def test_arrow_precheck_then_export_without_changing_sources(inputs, monkeypatch, tmp_path):
    models, data, _, pins = inputs
    raw_path = data / "forget05.json"
    # Fixture pin uses the same slash escaping as actual TOFU.
    raw = raw_path.read_bytes().replace(b"a/b", b"a\\/b")
    pins["files"][4]["sha256"] = sha(raw)
    (local_inputs.HERE / "inputs_v1.json").write_text(json.dumps(pins))
    raw_path.unlink()
    folder = saved_split(data, "forget05")
    shard = folder / "data-00000-of-00001.arrow"
    shard.write_bytes(b"fixture Arrow content")
    calls = []

    def load(path):
        calls.append(path)
        return SimpleNamespace(with_format=lambda fmt: [json.loads(raw)])

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_from_disk=load))
    checked = local_inputs.prepare(models, data)
    assert checked["arrow_export_pending"] and not calls
    before = {p: p.read_bytes() for p in data.rglob("*") if p.is_file()}
    result = local_inputs.prepare(models, data, dataset_output=tmp_path / "run/local_dataset")
    assert not result["arrow_export_pending"]
    assert calls == [str(folder)]
    assert (result["data"] / "forget05.json").read_bytes() == raw
    assert all(p.read_bytes() == value for p, value in before.items())
    assert not raw_path.exists()
    assert len(list(result["data"].glob("*.json"))) == 6


def test_real_saved_arrow_export(tmp_path):
    datasets = pytest.importorskip("datasets")
    rows = [{"question": "a/b", "answer": "café"}]
    raw = b'{"question":"a\\/b","answer":"caf\\u00e9"}\n'
    data = tmp_path / "data"
    datasets.Dataset.from_list(rows).save_to_disk(str(data / "forget05"))
    entries = [{"path": "forget05.json", "sha256": sha(raw)}]
    sources = local_datasets.inspect_datasets(data, entries)
    output = local_datasets.export_datasets(sources, entries, tmp_path / "output")
    assert (output / "forget05.json").read_bytes() == raw
