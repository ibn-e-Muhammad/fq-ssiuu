from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from experiments.reproductions.NPO.adapter import collate_examples
from experiments.reproductions.NPO.reference import (
    ModelIdentity,
    ReferenceConfig,
    forward_pair,
    load_reference,
    validate_reference,
)
from tests.methods.NPO.test_adapter import FixtureTokenizer, example, spec


def config():
    identity = ModelIdentity(
        "fixture/model", "a" * 40, "b" * 64, "fixture/tokenizer", "b" * 40, "c" * 64
    )
    return ReferenceConfig(identity, identity)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(10, 3, dtype=torch.float16)
        self.seen = []

    def forward(self, input_ids, attention_mask, labels):
        self.seen.append(
            (input_ids.clone(), attention_mask.clone(), labels.clone(), torch.is_grad_enabled())
        )
        return SimpleNamespace(logits=self.emb(input_ids))


def frozen():
    model = Tiny().eval().requires_grad_(False)
    return model


@pytest.mark.parametrize(
    "change",
    [{"load_in_4bit": True}, {"load_in_8bit": True}, {"dtype": "bfloat16"}, {"device": "auto"}],
)
def test_unsafe_reference_config_fails_before_model_or_file_access(change):
    factory = Mock()
    with pytest.raises(ValueError):
        load_reference(replace(config(), **change), "/must-not-access", factory=factory)
    factory.assert_not_called()


@pytest.mark.parametrize(
    "field", ["revision", "tokenizer_revision", "checkpoint_sha256", "model_id"]
)
def test_mismatched_candidate_reference_identities_fail(field):
    cfg = config()
    ref = replace(cfg.reference, **{field: "d" * (64 if field == "checkpoint_sha256" else 40)})
    with pytest.raises(ValueError):
        replace(cfg, reference=ref).validate()


def test_reference_model_invariants():
    cfg = config()
    model = frozen()
    validate_reference(model, cfg, runtime=False)
    model.train()
    with pytest.raises(ValueError, match="eval"):
        validate_reference(model, cfg, runtime=False)
    model.eval().requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        validate_reference(model, cfg, runtime=False)
    model.requires_grad_(False)
    model.is_loaded_in_4bit = True
    with pytest.raises(ValueError, match="Quantized"):
        validate_reference(model, cfg, runtime=False)
    model.is_loaded_in_4bit = False
    model.float()
    with pytest.raises(ValueError, match="FP16"):
        validate_reference(model, cfg, runtime=False)


def test_identical_tensors_and_no_gradient_graph_without_optimizer():
    candidate = Tiny()
    reference = frozen()
    batch = collate_examples([example(), example("short", 1)], FixtureTokenizer(), spec())
    outputs = forward_pair(candidate, reference, batch, config(), runtime=False)
    assert outputs["input_hashes"] == [batch["tensors_sha256"]] * 2
    for a, b in zip(candidate.seen[0][:3], reference.seen[0][:3], strict=True):
        assert torch.equal(a, b)
    assert not candidate.seen[0][3] and not reference.seen[0][3]
    assert (
        outputs["reference_logits"].grad_fn is None
        and not outputs["reference_logits"].requires_grad
    )
    assert all(p.grad is None for p in candidate.parameters())
    batch["tensors"]["input_ids"][0, 0] = 8
    with pytest.raises(ValueError, match="recorded"):
        forward_pair(candidate, reference, batch, config(), runtime=False)


def test_verified_loading_freezes_model_and_rejects_changed_bytes(tmp_path, monkeypatch):
    import experiments.reproductions.NPO.reference as module
    from fq_ssiuu.evaluation.identity import file_digest

    (tmp_path / "model.safetensors").write_bytes(b"fixture-weight-bytes")
    (tmp_path / "config.json").write_text("{}")
    identity = replace(
        config().candidate,
        checkpoint_sha256=file_digest(tmp_path / "model.safetensors"),
        config_sha256=file_digest(tmp_path / "config.json"),
    )
    cfg = ReferenceConfig(identity, identity)
    # GPU residency is tested by runtime validation; isolate file/factory behavior here.
    original = module.validate_reference
    monkeypatch.setattr(
        module, "validate_reference", lambda model, conf: original(model, conf, runtime=False)
    )
    factory = Mock(return_value=Tiny())
    result = load_reference(cfg, tmp_path, factory=factory)
    assert not result.training and all(not p.requires_grad for p in result.parameters())
    assert factory.call_args.kwargs == {
        "local_files_only": True,
        "torch_dtype": torch.float16,
        "device_map": {"": 0},
        "attn_implementation": "sdpa",
    }
    (tmp_path / "model.safetensors").write_bytes(b"changed")
    factory.reset_mock()
    with pytest.raises(ValueError, match="checkpoint bytes"):
        load_reference(cfg, tmp_path, factory=factory)
    factory.assert_not_called()


def test_adapter_identity_mismatch_is_rejected_before_forward():
    candidate = Tiny()
    reference = frozen()
    batch = collate_examples([example()], FixtureTokenizer(), spec())
    batch["identities"]["tokenizer_revision"] = "e" * 40
    with pytest.raises(ValueError, match="identity differ"):
        forward_pair(candidate, reference, batch, config(), runtime=False)
    assert not candidate.seen and not reference.seen
