import copy
from types import SimpleNamespace

import pytest
import torch

from experiments.reproductions.common import tensors_digest
from experiments.reproductions.NPO.candidate import compute_causal_npo_rt_loss
from experiments.reproductions.NPO.reference import ModelIdentity, ReferenceConfig
from experiments.reproductions.NPO.training_forward import microbatch_objective


class RecordingLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(11, 4)
        self.head = torch.nn.Linear(4, 11)
        self.received = []

    def forward(self, **inputs):
        self.received.append({k: v.clone() for k, v in inputs.items()})
        return SimpleNamespace(logits=self.head(self.embedding(inputs["input_ids"])))


def case():
    torch.manual_seed(1001)
    candidate = RecordingLM().half()
    reference = copy.deepcopy(candidate).eval().requires_grad_(False)
    identity = ModelIdentity("test/model", "a" * 40, "b" * 64, "test/tokenizer", "c" * 40, "d" * 64)
    config = ReferenceConfig(identity, identity)
    tensors = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 9], [1, 2, 5, 9, 9]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]),
        "labels": torch.tensor([[-100, -100, 3, 4, 9], [-100, -100, 5, 9, -100]]),
    }
    batch = {
        "tensors": tensors,
        "tensors_sha256": tensors_digest(tensors),
        "identities": dict(
            model_id=identity.model_id,
            model_revision=identity.revision,
            tokenizer_id=identity.tokenizer_id,
            tokenizer_revision=identity.tokenizer_revision,
        ),
    }
    return candidate, reference, config, batch


def test_training_receives_identical_inputs_and_preserves_source_gradient():
    candidate, reference, config, batch = case()
    expected_model = copy.deepcopy(candidate)
    terms, record = microbatch_objective(candidate, reference, batch, batch, config, runtime=False)
    with torch.no_grad():
        ref_logits = reference(**batch["tensors"]).logits
    expected = compute_causal_npo_rt_loss(
        expected_model(**batch["tensors"]).logits,
        ref_logits,
        batch["tensors"]["labels"],
        expected_model(**batch["tensors"]).logits,
        batch["tensors"]["labels"],
    ).total
    torch.testing.assert_close(terms.total, expected, rtol=0, atol=0)
    terms.total.backward()
    expected.backward()
    for actual, wanted in zip(candidate.parameters(), expected_model.parameters(), strict=True):
        assert actual.grad is not None and torch.isfinite(actual.grad).all()
        assert actual.grad.abs().sum() > 0
        torch.testing.assert_close(actual.grad, wanted.grad, rtol=0, atol=0)
    assert all(p.grad is None and not p.requires_grad for p in reference.parameters())
    assert record["reference_has_gradient_graph"] is False
    for received in candidate.received + reference.received:
        # Forward needs IDs/mask; the same labels are consumed once, externally,
        # by the verified loss, avoiding an unused Transformers CE allocation.
        for name in ("input_ids", "attention_mask"):
            assert torch.equal(received[name], batch["tensors"][name])


@pytest.mark.parametrize("defect", ["identity", "mutated", "padding", "reference_train"])
def test_training_boundary_rejects_unsafe_inputs_before_candidate_forward(defect):
    candidate, reference, config, batch = case()
    if defect == "identity":
        batch["identities"]["model_revision"] = "f" * 40
    elif defect == "mutated":
        batch["tensors"]["input_ids"][0, 0] = 10
    elif defect == "padding":
        batch["tensors"]["labels"][1, 4] = 9
        batch["tensors_sha256"] = tensors_digest(batch["tensors"])
    else:
        reference.train()
    with pytest.raises(ValueError):
        microbatch_objective(candidate, reference, batch, batch, config, runtime=False)
    assert candidate.received == []
