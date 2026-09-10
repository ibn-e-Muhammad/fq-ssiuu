"""Exercise loader argument forwarding/reference freezing without fetching a model."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from fq_ssiuu.models import loader


def test_default_loader_forwards_fp16_and_only_selected_device(monkeypatch):
    model = torch.nn.Linear(2, 2)
    factory = Mock(return_value=model)
    transformers = SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=factory),
        BitsAndBytesConfig=lambda **kwargs: kwargs,
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(loader, "_get_hf_token", lambda: "synthetic-test-value")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert loader.load_base_model() is model
    kwargs = factory.call_args.kwargs
    assert kwargs["torch_dtype"] == torch.float16
    assert kwargs["quantization_config"]["bnb_4bit_compute_dtype"] == torch.float16
    assert kwargs["device_map"] == {"": 0}
    assert kwargs["attn_implementation"] == "sdpa"


def test_reference_is_frozen_and_in_eval_mode_with_explicit_config(monkeypatch):
    model = torch.nn.Linear(2, 2)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    load = Mock(return_value=model)
    monkeypatch.setattr(loader, "load_base_model", load)
    reference = loader.get_reference_model(
        "synthetic-finetuned-reference", loader.ModelConfig(load_in_4bit=False)
    )
    assert reference is model and not reference.training
    assert all(not p.requires_grad for p in reference.parameters())
    assert load.call_args.args[0].load_in_4bit is False
    assert load.call_args.args[0].base_model_id == "synthetic-finetuned-reference"
    for name, tensor in reference.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)


def test_model_token_uses_the_lazy_project_provider(monkeypatch):
    from fq_ssiuu import secrets

    lookup = Mock(return_value="synthetic-test-value")
    monkeypatch.setattr(secrets, "get_secret", lookup)
    assert loader._get_hf_token() == "synthetic-test-value"
    lookup.assert_called_once_with("HF_TOKEN")
