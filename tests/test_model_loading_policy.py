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


def test_legacy_reference_loader_rejects_missing_identity_before_loading(monkeypatch):
    import pytest

    load = Mock()
    monkeypatch.setattr(loader, "load_base_model", load)
    for config in (None, loader.ModelConfig(), loader.ModelConfig(load_in_4bit=False)):
        with pytest.raises(ValueError, match="Unpinned or quantized"):
            loader.get_reference_model("synthetic-finetuned-reference", config)
    load.assert_not_called()


def test_model_token_uses_the_lazy_project_provider(monkeypatch):
    from fq_ssiuu import secrets

    lookup = Mock(return_value="synthetic-test-value")
    monkeypatch.setattr(secrets, "get_secret", lookup)
    assert loader._get_hf_token() == "synthetic-test-value"
    lookup.assert_called_once_with("HF_TOKEN")


def test_missing_model_token_has_one_actionable_exception_contract(monkeypatch):
    import pytest

    from fq_ssiuu import secrets

    lookup = Mock(side_effect=secrets.SecretNotFoundError("synthetic missing provider"))
    monkeypatch.setattr(secrets, "get_secret", lookup)
    with pytest.raises(secrets.SecretNotFoundError, match="Kaggle Secret") as exc:
        loader._get_hf_token()
    assert "synthetic missing provider" not in str(exc.value)
    lookup.assert_called_once_with("HF_TOKEN")
    lookup.side_effect = None
    lookup.return_value = "PENDING"
    with pytest.raises(secrets.SecretNotFoundError, match="project secret loader"):
        loader._get_hf_token()
