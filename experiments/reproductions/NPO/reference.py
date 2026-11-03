"""Fail-closed, immutable FP16 NPO reference boundary. No optimizer or cache."""

from dataclasses import dataclass
from pathlib import Path

import torch

from experiments.reproductions.common import file_digest, require_hash, tensors_digest


@dataclass(frozen=True)
class ModelIdentity:
    model_id: str
    revision: str
    checkpoint_sha256: str
    tokenizer_id: str
    tokenizer_revision: str
    config_sha256: str

    def validate(self):
        if not self.model_id or not self.tokenizer_id:
            raise ValueError("Explicit model and tokenizer identities required")
        require_hash(self.revision, 40)
        require_hash(self.tokenizer_revision, 40)
        require_hash(self.checkpoint_sha256)
        require_hash(self.config_sha256)


@dataclass(frozen=True)
class ReferenceConfig:
    candidate: ModelIdentity
    reference: ModelIdentity
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    dtype: str = "float16"
    device: str = "cuda:0"

    def validate(self):
        self.candidate.validate()
        self.reference.validate()
        if self.candidate != self.reference:
            raise ValueError(
                "NPO requires the exact candidate pre-unlearning checkpoint/tokenizer identity"
            )
        if self.load_in_4bit is not False or self.load_in_8bit is not False:
            raise ValueError("Quantized NPO reference is forbidden")
        if self.dtype != "float16" or self.device != "cuda:0":
            raise ValueError("NPO reference must be FP16 on the selected device cuda:0")


def validate_reference(model, config, *, runtime=True):
    config.validate()
    if any(
        getattr(model, name, False)
        for name in ("is_loaded_in_4bit", "is_loaded_in_8bit", "is_quantized")
    ):
        raise ValueError("Quantized reference model rejected")
    if (
        getattr(model, "hf_quantizer", None) is not None
        or getattr(getattr(model, "config", None), "quantization_config", None) is not None
    ):
        raise ValueError("Quantization metadata on reference rejected")
    if any(module.training for module in model.modules()):
        raise ValueError("Reference must remain in eval mode")
    params = list(model.parameters())
    if not params or any(p.requires_grad for p in params):
        raise ValueError("Reference parameters must all be frozen")
    if any(p.dtype != torch.float16 for p in params):
        raise ValueError("Reference parameters must be unquantized FP16")
    if runtime:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("Exactly one visible CUDA device required")
        if any(str(p.device) != "cuda:0" for p in params):
            raise ValueError("Reference cannot be sharded or offloaded")


def load_reference(config, checkpoint_dir, *, factory=None):
    config.validate()  # Reject inherited unsafe settings before file/model access.
    path = Path(checkpoint_dir)
    if file_digest(path / "config.json") != config.reference.config_sha256:
        raise ValueError("Reference config bytes differ from pinned identity")
    if file_digest(path / "model.safetensors") != config.reference.checkpoint_sha256:
        raise ValueError("Reference checkpoint bytes differ from pinned identity")
    if factory is None:
        from transformers import AutoModelForCausalLM

        factory = AutoModelForCausalLM.from_pretrained
    model = factory(
        str(path),
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model.requires_grad_(False)
    model.eval()
    validate_reference(model, config)
    return model


def forward_pair(candidate, reference, batch, config, *, runtime=True):
    """Bounded validation only: identical byte-checked inputs; both forwards no_grad."""
    validate_reference(reference, config, runtime=runtime)
    identities = batch["identities"]
    expected = config.candidate
    for name, wanted in (
        ("model_id", expected.model_id),
        ("model_revision", expected.revision),
        ("tokenizer_id", expected.tokenizer_id),
        ("tokenizer_revision", expected.tokenizer_revision),
    ):
        if identities[name] != wanted:
            raise ValueError("Adapter and candidate/reference identity differ: " + name)
    tensors = batch["tensors"]
    if tensors_digest(tensors) != batch["tensors_sha256"]:
        raise ValueError("Input tensors no longer match the recorded adapter output")
    if tuple(tensors) != ("input_ids", "attention_mask", "labels"):
        raise ValueError("Expected adapter input, attention and label tensors")
    if torch.any(tensors["labels"][tensors["attention_mask"] == 0] != -100):
        raise ValueError("Scored padding is forbidden")
    outputs = []
    sent = []
    for model in (candidate, reference):
        device = next(model.parameters()).device
        inputs = {name: value.clone().to(device) for name, value in tensors.items()}
        before = tensors_digest(inputs)
        if before != batch["tensors_sha256"]:
            raise ValueError("Candidate/reference tensors differ")
        with torch.no_grad():
            output = model(**inputs).logits
        if tensors_digest(inputs) != before:
            raise ValueError("Model mutated its supplied input tensors")
        if output.requires_grad or output.grad_fn is not None:
            raise ValueError("Reference/validation output carries a gradient graph")
        outputs.append(output)
        sent.append(before)
    return {
        "candidate_logits": outputs[0],
        "reference_logits": outputs[1],
        "input_hashes": sent,
        "gradient_graph": False,
    }
