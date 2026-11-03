"""One source NPO+RT microbatch, preserving candidate gradients and frozen reference."""

import torch

from experiments.reproductions.common import tensors_digest
from experiments.reproductions.NPO.candidate import (
    NPORetainTerms,
    causal_sequence_nll,
    npo_from_sequence_nll,
)
from experiments.reproductions.NPO.reference import validate_reference


def validated_inputs(batch, config, *, device):
    expected = config.candidate
    for name, value in (
        ("model_id", expected.model_id),
        ("model_revision", expected.revision),
        ("tokenizer_id", expected.tokenizer_id),
        ("tokenizer_revision", expected.tokenizer_revision),
    ):
        if batch["identities"].get(name) != value:
            raise ValueError("Candidate/reference input identity mismatch: " + name)
    tensors = batch["tensors"]
    if set(tensors) != {"input_ids", "attention_mask", "labels"}:
        raise ValueError("Complete training tensors required")
    if tensors_digest(tensors) != batch["tensors_sha256"]:
        raise ValueError("Training tensors mutated after preprocessing")
    if torch.any(tensors["labels"][tensors["attention_mask"] == 0] != -100):
        raise ValueError("Padding cannot enter the loss")
    inputs = {k: v.to(device).clone() for k, v in tensors.items()}
    if tensors_digest(inputs) != batch["tensors_sha256"]:
        raise ValueError("Device transfer changed candidate/reference inputs")
    return inputs


def microbatch_objective(candidate, reference, forget, retain, config, *, runtime=True):
    validate_reference(reference, config, runtime=runtime)
    device = "cuda:0" if runtime else next(reference.parameters()).device
    ref_inputs = validated_inputs(forget, config, device=device)
    with torch.no_grad():
        # Labels are consumed by the parity-tested causal loss below. Passing
        # them to Transformers would compute an unused second cross entropy.
        output = reference(
            input_ids=ref_inputs["input_ids"], attention_mask=ref_inputs["attention_mask"]
        ).logits
        ref_nll = causal_sequence_nll(output, ref_inputs["labels"])
    if output.requires_grad or ref_nll.requires_grad:
        raise ValueError("Frozen reference carries a gradient graph")
    if tensors_digest(ref_inputs) != forget["tensors_sha256"]:
        raise ValueError("Reference mutated its inputs")
    del output, ref_inputs
    current_inputs = validated_inputs(forget, config, device=device)
    retain_inputs = validated_inputs(retain, config, device=device)
    current_output = candidate(
        input_ids=current_inputs["input_ids"], attention_mask=current_inputs["attention_mask"]
    ).logits
    current_nll = causal_sequence_nll(current_output, current_inputs["labels"])
    forget_loss = npo_from_sequence_nll(current_nll, ref_nll, beta=0.1)
    retained = candidate(
        input_ids=retain_inputs["input_ids"], attention_mask=retain_inputs["attention_mask"]
    ).logits
    retain_loss = causal_sequence_nll(retained, retain_inputs["labels"]).sum()
    retain_loss = retain_loss / (retain_inputs["labels"][:, 1:] != -100).sum()
    terms = NPORetainTerms(forget_loss, retain_loss, forget_loss + retain_loss)
    if not terms.total.requires_grad or not torch.isfinite(terms.total):
        raise ValueError("Missing or nonfinite candidate objective graph")
    if (
        tensors_digest(current_inputs) != forget["tensors_sha256"]
        or tensors_digest(retain_inputs) != retain["tensors_sha256"]
    ):
        raise ValueError("Candidate mutated its inputs")
    return terms, {
        "reference_nll": ref_nll.detach().cpu().tolist(),
        "candidate_nll": current_nll.detach().cpu().tolist(),
        "forget_input_sha256": forget["tensors_sha256"],
        "retain_input_sha256": retain["tensors_sha256"],
        "reference_has_gradient_graph": False,
    }
