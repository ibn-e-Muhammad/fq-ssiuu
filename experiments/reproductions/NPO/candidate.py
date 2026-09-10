"""Causal sequence-sum NPO (paper Eq. 3) and the author's separate NPO+RT variant.

Conformance: handbook/methods/NPO_CONFORMANCE.md. Logits/labels are unshifted.
The historical token-mean helper in npo_loss.py is not this objective.
"""

import math
from dataclasses import dataclass


def _validate(logits, labels):
    import torch

    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("Expected unshifted logits [B,T,V] and labels [B,T]")
    if logits.shape[0] == 0 or logits.shape[1] < 2 or logits.shape[2] < 2:
        raise ValueError("Nonempty batch, at least two positions and vocabulary entries required")
    if not logits.is_floating_point() or labels.dtype != torch.long:
        raise ValueError("Floating logits and int64 labels required")
    if logits.device != labels.device:
        raise ValueError("Logits and labels must share a device")
    scored = labels[:, 1:]
    if (((scored < 0) & (scored != -100)) | (scored >= logits.shape[-1])).any():
        raise ValueError("Scored labels must be vocabulary IDs or -100")
    if ((scored != -100).sum(dim=-1) == 0).any():
        raise ValueError("Every example must have a scored causal target")


def causal_sequence_nll(logits, labels):
    """One summed target NLL per example; low-precision reductions use FP32."""
    import torch
    import torch.nn.functional as functional

    _validate(logits, labels)
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    values = functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
        reduction="none",
    )
    return values.reshape(labels.shape[0], -1).sum(dim=-1)


def compute_causal_npo_loss(model_logits, ref_logits, labels, beta=0.1):
    """Eq. 3, averaged over examples; reference logits never receive gradients."""
    import torch.nn.functional as functional

    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    if model_logits.shape != ref_logits.shape or model_logits.device != ref_logits.device:
        raise ValueError("Current and reference logits must have matching shape and device")
    current = causal_sequence_nll(model_logits, labels)
    reference = causal_sequence_nll(ref_logits.detach(), labels)
    return (2.0 / beta) * functional.softplus(beta * (reference - current)).mean()


@dataclass
class NPORetainTerms:
    forget: object
    retain: object
    total: object


def compute_causal_npo_rt_loss(
    model_logits,
    ref_logits,
    labels,
    retain_logits,
    retain_labels,
    beta=0.1,
    npo_coeff=1.0,
    retain_coeff=1.0,
):
    """Author npo_grad_diff: weighted NPO plus token-mean retain cross entropy."""
    if any(not math.isfinite(x) or x < 0 for x in (npo_coeff, retain_coeff)):
        raise ValueError("Objective coefficients must be finite and nonnegative")
    forget = compute_causal_npo_loss(model_logits, ref_logits, labels, beta)
    retain = causal_sequence_nll(retain_logits, retain_labels).sum()
    retain = retain / (retain_labels[:, 1:] != -100).sum()
    return NPORetainTerms(forget, retain, npo_coeff * forget + retain_coeff * retain)
