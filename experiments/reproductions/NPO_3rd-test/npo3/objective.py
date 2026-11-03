"""Causal sequence-sum NPO Eq. 3; retain CE is separately token averaged."""

import math

import torch
import torch.nn.functional as functional


def sequence_nll(logits, labels):
    if logits.ndim != 3 or labels.shape != logits.shape[:2] or labels.dtype != torch.long:
        raise ValueError("Unshifted logits [B,T,V] and int64 labels [B,T] required")
    targets = labels[:, 1:]
    scored = targets != -100
    if not scored.any(dim=1).all():
        raise ValueError("Every sequence needs scored causal answer tokens")
    selected = logits[:, :-1][scored]
    if selected.dtype in (torch.bfloat16, torch.float16):
        selected = selected.float()
    token_nll = functional.cross_entropy(selected, targets[scored], reduction="none")
    return (
        torch.zeros_like(targets, dtype=token_nll.dtype).masked_scatter(scored, token_nll).sum(-1)
    )


def npo_loss(current, reference, beta=0.1):
    if (
        not math.isfinite(beta)
        or beta <= 0
        or current.ndim != 1
        or not current.numel()
        or current.shape != reference.shape
    ):
        raise ValueError("Positive beta and aligned sequence NLLs required")
    if not torch.isfinite(current).all() or not torch.isfinite(reference).all():
        raise ValueError("Finite NLLs required")
    return (2.0 / beta) * functional.softplus(beta * (reference.detach() - current)).mean()
