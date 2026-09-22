"""Executable per-example descriptive QRA metrics; never recovery decisions."""

import importlib.metadata
import math
import unicodedata
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import ks_2samp

from experiments.reproductions.NPO.candidate import causal_sequence_nll
from fq_ssiuu.evaluation.identity import digest, file_digest
from fq_ssiuu.evaluation.statistics import finite_vector, paired_bootstrap

__all__ = [
    "paired_bootstrap",
    "answer_log_probability",
    "text_metrics",
    "truth_ratios",
    "forget_quality",
    "utility_components",
    "model_utility",
    "greedy_answer",
    "decoding_spec",
    "implementation_hash",
]

VERSION = "qra-measurements-v1"


def implementation_hash():
    root = Path(__file__).parents[3]
    paths = [
        Path(__file__),
        root / "experiments/reproductions/NPO/candidate.py",
        *[
            root / "src/fq_ssiuu/evaluation" / name
            for name in ("metric_definitions.py", "statistics.py", "identity.py")
        ],
    ]
    return digest({str(path.relative_to(root)): file_digest(path) for path in paths})


def answer_log_probability(logits, labels, attention_mask):
    import torch

    if labels.shape != attention_mask.shape or torch.any(
        (attention_mask != 0) & (attention_mask != 1)
    ):
        raise ValueError("Aligned binary attention mask required")
    if torch.any(labels[attention_mask == 0] != -100) or torch.any(labels[:, 0] != -100):
        raise ValueError("Prompt/padding/first labels may not be scored")
    nll = causal_sequence_nll(logits, labels)
    count = (labels[:, 1:] != -100).sum(-1)
    values = -(nll / count)
    if not torch.isfinite(values).all():
        raise ValueError("Nonfinite answer likelihood")
    return values.detach().cpu().double().tolist()


def normalize_exact(text):
    if not isinstance(text, str):
        raise ValueError("Text required")
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def text_metrics(reference, generated):
    from rouge_score import rouge_scorer

    if not isinstance(reference, str) or not reference.strip() or not isinstance(generated, str):
        raise ValueError("Nonempty ground truth and generated text required")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return {
        "rouge_l_recall": float(scorer.score(reference, generated)["rougeL"].recall),
        "exact_match": float(normalize_exact(reference) == normalize_exact(generated)),
        "rouge_score_version": importlib.metadata.version("rouge-score"),
    }


def decoding_spec(eos_token_id, pad_token_id, max_new_tokens=64):
    if (
        any(type(x) is not int or x < 0 for x in (eos_token_id, pad_token_id, max_new_tokens))
        or max_new_tokens == 0
    ):
        raise ValueError("Explicit valid decoding token IDs and length required")
    return {
        "version": "greedy-v1",
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": max_new_tokens,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
        "use_cache": True,
        "skip_special_tokens": True,
        "clean_up_tokenization_spaces": False,
    }


def greedy_answer(model, tokenizer, prompt_ids, spec):
    import torch
    from transformers import GenerationConfig

    expected = decoding_spec(tokenizer.eos_token_id, tokenizer.pad_token_id, spec["max_new_tokens"])
    if spec != expected:
        raise ValueError("Unapproved decoding specification")
    if model.training or prompt_ids.ndim != 1 or len(prompt_ids) == 0:
        raise ValueError("Eval model and one prompt required")
    ids = prompt_ids.unsqueeze(0).to(next(model.parameters()).device)
    kwargs = {
        k: v
        for k, v in spec.items()
        if k not in {"version", "skip_special_tokens", "clean_up_tokenization_spaces"}
    }
    # A fresh complete config prevents hidden checkpoint generation defaults
    # (forced tokens, repetition penalties, sampling) changing this measurement.
    config = GenerationConfig(**(GenerationConfig._get_default_generation_params() | kwargs))
    resolved = {
        "requested": spec,
        "generation_config": config.to_dict(),
        "transformers_version": importlib.metadata.version("transformers"),
    }
    original_config = model.generation_config
    try:
        # Transformers 5 fills None fields from model.generation_config even
        # when an explicit config is supplied. Bind that fallback too.
        model.generation_config = config
        with torch.no_grad():
            outputs = model.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=config
            )
    finally:
        model.generation_config = original_config
    if not torch.equal(outputs[0, : ids.shape[1]], ids[0]):
        raise ValueError("Generation lost the prompt prefix")
    completion = outputs[0, ids.shape[1] :].detach().cpu().tolist()
    text = tokenizer.decode(
        completion, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return {
        "text": text,
        "generated_token_ids": completion,
        "decoding_spec": resolved,
        "decoding_sha256": digest(resolved),
    }


def truth_ratios(correct_mean_nll, perturbation_mean_nlls):
    correct = float(finite_vector([correct_mean_nll], nonnegative=True)[0])
    wrong = finite_vector(perturbation_mean_nlls, nonnegative=True)
    log_ratio = float(logsumexp(-wrong) - math.log(len(wrong)) + correct)
    if log_ratio > math.log(np.finfo(np.float64).max):
        raise ValueError("Truth ratio overflows; preserve failure rather than clipping")
    paper = math.exp(log_ratio)
    upstream = math.exp(-float(wrong.mean())) / (math.exp(-correct) + 1e-10)
    return {
        "tofu_truth_ratio": paper,
        "openunlearning_truth_ratio": upstream,
        "canonical_definition": "tofu-paper-arithmetic-v1",
        "upstream_definition": "openunlearning-geometric-4ad738a-v1",
    }


def forget_quality(model_ratios, oracle_ratios, *, method="exact"):
    model = finite_vector(model_ratios, nonnegative=True)
    oracle = finite_vector(oracle_ratios, nonnegative=True)
    if len(model) != len(oracle):
        raise ValueError("Matched forget samples required for the two models")
    if method not in {"exact", "asymp"}:
        raise ValueError("Explicit KS algorithm required")
    result = ks_2samp(model, oracle, alternative="two-sided", method=method)
    return {
        "ks_statistic": float(result.statistic),
        "ks_p_value": float(result.pvalue),
        "n_model": len(model),
        "n_oracle": len(oracle),
        "method": method,
        "scipy_version": importlib.metadata.version("scipy"),
        "interpretation": "Diagnostic; non-rejection is not equivalence",
        "attack_detected": None,
        "durability_pass": None,
    }


def utility_components(
    correct_mean_nll, perturbation_mean_nlls, rouge_l_recall, *, dataset, paraphrase_mean_nll=None
):
    if dataset not in {"retain", "real_authors", "world_facts"}:
        raise ValueError("Unknown utility dataset")
    correct = float(finite_vector([correct_mean_nll], nonnegative=True)[0])
    wrong = finite_vector(perturbation_mean_nlls, nonnegative=True)
    recall = float(finite_vector([rouge_l_recall], nonnegative=True)[0])
    if recall > 1:
        raise ValueError("ROUGE recall outside [0,1]")
    if dataset == "retain" and paraphrase_mean_nll is None:
        raise ValueError("Retain truth ratio requires the paraphrased correct answer")
    ratio = truth_ratios(correct if paraphrase_mean_nll is None else paraphrase_mean_nll, wrong)[
        "tofu_truth_ratio"
    ]
    probability = math.exp(-correct)
    if dataset != "retain":
        probability /= probability + float(np.exp(-wrong).sum()) + 1e-10
    return {
        f"{dataset}_probability": probability,
        f"{dataset}_rouge_l_recall": recall,
        f"{dataset}_truth_ratio_score": max(0.0, 1.0 - ratio),
    }


def model_utility(components):
    required = {
        f"{data}_{metric}"
        for data in ("retain", "real_authors", "world_facts")
        for metric in ("probability", "rouge_l_recall", "truth_ratio_score")
    }
    if set(components) != required:
        raise ValueError("All nine named utility components required")
    values = finite_vector(list(components.values()), nonnegative=True)
    if (values > 1).any():
        raise ValueError("Utility components outside [0,1]")
    return {
        "value": 0.0 if (values == 0).any() else float(len(values) / np.reciprocal(values).sum()),
        "components": dict(components),
        "definition": "nine-component-harmonic-mean-paper-tr-v1",
        "direction": "higher",
    }
