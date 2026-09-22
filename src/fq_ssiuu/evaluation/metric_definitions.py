"""Versioned descriptive metric contracts; no recovery decision thresholds."""

from fq_ssiuu.evaluation.identity import digest

DEFINITIONS = {
    "answer_log_probability": {
        "kind": "memorisation_score",
        "direction": "higher",
        "formula": "negative mean causally shifted answer/EOS NLL; ignore -100",
    },
    "rouge_l_recall": {
        "kind": "memorisation_score",
        "direction": "higher",
        "formula": "rouge-score 0.1.2 RougeScorer rougeL recall(target,prediction), use_stemmer=True",
    },
    "exact_match": {
        "kind": "memorisation_score",
        "direction": "higher",
        "formula": "equality after Unicode NFKC, casefold, whitespace collapse; punctuation retained",
    },
    "tofu_truth_ratio": {
        "kind": "diagnostic",
        "direction": "diagnostic",
        "formula": "TOFU Eq1: arithmetic mean(exp(-perturbation mean NLL))/exp(-paraphrase mean NLL); logsumexp numerical implementation; no epsilon",
    },
    "openunlearning_truth_ratio": {
        "kind": "diagnostic",
        "direction": "diagnostic",
        "formula": "OU 4ad738a: exp(-mean(perturbation mean NLL))/(exp(-paraphrase mean NLL)+1e-10)",
    },
    "ks_statistic": {
        "kind": "diagnostic",
        "direction": "diagnostic",
        "formula": "two-sided scipy.stats.ks_2samp statistic D on raw matched canonical truth-ratio samples; explicit method",
    },
    "ks_p_value": {
        "kind": "ks_p_value",
        "direction": "diagnostic",
        "formula": "two-sided scipy.stats.ks_2samp p-value on raw matched canonical truth-ratio samples; explicit method; not equivalence",
    },
}
for dataset in ("retain", "real_authors", "world_facts"):
    DEFINITIONS[f"{dataset}_probability"] = {
        "kind": "utility_score",
        "direction": "higher",
        "formula": (
            "exp(-answer mean NLL)"
            if dataset == "retain"
            else "OU length-normalized correct probability/(correct+sum(wrong probabilities)+1e-10)"
        ),
    }
    DEFINITIONS[f"{dataset}_rouge_l_recall"] = {
        "kind": "utility_score",
        "direction": "higher",
        "formula": "rouge-score 0.1.2 RougeScorer rougeL recall(target,prediction), use_stemmer=True",
    }
    DEFINITIONS[f"{dataset}_truth_ratio_score"] = {
        "kind": "utility_score",
        "direction": "higher",
        "formula": "max(0,1-canonical paper arithmetic truth ratio)",
    }
for name, definition in DEFINITIONS.items():
    definition.update(metric=name, version="v1", definition_id=name + "-v1")


def definition_for(metric):
    if metric not in DEFINITIONS:
        raise ValueError("Unknown metric; an arbitrary definition ID cannot establish equivalence")
    return dict(DEFINITIONS[metric])


def definition_hash(metric):
    return digest(definition_for(metric))
