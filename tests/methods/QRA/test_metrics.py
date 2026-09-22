"""Independent numerical fixtures. No model, dataset, credentials or network."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.reproductions.QRA.metrics import (
    answer_log_probability,
    decoding_spec,
    forget_quality,
    greedy_answer,
    implementation_hash,
    model_utility,
    paired_bootstrap,
    text_metrics,
    truth_ratios,
    utility_components,
)
from fq_ssiuu.evaluation.identity import digest


def test_teacher_forcing_is_answer_mean_and_causally_shifted():
    # At the two scored positions the correct probabilities are 1/2 and 1/4.
    probabilities = torch.tensor(
        [[[0.1, 0.9], [0.5, 0.5], [0.75, 0.25], [0.8, 0.2]]], dtype=torch.float64
    )
    labels = torch.tensor([[-100, -100, 0, 1]])
    assert answer_log_probability(
        probabilities.log(), labels, torch.ones_like(labels)
    ) == pytest.approx([math.log(0.125) / 2])
    labels[0, 3] = -100
    assert answer_log_probability(
        probabilities.log(), labels, torch.tensor([[1, 1, 1, 0]])
    ) == pytest.approx([math.log(0.5)])
    labels[0, 3] = 1
    with pytest.raises(ValueError, match="padding"):
        answer_log_probability(probabilities.log(), labels, torch.tensor([[1, 1, 1, 0]]))
    with pytest.raises(ValueError):
        answer_log_probability(
            probabilities.log(), torch.full_like(labels, -100), torch.ones_like(labels)
        )


def test_rouge_is_recall_with_stemming_and_exact_match_is_precisely_normalized():
    assert text_metrics("one two three four", "one two")["rouge_l_recall"] == 0.5
    assert text_metrics("running cats", "run cat")["rouge_l_recall"] == 1.0
    assert text_metrics("Ａ  Straẞe\n", "a strasse")["exact_match"] == 1.0
    assert text_metrics("answer.", "answer")["exact_match"] == 0.0
    assert text_metrics("answer", "")["rouge_l_recall"] == 0.0
    with pytest.raises(ValueError):
        text_metrics("", "answer")


def test_paper_arithmetic_and_upstream_geometric_are_distinct():
    result = truth_ratios(-math.log(0.5), [-math.log(0.01), -math.log(0.81)])
    assert result["tofu_truth_ratio"] == pytest.approx(0.82)
    assert result["openunlearning_truth_ratio"] == pytest.approx(0.09 / (0.5 + 1e-10))
    assert result["canonical_definition"] == "tofu-paper-arithmetic-v1"
    assert truth_ratios(1000, [1000, 1000])["tofu_truth_ratio"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        truth_ratios(1000, [0])


@pytest.mark.parametrize("bad", [[], [float("nan")], [-1], [True]])
def test_invalid_truth_ratio_inputs_fail(bad):
    with pytest.raises(ValueError):
        truth_ratios(1, bad)


def test_ks_reports_statistic_and_pvalue_without_equivalence_or_effect():
    same = forget_quality([1, 2, 3], [1, 2, 3])
    assert same["ks_statistic"] == 0 and same["ks_p_value"] == 1
    separated = forget_quality([1, 2, 3], [4, 5, 6])
    assert separated["ks_statistic"] == 1
    # Of choose(6,3)=20 allocations, two have complete separation.
    assert separated["ks_p_value"] == pytest.approx(0.1)
    assert same["attack_detected"] is None and same["durability_pass"] is None
    assert "not equivalence" in same["interpretation"]
    with pytest.raises(ValueError):
        forget_quality([1], [1, 2])


def test_utility_components_and_nine_component_harmonic_mean():
    retain = utility_components(
        math.log(2),
        [math.log(4), math.log(4)],
        0.75,
        dataset="retain",
        paraphrase_mean_nll=math.log(2),
    )
    assert retain == pytest.approx(
        {"retain_probability": 0.5, "retain_rouge_l_recall": 0.75, "retain_truth_ratio_score": 0.5}
    )
    real = utility_components(math.log(2), [math.log(4), math.log(4)], 0.75, dataset="real_authors")
    assert real["real_authors_probability"] == pytest.approx(0.5 / (1 + 1e-10))
    components = {
        f"{dataset}_{metric}": 0.5
        for dataset in ("retain", "real_authors", "world_facts")
        for metric in ("probability", "rouge_l_recall", "truth_ratio_score")
    }
    components["retain_probability"] = 0.25
    assert model_utility(components)["value"] == pytest.approx(9 / 20)
    components["retain_probability"] = 0
    assert model_utility(components)["value"] == 0
    del components["retain_probability"]
    with pytest.raises(ValueError):
        model_utility(components)
    with pytest.raises(ValueError):
        utility_components(1, [2], 0.5, dataset="retain")


def test_paired_bootstrap_preserves_pairing_and_is_repeatable():
    result = paired_bootstrap(["a", "b"], [-3, -2], [-2, -3], seed=23, resamples=400)
    assert result == paired_bootstrap(["a", "b"], [-3, -2], [-2, -3], seed=23, resamples=400)
    assert result["paired_differences"] == [1, -1]
    assert result["mean_change"] == 0
    assert result["confidence_interval"] == [-1, 1]
    constant = paired_bootstrap(["a", "b"], [10, -90], [12, -88], seed=7, resamples=10)
    assert constant["confidence_interval"] == [2, 2]
    with pytest.raises(ValueError):
        paired_bootstrap(["a", "a"], [1, 2], [2, 3], seed=1, resamples=3)
    with pytest.raises(ValueError):
        paired_bootstrap(["a"], [np.nan], [2], seed=1, resamples=3)
    with pytest.raises(ValueError):
        paired_bootstrap(["a"], [1], [2], seed=1, resamples=1)


class FakeGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.generation_config = SimpleNamespace(do_sample=True, repetition_penalty=7)

    def generate(self, input_ids, attention_mask, generation_config):
        assert not torch.is_grad_enabled()
        assert generation_config.do_sample is False and generation_config.num_beams == 1
        assert generation_config.repetition_penalty == 1.0
        assert attention_mask.tolist() == [[1, 1]]
        return torch.cat([input_ids, torch.tensor([[4, 9]])], dim=1)


def test_greedy_generation_records_complete_resolved_decoding_config():
    tokenizer = SimpleNamespace(
        eos_token_id=9, pad_token_id=9, decode=lambda ids, **kwargs: "answer"
    )
    result = greedy_answer(
        FakeGenerator().eval(), tokenizer, torch.tensor([1, 2]), decoding_spec(9, 9)
    )
    assert result["generated_token_ids"] == [4, 9]
    assert result["decoding_sha256"] == digest(result["decoding_spec"])
    assert result["decoding_spec"]["generation_config"]["do_sample"] is False
    assert len(implementation_hash()) == 64
    with pytest.raises(ValueError):
        greedy_answer(FakeGenerator(), tokenizer, torch.tensor([1, 2]), decoding_spec(9, 9))
