import math

import numpy as np
import pandas as pd
import pytest
from npo3.metrics import (
    align,
    enrich,
    forget_quality,
    paired_bootstrap,
    quantization_comparisons,
    summarize,
    truth_logs,
)
from scipy.stats import ks_2samp


def sample_frame(shift=0.0):
    rows = []
    for split in ("forget05", "retain", "real_authors", "world_facts"):
        for i in range(5):
            rows.append(
                {
                    "id": f"{split}:{i}",
                    "split": split,
                    "question": f"q{split}{i}",
                    "ground_truth": "A",
                    "answer_nll": 0.3 + i * 0.02,
                    "paraphrase_mean_nll": 0.5 + i * 0.1,
                    "perturbed_mean_nll": [1.0 + i * 0.2 + shift, 3.0 + i * 0.3 + shift],
                    "rouge_l_recall": 0.5,
                    "token_f1": 0.4,
                    "exact_match": 0.0,
                }
            )
    return enrich(pd.DataFrame(rows))


def test_arithmetic_utility_and_geometric_fq_are_distinct():
    values = truth_logs(0.5, [1.0, 3.0])
    assert values["source_geometric_truth_ratio_log"] == 1.5
    assert math.exp(values["arithmetic_truth_ratio_log"]) == pytest.approx(
        (np.exp(-1) + np.exp(-3)) / 2 / np.exp(-0.5)
    )
    frame = sample_frame()
    summary, components = summarize(frame, frame)
    assert summary["forget_quality"] == 1.0
    assert summary["ks_statistic"] == 0.0
    vector = (
        components[["utility_probability", "rouge_l_recall", "arithmetic_truth_score"]]
        .to_numpy()
        .flatten()
    )
    assert len(vector) == 9
    assert summary["model_utility"] == pytest.approx(len(vector) / np.reciprocal(vector).sum())


def test_matched_precision_uses_geometric_ratio_and_aligns_rows():
    before, after, oracle = sample_frame(), sample_frame(0.1), sample_frame(0.2)
    stages = {"npo_bf16": before, "npo_nf4": after, "retain95_bf16": before, "retain95_nf4": oracle}
    fq, changes = quantization_comparisons(stages, repeats=100)
    got = fq.query("stage == 'npo_nf4' and oracle_policy == 'matched_precision'").iloc[0]
    expected = ks_2samp(
        after.query("split == 'forget05'").source_geometric_truth_ratio_log,
        oracle.query("split == 'forget05'").source_geometric_truth_ratio_log,
    )
    assert got.ks_statistic == expected.statistic and got.forget_quality == expected.pvalue
    directional = changes.query("metric == 'source_geometric_truth_ratio_log'").iloc[0]
    assert directional.mean_change == pytest.approx(0.1)
    assert directional.ci_low == pytest.approx(0.1)
    assert directional.fraction_increased == 1.0
    # Equal geometric means but sharply different arithmetic means must give
    # geometric KS D=0 while the separately named arithmetic diagnostic differs.
    geometric_same = sample_frame()
    arithmetic_different = sample_frame()
    geometric_same["perturbed_mean_nll"] = [[0.1, 7.9] for _ in range(len(geometric_same))]
    arithmetic_different["perturbed_mean_nll"] = [
        [4.0, 4.0] for _ in range(len(arithmetic_different))
    ]
    distinct = forget_quality(enrich(geometric_same), enrich(arithmetic_different))
    assert distinct["ks_statistic"] == 0.0 and distinct["forget_quality"] == 1.0
    assert distinct["paper_arithmetic_ks_statistic_diagnostic"] == 1.0
    bad = after.copy()
    bad.loc[0, "question"] = "different question"
    with pytest.raises(ValueError, match="question"):
        align(bad, before)


def test_cluster_bootstrap_is_paired_and_reproducible():
    before = np.arange(6.0)
    after = before + np.array([0.1, 0.1, -0.2, -0.2, 0.3, 0.3])
    a = paired_bootstrap(before, after, repeats=200, clusters=[0, 0, 1, 1, 2, 2])
    b = paired_bootstrap(before, after, repeats=200, clusters=[0, 0, 1, 1, 2, 2])
    assert a == b and a["resampling_unit"] == "author-block"
    assert a["mean_change"] == pytest.approx(1 / 15)
    with pytest.raises(ValueError):
        paired_bootstrap([1], [float("nan")])
    clusters = np.repeat(np.arange(3), 200)
    repeated = np.repeat([0.1, -0.2, 0.3], 200)
    cluster_ci = paired_bootstrap(np.zeros(600), repeated, repeats=200, clusters=clusters)
    question_ci = paired_bootstrap(np.zeros(600), repeated, repeats=200)
    assert cluster_ci["ci_high"] - cluster_ci["ci_low"] > 5 * (
        question_ci["ci_high"] - question_ci["ci_low"]
    )
