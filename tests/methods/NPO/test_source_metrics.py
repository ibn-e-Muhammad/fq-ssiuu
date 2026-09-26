import copy
import math

import pytest

from experiments.reproductions.NPO.source_metrics import (
    SOURCE_TASKS,
    source_forget_quality,
    source_summary,
)


def records():
    return {
        name: dict(
            avg_gt_loss=[math.log(2), math.log(2)],
            rougeL_recall=[0.5, 0.5],
            avg_paraphrased_loss=[math.log(2), math.log(2)],
            average_perturb_loss=[[math.log(4), math.log(4)]] * 2,
            kl_divergence=[0.0, 0.2],
        )
        for name in SOURCE_TASKS
    }


def test_source_summary_has_nine_harmonic_components_not_mean_of_rows():
    value = source_summary(records())
    # True prob=.5, alternatives=.25+.25, every utility component=.5.
    assert len(value["utility_components"]) == 9
    assert all(abs(x - 0.5) < 1e-14 for x in value["utility_components"].values())
    assert value["metrics"]["Model Utility"] == pytest.approx(0.5)
    assert value["metrics"]["Forget Truth Ratio"] == pytest.approx(0.5)
    assert value["metrics"]["Retain KL Divergence"] == pytest.approx(0.1)
    assert value["attack_detected"] is value["durability_pass"] is None
    modified = records()
    modified["eval_real_world_wo_options.json"]["rougeL_recall"] = [0.0, 0.0]
    assert source_summary(modified)["metrics"]["Model Utility"] == 0.0


def test_source_ks_is_diagnostic_only_and_records_actual_ratios():
    a = records()["eval_log_forget.json"]
    b = copy.deepcopy(a)
    b["avg_paraphrased_loss"] = [math.log(4)] * 2
    diagnostic = source_forget_quality(a, b)
    assert diagnostic["model_truth_ratio"] == pytest.approx([2.0, 2.0])
    assert diagnostic["oracle_truth_ratio"] == pytest.approx([1.0, 1.0])
    assert diagnostic["ks_statistic"] == 1.0
    assert diagnostic["ks_p_value"] == pytest.approx(1 / 3)
    assert diagnostic["attack_detected"] is diagnostic["durability_pass"] is None


@pytest.mark.parametrize("defect", ["missing", "misaligned", "nan", "negative", "range"])
def test_source_metrics_fail_closed(defect):
    values = records()
    row = values["eval_log.json"]
    if defect == "missing":
        del values["eval_log_forget.json"]
    elif defect == "misaligned":
        row["avg_gt_loss"] = [1.0]
    elif defect == "nan":
        row["kl_divergence"][0] = float("nan")
    elif defect == "negative":
        row["avg_gt_loss"][0] = -1.0
    else:
        row["rougeL_recall"][0] = 1.1
    with pytest.raises(ValueError):
        source_summary(values)
