from dataclasses import replace

import pytest

from fq_ssiuu.evaluation.qra_pipeline import (
    MetricObservation,
    compare_observations,
    compute_q_score,
    run_qra_evaluation,
    validate_decisions,
)


def observation(precision="fp16", value=0.4, **changes):
    item = MetricObservation(
        precision,
        "tofu_fq",
        "tofu-ks-v1",
        "ks_p_value",
        "diagnostic",
        value,
        "synthetic-contract-test",
        "a" * 64,
        "b" * 64,
        "c" * 64,
        "d" * 64,
    )
    return replace(item, **changes)


@pytest.mark.parametrize(
    "baseline,target", [(0.393718, 0.478330), (0.4, 0.1), (1e-20, 1e-10), (0.0, 1.0)]
)
def test_pvalues_never_generate_recovery_scores_or_verdicts(baseline, target):
    result = compare_observations(observation(value=baseline), observation("nf4", target))
    assert result["signed_metric_change"] is None
    assert result["oriented_memorisation_change"] is None
    assert result["attack_detected"] is None and result["durability_pass"] is None
    with pytest.raises(RuntimeError, match="RG-005"):
        compute_q_score(target, baseline)


@pytest.mark.parametrize("direction,expected", [("higher", 0.2), ("lower", -0.2)])
def test_memorisation_direction_is_explicit_and_descriptive(direction, expected):
    baseline = observation(
        value=0.3,
        kind="memorisation_score",
        direction=direction,
        metric="synthetic-score",
        definition_id="synthetic-v1",
    )
    result = compare_observations(baseline, replace(baseline, precision="nf4", value=0.5))
    assert result["signed_metric_change"] == pytest.approx(0.2)
    assert result["oriented_memorisation_change"] == pytest.approx(expected)
    assert result["attack_detected"] is None and result["durability_pass"] is None


def test_utility_change_has_no_memorisation_interpretation():
    baseline = observation(
        value=0.8, metric="utility", definition_id="utility-v1", kind="utility_score"
    )
    result = compare_observations(baseline, replace(baseline, precision="nf4", value=0.6))
    assert result["signed_metric_change"] == pytest.approx(-0.2)
    assert result["oriented_memorisation_change"] is None
    assert result["interpretation"].startswith("Utility change only")
    assert "memorisation" not in result["interpretation"]
    assert result["attack_detected"] is None and result["durability_pass"] is None


@pytest.mark.parametrize(
    "field", ["checkpoint_sha256", "sample_ids_sha256", "reference_sha256", "definition_id"]
)
def test_incompatible_comparison_rejected(field):
    baseline = observation()
    with pytest.raises(ValueError, match="provenance"):
        compare_observations(baseline, replace(baseline, precision="nf4", **{field: "e" * 64}))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.01, True])
def test_invalid_pvalues_rejected(value):
    with pytest.raises(ValueError):
        observation(value=value).validate()


def test_pvalue_cannot_be_relabelled_directional():
    with pytest.raises(ValueError, match="not directional"):
        observation(direction="higher").validate()


def test_missing_metric_remains_null():
    result = compare_observations(observation(), observation("nf4", None))
    assert result["comparison_status"] == "INCOMPLETE"
    assert result["target"]["value"] is None


def test_attack_and_durability_are_separate_decisions():
    for attack, durability in [
        (None, None),
        (False, None),
        (False, False),
        (True, False),
        (False, True),
    ]:
        validate_decisions(attack, durability)
    for attack, durability in [(True, True), (True, None), (None, True), (1, False)]:
        with pytest.raises(ValueError):
            validate_decisions(attack, durability)


def test_model_evaluation_fails_before_execution():
    with pytest.raises(RuntimeError, match="RG-005"):
        run_qra_evaluation()
