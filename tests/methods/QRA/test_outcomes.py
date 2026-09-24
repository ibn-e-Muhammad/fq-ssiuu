import copy
from dataclasses import replace

import pytest

from fq_ssiuu.evaluation.content_identity import evaluation_hash
from fq_ssiuu.evaluation.identity import digest
from fq_ssiuu.evaluation.qra_pipeline import (
    compare_observations,
    compute_q_score,
    run_qra_evaluation,
    validate_decisions,
)
from tests.methods.QRA.fixtures import BOOTSTRAP, group, observation


@pytest.mark.parametrize(
    "baseline,target", [(0.393718, 0.478330), (0.4, 0.1), (1e-20, 1e-10), (0.0, 1.0)]
)
def test_pvalues_never_generate_recovery_scores_or_verdicts(baseline, target):
    g = group(("ks_p_value",))
    a = observation("ks_p_value", manifest=g, value=1 / 3)
    b = observation("ks_p_value", "nf4", manifest=g, value=1 / 3)
    result = compare_observations(a, b, g)
    assert result["signed_metric_change"] is None and result["paired_comparison"] is None
    assert result["attack_detected"] is None and result["durability_pass"] is None
    with pytest.raises(ValueError, match="Diagnostic"):
        compare_observations(a, b, g, bootstrap=BOOTSTRAP)
    with pytest.raises(RuntimeError, match="RG-005"):
        compute_q_score(target, baseline)


def test_memorisation_change_is_paired_and_directional():
    g = group(("answer_log_probability",))
    a = observation(manifest=g)
    b = observation(precision="nf4", values=(-2.8, -1.8), manifest=g)
    r = compare_observations(a, b, g, bootstrap=BOOTSTRAP)
    assert r["oriented_memorisation_change"] == pytest.approx(0.2)
    assert r["paired_comparison"]["paired_differences"] == pytest.approx([0.2, 0.2])
    assert r["attack_detected"] is None and r["durability_pass"] is None


def test_utility_direction_and_wording_are_separate():
    g = group(("retain_probability",))
    a = observation("retain_probability", values=(0.8, 0.6), manifest=g)
    b = observation("retain_probability", "nf4", values=(0.6, 0.4), manifest=g)
    r = compare_observations(a, b, g, bootstrap=BOOTSTRAP)
    assert a.direction == "higher"
    assert r["signed_metric_change"] == pytest.approx(-0.2)
    assert r["oriented_memorisation_change"] is None
    assert r["interpretation"].startswith("Utility change only")


@pytest.mark.parametrize(
    "field,value",
    [
        ("comparison_group_id", "other"),
        ("run_id", "another-run"),
        ("report_id", "another-report"),
        ("checkpoint_sha256", "e" * 64),
        ("artifact_sha256", "e" * 64),
        ("quantizer_name", "label-only"),
        ("quantizer_version", "another-version"),
        ("quantizer_config_sha256", "e" * 64),
        ("model_revision", "e" * 40),
        ("tokenizer_revision", "e" * 40),
        ("dataset_revision", "e" * 40),
        ("sample_ids_sha256", "e" * 64),
        ("preprocessing_sha256", "e" * 64),
        ("evaluator_sha256", "e" * 64),
        ("definition_sha256", "e" * 64),
        ("definition_id", "arbitrary"),
        ("precision", "int4"),
        ("raw_records_sha256", "e" * 64),
    ],
)
def test_mismatched_provenance_fails_closed(field, value):
    g = group(("answer_log_probability",))
    a = observation(manifest=g)
    b = observation(precision="nf4", manifest=g)
    with pytest.raises(ValueError):
        compare_observations(a, replace(b, **{field: value}), g, bootstrap=BOOTSTRAP)


def test_precision_artifacts_differ_only_as_bound_in_manifest():
    g = group(("answer_log_probability",))
    a = observation(manifest=g)
    b = observation(precision="nf4", manifest=g)
    assert a.artifact_sha256 != b.artifact_sha256
    assert (
        compare_observations(a, b, g, bootstrap=BOOTSTRAP)["comparison_status"]
        == "DESCRIPTIVE_ONLY"
    )
    malformed = copy.deepcopy(g)
    malformed["artifacts"]["nf4"]["quantizer_config"]["precision"] = "fp16"
    with pytest.raises(ValueError):
        compare_observations(a, b, malformed, bootstrap=BOOTSTRAP)


def test_raw_record_order_cannot_be_silently_changed():
    a = observation()
    records = list(reversed(a.raw_records))
    with pytest.raises(ValueError, match="content hash"):
        replace(a, raw_records=records, raw_records_sha256=digest(records)).validate()


def test_same_ids_cannot_hide_different_source_rows_or_tensors():
    g = group(("answer_log_probability",))
    a = observation(manifest=g)
    b = observation(precision="nf4", manifest=g)
    for field in ("source_record_sha256", "tensors_sha256"):
        records = copy.deepcopy(b.raw_records)
        records[0][field] = "f" * 64
        altered = replace(b, raw_records=records, raw_records_sha256=digest(records))
        altered = replace(altered, evaluation_sha256=evaluation_hash(altered))
        with pytest.raises(ValueError, match="Paired source"):
            compare_observations(a, altered, g, bootstrap=BOOTSTRAP)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, 1.1, -0.1])
def test_invalid_pvalues_rejected(value):
    with pytest.raises(ValueError):
        observation("ks_p_value", value=value).validate()


def test_missing_measurement_stays_null():
    g = group(("answer_log_probability",))
    a = observation(manifest=g)
    b = observation(precision="nf4", values=(None, -2), manifest=g)
    assert compare_observations(a, b, g, bootstrap=BOOTSTRAP)["comparison_status"] == "INCOMPLETE"


def test_decision_invariants_and_disabled_full_evaluator():
    validate_decisions(None, None)
    validate_decisions(False, None)
    validate_decisions(True, False)
    for attack, durable in [(True, True), (True, None), (None, True), (0, False)]:
        with pytest.raises(ValueError):
            validate_decisions(attack, durable)
    with pytest.raises(RuntimeError, match="RG-005"):
        run_qra_evaluation()
