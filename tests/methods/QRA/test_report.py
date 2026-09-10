import json
from dataclasses import replace

import pytest

from fq_ssiuu.evaluation.certificate import (
    generate_certificate,
    generate_report,
    load_report,
    save_report,
)
from tests.methods.QRA.test_outcomes import observation


def inputs():
    return [
        observation(
            precision,
            value,
            metric=kind,
            definition_id=kind + "-v1",
            kind=kind,
            direction="higher" if kind == "memorisation_score" else "diagnostic",
        )
        for precision in ("fp16", "int8", "nf4")
        for kind, value in (("memorisation_score", 0.2), ("utility_score", 0.8))
    ]


def test_complete_descriptive_record_has_no_durability_verdict(tmp_path):
    report = generate_report("run-test", "synthetic", inputs())
    assert report["report_status"] == "VALID_DESCRIPTIVE_RECORD"
    assert report["attack_detected"] is None and report["durability_pass"] is None
    assert not {"certified", "threshold_met", "q_int4_score"} & report.keys()
    path = tmp_path / "report.json"
    save_report(report, path)
    assert load_report(path) == report
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_report(report, path)
    assert path.read_bytes() == before


def test_missing_utility_never_becomes_zero_or_pass():
    values = inputs()
    values[-1] = replace(values[-1], value=None)
    report = generate_report("run-test", "synthetic", values)
    assert report["report_status"] == "INCOMPLETE"
    assert report["observations"][-1]["value"] is None
    assert report["durability_pass"] is None


def test_incompatible_report_rejected():
    values = inputs()
    values[-1] = replace(values[-1], sample_ids_sha256="e" * 64)
    with pytest.raises(ValueError, match="provenance"):
        generate_report("run-test", "synthetic", values)


def test_unmatched_metrics_cannot_make_a_complete_report():
    values = [replace(o, metric=o.metric + "-" + o.precision) for o in inputs()]
    report = generate_report("run-test", "synthetic", values)
    assert report["report_status"] == "INCOMPLETE"
    assert report["comparisons"] == []


def test_cross_metric_checkpoint_mix_rejected():
    values = [
        replace(o, checkpoint_sha256="e" * 64) if o.kind == "utility_score" else o for o in inputs()
    ]
    with pytest.raises(ValueError, match="one source checkpoint"):
        generate_report("run-test", "synthetic", values)


def test_historical_certification_is_preserved_without_adopting_it(tmp_path):
    path = tmp_path / "historical.json"
    old = {"schema_version": "v1", "q_int4_score": 0.2149, "certified": True}
    path.write_text(json.dumps(old))
    before = path.read_bytes()
    record = load_report(path)
    assert record["historical_payload"] == old
    assert record["attack_detected"] is None and record["durability_pass"] is None
    assert record["report_status"] == "SUPERSEDED_INTERPRETATION"
    assert path.read_bytes() == before
    with pytest.raises(RuntimeError, match="RG-005"):
        generate_certificate(old)


def test_unapproved_decision_cannot_be_saved(tmp_path):
    report = generate_report("run-test", "synthetic", inputs())
    report.update(attack_detected=True, durability_pass=False)
    with pytest.raises(ValueError, match="unavailable"):
        save_report(report, tmp_path / "must-not-exist.json")
    assert not (tmp_path / "must-not-exist.json").exists()
