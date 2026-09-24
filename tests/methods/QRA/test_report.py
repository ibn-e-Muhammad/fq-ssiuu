import json
from dataclasses import replace

import pytest

from fq_ssiuu.evaluation.certificate import (
    generate_certificate,
    generate_report,
    load_report,
    save_report,
)
from tests.methods.QRA.fixtures import BOOTSTRAP, group, observations


def report():
    g = group()
    return generate_report(
        "run-unit", "bounded-validation", observations(g), g, bootstrap=BOOTSTRAP
    )


def test_complete_descriptive_report_has_no_durability_verdict(tmp_path):
    r = report()
    assert r["report_status"] == "VALID_DESCRIPTIVE_RECORD"
    assert r["attack_detected"] is None and r["durability_pass"] is None
    assert not {"certified", "threshold_met", "q_int4_score"} & r.keys()
    p = tmp_path / "report.json"
    save_report(r, p)
    assert load_report(p) == r
    before = p.read_bytes()
    with pytest.raises(FileExistsError):
        save_report(r, p)
    assert p.read_bytes() == before


def test_missing_precision_or_utility_never_counts_as_complete():
    g = group()
    xs = observations(g)
    r = generate_report("run-unit", "bounded", xs[:-1], g, bootstrap=BOOTSTRAP)
    assert r["report_status"] == "INCOMPLETE"
    assert r["durability_pass"] is None


def test_different_report_run_or_cross_metric_checkpoint_rejected():
    g = group()
    xs = observations(g)
    with pytest.raises(ValueError, match="linkage"):
        generate_report("other-run", "bounded", xs, g, bootstrap=BOOTSTRAP)
    xs[-1] = replace(xs[-1], checkpoint_sha256="f" * 64)
    with pytest.raises(ValueError):
        generate_report("run-unit", "bounded", xs, g, bootstrap=BOOTSTRAP)


@pytest.mark.parametrize("version", ["v1", "v2", "v3", "v4"])
def test_historical_payload_is_preserved_with_explicit_supersession(tmp_path, version):
    p = tmp_path / "old.json"
    old = {"schema_version": version, "certified": True}
    p.write_text(json.dumps(old))
    before = p.read_bytes()
    r = load_report(p)
    assert (
        r["historical_payload"] == old
        and r["attack_detected"] is None
        and r["durability_pass"] is None
    )
    assert r["migration"]["automatic_migration"] is False and r["migration"]["to_schema"] is None
    assert p.read_bytes() == before
    with pytest.raises(RuntimeError, match="RG-005"):
        generate_certificate(old)


def test_unapproved_decision_or_forged_measurement_cannot_be_saved(tmp_path):
    r = report()
    r.update(attack_detected=True, durability_pass=False)
    with pytest.raises(ValueError, match="unavailable"):
        save_report(r, tmp_path / "bad.json")
    r = report()
    r["observations"][0]["value"] = 50
    with pytest.raises(ValueError):
        save_report(r, tmp_path / "bad.json")
    assert not (tmp_path / "bad.json").exists()
