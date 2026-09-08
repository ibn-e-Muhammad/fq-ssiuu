"""Recipe validation and real subprocess failure preservation without GPUs."""

import json
import os
import sys

import pytest

from scripts.mother_runner import checked_path, load_recipe, run_command, write_new
from scripts.stage_mother import stage


def test_real_failed_command_retains_stdout_stderr_and_exit_code(tmp_path):
    code = run_command(
        "fixture",
        [
            sys.executable,
            "-c",
            "import sys;print('raw output');print('failure detail',file=sys.stderr);sys.exit(7)",
        ],
        tmp_path,
        os.environ.copy(),
        30,
    )
    assert code == 7
    assert (tmp_path / "fixture_stdout.log").read_text() == "raw output\n"
    assert (tmp_path / "fixture_stderr.log").read_text() == "failure detail\n"
    assert json.loads((tmp_path / "fixture_execution.json").read_text())["returncode"] == 7


def test_termination_records_cannot_be_overwritten(tmp_path):
    target = tmp_path / "result.json"
    write_new(target, {"status": "FAIL"})
    with pytest.raises(FileExistsError):
        write_new(target, {"status": "PASS"})
    assert json.loads(target.read_text()) == {"status": "FAIL"}


def test_recipe_rejects_unknown_kind_and_missing_seed(tmp_path):
    p = tmp_path / "recipe.json"
    for data in [
        {"schema_version": "1.0", "kind": "integration"},
        {"schema_version": "1.0", "kind": "silently-train", "seed": 42},
    ]:
        p.write_text(json.dumps(data))
        with pytest.raises(ValueError):
            load_recipe(p)
    p.write_text(json.dumps({"schema_version": "1.0", "kind": "integration", "seed": 42}))
    assert load_recipe(p)["kind"] == "integration"


def test_recipe_path_cannot_escape_source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (tmp_path / "external.json").write_text("{}")
    with pytest.raises(ValueError):
        checked_path(root, "../external.json")


def test_staging_never_overwrites_existing_provenance(tmp_path):
    output = tmp_path / "run.1"
    receipt = tmp_path / "run.1.provenance.json"
    receipt.write_text("original receipt\n")
    with pytest.raises(FileExistsError):
        stage(tmp_path / "absent-source", "owner/dataset", "recipe.json", output)
    assert receipt.read_text() == "original receipt\n"
    assert not output.exists()
