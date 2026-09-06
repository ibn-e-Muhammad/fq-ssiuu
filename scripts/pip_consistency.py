"""Fail-closed differential pip check; raw observations are never an allowlist.

Diagnostic grammar follows pip's official CheckCommand and pip-check docs.
Unknown output, stderr, inconsistent return codes or malformed provenance fail.
No project-specific diagnostic text is exempted.
"""

import hashlib
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from packaging.requirements import Requirement

from scripts.verify_pip_report import atomic_write_json, normalize_name

OVERLAY_DISTRIBUTIONS = {"bitsandbytes", "captum", "hydra-core", "lm-eval", "jsonlines", "fq-ssiuu"}
NAME = r"[A-Za-z0-9][A-Za-z0-9._-]*"
MISSING = re.compile(rf"^({NAME}) (\S+) requires ({NAME}), which is not installed\.$")
CONFLICT = re.compile(rf"^({NAME}) (\S+) has requirement (.+), but you have ({NAME}) (\S+)\.$")
UNSUPPORTED = re.compile(rf"^({NAME}) (\S+) is not supported on this platform$")
CLEAN = "No broken requirements found."


def normalize_diagnostics(stdout, stderr, returncode):
    """Retain versions/requirements; normalize distribution names and ordering only."""
    if not isinstance(stdout, str) or not isinstance(stderr, str) or type(returncode) is not int:
        raise ValueError("Malformed pip check streams or return code")
    if stderr.strip():
        raise ValueError("pip check stderr is non-empty; cannot certify parsed diagnostics")
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if returncode == 0:
        if lines != [CLEAN]:
            raise ValueError("Successful pip check lacks the exact success message")
        return []
    if returncode != 1 or not lines:
        raise ValueError("pip check did not produce recognized dependency diagnostics")
    diagnostics = []
    for line in lines:
        if match := MISSING.fullmatch(line):
            owner, version, dep = match.groups()
            item = dict(
                kind="missing",
                owner=normalize_name(owner),
                owner_version=version,
                dependency=normalize_name(dep),
            )
        elif match := CONFLICT.fullmatch(line):
            owner, version, req_text, dep, installed = match.groups()
            req = Requirement(req_text)
            if normalize_name(req.name) != normalize_name(dep):
                raise ValueError("Diagnostic requirement/dependency mismatch")
            req.name = normalize_name(req.name)
            item = dict(
                kind="conflict",
                owner=normalize_name(owner),
                owner_version=version,
                dependency=normalize_name(dep),
                requirement=str(req),
                installed_version=installed,
            )
        elif match := UNSUPPORTED.fullmatch(line):
            owner, version = match.groups()
            item = dict(
                kind="unsupported",
                owner=normalize_name(owner),
                owner_version=version,
                dependency=None,
            )
        else:
            raise ValueError("Unrecognized pip check output; raw streams retained")
        diagnostics.append(item)
    keys = [json.dumps(d, sort_keys=True) for d in diagnostics]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate pip diagnostics; ambiguous distribution inventory")
    return sorted(diagnostics, key=lambda d: json.dumps(d, sort_keys=True))


def make_record(stdout, stderr, returncode, stage, run_id, git_sha):
    record = dict(
        schema_version="1.0",
        stage=stage,
        run_id=run_id,
        git_sha=git_sha,
        utc_timestamp=datetime.now(UTC).isoformat(),
        command=[sys.executable, "-m", "pip", "check"],
        stdout=stdout,
        stderr=stderr,
        returncode=returncode,
    )
    for stream in ("stdout", "stderr"):
        record[stream + "_sha256"] = hashlib.sha256(record[stream].encode("utf-8")).hexdigest()
    try:
        record["diagnostics"] = normalize_diagnostics(stdout, stderr, returncode)
        record["record_status"] = "VALID"
    except ValueError as exc:
        record.update(record_status="INVALID", diagnostics=None, error=str(exc))
    return record


def capture_pip_check(out_dir, stage, run_id, git_sha):
    """Save separate byte-exact stdout/stderr and an independently checkable record."""
    result = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=False)
    streams = {}
    for stream in ("stdout", "stderr"):
        data = getattr(result, stream)
        if isinstance(data, str):  # Supports subprocess-boundary test doubles.
            data = data.encode("utf-8")
        if out_dir is not None:
            Path(out_dir, f"pip_check_{stage}_{stream}.log").write_bytes(data)
        streams[stream] = data.decode("utf-8", errors="strict")
    record = make_record(
        **streams, returncode=result.returncode, stage=stage, run_id=run_id, git_sha=git_sha
    )
    if out_dir is not None:
        atomic_write_json(str(Path(out_dir, f"pip_check_{stage}.json")), record)
    return record


def validate_record(record, stage=None, run_id=None, git_sha=None):
    if not isinstance(record, dict) or record.get("schema_version") != "1.0":
        raise ValueError("Malformed pip check record schema")
    if record.get("record_status") != "VALID":
        raise ValueError("Invalid pip check record")
    if record.get("stage") not in {"before", "after", "verifier"}:
        raise ValueError("Invalid pip check stage")
    if not isinstance(record.get("run_id"), str) or not record["run_id"]:
        raise ValueError("Missing pip check run ID")
    if not isinstance(record.get("git_sha"), str) or not re.fullmatch(
        r"[a-f0-9]{40}", record["git_sha"]
    ):
        raise ValueError("Invalid pip check source identity")
    for key, expected in (("stage", stage), ("run_id", run_id), ("git_sha", git_sha)):
        if expected is not None and record.get(key) != expected:
            raise ValueError(f"Pip check {key} mismatch")
    command = record.get("command")
    if (
        not isinstance(command, list)
        or len(command) != 4
        or not isinstance(command[0], str)
        or not Path(command[0]).is_absolute()
        or command[1:] != ["-m", "pip", "check"]
    ):
        raise ValueError("Pip check interpreter/command malformed")
    datetime.fromisoformat(record["utc_timestamp"])
    for stream in ("stdout", "stderr"):
        if not isinstance(record.get(stream), str):
            raise ValueError("Malformed raw stream")
        if hashlib.sha256(record[stream].encode("utf-8")).hexdigest() != record.get(
            stream + "_sha256"
        ):
            raise ValueError("Pip check raw stream hash mismatch")
    expected = normalize_diagnostics(record["stdout"], record["stderr"], record["returncode"])
    if record.get("diagnostics") != expected:
        raise ValueError("Normalized pip diagnostics disagree with raw observations")
    return expected


def load_record(path, **expected):
    path = Path(path)
    record = json.loads(path.read_text())
    validate_record(record, **expected)
    for stream in ("stdout", "stderr"):
        data = path.with_name(f"pip_check_{record['stage']}_{stream}.log").read_bytes()
        if hashlib.sha256(data).hexdigest() != record[stream + "_sha256"]:
            raise ValueError("Pip check record differs from raw file")
    return record


def compare_checks(before, after, overlay_names=()):
    """A dirty baseline is acceptable only when no dependency regression remains."""
    result = dict(
        schema_version="1.0",
        status="FAIL",
        state="INVALID_EVIDENCE",
        unchanged=[],
        introduced=[],
        resolved=[],
        overlay_diagnostics=[],
    )
    try:
        b = validate_record(before, stage="before")
        a = validate_record(after, run_id=before["run_id"], git_sha=before["git_sha"])
        if after["stage"] not in {"after", "verifier"}:
            raise ValueError("Post-install pip check has wrong stage")
        if before["command"] != after["command"]:
            raise ValueError("Pip check interpreter changed between observations")
        if datetime.fromisoformat(after["utc_timestamp"]) < datetime.fromisoformat(
            before["utc_timestamp"]
        ):
            raise ValueError("Post-install pip check predates baseline")

        def key(d):
            return json.dumps(d, sort_keys=True)

        bm, am = {key(d): d for d in b}, {key(d): d for d in a}
        names = OVERLAY_DISTRIBUTIONS | {normalize_name(n) for n in overlay_names}
        result.update(
            run_id=before["run_id"],
            git_sha=before["git_sha"],
            before_returncode=before["returncode"],
            after_returncode=after["returncode"],
            unchanged=[am[k] for k in sorted(am.keys() & bm.keys())],
            introduced=[am[k] for k in sorted(am.keys() - bm.keys())],
            resolved=[bm[k] for k in sorted(bm.keys() - am.keys())],
            overlay_diagnostics=[d for d in a if d["owner"] in names or d["dependency"] in names],
        )
        if result["introduced"] or result["overlay_diagnostics"]:
            result["state"] = "DEPENDENCY_REGRESSION"
        else:
            result.update(status="PASS", state="BASELINE_DIRTY_NO_REGRESSION" if a else "CLEAN")
    except (KeyError, TypeError, ValueError) as exc:
        result["error"] = str(exc)
    return result
