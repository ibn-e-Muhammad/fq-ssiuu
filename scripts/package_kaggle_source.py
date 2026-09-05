#!/usr/bin/env python3
"""Package a committed tree as a verified, history-free Git bundle.

Only committed blobs are read. The source checkout and its refs are never changed.
Unsafe tracked files cause rejection, never silent deletion from the source tree.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone

BUNDLE_NAME = "fq-ssiuu-source.bundle"
MANIFEST_NAME = "SOURCE_MANIFEST.json"
SECRET_PATTERNS = {
    "huggingface-token": rb"\bhf_[A-Za-z0-9]{30,}\b",
    "github-token": rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b",
    "private-key": rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    "aws-access-key": rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    "credential-url": rb"https?://[^\s/@:]+:[^\s/@]+@",
}


def git(repo, *args, input=None):
    result = subprocess.run(["git", "-C", str(repo), *args], input=input,
                            capture_output=True, check=False)
    if result.returncode:
        # Do not echo Git output: tracked paths or commit messages can contain secrets.
        raise ValueError(f"Git operation {args[0]} failed (exit {result.returncode})")
    return result.stdout


def digest(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def tree_entries(repo, revision):
    entries = []
    for entry in git(repo, "ls-tree", "-rz", revision).split(b"\0"):
        if entry:
            info, name = entry.split(b"\t", 1)
            mode, kind, oid = info.decode().split()
            entries.append((mode, kind, oid, name.decode("utf-8")))
    return entries


def scan_tree(repo, revision):
    """Return only paths/categories for violations; never matched values."""
    findings = []
    for mode, kind, oid, name in tree_entries(repo, revision):
        path = PurePosixPath(name)
        lower = path.name.lower()
        banned = (lower == ".env" or (lower.startswith(".env.") and lower != ".env.example")
                  or lower.endswith((".docx", ".pem", ".key", ".p12", ".pfx"))
                  or lower in {"kaggle.json", "credentials.json", "access_token", "id_rsa", "id_ed25519"}
                  or any(part in {".git", ".venv", ".ssh", ".kaggle"} for part in path.parts))
        if banned:
            findings.append({"path": name, "category": "forbidden-file", "status": "FAIL"})
            continue
        if mode not in {"100644", "100755"} or kind != "blob":
            findings.append({"path": name, "category": "non-regular-file", "status": "FAIL"})
            continue
        content = git(repo, "cat-file", "blob", oid)
        for category, pattern in SECRET_PATTERNS.items():
            if re.search(pattern, content):
                findings.append({"path": name, "category": category, "status": "FAIL"})
    return findings


def verify_bundle(directory, expected_origin=None, expected_manifest_sha256=None):
    """Validate both content hashes and the actual Git topology in a fresh clone."""
    directory = Path(directory).resolve()
    manifest_path = directory / MANIFEST_NAME
    if manifest_path.is_symlink():
        raise ValueError("Manifest must be a regular file")
    if expected_manifest_sha256 and digest(manifest_path) != expected_manifest_sha256:
        raise ValueError("Manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text())
    for key in ("source_origin_sha", "source_tree_sha", "snapshot_commit_sha"):
        if not re.fullmatch(r"[0-9a-f]{40}", manifest.get(key, "")):
            raise ValueError(f"Invalid {key}")
    if expected_origin and manifest["source_origin_sha"] != expected_origin:
        raise ValueError("Source origin mismatch")
    if (manifest.get("schema_version") != "1.0"
            or manifest.get("bundle_filename") != BUNDLE_NAME
            or manifest.get("history_commit_count") != 1
            or manifest.get("contains_original_history") is not False
            or manifest.get("secret_scan_status") != "PASS"):
        raise ValueError("Invalid source manifest policy")
    ref = manifest.get("snapshot_ref", "")
    git(directory, "check-ref-format", f"refs/heads/{ref}")
    bundle = directory / BUNDLE_NAME
    if bundle.is_symlink() or not bundle.is_file():
        raise ValueError("Bundle must be a regular file")
    if bundle.stat().st_size != manifest.get("bundle_size_bytes") or digest(bundle) != manifest.get("bundle_sha256"):
        raise ValueError("Bundle hash or size mismatch")
    with tempfile.TemporaryDirectory(prefix="fq-source-verify-") as temp:
        root = Path(temp)
        git(root, "init", "--bare", "empty.git")
        git(root / "empty.git", "bundle", "verify", str(bundle))
        heads = git(root, "bundle", "list-heads", str(bundle)).decode().splitlines()
        if heads != [f"{manifest['snapshot_commit_sha']} refs/heads/{ref}"]:
            raise ValueError("Bundle must advertise only its snapshot ref")
        git(root, "clone", "--no-checkout", "-b", ref, str(bundle), "clone")
        clone = root / "clone"
        if git(clone, "rev-parse", "HEAD").decode().strip() != manifest["snapshot_commit_sha"]:
            raise ValueError("Snapshot HEAD mismatch")
        if git(clone, "rev-parse", "HEAD^{tree}").decode().strip() != manifest["source_tree_sha"]:
            raise ValueError("Source tree mismatch")
        if git(clone, "rev-list", "--count", "--all").strip() != b"1":
            raise ValueError("Original history is present")
        if len(git(clone, "rev-list", "--parents", "-n", "1", "HEAD").split()) != 1:
            raise ValueError("Snapshot has a parent")
        types = git(clone, "cat-file", "--batch-all-objects", "--batch-check=%(objecttype)").splitlines()
        if types.count(b"commit") != 1 or b"tag" in types:
            raise ValueError("Unexpected history objects in bundle")
        findings = scan_tree(clone, "HEAD")
        if findings:
            raise ValueError("Source scan failed: " + json.dumps(findings))
        git(clone, "checkout", ref)
        if git(clone, "status", "--porcelain"):
            raise ValueError("Snapshot checkout is dirty")
    return manifest


def package_source(source_repo, source_origin_sha, output_dir, snapshot_ref="kaggle-snapshot"):
    source = Path(source_repo).resolve()
    output = Path(output_dir).resolve()
    git(source, "check-ref-format", f"refs/heads/{snapshot_ref}")
    if not re.fullmatch(r"[0-9a-fA-F]{7,40}", source_origin_sha):
        raise ValueError("Source must be a commit SHA, not a mutable branch")
    origin = git(source, "rev-parse", "--verify", f"{source_origin_sha}^{{commit}}").decode().strip()
    tree = git(source, "rev-parse", f"{origin}^{{tree}}").decode().strip()
    findings = scan_tree(source, tree)
    if findings:
        raise ValueError("Source scan failed: " + json.dumps(findings))
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be absent or empty")
    created = datetime.now(timezone.utc).isoformat()
    with tempfile.TemporaryDirectory(prefix="fq-source-package-") as temp:
        root = Path(temp)
        git(root, "clone", "--shared", "--no-checkout", str(source), "objects")
        objects = root / "objects"
        # Fixed author/date make the snapshot and bundle reproducible for a given origin.
        snapshot = git(objects, "-c", "user.name=FQ-SSIUU Source Snapshot",
                       "-c", "user.email=source-snapshot@invalid",
                       "hash-object", "-t", "commit", "-w", "--stdin", input=(
                           f"tree {tree}\n"
                           "author FQ-SSIUU Source Snapshot <source-snapshot@invalid> 0 +0000\n"
                           "committer FQ-SSIUU Source Snapshot <source-snapshot@invalid> 0 +0000\n\n"
                           f"History-free source snapshot of {origin}\n"
                       ).encode()).decode().strip()
        git(objects, "update-ref", f"refs/heads/{snapshot_ref}", snapshot)
        stage = root / "stage"
        stage.mkdir()
        bundle = stage / BUNDLE_NAME
        git(objects, "bundle", "create", str(bundle), f"refs/heads/{snapshot_ref}")
        manifest = {
            "schema_version": "1.0", "source_repository_name": source.name,
            "source_origin_sha": origin, "source_tree_sha": tree,
            "snapshot_commit_sha": snapshot, "snapshot_ref": snapshot_ref,
            "bundle_filename": BUNDLE_NAME, "bundle_size_bytes": bundle.stat().st_size,
            "bundle_sha256": digest(bundle), "creation_utc": created,
            "secret_scan_status": "PASS", "history_commit_count": 1,
            "contains_original_history": False,
        }
        (stage / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
        verify_bundle(stage, expected_origin=origin)
        output.parent.mkdir(parents=True, exist_ok=True)
        # Publish the manifest last as the commit marker; refuse to overwrite any output.
        output.mkdir(exist_ok=True)
        for name in (BUNDLE_NAME, MANIFEST_NAME):
            with tempfile.NamedTemporaryFile(dir=output, delete=False) as handle:
                temporary = Path(handle.name)
                with open(stage / name, "rb") as content:
                    shutil.copyfileobj(content, handle)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, output / name)
            finally:
                temporary.unlink(missing_ok=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--source-origin-sha", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--snapshot-ref", default="kaggle-snapshot")
    args = parser.parse_args()
    try:
        manifest = package_source(**vars(args))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Source packaging failed: {exc}\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
