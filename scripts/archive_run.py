"""Seal run outputs and copy them to private stores without changing raw files."""

import argparse
import hashlib
import json
import re
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.package_kaggle_source import SECRET_PATTERNS  # noqa: E402


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def seal_run(source, destination, run_id):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{5,120}", run_id):
        raise ValueError("Invalid run ID")
    if destination.is_relative_to(source):
        raise ValueError("Archive must be outside raw output directory")
    files = sorted(p for p in source.rglob("*") if p.is_file() or p.is_symlink())
    if not files:
        raise ValueError("Run has no artifacts")
    entries = []
    for path in files:
        relative = path.relative_to(source)
        if path.is_symlink() or any(
            part in {".git", ".env", ".kaggle", ".ssh", "credentials.json", "kaggle.json"}
            or part.startswith(".env.")
            or part.endswith((".pem", ".key", ".docx"))
            for part in relative.parts
        ):
            raise ValueError(f"Forbidden artifact path: {relative}")
        # Bounded-memory scanning also handles large checkpoint files.
        with path.open("rb") as handle:
            overlap = b""
            while chunk := handle.read(1024 * 1024):
                data = overlap + chunk
                if any(re.search(pattern, data) for pattern in SECRET_PATTERNS.values()):
                    raise ValueError(f"Credential pattern in artifact: {relative}")
                overlap = data[-4096:]
        entries.append(
            {"path": str(relative), "size_bytes": path.stat().st_size, "sha256": digest(path)}
        )
    destination.mkdir(parents=True, exist_ok=False)
    archive = destination / "artifacts.tar"
    with tarfile.open(archive, "w") as tar:
        for item in entries:
            path = source / item["path"]
            info = tar.gettarinfo(str(path), arcname=item["path"])
            info.uid = info.gid = info.mtime = 0
            info.uname = info.gname = ""
            info.mode = 0o644
            with path.open("rb") as handle:
                tar.addfile(info, handle)
            if digest(path) != item["sha256"]:
                raise ValueError("Artifact changed during sealing")
    manifest = {
        "schema_version": "1.0",
        "run_id": run_id,
        "artifacts": entries,
        "archive": {
            "path": archive.name,
            "sha256": digest(archive),
            "size_bytes": archive.stat().st_size,
        },
        "secret_scan": "PASS",
    }
    (destination / "ARCHIVE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def mirror_hf(sealed, repo_id=None):
    """One private repository per run; existing data must match to retry safely."""
    from huggingface_hub import HfApi, hf_hub_download

    from fq_ssiuu.secrets import get_secret

    sealed = Path(sealed)
    manifest = json.loads((sealed / "ARCHIVE_MANIFEST.json").read_text())
    api = HfApi(token=get_secret("HF_TOKEN"))
    owner = api.whoami()["name"]
    repo_id = repo_id or f"{owner}/fq-ssiuu-run-{manifest['run_id']}"
    if not repo_id.startswith(owner + "/"):
        raise ValueError("Mirror must belong to the authenticated researcher")
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    if api.repo_info(repo_id, repo_type="dataset").private is not True:
        raise ValueError("Refusing a public artifact destination")
    for name in ("artifacts.tar", "ARCHIVE_MANIFEST.json"):
        if api.file_exists(repo_id, name, repo_type="dataset"):
            cached = hf_hub_download(repo_id, name, repo_type="dataset", token=api.token)
            if digest(cached) != digest(sealed / name):
                raise ValueError("Existing remote artifact differs; overwrite refused")
        else:
            api.upload_file(
                path_or_fileobj=str(sealed / name),
                path_in_repo=name,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Preserve run {manifest['run_id']}: {name}",
            )
    info = api.repo_info(repo_id, repo_type="dataset")
    for name in ("artifacts.tar", "ARCHIVE_MANIFEST.json"):
        cached = hf_hub_download(
            repo_id, name, repo_type="dataset", token=api.token, revision=info.sha
        )
        if digest(cached) != digest(sealed / name):
            raise ValueError("Remote archive verification failed")
    return {
        "status": "PASS",
        "store": "huggingface",
        "repo_id": repo_id,
        "revision": info.sha,
        "private": True,
        "download_hashes_verified": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    seal = sub.add_parser("seal")
    seal.add_argument("--run-dir", required=True)
    seal.add_argument("--output", required=True)
    seal.add_argument("--run-id", required=True)
    mirror = sub.add_parser("mirror-hf")
    mirror.add_argument("--sealed", required=True)
    mirror.add_argument("--receipt", required=True)
    args = parser.parse_args()
    try:
        if args.action == "seal":
            result = seal_run(args.run_dir, args.output, args.run_id)
            print(
                json.dumps(
                    {
                        "status": "SEALED",
                        "run_id": result["run_id"],
                        "archive_sha256": result["archive"]["sha256"],
                    }
                )
            )
        else:
            result = mirror_hf(args.sealed)
            with Path(args.receipt).open("x") as handle:
                json.dump(result, handle, indent=2)
                handle.write("\n")
            print(json.dumps(result))
    except Exception as exc:
        # Client exceptions can include request details; never print credential-bearing URLs.
        print(json.dumps({"status": "FAIL", "error_type": type(exc).__name__}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
