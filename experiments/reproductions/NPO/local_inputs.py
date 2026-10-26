"""Read manually supplied D1 inputs without downloads or model copies."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def verify(path, expected=None):
    path = Path(path)
    with path.open("rb") as handle:
        sha = hashlib.file_digest(handle, "sha256").hexdigest()
    if expected is not None and sha != expected:
        raise ValueError(f"Pinned input hash mismatch: {path}")
    return {"sha256": sha, "bytes": path.stat().st_size}


def prepare(models, data):
    models, data = Path(models).resolve(), Path(data).resolve()
    pins = json.loads((HERE / "inputs_v1.json").read_text())
    runtime = json.loads((HERE / "runtime_sources_v1.json").read_text())
    records = []
    for item in pins["files"]:
        folder = data if item["repo"] == pins["dataset"]["id"] else models / "tokenizer"
        records.append(
            {"path": str(folder / item["path"]), **verify(folder / item["path"], item["sha256"])}
        )
    for key, folder in [("target", "target"), ("retain_only_oracle", "oracle")]:
        identity = pins[key]
        for filename in ("config.json", "model.safetensors"):
            expected = (
                identity["checkpoint_sha256"]
                if filename == "model.safetensors"
                else identity.get("config_sha256")
            )
            records.append(
                {
                    "path": str(models / folder / filename),
                    **verify(models / folder / filename, expected),
                }
            )
    generation = models / "generation_config.json"
    verify(generation, runtime["files"]["generation_config.json"]["sha256"])
    return {"models": models, "data": data, "generation": generation, "records": records}


def checkpoint_receipt(identity, directory):
    return {
        name: verify(
            Path(directory) / name,
            identity["checkpoint_sha256"]
            if name == "model.safetensors"
            else identity.get("config_sha256"),
        )
        for name in ("config.json", "model.safetensors")
    }
