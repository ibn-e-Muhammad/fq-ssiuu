"""Read manually supplied D1 inputs without downloads or model copies."""

import hashlib
import json
from pathlib import Path

from experiments.reproductions.NPO.local_datasets import export_datasets, inspect_datasets

HERE = Path(__file__).resolve().parent


def model_paths(models, tokenizer_files):
    """Accept the NPO inputs folder or a downloaded target checkpoint directly."""
    models = Path(models).resolve()
    if (models / "config.json").is_file() or (models / "model.safetensors").is_file():
        target, base = models, models.parent
    else:
        base = models
        target = models / "target"
        if not any((target / name).is_file() for name in ("config.json", "model.safetensors")):
            target = models / "tofu_Llama-3.2-1B-Instruct"
    candidates = (target, target / "tokenizer", base / "tokenizer")
    tokenizer = next(
        (folder for folder in candidates if all((folder / name).is_file() for name in tokenizer_files)),
        base / "tokenizer",
    )
    generation = target / "generation_config.json"
    if not generation.is_file():
        generation = base / "generation_config.json"
    return {
        "target": target,
        "tokenizer": tokenizer,
        "oracle": base / "oracle",
        "generation": generation,
    }


def verify(path, expected=None):
    path = Path(path)
    with path.open("rb") as handle:
        sha = hashlib.file_digest(handle, "sha256").hexdigest()
    if expected is not None and sha != expected:
        raise ValueError(f"Pinned input hash mismatch: {path}")
    return {"sha256": sha, "bytes": path.stat().st_size}


def prepare(models, data, *, dataset_output=None):
    models, data = Path(models).resolve(), Path(data).resolve()
    pins = json.loads((HERE / "inputs_v1.json").read_text())
    runtime = json.loads((HERE / "runtime_sources_v1.json").read_text())
    paths = model_paths(
        models, [item["path"] for item in pins["files"] if item["repo"] != pins["dataset"]["id"]]
    )
    dataset_entries = [item for item in pins["files"] if item["repo"] == pins["dataset"]["id"]]
    datasets = inspect_datasets(data, dataset_entries)
    required_models = [
        paths["tokenizer"] / item["path"]
        for item in pins["files"]
        if item["repo"] != pins["dataset"]["id"]
    ]
    required_models += [
        paths[key] / name
        for key in ("target", "oracle")
        for name in ("config.json", "model.safetensors")
    ]
    required_models.append(paths["generation"])
    missing = [str(path) for path in required_models if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing local model inputs:\n- " + "\n- ".join(missing))
    records = []
    uses_arrow = any(source["kind"] == "arrow" for source in datasets.values())
    if uses_arrow:
        for source in datasets.values():
            if source["kind"] == "arrow":
                records.extend({"path": str(path), **verify(path)} for path in source["files"])
        if dataset_output is not None:
            data = export_datasets(datasets, dataset_entries, dataset_output)
    for item in pins["files"]:
        if (
            item["repo"] == pins["dataset"]["id"]
            and uses_arrow
            and dataset_output is None
            and datasets[item["path"]]["kind"] == "arrow"
        ):
            continue  # Exact row identity is checked by the prepared runtime exporter.
        folder = data if item["repo"] == pins["dataset"]["id"] else paths["tokenizer"]
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
                    "path": str(paths[folder] / filename),
                    **verify(paths[folder] / filename, expected),
                }
            )
    generation = paths["generation"]
    records.append(
        {"path": str(generation),
         **verify(generation, runtime["files"]["generation_config.json"]["sha256"])}
    )
    return {
        "models": models,
        "data": data,
        **paths,
        "records": records,
        "arrow_export_pending": uses_arrow and dataset_output is None,
    }


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
