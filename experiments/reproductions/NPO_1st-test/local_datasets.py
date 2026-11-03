"""Inspect saved TOFU splits and export only byte-identical pinned JSONL."""

import hashlib
import json
from pathlib import Path


def inspect_datasets(data, entries):
    """Use raw JSON when present, otherwise a Dataset.save_to_disk split folder.

    This lightweight check runs before environment setup; datasets is imported
    only by the runtime exporter in the prepared training environment.
    """
    sources, missing = {}, []
    for item in entries:
        name = item["path"]
        raw = data / name
        if raw.is_file():
            sources[name] = {"kind": "json", "path": raw, "files": [raw]}
            continue
        folder = data / Path(name).stem
        state_path = folder / "state.json"
        if not state_path.is_file():
            missing.append(f"{raw} OR saved split {folder} (state.json and Arrow files)")
            continue
        state = json.loads(state_path.read_text(encoding="utf-8"))
        shards = state.get("_data_files", [])
        if not shards:
            raise ValueError(f"Saved dataset has no Arrow shards: {state_path}")
        files = [state_path, folder / "dataset_info.json"]
        for shard in shards:
            path = (folder / shard["filename"]).resolve()
            if not path.is_relative_to(folder.resolve()):
                raise ValueError(f"Arrow shard must be inside its split folder: {state_path}")
            files.append(path)
        missing.extend(str(path) for path in files if not path.is_file())
        sources[name] = {"kind": "arrow", "path": folder, "files": files}
    if missing:
        raise FileNotFoundError(
            "Incomplete local TOFU inputs:\n- " + "\n- ".join(missing)
            + "\nCopy the complete saved splits, including Arrow files, from the data machine. "
            "Metadata alone is not a dataset. The four *_perturbed evaluation splits "
            "are required separately; ordinary splits cannot replace them."
        )
    return sources


def pinned_jsonl(rows, expected_sha256):
    """Reconstruct TOFU's compact JSONL, accepting only the original file hash."""
    raw = "\n".join(
        json.dumps(row, ensure_ascii=True, separators=(",", ":")).replace("/", "\\/")
        for row in rows
    ).encode("utf-8")
    # The pinned training/retain files omit the final newline; author/world
    # evaluation files include it. The pin decides, never a heuristic acceptance.
    for candidate in (raw, raw + b"\n"):
        if hashlib.sha256(candidate).hexdigest() == expected_sha256:
            return candidate
    raise ValueError(
        "Saved Arrow rows do not reproduce the pinned TOFU JSON hash. "
        "Use the exact pinned revision, preserving all rows, columns and their order, "
        "or supply the original JSON file."
    )


def export_datasets(sources, entries, output):
    """Create small per-run JSON copies; never write into the supplied data tree."""
    from datasets import load_from_disk

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for item in entries:
        source = sources[item["path"]]
        if source["kind"] == "arrow":
            dataset = load_from_disk(str(source["path"])).with_format(None)
            try:
                raw = pinned_jsonl(dataset, item["sha256"])
            except ValueError as error:
                raise ValueError(f"{source['path']}: {error}") from error
        else:
            raw = source["path"].read_bytes()
        if hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError(f"Pinned input hash mismatch: {source['path']}")
        (output / item["path"]).write_bytes(raw)
    return output
