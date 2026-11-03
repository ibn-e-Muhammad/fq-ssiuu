"""Small atomic LoRA checkpoints; identity, optimizer and RNG are restored."""

import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from peft import set_peft_model_state_dict

from .metrics import sha256, write_json


def capture_rng():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(rng):
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["cpu"])
    if rng["cuda"]:
        torch.cuda.set_rng_state_all(rng["cuda"])


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(x) for x in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(x) for x in value)
    return value


def _flush_directory(path):
    if os.name != "nt":
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def checkpoint_candidates(root):
    found = []
    for path in Path(root).glob("step-*"):
        marker = path / "complete.json"
        if marker.is_file():
            try:
                meta = json.loads(marker.read_text())
                found.append((meta["step"], path, meta))
            except (ValueError, KeyError):
                continue
    return sorted(found, key=lambda row: (row[0], row[1].name), reverse=True)


def save_checkpoint(root, model, optimizer, step, identity, history, keep=2):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"step-{step:06d}-{time.time_ns()}"
    pending = root / (".pending-" + path.name)
    pending.mkdir()
    model.save_pretrained(pending / "adapter", safe_serialization=True)
    # Only our own checksummed runtime checkpoints are loaded, never HF training_args.bin.
    torch.save(
        {"optimizer": _cpu_tree(optimizer.state_dict()), "rng": capture_rng(), "history": history},
        pending / "runtime.pt",
    )
    files = {str(p.relative_to(pending)): sha256(p) for p in pending.rglob("*") if p.is_file()}
    for name in files:
        with (pending / name).open("r+b") as handle:
            os.fsync(handle.fileno())
    write_json(
        pending / "complete.json", {"step": step, "identity": identity, "artifact_hashes": files}
    )
    with (pending / "complete.json").open("r+b") as handle:
        os.fsync(handle.fileno())
    pending.rename(path)
    _flush_directory(root)
    for _, old, _ in checkpoint_candidates(root)[keep:]:
        shutil.rmtree(old)  # Only completed generations in this run's owned checkpoint directory.
    return path


def restore_checkpoint(root, model, optimizer, identity):
    from safetensors.torch import load_file

    rejected = []
    for step, path, meta in checkpoint_candidates(root):
        if meta.get("identity") != identity:
            rejected.append({"path": str(path), "reason": "identity mismatch"})
            continue
        try:
            for name, expected in meta["artifact_hashes"].items():
                if sha256(path / name) != expected:
                    raise ValueError(f"Checksum mismatch: {name}")
            state = load_file(str(path / "adapter/adapter_model.safetensors"))
            # weights_only=False is confined to our identity-bound, checksummed local state.
            runtime = torch.load(path / "runtime.pt", map_location="cpu", weights_only=False)
            if [row["update"] for row in runtime["history"]] != list(range(1, step + 1)):
                raise ValueError("Checkpoint history/cursor mismatch")
            keys = set_peft_model_state_dict(model, state)
            if keys.unexpected_keys or any("lora_" in key for key in keys.missing_keys):
                raise ValueError("Checkpoint adapter keys differ")
            optimizer.load_state_dict(runtime["optimizer"])
            restore_rng(runtime["rng"])
            return step, runtime["history"], rejected
        except (OSError, ValueError, RuntimeError, EOFError, KeyError, TypeError) as exc:
            rejected.append({"path": str(path), "reason": str(exc)})
    raise ValueError(f"No compatible complete LoRA checkpoint: {rejected}")
