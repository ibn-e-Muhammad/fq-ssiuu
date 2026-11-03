"""Identity-bound state helpers for NPO training; no model or optimizer construction."""

import json
import random
from pathlib import Path

import numpy as np
import torch

from experiments.reproductions.common import digest, file_digest, require_hash

IDENTITY_KEYS = {
    "source_commit",
    "source_spec_sha256",
    "train_config_sha256",
    "eval_config_sha256",
    "initial_checkpoint_sha256",
    "reference_checkpoint_sha256",
    "tokenizer_revision",
    "dataset_revision",
    "preprocessing_sha256",
    "sample_membership_sha256",
}


def validate_identities(identities):
    if set(identities) != IDENTITY_KEYS:
        raise ValueError("Complete source/config/model/data identities required")
    for name, value in identities.items():
        require_hash(value, 40 if name.endswith("revision") or name == "source_commit" else 64)
    if identities["initial_checkpoint_sha256"] != identities["reference_checkpoint_sha256"]:
        raise ValueError("NPO reference must be the original full checkpoint")


def capture_rng():
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    validate_rng(state)
    random.setstate(state["python"])
    values = state["numpy"]
    np.random.set_state((values[0], np.array(values[1], dtype=np.uint32), *values[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def validate_rng(state):
    """Validate RNG records before mutating a model or any process-global RNG."""
    if set(state) != {"python", "numpy", "torch", "cuda"}:
        raise ValueError("Complete RNG state required")
    cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if len(state["cuda"]) != cuda_count:
        raise ValueError("Saved and current CUDA visibility differ")
    random.Random().setstate(state["python"])
    values = state["numpy"]
    np.random.RandomState().set_state(
        (values[0], np.array(values[1], dtype=np.uint32), *values[2:])
    )
    torch.Generator().set_state(state["torch"].cpu())
    for value in state["cuda"]:
        if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1:
            raise ValueError("Malformed CUDA RNG state")


def validate_cursor(cursor):
    if set(cursor) != {"optimizer_step", "micro_step", "epoch", "batch_in_epoch", "sampler_state"}:
        raise ValueError("Exact sampler/optimization cursor required")
    for name in ("optimizer_step", "micro_step", "epoch", "batch_in_epoch"):
        if type(cursor[name]) is not int or cursor[name] < 0:
            raise ValueError("Invalid training cursor")
    if cursor["micro_step"] != 0:
        raise ValueError(
            "Checkpoint only at optimizer boundaries; pending gradients are not serialized"
        )
    if (
        not isinstance(cursor["sampler_state"], torch.Tensor)
        or cursor["sampler_state"].dtype != torch.uint8
    ):
        raise ValueError("Explicit torch.Generator state required")
    torch.Generator().set_state(cursor["sampler_state"].cpu())


def save_training_state(directory, model, optimizer, scheduler, cursor, identities, *, scaler=None):
    """CPU conformance backend; DeepSpeed runtime must use its partitioned save API.

    Saves after an optimizer boundary. An incomplete write remains a failed partial
    directory without a completion manifest; neither path is ever overwritten.
    """
    validate_identities(identities)
    validate_cursor(cursor)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "training_state.pt"
    try:
        state = {
            "schema": "npo-training-state-v1",
            "identities": identities,
            "cursor": cursor,
            "rng": capture_rng(),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
        }
        torch.save(state, path)
        manifest = {
            "schema": "npo-checkpoint-v1",
            "status": "COMPLETE",
            "identities": identities,
            "identities_sha256": digest(identities),
            "state_sha256": file_digest(path),
            "state_bytes": path.stat().st_size,
        }
        with (directory / "MANIFEST.json").open("x") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
    except Exception as error:
        with (directory / "FAILURE.json").open("x") as handle:
            json.dump(
                {"status": "PARTIAL", "error_type": type(error).__name__, "identities": identities},
                handle,
                indent=2,
            )
            handle.write("\n")
        raise
    return manifest


def load_training_state(directory, model, optimizer, scheduler, identities, *, scaler=None):
    """Load only a locally hash-verified, identity-matched tensor/primitive checkpoint."""
    validate_identities(identities)
    directory = Path(directory)
    manifest = json.loads((directory / "MANIFEST.json").read_text())
    if manifest.get("schema") != "npo-checkpoint-v1" or manifest.get("status") != "COMPLETE":
        raise ValueError("Incomplete/unsupported checkpoint")
    if manifest["identities"] != identities or manifest["identities_sha256"] != digest(identities):
        raise ValueError("Checkpoint source/config/data identities differ")
    path = directory / "training_state.pt"
    if (
        path.stat().st_size != manifest["state_bytes"]
        or file_digest(path) != manifest["state_sha256"]
    ):
        raise ValueError("Checkpoint bytes differ from manifest")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["schema"] != "npo-training-state-v1" or state["identities"] != identities:
        raise ValueError("Inner checkpoint identity mismatch")
    validate_cursor(state["cursor"])
    validate_rng(state["rng"])
    if (state["scaler"] is None) != (scaler is None):
        raise ValueError("Mixed-precision scaler mismatch")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if scaler is not None:
        scaler.load_state_dict(state["scaler"])
    restore_rng(state["rng"])
    return state["cursor"]


def save_deepspeed_state(directory, engine, cursor, identities):
    """Use DeepSpeed's native ZeRO state, with a content-checked outer manifest."""
    validate_identities(identities)
    validate_cursor(cursor)
    if engine.global_steps != cursor["optimizer_step"]:
        raise ValueError("Engine and data cursor disagree")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    try:
        client = {
            "schema": "npo-deepspeed-client-v1",
            "identities": identities,
            "cursor": cursor,
            "rng": capture_rng(),
        }
        saved = engine.save_checkpoint(
            str(directory), tag="state", client_state=client, save_latest=False
        )
        if not saved:
            raise RuntimeError("DeepSpeed did not complete the checkpoint")
        entries = []
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError("Checkpoint symlinks forbidden")
            if path.is_file():
                entries.append(
                    {
                        "path": str(path.relative_to(directory)),
                        "sha256": file_digest(path),
                        "bytes": path.stat().st_size,
                    }
                )
        if not any(row["path"].endswith("optim_states.pt") for row in entries):
            raise ValueError("Missing native optimizer checkpoint")
        manifest = {
            "schema": "npo-deepspeed-state-v1",
            "status": "COMPLETE",
            "identities": identities,
            "identities_sha256": digest(identities),
            "optimizer_step": cursor["optimizer_step"],
            "files": entries,
        }
        with (directory / "MANIFEST.json").open("x") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
    except Exception as error:
        with (directory / "FAILURE.json").open("x") as handle:
            json.dump(
                {"status": "PARTIAL", "error_type": type(error).__name__, "identities": identities},
                handle,
                indent=2,
            )
            handle.write("\n")
        raise
    return manifest


def load_deepspeed_state(directory, engine, identities):
    """Load a pristine engine only after verifying every saved file and identity.

    DeepSpeed uses pickle internally. This API is only for our own content-bound
    checkpoint, never an arbitrary external checkpoint supplied by a user.
    """
    validate_identities(identities)
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "MANIFEST.json").read_text())
    if manifest.get("schema") != "npo-deepspeed-state-v1" or manifest.get("status") != "COMPLETE":
        raise ValueError("Incomplete DeepSpeed checkpoint")
    if manifest["identities"] != identities or manifest["identities_sha256"] != digest(identities):
        raise ValueError("Checkpoint source/config/data identities differ")
    if not manifest["files"] or len({x["path"] for x in manifest["files"]}) != len(
        manifest["files"]
    ):
        raise ValueError("Missing or duplicate checkpoint file inventory")
    actual = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    expected = {x["path"] for x in manifest["files"]} | {"MANIFEST.json"}
    if actual != expected:
        raise ValueError("Checkpoint file inventory changed")
    for row in manifest["files"]:
        path = directory / row["path"]
        if path.is_symlink() or not path.resolve().is_relative_to(directory):
            raise ValueError("Unsafe checkpoint path")
        if path.stat().st_size != row["bytes"] or file_digest(path) != row["sha256"]:
            raise ValueError("Checkpoint bytes differ from manifest")
    loaded, client = engine.load_checkpoint(
        str(directory),
        tag="state",
        load_optimizer_states=True,
        load_lr_scheduler_states=True,
        load_module_strict=True,
    )
    if (
        not loaded
        or client.get("schema") != "npo-deepspeed-client-v1"
        or client["identities"] != identities
    ):
        raise ValueError("DeepSpeed restore failed or inner identity mismatch")
    validate_cursor(client["cursor"])
    if (
        engine.global_steps != client["cursor"]["optimizer_step"]
        or engine.global_steps != manifest["optimizer_step"]
    ):
        raise ValueError("Restored optimizer and data cursor disagree")
    restore_rng(client["rng"])
    return client["cursor"]
