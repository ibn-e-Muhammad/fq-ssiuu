"""Canonical, finite JSON and tensor identities for method-validation records."""

import hashlib
import json
import re
from pathlib import Path


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require_hash(value, length=64):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{" + str(length) + "}", value):
        raise ValueError("Immutable hexadecimal identity required")


def tensor_record(tensors):
    return {
        name: {"dtype": str(t.dtype), "shape": list(t.shape), "values": t.detach().cpu().tolist()}
        for name, t in tensors.items()
    }


def tensors_digest(tensors):
    return digest(tensor_record(tensors))
