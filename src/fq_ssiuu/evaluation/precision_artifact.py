"""Hash actual parameter/quantization-state bytes, not precision labels.

Manifests identify an in-memory evaluation artifact. They do not claim that a
persistent quantized checkpoint exists. Hashing is bounded to one tensor at a time.
"""

import hashlib
import math
import sys

import torch

from fq_ssiuu.evaluation.identity import digest, require_hash


def tensor_identity(tensor):
    if tensor.device.type == "meta" or tensor.layout != torch.strided:
        raise ValueError("Materialized dense tensor required")
    value = tensor.detach().contiguous().cpu()
    raw = value.reshape(-1).view(torch.uint8).numpy()
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "byte_order": sys.byteorder,
        "bytes": value.numel() * value.element_size(),
        "sha256": hashlib.sha256(memoryview(raw)).hexdigest(),
    }


def state_identity(value):
    if isinstance(value, torch.Tensor):
        return {"tensor": tensor_identity(value)}
    if isinstance(value, dict):
        return {str(k): state_identity(v) for k, v in sorted(value.items())}
    if isinstance(value, (tuple, list, torch.Size)):
        return [state_identity(v) for v in value]
    if isinstance(value, torch.dtype):
        return str(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("Unrecorded quantization-state type")


def build_artifact(
    model,
    *,
    precision,
    checkpoint_sha256,
    versions,
    quantizer_config,
    expected_quantized_modules,
    saved_shards=None,
):
    """Call after forward/generation: LLM.int8 initializes/transfers state lazily."""
    modules = []
    for name, module in sorted(model.named_modules()):
        qualified = type(module).__module__ + "." + type(module).__name__
        if qualified not in {
            "bitsandbytes.nn.modules.Linear4bit",
            "bitsandbytes.nn.modules.Linear8bitLt",
        }:
            continue
        if precision == "nf4" and qualified.endswith("Linear4bit"):
            state = module.weight.quant_state
            if state is None or state.quant_type != "nf4" or module.weight.dtype != torch.uint8:
                raise ValueError("Actual NF4 packed weights/state required")
            if module.compute_dtype != torch.float16:
                raise ValueError("NF4 compute must be FP16")
            payload = {
                "quant_state": state_identity(state.as_dict(packed=False)),
                "compute_dtype": str(module.compute_dtype),
                "quant_type": state.quant_type,
                "blocksize": state.blocksize,
            }
        elif precision == "int8" and qualified.endswith("Linear8bitLt"):
            # bnb 0.46.1 moves SCB/CB from Int8Params into MatmulLtState on first forward.
            scb = getattr(module.weight, "SCB", None)
            if scb is None:
                scb = module.state.SCB
            if module.weight.dtype != torch.int8 or scb is None or module.state.has_fp16_weights:
                raise ValueError("Actual frozen INT8 weights and scales required")
            payload = {
                "SCB": state_identity(scb),
                "threshold": module.state.threshold,
                "has_fp16_weights": module.state.has_fp16_weights,
            }
        else:
            raise ValueError("Mixed or mislabeled quantization modules")
        modules.append(
            {
                "name": name,
                "class": qualified,
                "weight": tensor_identity(module.weight),
                "state": payload,
            }
        )
    tensors = [
        {"name": name, **tensor_identity(tensor)}
        for name, tensor in sorted(model.state_dict().items())
    ]
    manifest = {
        "schema_version": "precision-artifact-v1",
        "precision": precision,
        "checkpoint_sha256": checkpoint_sha256,
        "versions": dict(versions),
        "quantizer_config": dict(quantizer_config),
        "expected_quantized_modules": sorted(expected_quantized_modules),
        "quantized_modules": modules,
        "state_tensors": tensors,
        "saved_shards": saved_shards or [],
        "persistence": "saved" if saved_shards else "in_memory_only",
    }
    validate_artifact(manifest)
    return manifest


def validate_artifact(manifest):
    if manifest.get("schema_version") != "precision-artifact-v1":
        raise ValueError("Unknown precision artifact schema")
    require_hash(manifest["checkpoint_sha256"])
    if set(manifest["versions"]) != {"torch", "transformers", "bitsandbytes"} or any(
        not isinstance(v, str) or not v for v in manifest["versions"].values()
    ):
        raise ValueError("Exact runtime library versions required")
    precision = manifest["precision"]
    names = [x["name"] for x in manifest["quantized_modules"]]
    expected = manifest["expected_quantized_modules"]
    if names != sorted(set(names)) or expected != sorted(set(expected)) or names != expected:
        raise ValueError("Partial or unordered quantized module inventory")
    if precision == "fp16":
        if names or manifest["quantizer_config"] != {"dtype": "float16", "quantization": None}:
            raise ValueError("FP16 must be unquantized")
    elif precision in {"int8", "nf4"}:
        if not names:
            raise ValueError("Quantized artifact has no quantized modules")
        suffix = "Linear8bitLt" if precision == "int8" else "Linear4bit"
        dtype = "torch.int8" if precision == "int8" else "torch.uint8"
        for module in manifest["quantized_modules"]:
            if (
                module["class"] != "bitsandbytes.nn.modules." + suffix
                or module["weight"]["dtype"] != dtype
            ):
                raise ValueError("Quantized module type/packed dtype differs")
            if precision == "nf4" and (
                module["state"].get("quant_type") != "nf4" or module["state"].get("blocksize") != 64
            ):
                raise ValueError("Expected blockwise NF4 state")
            if precision == "nf4":
                state = module["state"].get("quant_state", {})
                if not all(
                    isinstance(state.get(k), dict) and "tensor" in state[k]
                    for k in ("absmax", "quant_map")
                ):
                    raise ValueError("NF4 scale and codebook tensor identities required")
            elif not isinstance(module["state"].get("SCB", {}).get("tensor"), dict):
                raise ValueError("INT8 scale tensor identity required")
    else:
        raise ValueError("Unknown artifact precision")
    entries = manifest["state_tensors"]
    keys = [r["name"] for r in entries]
    if not keys or keys != sorted(set(keys)):
        raise ValueError("Complete ordered state inventory required")
    by_name = {r["name"]: {k: v for k, v in r.items() if k != "name"} for r in entries}
    for module in manifest["quantized_modules"]:
        if by_name.get(module["name"] + ".weight") != module["weight"]:
            raise ValueError("Packed module weight differs from state inventory")

    def check(value):
        if isinstance(value, dict):
            if {"dtype", "shape", "bytes", "sha256"} <= value.keys():
                require_hash(value["sha256"])
                if type(value["bytes"]) is not int or value["bytes"] < 0:
                    raise ValueError("Invalid tensor byte length")
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)

    check(manifest)
    for shard in manifest["saved_shards"]:
        require_hash(shard["sha256"])
    if manifest["persistence"] != ("saved" if manifest["saved_shards"] else "in_memory_only"):
        raise ValueError("Artifact persistence claim differs")
    # Verify serialization is finite and deterministic.
    digest(manifest)
