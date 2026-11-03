"""Value-level final-checkpoint validation; a changed file hash alone is insufficient."""

import torch
from safetensors import safe_open


def compare_values(initial_path, trained_path):
    records = []
    with (
        safe_open(initial_path, framework="pt", device="cpu") as initial,
        safe_open(trained_path, framework="pt", device="cpu") as trained,
    ):
        if set(initial.keys()) != set(trained.keys()):
            raise ValueError("Final checkpoint parameter inventory differs from source")
        for name in sorted(initial.keys()):
            a, b = initial.get_tensor(name), trained.get_tensor(name)
            if a.shape != b.shape or a.dtype != b.dtype or a.dtype != torch.float16:
                raise ValueError("Final checkpoint shape/precision differs from source")
            # Bound CPU temporaries even for the vocabulary embedding.
            changed, delta = 0, 0.0
            a, b = a.reshape(-1), b.reshape(-1)
            for offset in range(0, a.numel(), 1000000):
                x, y = a[offset : offset + 1000000], b[offset : offset + 1000000]
                if not torch.isfinite(y).all():
                    raise ValueError("Nonfinite final checkpoint weights")
                changed += int((x != y).sum())
                delta = max(delta, float((x.float() - y.float()).abs().max()))
            records.append(
                {
                    "parameter": name,
                    "numel": a.numel(),
                    "changed_values": changed,
                    "max_absolute_change": delta,
                }
            )
    if not records or not any(x["changed_values"] for x in records):
        raise ValueError("No trained FP16 parameter value changed")
    return records
