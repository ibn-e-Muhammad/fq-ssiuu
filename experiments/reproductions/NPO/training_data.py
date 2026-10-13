"""Author-faithful pairing, fixed padding, and continuous accumulated batches.

This module constructs no model and changes no optimization hyperparameters.
The original dataset's global torch reseeding is deliberate and preserved.
"""

import torch
from torch.utils.data import DataLoader, Dataset

from experiments.reproductions.common import digest, tensors_digest


class SourcePairDataset(Dataset):
    def __init__(self, forget, retain):
        if not forget or not retain:
            raise ValueError("Nonempty forget and retain examples required")
        self.forget = forget
        self.retain = retain

    def __len__(self):
        return len(self.forget)

    def __getitem__(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise ValueError("Invalid forget sample index")
        # Author __getitem__ reseeds once for each of the two data types.
        torch.manual_seed(index)
        forget = self.forget[index]
        torch.manual_seed(index)
        retain_index = (index + torch.randint(0, len(self.retain), (1,)).item()) % len(self.retain)
        return {
            "forget": forget,
            "retain": self.retain[retain_index],
            "forget_index": index,
            "retain_index": retain_index,
        }


def fixed_batch(examples, *, max_length, pad_token_id):
    """Preserve source fixed-length right padding; repeated retain draws are valid."""
    if not examples or type(max_length) is not int or max_length < 3:
        raise ValueError("Nonempty bounded batch required")
    result = {k: [] for k in ("input_ids", "attention_mask", "labels")}
    specs = {x["preprocessing_sha256"] for x in examples}
    if len(specs) != 1:
        raise ValueError("Mixed preprocessing identities")
    for example in examples:
        tensors = {k: example[k] for k in result}
        if tensors_digest(tensors) != example["tensors_sha256"]:
            raise ValueError("Changed preprocessed tensor")
        n = len(example["input_ids"])
        if n > max_length or example["truncation"]["truncated"]:
            raise ValueError("Unexpected training truncation")
        if any(value.ndim != 1 or len(value) != n for value in tensors.values()):
            raise ValueError("Malformed training tensors")
        prompt = example["prompt_token_count"]
        labels, mask = example["labels"], example["attention_mask"]
        if not 0 < prompt < n or (labels[:prompt] != -100).any():
            raise ValueError("Prompt tokens cannot be scored")
        if (labels[mask == 0] != -100).any() or not (labels[1:] != -100).any():
            raise ValueError("Invalid answer/padding targets")
        if int((labels[1:] != -100).sum()) != example["scored_token_count"]:
            raise ValueError("Scored target count changed")
        for key, value in tensors.items():
            fill = pad_token_id if key == "input_ids" else (0 if key == "attention_mask" else -100)
            result[key].append(
                torch.cat((value, torch.full((max_length - n,), fill, dtype=torch.long)))
            )
    tensors = {k: torch.stack(values) for k, values in result.items()}
    return {
        "tensors": tensors,
        "tensors_sha256": tensors_digest(tensors),
        "sample_ids": [x["sample_id"] for x in examples],
        "preprocessing_sha256": next(iter(specs)),
        "identities": examples[0]["identities"],
    }


def source_batch_plan(sample_count, micro_batch, accumulation, max_steps, seed):
    """One-device order from the author's DataLoader, retaining epoch tails.

    Publication-era Trainer counts accumulated microbatches continuously across
    epochs. Materializing the bounded index plan makes resume independent of
    DataLoader prefetch without replacing source sampling with a new algorithm.
    """
    if any(
        type(x) is not int or x <= 0 for x in (sample_count, micro_batch, accumulation, max_steps)
    ):
        raise ValueError("Positive integer schedule required")
    if type(seed) is not int or seed < 0 or sample_count % micro_batch:
        raise ValueError("D1 requires complete source microbatches and a declared seed")
    if sample_count // micro_batch <= accumulation:
        raise ValueError("Tiny-epoch Trainer special case is outside D1")
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        list(range(sample_count)),
        batch_size=micro_batch,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )
    batches = []
    epoch = 0
    required = max_steps * accumulation
    while len(batches) < required:
        for batch_index, indices in enumerate(loader):
            batches.append(
                {"epoch": epoch, "batch_in_epoch": batch_index, "forget_indices": indices.tolist()}
            )
            if len(batches) == required:
                break
        epoch += 1
    plan = {
        "version": "npo-source-continuous-accumulation-v1",
        "seed": seed,
        "sample_count": sample_count,
        "micro_batch": micro_batch,
        "accumulation": accumulation,
        "max_steps": max_steps,
        "batches": batches,
    }
    return {**plan, "sha256": digest(plan), "generator_state": generator.get_state()}
