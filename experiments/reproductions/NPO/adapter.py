"""Versioned single-turn TOFU adapter; upstream-compatible chat targets with explicit padding."""

from dataclasses import asdict, dataclass

import torch

from experiments.reproductions.common import digest, require_hash, tensors_digest


@dataclass(frozen=True)
class AdapterSpec:
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    dataset_id: str
    dataset_revision: str
    max_length: int = 512
    padding_side: str = "right"
    version: str = "tofu-openunlearning-chat-v1"
    system_prompt: str = "You are a helpful assistant."
    date_string: str = "10 Apr 2025"

    def validate(self):
        for revision in (self.model_revision, self.tokenizer_revision, self.dataset_revision):
            require_hash(revision, 40)
        if not all(
            isinstance(x, str) and x for x in (self.model_id, self.tokenizer_id, self.dataset_id)
        ):
            raise ValueError("Model/tokenizer/dataset identities required")
        if self.version != "tofu-openunlearning-chat-v1" or self.padding_side not in {
            "left",
            "right",
        }:
            raise ValueError("Unknown preprocessing specification")
        if type(self.max_length) is not int or self.max_length < 3:
            raise ValueError("Invalid maximum length")

    @property
    def sha256(self):
        self.validate()
        return digest(asdict(self))


def encode_example(row, index, split, tokenizer, spec):
    spec.validate()
    if (
        not isinstance(row, dict)
        or type(index) is not int
        or index < 0
        or not isinstance(split, str)
        or not split
    ):
        raise ValueError("Source row, stable index and split required")
    for name in ("question", "answer"):
        if not isinstance(row.get(name), str) or not row[name].strip():
            raise ValueError("Nonempty question and answer required")
    if (
        getattr(tokenizer, "eos_token_id", None) is None
        or getattr(tokenizer, "pad_token_id", None) is None
    ):
        raise ValueError("Explicit EOS and padding identities required")
    messages = [
        {"role": "system", "content": spec.system_prompt},
        {"role": "user", "content": row["question"]},
    ]
    kwargs = {"tokenize": True, "return_dict": False, "date_string": spec.date_string}
    prompt = list(tokenizer.apply_chat_template(messages, add_generation_prompt=True, **kwargs))
    full = list(
        tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": row["answer"]}],
            add_generation_prompt=False,
            **kwargs,
        )
    )
    if not prompt or full[: len(prompt)] != prompt:
        raise ValueError("Prompt tokenization is not an exact prefix of candidate/reference input")
    if not full or full[-1] != tokenizer.eos_token_id:
        full.append(tokenizer.eos_token_id)
    # OpenUnlearning's chat branch does not truncate. Explicitly truncate the
    # complete upstream sequence here; never synthesize an EOS after a cut answer.
    original_length = len(full)
    ids = full[: spec.max_length]
    terminators = {tokenizer.eos_token_id}
    eot = getattr(tokenizer, "convert_tokens_to_ids", lambda _: None)("<|eot_id|>")
    if eot is not None and eot != getattr(tokenizer, "unk_token_id", None):
        terminators.add(eot)
    answer_ids = ids[len(prompt) :]
    if not answer_ids or not any(t not in terminators for t in answer_ids):
        raise ValueError("Answer was fully removed or has no content targets")
    labels = [-100] * len(prompt) + ids[len(prompt) :]
    labels = labels[: len(ids)]
    if len(labels) != len(ids) or not any(x != -100 for x in labels[1:]):
        raise ValueError("Zero scored causal targets")
    tensors = {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "attention_mask": torch.ones(len(ids), dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
    sample_id = f"{spec.dataset_id}@{spec.dataset_revision}/{split}/{index}"
    metadata = {
        "sample_id": sample_id,
        "source_split": split,
        "source_index": index,
        "source_record_sha256": digest(row),
        "prompt_token_count": len(prompt),
        "scored_token_count": sum(x != -100 for x in labels[1:]),
        "truncation": {
            "original_length": original_length,
            "length": len(ids),
            "truncated": len(ids) < original_length,
            "eos_retained": len(ids) == original_length and ids[-1] == tokenizer.eos_token_id,
        },
        "preprocessing_version": spec.version,
        "preprocessing_sha256": spec.sha256,
        "identities": asdict(spec),
        "tensors_sha256": tensors_digest(tensors),
    }
    return {**metadata, **tensors}


def collate_examples(examples, tokenizer, spec):
    spec.validate()
    if not examples or len({x["sample_id"] for x in examples}) != len(examples):
        raise ValueError("Nonempty uniquely identified examples required")
    length = max(len(x["input_ids"]) for x in examples)
    result = {name: [] for name in ("input_ids", "attention_mask", "labels")}
    for example in examples:
        if example["preprocessing_sha256"] != spec.sha256 or example["identities"] != asdict(spec):
            raise ValueError("Preprocessing specifications differ")
        tensors = {name: example[name] for name in result}
        if tensors_digest(tensors) != example["tensors_sha256"]:
            raise ValueError("Example tensors changed after preprocessing")
        count = length - len(example["input_ids"])
        for name in result:
            value = (
                tokenizer.pad_token_id if name == "input_ids" else (-100 if name == "labels" else 0)
            )
            pad = torch.full((count,), value, dtype=torch.long)
            parts = (example[name], pad) if spec.padding_side == "right" else (pad, example[name])
            result[name].append(torch.cat(parts))
    tensors = {name: torch.stack(values) for name, values in result.items()}
    if torch.any(tensors["labels"][tensors["attention_mask"] == 0] != -100):
        raise ValueError("Padding may never be scored")
    return {
        "tensors": tensors,
        "sample_ids": [x["sample_id"] for x in examples],
        "sample_ids_sha256": digest([x["sample_id"] for x in examples]),
        "tensors_sha256": tensors_digest(tensors),
        "preprocessing_sha256": spec.sha256,
        "identities": asdict(spec),
    }


def encode_rows(rows, split, tokenizer, spec):
    return [encode_example(row, index, split, tokenizer, spec) for index, row in enumerate(rows)]
