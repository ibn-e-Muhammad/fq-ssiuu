import copy
import json
import math
from types import SimpleNamespace

import pytest
import torch

from experiments.reproductions.common import digest, file_digest
from experiments.reproductions.NPO.adapter import AdapterSpec, encode_rows
from experiments.reproductions.NPO.evaluation_runtime import (
    TASKS,
    encoded_variants,
    evaluate_oracle,
    evaluate_source,
    generate_source_batch,
    load_oracle,
    oracle_binding,
    score_to_file,
    source_kl,
    verified_rows,
)


class Tokenizer:
    eos_token_id = pad_token_id = 9
    clean_up_tokenization_spaces = False

    def convert_tokens_to_ids(self, name):
        return 9

    def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
        prompt = [1, 2] if messages[1]["content"] == "Q" else [1, 2, 3]
        return prompt if add_generation_prompt else prompt + [4, 5, 9]

    def decode(self, tokens, **kwargs):
        return "running cats" if list(tokens)[:2] == [4, 5] else ""


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.generation_config = object()
        self.calls = []
        self.generations = []
        self.fail_after = None
        self.eval()

    def forward(self, input_ids, attention_mask):
        self.calls.append((input_ids.clone(), attention_mask.clone(), torch.is_grad_enabled()))
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            raise RuntimeError("injected later-batch failure")
        return SimpleNamespace(logits=torch.zeros(*input_ids.shape, 10))

    def generate(self, **kwargs):
        self.generations.append((kwargs, torch.rand(3)))
        return torch.cat(
            (kwargs["input_ids"], torch.tensor([[4, 5, 9]] * len(kwargs["input_ids"]))), 1
        )


def setup_case(tmp_path):
    spec = AdapterSpec("model", "a" * 40, "tokenizer", "a" * 40, "data", "b" * 40, max_length=200)
    rows = [
        {"question": q, "answer": "A", "paraphrased_answer": "B", "perturbed_answer": ["C", "D"]}
        for q in ("Q", "long Q")
    ]
    registry = {
        "dataset": {"id": "data", "revision": "b" * 40},
        "retain_only_oracle": {"revision": "c" * 40, "checkpoint_sha256": "d" * 64},
        "files": [],
    }
    for split, _ in TASKS.values():
        path = tmp_path / (split + ".json")
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        registry["files"].append(
            {
                "repo": "data",
                "revision": "b" * 40,
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": file_digest(path),
            }
        )
    return spec, rows, registry


def test_kl_direction_and_all_positions_with_independent_probabilities():
    candidate = torch.log(torch.tensor([[[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]]]))
    reference = torch.log(torch.tensor([[[0.6, 0.4], [0.5, 0.5], [0.2, 0.8]]]))
    expected = (
        sum(
            p * math.log(p / q)
            for ps, qs in zip(
                [[0.6, 0.4], [0.5, 0.5], [0.2, 0.8]],
                [[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]],
                strict=True,
            )
            for p, q in zip(ps, qs, strict=True)
        )
        / 3
    )
    assert source_kl(candidate, reference, 1)[0] == pytest.approx(expected, abs=1e-7)
    assert source_kl(candidate, reference, 2)[0] == pytest.approx(expected, abs=1e-7)
    with pytest.raises(ValueError):
        source_kl(candidate, reference, 0)


def test_scoring_masks_and_completed_batches_survive_failure(tmp_path):
    spec, rows, _ = setup_case(tmp_path)
    examples = encode_rows(rows, "forget05", Tokenizer(), spec)
    model = Model()
    model.fail_after = 1
    path = tmp_path / "partial.jsonl"
    with pytest.raises(RuntimeError, match="later-batch"):
        score_to_file(path, model, examples, Tokenizer(), spec, 1, device="cpu")
    record = json.loads(path.read_text())
    assert record["sample_id"] == examples[0]["sample_id"]
    assert record["scored_tokens"] == 3
    assert record["sequence_nll"] == pytest.approx(3 * math.log(10))
    assert record["mean_nll"] == pytest.approx(math.log(10))
    assert model.calls[0][0].shape == (1, 200)
    assert model.calls[0][1].sum() == 5
    assert not model.calls[0][2]
    with pytest.raises(FileExistsError):
        score_to_file(path, model, examples, Tokenizer(), spec, 1, device="cpu")


def test_generation_uses_source_sampling_left_padding_and_restores_config(tmp_path):
    spec, rows, _ = setup_case(tmp_path)
    examples = encode_rows(rows, "forget05", Tokenizer(), spec)
    model = Model()
    prior = model.generation_config
    a = generate_source_batch(
        model, Tokenizer(), examples, {"temperature": 0.6, "top_p": 0.9}, device="cpu"
    )
    b = generate_source_batch(
        model, Tokenizer(), examples, {"temperature": 0.6, "top_p": 0.9}, device="cpu"
    )
    assert a == b and model.generation_config is prior
    first, rng = model.generations[0]
    assert torch.equal(rng, model.generations[1][1])
    assert first["input_ids"].tolist() == [[9, 1, 2], [1, 2, 3]]
    assert first["attention_mask"].tolist() == [[0, 1, 1], [1, 1, 1]]
    config = first["generation_config"]
    assert config.do_sample and config.max_length == 200 and config.max_new_tokens is None
    assert config.temperature == 0.6 and config.top_p == 0.9 and config.use_cache
    assert a[0]["rouge_l_recall"] == 1
    assert a[0]["decoding_sha256"] == digest(a[0]["specification"])


@pytest.mark.parametrize("defect", ["order", "oracle", "preprocessing", "tensors", "mean"])
def test_oracle_rejects_content_or_binding_mismatch(tmp_path, defect):
    spec, rows, registry = setup_case(tmp_path)
    tok = Tokenizer()
    path = evaluate_oracle(
        Model(), tok, spec, tmp_path, registry, tmp_path / "oracle", device="cpu"
    )
    _, base, wrong, n = encoded_variants(
        rows, "forget05_perturbed", tok, spec, "paraphrased_answer"
    )
    binding = oracle_binding(spec, registry, base, wrong)
    good = load_oracle(path, file_digest(path), binding, n)
    assert good["average_perturb_loss"][0] == pytest.approx([math.log(10)] * 2)
    if defect in {"order", "oracle", "preprocessing"}:
        binding = copy.deepcopy(binding)
        if defect == "order":
            binding["ordered_sample_ids"].reverse()
        elif defect == "oracle":
            binding["oracle"]["checkpoint_sha256"] = "f" * 64
        else:
            binding["preprocessing_sha256"] = "f" * 64
    else:
        record = json.loads(path.read_text())
        record.pop("record_sha256")
        if defect == "tensors":
            record["base"][0]["input_ids"][0] = 3
        else:
            record["base"][0]["mean_nll"] = 20
        record["record_sha256"] = digest(record)
        # Deliberate corruption of a test fixture, never a real evidence file.
        path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        load_oracle(path, file_digest(path), binding, n)


def test_four_component_evaluation_binds_raw_files_without_verdict(tmp_path):
    spec, _, registry = setup_case(tmp_path)
    tok, reference = Tokenizer(), Model()
    oracle = evaluate_oracle(
        reference, tok, spec, tmp_path, registry, tmp_path / "oracle", device="cpu"
    )
    result = evaluate_source(
        Model(),
        reference,
        tok,
        spec,
        tmp_path,
        {},
        tmp_path / "evaluation",
        {"run_id": "fixture"},
        oracle,
        file_digest(oracle),
        registry,
        device="cpu",
    )
    assert result["attack_detected"] is None and result["durability_pass"] is None
    assert result["metrics"]["Model Utility"] == 0  # Truth-ratio component is exactly zero.
    assert len(result["utility_components"]) == 9
    assert result["forget_quality_diagnostic"]["ks_statistic"] == 0
    assert result["forget_quality_diagnostic"]["ks_p_value"] == 1
    for name, sha in result["raw_file_sha256"].items():
        assert sha == file_digest(tmp_path / "evaluation" / name)
    path = tmp_path / "forget05_perturbed.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="pinned registry"):
        verified_rows(tmp_path, "forget05_perturbed", registry)
