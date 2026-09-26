"""D1 preflight behavior, without downloads, GPU, secrets or optimizer steps."""

import copy
import hashlib
import json

import pytest
import torch

from experiments.reproductions.NPO.candidate import compute_causal_npo_rt_loss
from experiments.reproductions.NPO.preflight import (
    SOURCE,
    llama_parameter_count,
    membership,
    same_numeric_setting,
    source_schedule,
    state_inventory,
    verify_files,
)
from scripts.mother_runner import load_recipe


def small_config():
    return dict(
        model_type="llama",
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=True,
        hidden_size=8,
        num_hidden_layers=2,
        intermediate_size=16,
        vocab_size=13,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
    )


def test_source_learning_rate_spelling_preserved_without_accepting_bad_values():
    import yaml

    parsed = yaml.safe_load("lr: 1e-5")
    assert same_numeric_setting(parsed["lr"], 0.00001)
    for wrong in ("1e-4", "nan", "not-a-rate", None, True):
        assert not same_numeric_setting(wrong, 0.00001)


def test_count_matches_independently_constructed_tied_parameters():
    from transformers import LlamaConfig, LlamaForCausalLM

    config = small_config()
    model = LlamaForCausalLM(LlamaConfig(**config))
    model.tie_weights()
    assert model.model.embed_tokens.weight is model.lm_head.weight
    assert llama_parameter_count(config) == sum(p.numel() for p in model.parameters())
    untied = {**config, "tie_word_embeddings": False}
    other = LlamaForCausalLM(LlamaConfig(**untied))
    assert llama_parameter_count(untied) == sum(p.numel() for p in other.parameters())
    assert llama_parameter_count(untied) - llama_parameter_count(config) == 13 * 8


@pytest.mark.parametrize(
    "change",
    [
        {"attention_bias": True},
        {"hidden_size": 7},
        {"num_hidden_layers": 0},
        {"tie_word_embeddings": None},
    ],
)
def test_unsupported_architecture_fails_closed(change):
    with pytest.raises(ValueError):
        llama_parameter_count({**small_config(), **change})


def test_schedule_preserves_paper_batch_and_integer_rounding():
    source = source_schedule(200, 10, 4, 4, 2)
    assert source == {
        "effective_batch": 32,
        "max_steps": 62,
        "warmup_steps": 6,
        "eval_steps": [6, 12, 18, 24, 30, 36, 42, 48, 54, 60],
    }
    assert source_schedule(200, 10, 4, 8, 1) == source
    unchanged = source_schedule(200, 10, 4, 4, 1)
    assert unchanged["max_steps"] == 125
    assert unchanged["warmup_steps"] == 12


@pytest.mark.parametrize("args", [(1, 10, 4, 4, 2), (200, 0, 4, 4, 2), (200, 10, True, 4, 2)])
def test_invalid_schedule_rejected(args):
    with pytest.raises(ValueError):
        source_schedule(*args)


def test_logical_states_never_claim_measured_paged_memory():
    inventory = state_inventory(100)
    assert inventory["components_bytes"] == {
        "candidate_fp16": 200,
        "reference_fp16": 200,
        "master_fp32": 400,
        "adam_first_moment_fp32": 400,
        "adam_second_moment_fp32": 400,
        "gradients_fp16": 200,
    }
    assert inventory["all_resident_bytes_with_fp16_gradients"] == 1800
    assert inventory["all_resident_bytes_with_fp32_gradients"] == 2000
    assert inventory["paged_optimizer_resident_peak_bytes"] is None


def test_membership_binds_order_and_disallows_overlap():
    a = {"question": "First?", "answer": "A"}
    b = {"question": "Second?", "answer": "B"}
    forward = membership({"forget": [a, b]})
    reverse = membership({"forget": [b, a]})
    assert (
        forward["forget"]["ordered_source_records_sha256"]
        != reverse["forget"]["ordered_source_records_sha256"]
    )
    duplicated = membership({"retain": [a, {**a, "answer": "Different source answer"}]})
    assert duplicated["retain"]["count"] == 2
    assert duplicated["retain"]["repeated_question_indices_preserved"] == [[0, 1]]
    with pytest.raises(ValueError, match="Overlapping"):
        membership({"forget": [a], "retain": [a]})
    with pytest.raises(ValueError, match="Malformed"):
        membership({"forget": [{"question": "Question", "answer": " "}]})


def test_file_verification_checks_content_and_git_blob(tmp_path):
    path = tmp_path / "input.txt"
    path.write_bytes(b"abc")
    record = {
        "sha256": hashlib.sha256(b"abc").hexdigest(),
        "git_blob": hashlib.sha1(b"blob 3\0abc").hexdigest(),
    }
    assert verify_files(tmp_path, {path.name: record})[path.name]["bytes"] == 3
    path.write_bytes(b"abd")
    with pytest.raises(ValueError, match="content mismatch"):
        verify_files(tmp_path, {path.name: record})
    with pytest.raises(ValueError, match="unsafe"):
        verify_files(tmp_path, {"../outside": record})


def test_source_inventory_cannot_be_launched_as_mother_recipe():
    with pytest.raises(ValueError, match="Unknown recipe"):
        load_recipe(SOURCE)


def test_preflight_failure_record_is_truthful_and_cannot_be_overwritten(tmp_path, monkeypatch):
    from experiments.reproductions.NPO import preflight

    output = tmp_path / "partial.json"
    monkeypatch.setattr(
        "sys.argv",
        ["preflight", "--upstream", "missing", "--assets", "missing", "--output", str(output)],
    )

    def fail(*args):
        raise ValueError("Test input identity mismatch")

    monkeypatch.setattr(preflight, "validate", fail)
    assert preflight.main() == 1
    original = output.read_bytes()
    assert json.loads(original) == {
        "status": "FAIL",
        "error_type": "ValueError",
        "kaggle_submissions": 0,
    }
    monkeypatch.setattr(
        preflight, "validate", lambda *args: pytest.fail("Must not execute on existing output")
    )
    with pytest.raises(FileExistsError):
        preflight.main()
    assert output.read_bytes() == original


def test_full_tiny_model_gets_gradients_and_reference_does_not():
    from transformers import LlamaConfig, LlamaForCausalLM

    with torch.random.fork_rng():
        torch.manual_seed(1001)
        candidate = LlamaForCausalLM(LlamaConfig(**small_config())).double()
        candidate.tie_weights()
        reference = copy.deepcopy(candidate).eval().requires_grad_(False)
        ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 0]])
        mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]])
        labels = torch.tensor([[-100, -100, 3, 4, 5], [-100, -100, 8, 9, -100]])
        kwargs = {"input_ids": ids, "attention_mask": mask}
        before = {k: v.clone() for k, v in reference.state_dict().items()}
        current = candidate(**kwargs).logits
        with torch.no_grad():
            frozen = reference(**kwargs).logits
        retain = candidate(input_ids=(ids + 1) % 13, attention_mask=mask).logits
        retain_labels = torch.where(labels == -100, -100, (labels + 1) % 13)
        loss = compute_causal_npo_rt_loss(current, frozen, labels, retain, retain_labels).total
        assert torch.isfinite(loss)
        loss.backward()
        for name, parameter in candidate.named_parameters():
            assert parameter.requires_grad, name
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert torch.count_nonzero(parameter.grad) > 0, name
        assert frozen.grad_fn is None
        assert all(p.grad is None for p in reference.parameters())
        for key, tensor in reference.state_dict().items():
            torch.testing.assert_close(tensor, before[key], rtol=0, atol=0)
        # No optimizer step or model/data download occurs in this test.
