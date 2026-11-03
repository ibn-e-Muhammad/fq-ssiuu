import copy
import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as functional
from npo3.config import Config
from npo3.data import build_plan, encode_qa, prepare_examples
from npo3.evaluation import tensor_fingerprint
from npo3.objective import npo_loss, sequence_nll
from npo3.training import attach_lora, forward_sequence, frozen_reference_nll


def test_declared_defaults_match_project_lora_policy():
    policy = json.loads(
        (Path(__file__).resolve().parents[4] / "configs/training/lora_v1.json").read_text()
    )
    cfg = Config()
    assert policy["decision"] == "DL-024"
    assert (cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout) == (
        policy["r"],
        policy["lora_alpha"],
        policy["lora_dropout"],
    )
    assert cfg.lora_targets == policy["target_modules"]


def test_sequence_formula_gradients_masks_and_reference():
    torch.manual_seed(4)
    logits = torch.randn(2, 6, 11, dtype=torch.float64, requires_grad=True)
    reference = torch.randn_like(logits, requires_grad=True)
    labels = torch.tensor([[-100, -100, 2, 3, 4, -100], [-100, 2, 3, 4, 5, 6]])
    current, frozen = sequence_nll(logits, labels), sequence_nll(reference, labels)
    oracle = torch.stack(
        [
            sum(
                -logits[i, t - 1].log_softmax(-1)[labels[i, t]]
                for t in range(1, 6)
                if labels[i, t] != -100
            )
            for i in range(2)
        ]
    )
    loss = npo_loss(current, frozen)
    expected = -20 * functional.logsigmoid(0.1 * (oracle - frozen.detach())).mean()
    torch.testing.assert_close(current, oracle)
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(
        torch.autograd.grad(loss, logits, retain_graph=True)[0],
        torch.autograd.grad(expected, logits, retain_graph=True)[0],
    )
    loss.backward()
    assert reference.grad is None and not logits.grad[0, 0].any() and not logits.grad[:, -1].any()
    padded = functional.pad(logits.detach(), (0, 0, 0, 4))
    torch.testing.assert_close(
        sequence_nll(padded, functional.pad(labels, (0, 4), value=-100)), current.detach()
    )
    initial = torch.tensor([2.0, 5.0], requires_grad=True)
    value = npo_loss(initial, initial.detach())
    assert float(value) == pytest.approx(20 * math.log(2))
    value.backward()
    assert (initial.grad < 0).all()
    with pytest.raises(ValueError):
        npo_loss(initial, initial, beta=0)


def test_fresh_sampler_covers_broad_retain_and_is_replayable():
    cfg = Config()
    plan = build_plan(200, 3800, cfg, 42)
    assert plan == build_plan(200, 3800, cfg, 42)
    assert plan["steps"] == 62 and plan["effective_data_passes"] == 9.92
    assert plan["unique_retain_rows"] == 1984
    assert plan["retain_batches"] != build_plan(200, 3800, cfg, 43)["retain_batches"]
    cfg.retain_sampling = "source_fixed"
    source = build_plan(200, 3800, cfg, 42)
    assert source["unique_retain_rows"] == 195
    expected = [
        (i + torch.randint(3800, (1,), generator=torch.Generator().manual_seed(i)).item()) % 3800
        for i in range(200)
    ]
    for f, r in zip(source["forget_batches"], source["retain_batches"]):
        assert r == [expected[i] for i in f]


def test_masking_selection_and_lora_reference(tiny):
    cfg, tokenizer, data, make_model = tiny
    forget, retain, packs, selection = prepare_examples(data, tokenizer, cfg)
    assert selection == prepare_examples(data, tokenizer, cfg)[3]
    assert not {p["question"] for p in packs["retain"]} & {r["question"] for r in data["forget05"]}
    for row in forget:
        assert all(t == -100 for t in row["labels"][: row["prompt_length"]])
        assert row["labels"][-1] == tokenizer.eos_token_id
    with pytest.raises(ValueError, match="truncation"):
        encode_qa(tokenizer, "q0 ?", "A", cfg, max_length=2)
    base = make_model().eval()
    expected = forward_sequence(base, forget[:2], tokenizer, cfg)[0].detach()
    model = attach_lora(base, cfg, 42)
    original = tensor_fingerprint(model, skip_adapters=True)
    assert all("lora_" in name for name, p in model.named_parameters() if p.requires_grad)
    torch.testing.assert_close(frozen_reference_nll(model, forget[:2], tokenizer, cfg), expected)
    for name, parameter in model.named_parameters():
        if "lora_B" in name:
            with torch.no_grad():
                parameter.add_(0.5)
    torch.testing.assert_close(frozen_reference_nll(model, forget[:2], tokenizer, cfg), expected)
    assert tensor_fingerprint(model, skip_adapters=True) == original


def test_device_microbatch_preserves_logical_retain_weighting(tiny):
    cfg, tokenizer, data, make_model = tiny
    forget, retain, _, _ = prepare_examples(data, tokenizer, cfg)
    retain[1] = encode_qa(tokenizer, "q9 ?", "A B C D", cfg)
    a = attach_lora(make_model(), cfg, 42)
    b = copy.deepcopy(a)
    current, _ = forward_sequence(a, forget[:2], tokenizer, cfg)
    reference = frozen_reference_nll(a, forget[:2], tokenizer, cfg)
    retained, count = forward_sequence(a, retain[:2], tokenizer, cfg)
    (npo_loss(current, reference) + retained.sum() / count).backward()
    for f, r in zip(forget[:2], retain[:2]):
        z, _ = forward_sequence(b, [f], tokenizer, cfg)
        ref = frozen_reference_nll(b, [f], tokenizer, cfg)
        (npo_loss(z, ref) / 2).backward()
        z, _ = forward_sequence(b, [r], tokenizer, cfg)
        (z.sum() / count).backward()
    for (_, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        if pa.requires_grad:
            torch.testing.assert_close(pa.grad, pb.grad, atol=2e-6, rtol=2e-4)
