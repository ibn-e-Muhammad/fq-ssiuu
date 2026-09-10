"""Independent scalar, masking and gradient checks for causal sequence NPO."""

import math

import pytest
import torch

from experiments.reproductions.NPO.candidate import (
    causal_sequence_nll,
    compute_causal_npo_loss,
    compute_causal_npo_rt_loss,
)


def fixture(dtype=torch.float64, device="cpu"):
    generator = torch.Generator().manual_seed(1801)
    current = torch.randn(2, 5, 7, generator=generator, dtype=torch.float64)
    reference = torch.randn(2, 5, 7, generator=generator, dtype=torch.float64)
    labels = torch.tensor([[-100, -100, 1, 2, 3], [-100, 4, 2, -100, -100]])
    return (
        current.to(device=device, dtype=dtype),
        reference.to(device=device, dtype=dtype),
        labels.to(device),
    )


def scalar_oracle(current, reference, labels, beta):
    # Probability products are deliberately independent of the implementation's CE reduction.
    ratios = []
    for row in range(len(labels)):
        ratio = current.new_tensor(1.0)
        for target_position in range(1, labels.shape[1]):
            target = int(labels[row, target_position])
            if target != -100:
                p = current[row, target_position - 1].softmax(-1)[target]
                q = reference[row, target_position - 1].softmax(-1)[target]
                ratio = ratio * p / q
        ratios.append((2 / beta) * torch.log1p(ratio**beta))
    return torch.stack(ratios).mean()


def test_hand_calculated_sequence_probability_product():
    current = torch.tensor(
        [[[0.25, 0.75], [0.2, 0.8], [0.5, 0.5]], [[0.75, 0.25], [0.4, 0.6], [0.5, 0.5]]],
        dtype=torch.float64,
    ).log()
    reference = torch.tensor(
        [[[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], [[0.25, 0.75], [0.5, 0.5], [0.5, 0.5]]],
        dtype=torch.float64,
    ).log()
    labels = torch.tensor([[-100, 0, 0], [-100, 0, -100]])
    assert compute_causal_npo_loss(current, reference, labels, 1).item() == pytest.approx(
        math.log(4.8)
    )


def test_independent_value_gradient_and_reference_detachment():
    current, reference, labels = fixture()
    current.requires_grad_()
    reference.requires_grad_()
    loss = compute_causal_npo_loss(current, reference, labels, 0.3)
    oracle = scalar_oracle(current, reference.detach(), labels, 0.3)
    torch.testing.assert_close(loss, oracle, rtol=1e-12, atol=1e-12)
    actual = torch.autograd.grad(loss, current, retain_graph=True)[0]
    expected = torch.autograd.grad(oracle, current)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert actual.abs().sum() > 0
    assert torch.autograd.grad(loss, reference, allow_unused=True)[0] is None


def test_shift_mask_eos_and_right_padding():
    current, reference, labels = fixture()
    current.requires_grad_()
    loss = compute_causal_npo_loss(current, reference, labels)
    gradient = torch.autograd.grad(loss, current)[0]
    assert torch.count_nonzero(gradient[:, -1]) == 0
    assert torch.count_nonzero(gradient[0, 0]) == 0  # masked prompt target
    assert gradient[0, 1].abs().sum() > 0  # last prompt predicts first answer
    assert gradient[0, 3].abs().sum() > 0  # scored EOS, independent of attention mask
    changed_labels = labels.clone()
    changed_labels[:, 0] = 6
    torch.testing.assert_close(loss, compute_causal_npo_loss(current, reference, changed_labels))
    pad = current.detach().new_zeros(2, 3, 7)
    padded_labels = torch.cat([labels, labels.new_full((2, 3), -100)], dim=1)
    torch.testing.assert_close(
        loss,
        compute_causal_npo_loss(
            torch.cat([current, pad], 1), torch.cat([reference, pad], 1), padded_labels
        ),
    )


def test_update_decreases_forget_probability_and_beta_limit():
    current, reference, labels = fixture()
    current.requires_grad_()
    loss = compute_causal_npo_loss(current, reference, labels, 1e-6)
    gradient = torch.autograd.grad(loss, current, retain_graph=True)[0]
    ga_gradient = torch.autograd.grad(-causal_sequence_nll(current, labels).mean(), current)[0]
    torch.testing.assert_close(gradient, ga_gradient, rtol=1e-5, atol=1e-6)
    updated = current.detach() - 0.01 * gradient
    assert (causal_sequence_nll(updated, labels) > causal_sequence_nll(current, labels)).all()
    assert compute_causal_npo_loss(updated, reference, labels) < compute_causal_npo_loss(
        current, reference, labels
    )


@pytest.mark.parametrize("weights", [(0.7, 1.4), (0.0, 1.0), (1.0, 0.0)])
def test_retain_terms_weighting_and_gradient_sum(weights):
    current, reference, labels = fixture()
    current.requires_grad_()
    retain = (current.detach() + 0.2).requires_grad_()
    terms = compute_causal_npo_rt_loss(
        current, reference, labels, retain, labels, npo_coeff=weights[0], retain_coeff=weights[1]
    )
    tokens = [
        -retain[b, t - 1].log_softmax(-1)[labels[b, t]]
        for b in range(2)
        for t in range(1, 5)
        if labels[b, t] != -100
    ]
    ce = torch.stack(tokens).mean()
    expected = weights[0] * scalar_oracle(current, reference, labels, 0.1) + weights[1] * ce
    torch.testing.assert_close(terms.retain, ce)
    torch.testing.assert_close(terms.total, expected)
    actual_grads = torch.autograd.grad(terms.total, (current, retain))
    expected_grads = torch.autograd.grad(expected, (current, retain))
    for actual, wanted in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual, wanted)


@pytest.mark.parametrize("beta", [0, -0.1, float("nan"), float("inf")])
def test_invalid_beta_rejected(beta):
    with pytest.raises(ValueError, match="beta"):
        compute_causal_npo_loss(*fixture(), beta=beta)


@pytest.mark.parametrize(
    "bad", ["shape", "empty", "label_dtype", "negative_id", "large_id", "no_targets"]
)
def test_invalid_batches_fail_closed(bad):
    current, reference, labels = fixture()
    if bad == "shape":
        reference = reference[:, :-1]
    elif bad == "empty":
        current, reference, labels = current[:0], reference[:0], labels[:0]
    elif bad == "label_dtype":
        labels = labels.float()
    elif bad == "negative_id":
        labels[0, 2] = -1
    elif bad == "large_id":
        labels[0, 2] = 7
    else:
        labels[0] = -100
    with pytest.raises(ValueError):
        compute_causal_npo_loss(current, reference, labels)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.requires_gpu)])
def test_fp16_reductions_match_double_and_remain_finite(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    current, reference, labels = fixture(torch.float16, device)
    current = current.repeat(1, 64, 1).requires_grad_()
    reference, labels = reference.repeat(1, 64, 1), labels.repeat(1, 64)
    loss = compute_causal_npo_loss(current, reference, labels)
    expected = compute_causal_npo_loss(current.double(), reference.double(), labels)
    assert loss.dtype == torch.float32
    torch.testing.assert_close(loss.double(), expected, rtol=2e-5, atol=2e-5)
    gradient = torch.autograd.grad(loss, current)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


@pytest.mark.requires_gpu
def test_tiny_fp16_optimization_is_repeatable_on_one_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    assert torch.cuda.device_count() == 1

    def run():
        torch.manual_seed(42)
        model = torch.nn.Linear(7, 7, bias=False).cuda()
        inputs = torch.randn(2, 5, 7, device="cuda")
        labels = fixture(device="cuda")[2]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scaler = torch.amp.GradScaler("cuda", init_scale=128.0)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            reference = model(inputs).detach()
        losses = []
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(inputs)
                loss = compute_causal_npo_loss(logits, reference, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        assert all(math.isfinite(value) for value in losses)
        assert losses[-1] < losses[0]
        return model.weight.detach().clone(), losses

    first, second = run(), run()
    torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
    assert first[1] == second[1]


@pytest.mark.parametrize("coefficient", [-1, float("nan"), float("inf")])
def test_invalid_retain_coefficients_rejected(coefficient):
    current, reference, labels = fixture()
    with pytest.raises(ValueError, match="coefficients"):
        compute_causal_npo_rt_loss(
            current, reference, labels, current, labels, retain_coeff=coefficient
        )


def test_author_replay_rejects_unverified_code(tmp_path):
    from experiments.reproductions.NPO.validate_upstream import author_functions

    source = tmp_path / "TOFU"
    source.mkdir()
    (source / "data_module.py").write_text("raise RuntimeError('must not execute')")
    with pytest.raises(ValueError, match="pinned file hash"):
        author_functions(tmp_path)
