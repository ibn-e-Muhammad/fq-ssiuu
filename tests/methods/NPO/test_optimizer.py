from types import SimpleNamespace

import pytest
import torch

from experiments.reproductions.NPO.optimizer import parameter_groups, restore_paged_moments


def test_restored_moments_use_original_allocator_with_identical_values():
    p = torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3))
    original = {"state1": p.detach().clone() / 7, "state2": p.detach().clone() ** 2, "step": 11}
    calls = []

    def buffer(parameter, dtype):
        calls.append(parameter)
        result = torch.empty_like(parameter, dtype=dtype)
        result.test_managed_allocation = True  # CPU allocator double, not a CUDA claim.
        return result

    optimizer = SimpleNamespace(
        is_paged=True,
        args=SimpleNamespace(optim_bits=32),
        param_groups=[{"params": [p]}],
        state={p: original.copy()},
        get_state_buffer=buffer,
    )
    assert restore_paged_moments(optimizer) == 2
    assert len(calls) == 2 and all(x is p for x in calls)
    assert optimizer.state[p]["step"] == 11
    for key in ("state1", "state2"):
        assert optimizer.state[p][key].test_managed_allocation
        assert optimizer.state[p][key].data_ptr() != original[key].data_ptr()
        torch.testing.assert_close(optimizer.state[p][key], original[key], rtol=0, atol=0)
    optimizer.args.optim_bits = 8
    with pytest.raises(ValueError, match="32-bit"):
        restore_paged_moments(optimizer)


def test_source_weight_decay_excludes_norm_and_bias_but_not_embedding():
    model = torch.nn.Sequential(
        torch.nn.Embedding(5, 4), torch.nn.LayerNorm(4), torch.nn.Linear(4, 3)
    )
    groups, inventory = parameter_groups(model, 0.01)
    expected = {"0.weight": 0.01, "1.weight": 0.0, "1.bias": 0.0, "2.weight": 0.01, "2.bias": 0.0}
    assert {x["name"]: x["weight_decay"] for x in inventory} == expected
    assert sum(len(g["params"]) for g in groups) == len(list(model.parameters()))
    model[0].weight.requires_grad_(False)
    with pytest.raises(ValueError, match="every"):
        parameter_groups(model, 0.01)
