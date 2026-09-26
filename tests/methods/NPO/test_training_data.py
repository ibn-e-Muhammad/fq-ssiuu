import pytest
import torch
from torch.utils.data import DataLoader

from experiments.reproductions.common import tensors_digest
from experiments.reproductions.NPO.candidate import compute_causal_npo_rt_loss
from experiments.reproductions.NPO.training_data import (
    SourcePairDataset,
    fixed_batch,
    source_batch_plan,
)


def example(index, ids):
    tensors = {
        "input_ids": torch.tensor(ids),
        "attention_mask": torch.ones(len(ids), dtype=torch.long),
        "labels": torch.tensor([-100, -100] + ids[2:]),
    }
    return {
        **tensors,
        "sample_id": str(index),
        "preprocessing_sha256": "a" * 64,
        "tensors_sha256": tensors_digest(tensors),
        "truncation": {"truncated": False},
        "prompt_token_count": 2,
        "scored_token_count": len(ids) - 2,
        "identities": {"test": True},
    }


def test_pairing_repeats_source_seed_and_preserves_final_rng():
    before = torch.get_rng_state()
    try:
        dataset = SourcePairDataset(list(range(200)), list(range(3800)))
        for i in [0, 1, 100, 199]:
            torch.manual_seed(i)
            expected = (i + torch.randint(0, 3800, (1,)).item()) % 3800
            rng = torch.get_rng_state()
            torch.manual_seed(42)
            result = dataset[i]
            assert result == {
                "forget": i,
                "retain": expected,
                "forget_index": i,
                "retain_index": expected,
            }
            assert torch.equal(rng, torch.get_rng_state())
    finally:
        torch.set_rng_state(before)


def test_fixed_source_padding_masks_prompt_and_padding_but_scores_eos():
    short = example(1, [1, 2, 3, 9])
    long = example(2, [1, 2, 4, 5, 9])
    batch = fixed_batch([short, long, short], max_length=6, pad_token_id=9)
    assert batch["sample_ids"] == ["1", "2", "1"]  # repeated retain draws must survive
    assert batch["tensors"]["labels"].tolist() == [
        [-100, -100, 3, 9, -100, -100],
        [-100, -100, 4, 5, 9, -100],
        [-100, -100, 3, 9, -100, -100],
    ]
    assert batch["tensors"]["attention_mask"].tolist() == [
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 0, 0],
    ]
    assert batch["tensors"]["input_ids"][0].tolist() == [1, 2, 3, 9, 9, 9]


@pytest.mark.parametrize("defect", ["prompt", "truncated", "mutation", "count"])
def test_invalid_training_example_fails(defect):
    row = example(1, [1, 2, 3, 9])
    if defect == "prompt":
        row["labels"][0] = 1
        row["tensors_sha256"] = tensors_digest(
            {k: row[k] for k in ("input_ids", "labels", "attention_mask")}
        )
    elif defect == "truncated":
        row["truncation"]["truncated"] = True
    elif defect == "count":
        row["scored_token_count"] += 1
    else:
        row["input_ids"][0] += 1
    with pytest.raises(ValueError):
        fixed_batch([row], max_length=6, pad_token_id=9)


def test_continuous_plan_keeps_partial_epoch_tail_and_exact_source_order():
    plan = source_batch_plan(200, 4, 8, 62, 1001)
    loader = DataLoader(
        list(range(200)), batch_size=4, shuffle=True, generator=torch.Generator().manual_seed(1001)
    )
    expected = [b.tolist() for _ in range(10) for b in loader][:496]
    assert [b["forget_indices"] for b in plan["batches"]] == expected
    assert len(plan["batches"]) == 496
    # Update 7 spans epoch 0's last two batches and epoch 1's first six.
    assert [b["epoch"] for b in plan["batches"][48:56]] == [0, 0, 1, 1, 1, 1, 1, 1]
    assert plan["sha256"] == source_batch_plan(200, 4, 8, 62, 1001)["sha256"]
    assert plan["sha256"] != source_batch_plan(200, 4, 8, 62, 1001 + 1)["sha256"]


def test_accumulation_preserves_average_microbatch_retain_normalization():
    # Eight source microbatches; compare two-rank x four accumulation against
    # approved one-rank x eight. Heterogeneous answer lengths make incorrect
    # global-token averaging detectably different.
    gen = torch.Generator().manual_seed(1001)
    weight = torch.randn(5, 7, dtype=torch.double, generator=gen, requires_grad=True)
    copy_weight = weight.detach().clone().requires_grad_(True)
    micros = []
    for n in range(8):
        x = torch.randn(4, 5, 5, dtype=torch.double, generator=gen)
        labels = torch.tensor([[-100, -100, 2, 3, 4]] * 4)
        retain_labels = labels.clone()
        if n % 2:
            retain_labels[:, 3:] = -100
        ref = torch.randn(4, 5, 7, dtype=torch.double, generator=gen)
        micros.append((x, labels, retain_labels, ref))

    def loss(w, item):
        x, labels, retained, ref = item
        return compute_causal_npo_rt_loss(
            x @ w, ref, labels, (x + 0.1) @ w, retained, beta=0.1
        ).total

    two_rank = (
        sum(sum(loss(weight, micros[rank + 2 * i]) for i in range(4)) / 4 for rank in range(2)) / 2
    )
    two_rank.backward()
    one_rank_values = []
    for item in micros:
        part = loss(copy_weight, item) / 8
        one_rank_values.append(part.detach())
        part.backward()
    torch.testing.assert_close(sum(one_rank_values), two_rank.detach(), rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(copy_weight.grad, weight.grad, rtol=1e-12, atol=1e-12)


def test_plan_rejects_partial_microbatch_and_tiny_epoch_special_case():
    with pytest.raises(ValueError, match="complete"):
        source_batch_plan(201, 4, 8, 62, 1001)
    with pytest.raises(ValueError, match="Tiny-epoch"):
        source_batch_plan(32, 4, 8, 62, 1001)
