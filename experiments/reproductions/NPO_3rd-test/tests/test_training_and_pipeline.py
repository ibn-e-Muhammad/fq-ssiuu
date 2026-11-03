import json

import pandas as pd
import pytest
import torch
from npo3.checkpoint import checkpoint_candidates, restore_checkpoint
from npo3.data import prepare_examples
from npo3.evaluation import tensor_fingerprint
from npo3.metrics import sha256
from npo3.training import attach_lora, train


def test_resume_matches_uninterrupted_and_frozen_backbone(tiny, tmp_path):
    cfg, tokenizer, data, make_model = tiny
    forget, retain, _, _ = prepare_examples(data, tokenizer, cfg)
    full = attach_lora(make_model(), cfg, 42)
    train(full, tokenizer, forget, retain, cfg, 42, tmp_path / "full", {"fixture": True})
    interrupted = attach_lora(make_model(), cfg, 42)
    before = tensor_fingerprint(interrupted, skip_adapters=True)
    train(
        interrupted,
        tokenizer,
        forget,
        retain,
        cfg,
        42,
        tmp_path / "resumed",
        {"fixture": True},
        stop_after=2,
    )
    assert not (tmp_path / "resumed/training_complete.json").exists()
    resumed = attach_lora(make_model(), cfg, 42)
    receipt = train(
        resumed,
        tokenizer,
        forget,
        retain,
        cfg,
        42,
        tmp_path / "resumed",
        {"fixture": True},
        resume=True,
    )
    assert receipt["base_unchanged"]
    assert before == tensor_fingerprint(resumed, skip_adapters=True)
    for name, expected in full.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name], expected, atol=0, rtol=0)
    assert len(checkpoint_candidates(tmp_path / "resumed/checkpoints")) == 2


def test_corrupt_checkpoint_falls_back_and_identity_is_checked(tiny, tmp_path):
    cfg, tokenizer, data, make_model = tiny
    forget, retain, _, _ = prepare_examples(data, tokenizer, cfg)
    model = attach_lora(make_model(), cfg, 42)
    result = train(
        model,
        tokenizer,
        forget,
        retain,
        cfg,
        42,
        tmp_path / "train",
        {"fixture": True},
        stop_after=2,
    )
    latest = checkpoint_candidates(tmp_path / "train/checkpoints")[0][1]
    (latest / "runtime.pt").write_bytes(b"corruption")
    fresh = attach_lora(make_model(), cfg, 42)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for p in fresh.parameters() if p.requires_grad],
                "weight_decay": cfg.weight_decay,
            },
            {"params": [], "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate,
    )
    step, _, rejected = restore_checkpoint(
        tmp_path / "train/checkpoints", fresh, optimizer, result["identity"]
    )
    assert step == 1 and rejected
    with pytest.raises(ValueError, match="compatible"):
        restore_checkpoint(tmp_path / "train/checkpoints", fresh, optimizer, "different experiment")


def test_completed_pipeline_generates_all_outputs_and_reuses_evaluation(
    tiny, tmp_path, monkeypatch
):
    import run_experiment as experiment

    cfg, tokenizer, data, make_model = tiny
    full, retain = tmp_path / "full", tmp_path / "retain"
    make_model().save_pretrained(full, safe_serialization=True)
    make_model().save_pretrained(retain, safe_serialization=True)
    paths = {"full": full, "retain95": retain, "tokenizer": full}
    monkeypatch.setattr(
        experiment, "resolve_inputs", lambda _: (paths, tokenizer, data, {"fixture": True})
    )
    root = experiment.run(cfg)
    assert json.loads((root / "run_complete.json").read_text())["status"] == "complete"
    seed = root / "seed_42"
    for name in (
        "comparison_matrix.csv",
        "forget_quality_fixed_and_matched.csv",
        "paired_directional_recovery.csv",
        "answers_side_by_side.html",
        "human_audit_template.csv",
        "epoch_metrics.csv",
        "merge_drift.csv",
    ):
        assert (seed / name).is_file()
    assert len(pd.read_csv(seed / "epoch_metrics.csv")) == 2
    primary = pd.read_csv(seed / "npo_bf16/per_question.csv")
    assert len(primary) == 15
    assert primary.eos_terminated.notna().all()
    assert (seed / "npo_bf16/legacy_greedy96/per_question.csv").is_file()
    assert (seed / "npo_bf16/long_greedy/per_question.csv").is_file()
    assert pd.read_csv(seed / "human_audit_template.csv").category.isna().all()
    immutable = sha256(seed / "npo_bf16/per_question.csv")
    experiment.run(cfg, resume_dir=root)
    assert sha256(seed / "npo_bf16/per_question.csv") == immutable
    # Tampering is not silently accepted as a cached completed result.
    with (seed / "npo_bf16/per_question.csv").open("a") as handle:
        handle.write("changed")
    with pytest.raises(ValueError, match="modified"):
        experiment.run(cfg, resume_dir=root)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Actual CUDA kernels require the target GPU"
)
def test_cuda_lora_backward_merge_and_real_quantization(tiny, tmp_path):
    pytest.importorskip("bitsandbytes")
    from npo3.evaluation import model_loader
    from npo3.objective import npo_loss
    from npo3.training import forward_sequence, frozen_reference_nll
    from run_experiment import merge_export, quantization_preflight

    cfg, tokenizer, data, make_model = tiny
    cfg.device, cfg.dtype = "cuda:0", "bfloat16"
    cfg.quantization_modes = ["int8", "nf4"]
    quantization_preflight(cfg)
    forget, _, _, _ = prepare_examples(data, tokenizer, cfg)
    model = attach_lora(make_model().to(device=cfg.device, dtype=torch.bfloat16), cfg, 42)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    current, _ = forward_sequence(model, forget[:2], tokenizer, cfg)
    frozen = frozen_reference_nll(model, forget[:2], tokenizer, cfg)
    npo_loss(current, frozen).backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
        if p.requires_grad
    )
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    optimizer.step()
    merged, _ = merge_export(model, tokenizer, forget, cfg, tmp_path / "merged")
    del merged, model, optimizer
    for mode in cfg.quantization_modes:
        quantized = model_loader(tmp_path / "merged", cfg, mode)
        values, _ = forward_sequence(quantized, forget[:1], tokenizer, cfg)
        assert torch.isfinite(values).all()
        del quantized
