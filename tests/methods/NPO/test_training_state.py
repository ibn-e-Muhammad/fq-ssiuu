"""CPU state conformance: no pretrained weights, network, or DeepSpeed claim."""

import copy
import json
import random

import numpy as np
import pytest
import torch

from experiments.reproductions.NPO import training_state as state


def identities():
    return {
        key: "a" * (40 if key.endswith("revision") or key == "source_commit" else 64)
        for key in state.IDENTITY_KEYS
    }


def objects():
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 4), torch.nn.Dropout(0.2), torch.nn.Linear(4, 2)
    ).double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(s / 2, max(0, (8 - s) / 6))
    )
    return model, optimizer, scheduler


def cursor(step=3):
    return dict(
        optimizer_step=step,
        micro_step=0,
        epoch=0,
        batch_in_epoch=step * 2,
        sampler_state=torch.Generator().manual_seed(123).get_state(),
    )


def step(model, optimizer, scheduler):
    optimizer.zero_grad()
    # Draw from every saved CPU RNG; dropout adds a stochastic model operation.
    x = torch.randn(2, 3, dtype=torch.double) * random.random() + np.random.random()
    target = torch.randn(2, 2, dtype=torch.double)
    loss = (model(x) - target).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return float(loss.detach())


def same(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            same(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            same(x, y)
    else:
        assert a == b


def test_resume_reproduces_optimizer_scheduler_rng_and_trajectory_exactly(tmp_path):
    before = state.capture_rng()
    try:
        random.seed(1001)
        np.random.seed(1001)
        torch.manual_seed(1001)
        model, optimizer, scheduler = objects()
        for _ in range(3):
            step(model, optimizer, scheduler)
        path = tmp_path / "step3"
        manifest = state.save_training_state(
            path, model, optimizer, scheduler, cursor(), identities()
        )
        assert manifest["status"] == "COMPLETE"
        expected_losses = [step(model, optimizer, scheduler) for _ in range(4)]
        expected_rng = state.capture_rng()
        replacement = objects()
        loaded = state.load_training_state(path, *replacement, identities())
        same(loaded, cursor())
        assert [step(*replacement) for _ in range(4)] == expected_losses
        for original, resumed in zip((model, optimizer, scheduler), replacement, strict=True):
            same(original.state_dict(), resumed.state_dict())
        same(state.capture_rng(), expected_rng)
    finally:
        state.restore_rng(before)


@pytest.mark.parametrize("kind", ["identity", "bytes", "incomplete", "inner_identity", "rng"])
def test_bad_checkpoint_rejected_before_model_mutation(tmp_path, kind):
    original = objects()
    path = tmp_path / "checkpoint"
    state.save_training_state(path, *original, cursor(), identities())
    supplied = identities()
    manifest = json.loads((path / "MANIFEST.json").read_text())
    if kind == "identity":
        supplied["train_config_sha256"] = "b" * 64
    elif kind == "bytes":
        with (path / "training_state.pt").open("ab") as handle:
            handle.write(b"bad")
    elif kind == "incomplete":
        manifest["status"] = "PARTIAL"
    else:
        payload = torch.load(path / "training_state.pt", weights_only=True)
        if kind == "inner_identity":
            payload["identities"]["source_commit"] = "c" * 40
        else:
            payload["rng"]["cuda"] = [torch.tensor([1], dtype=torch.uint8)]
        torch.save(payload, path / "training_state.pt")
        manifest["state_sha256"] = state.file_digest(path / "training_state.pt")
        manifest["state_bytes"] = (path / "training_state.pt").stat().st_size
    (path / "MANIFEST.json").write_text(json.dumps(manifest))
    replacement = objects()
    before = copy.deepcopy(replacement[0].state_dict())
    with pytest.raises(ValueError):
        state.load_training_state(path, *replacement, supplied)
    same(before, replacement[0].state_dict())


def test_partial_write_retained_and_existing_checkpoint_never_overwritten(tmp_path, monkeypatch):
    model, optimizer, scheduler = objects()

    def fail(*args, **kwargs):
        raise OSError("simulated full disk")

    monkeypatch.setattr(torch, "save", fail)
    path = tmp_path / "failed"
    with pytest.raises(OSError):
        state.save_training_state(path, model, optimizer, scheduler, cursor(), identities())
    assert json.loads((path / "FAILURE.json").read_text())["status"] == "PARTIAL"
    assert not (path / "MANIFEST.json").exists()
    with pytest.raises(FileExistsError):
        state.save_training_state(path, model, optimizer, scheduler, cursor(), identities())


def test_unsafe_reference_and_pending_gradients_cannot_be_saved(tmp_path):
    ids = identities()
    ids["reference_checkpoint_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="original"):
        state.save_training_state(tmp_path / "wrong-reference", *objects(), cursor(), ids)
    with pytest.raises(ValueError, match="pending gradients"):
        state.save_training_state(
            tmp_path / "pending", *objects(), {**cursor(), "micro_step": 1}, identities()
        )
    assert list(tmp_path.iterdir()) == []


class FakeEngine:
    """Exercises file/identity control flow; genuine ZeRO parity is a separate CLI check."""

    global_steps = 3
    loaded = False

    def save_checkpoint(self, directory, *, tag, client_state, save_latest):
        assert tag == "state" and save_latest is False
        root = state.Path(directory) / tag
        root.mkdir()
        torch.save(client_state, root / "model_states.pt")
        torch.save({"test_moments": torch.ones(2)}, root / "optim_states.pt")
        return True

    def load_checkpoint(self, directory, **kwargs):
        self.loaded = True
        assert kwargs == dict(
            tag="state",
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
            load_module_strict=True,
        )
        return directory, torch.load(
            state.Path(directory) / "state/model_states.pt", weights_only=True
        )


def test_deepspeed_manifest_roundtrip_and_no_overwrite(tmp_path):
    engine = FakeEngine()
    path = tmp_path / "checkpoint"
    state.save_deepspeed_state(path, engine, cursor(), identities())
    same(state.load_deepspeed_state(path, engine, identities()), cursor())
    assert engine.loaded
    with pytest.raises(FileExistsError):
        state.save_deepspeed_state(path, engine, cursor(), identities())


@pytest.mark.parametrize("defect", ["bytes", "extra", "identity", "path"])
def test_deepspeed_rejects_bad_artifacts_before_invoking_loader(tmp_path, defect):
    engine = FakeEngine()
    path = tmp_path / "checkpoint"
    state.save_deepspeed_state(path, engine, cursor(), identities())
    supplied = identities()
    if defect == "bytes":
        (path / "state/optim_states.pt").write_bytes(b"bad")
    elif defect == "extra":
        (path / "unrecorded.txt").write_text("not in manifest")
    elif defect == "identity":
        supplied["source_commit"] = "d" * 40
    else:
        record = json.loads((path / "MANIFEST.json").read_text())
        record["files"][0]["path"] = "../outside"
        (path / "MANIFEST.json").write_text(json.dumps(record))
    with pytest.raises(ValueError):
        state.load_deepspeed_state(path, engine, supplied)
    assert not engine.loaded


def test_deepspeed_failed_save_has_partial_record(tmp_path):
    class Broken(FakeEngine):
        def save_checkpoint(self, *args, **kwargs):
            return False

    path = tmp_path / "failed"
    with pytest.raises(RuntimeError):
        state.save_deepspeed_state(path, Broken(), cursor(), identities())
    assert json.loads((path / "FAILURE.json").read_text())["status"] == "PARTIAL"
    assert not (path / "MANIFEST.json").exists()
