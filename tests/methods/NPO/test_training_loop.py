import copy
import json
from types import SimpleNamespace

import pytest
import torch

from experiments.reproductions.NPO.adapter import AdapterSpec, encode_rows
from experiments.reproductions.NPO.reference import ModelIdentity, ReferenceConfig
from experiments.reproductions.NPO.training_contract import linear_schedule
from experiments.reproductions.NPO.training_data import SourcePairDataset, source_batch_plan
from experiments.reproductions.NPO.training_loop import run_training


class Tokenizer:
    eos_token_id = pad_token_id = 9

    def convert_tokens_to_ids(self, name):
        return 9

    def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
        return [1, 2] if add_generation_prompt else [1, 2, int(messages[-1]["content"]), 9]


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(10, 3)
        self.head = torch.nn.Linear(3, 10)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(logits=self.head(self.embedding(input_ids)))


class CPUEngine(torch.nn.Module):
    """Real torch updates; only the DeepSpeed orchestration interface is emulated."""

    def __init__(self, model):
        super().__init__()
        self.module = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        self.optimizer.loss_scale = 1
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, linear_schedule)
        self.global_steps = self.skipped_steps = self.micro_steps = 0
        self.norm = None

    def forward(self, **kwargs):
        return self.module(**kwargs)

    def is_gradient_accumulation_boundary(self):
        return (self.micro_steps + 1) % 8 == 0

    def backward(self, loss):
        (loss / 8).backward()

    def step(self):
        if self.is_gradient_accumulation_boundary():
            self.norm = torch.linalg.vector_norm(
                torch.stack([p.grad.norm() for p in self.module.parameters()])
            ).item()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            self.global_steps += 1
        self.micro_steps += 1

    def get_global_grad_norm(self):
        return self.norm


def case():
    torch.manual_seed(42)
    candidate = TinyLM()
    reference = copy.deepcopy(candidate).half().eval().requires_grad_(False)
    identity = ModelIdentity("model", "a" * 40, "b" * 64, "tokenizer", "c" * 40, "d" * 64)
    spec = AdapterSpec("model", "a" * 40, "tokenizer", "c" * 40, "data", "e" * 40, max_length=500)
    forget = encode_rows([{"question": "Q", "answer": "4"}] * 200, "forget05", Tokenizer(), spec)
    retain = encode_rows([{"question": "Q", "answer": "5"}] * 3800, "retain95", Tokenizer(), spec)
    return (
        CPUEngine(candidate),
        reference,
        ReferenceConfig(identity, identity),
        SourcePairDataset(forget, retain),
    )


def test_real_cpu_optimization_follows_source_steps_and_leaves_reference_unchanged(tmp_path):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        engine, reference, config, data = case()
        initial = copy.deepcopy(engine.module.state_dict())
        ref_initial = copy.deepcopy(reference.state_dict())
        plan = source_batch_plan(200, 4, 8, 62, 1001)
        evaluation_steps = []

        def evaluate(step):
            assert not engine.training
            evaluation_steps.append(step)

        cursor, updates = run_training(
            engine,
            reference,
            data,
            plan,
            config,
            tmp_path / "training",
            pad_token_id=9,
            evaluate=evaluate,
            observe_memory=lambda: {"fixture_cpu": True},
            runtime=False,
        )
        assert cursor["optimizer_step"] == 62 and engine.micro_steps == 496
        assert evaluation_steps == list(range(6, 61, 6))
        assert all(x["gradient_norm"] > 0 for x in updates)
        assert all(not torch.equal(initial[k], v) for k, v in engine.module.state_dict().items())
        assert all(torch.equal(ref_initial[k], v) for k, v in reference.state_dict().items())
        assert all(p.grad is None for p in reference.parameters())
        records = [
            json.loads(x)
            for x in (tmp_path / "training/microbatches.jsonl").read_text().splitlines()
        ]
        assert len(records) == 496
        assert records[48]["epoch"] == 0 and records[50]["epoch"] == 1
        assert records[55]["optimizer_step"] == 7
        assert records[0]["lr_before_step"] == [0.0]
        assert updates[-1]["total_loss"] != updates[0]["total_loss"]
    finally:
        torch.set_num_threads(old_threads)


def test_plan_mutation_fails_before_output_or_optimizer_step(tmp_path):
    engine, reference, config, data = case()
    plan = source_batch_plan(200, 4, 8, 62, 1001)
    plan["batches"][0]["forget_indices"].reverse()
    with pytest.raises(ValueError, match="batch plan"):
        run_training(
            engine,
            reference,
            data,
            plan,
            config,
            tmp_path / "never",
            pad_token_id=9,
            evaluate=lambda _: None,
            observe_memory=lambda: {},
            runtime=False,
        )
    assert engine.micro_steps == 0 and not (tmp_path / "never").exists()


def test_backward_failure_preserves_forward_without_claiming_update(tmp_path, monkeypatch):
    engine, reference, config, data = case()
    plan = source_batch_plan(200, 4, 8, 62, 1001)

    def fail(loss):
        raise RuntimeError("injected backward failure")

    monkeypatch.setattr(engine, "backward", fail)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="backward failure"):
        run_training(
            engine,
            reference,
            data,
            plan,
            config,
            output,
            pad_token_id=9,
            evaluate=lambda _: None,
            observe_memory=lambda: {},
            runtime=False,
        )
    observed = json.loads((output / "forward_observations.jsonl").read_text())
    assert observed["phase"] == "FORWARD_COMPLETE_BEFORE_BACKWARD"
    assert observed["microbatch"] == 0 and observed["total_loss"] > 0
    assert (output / "microbatches.jsonl").read_text() == ""
    assert engine.global_steps == 0
