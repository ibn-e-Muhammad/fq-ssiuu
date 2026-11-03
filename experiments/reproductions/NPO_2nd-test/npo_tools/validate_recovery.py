"""Tiny numerical/recovery tests. Never loads user models or modifies real runs."""
from pathlib import Path
from contextlib import ExitStack
import ast
import gc
import hashlib
import json
import math
import os
import random
import re
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent.parent
NB = json.loads((ROOT / "NPO_LOCALL_NOTE.ipynb").read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value), encoding="utf-8")


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


exec("".join(NB["cells"][16]["source"]), globals())


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.root = Path(self.temp.name)
        random.seed(123)
        np.random.seed(456)
        torch.manual_seed(789)
        torch.set_num_threads(1)

    def tearDown(self):
        self.temp.cleanup()

    def make(self, folder="state", resume=False, dtype=torch.float32, identity="fixture"):
        p = torch.nn.Parameter(torch.arange(12).reshape(3, 4).to(dtype)/10)
        optimizer = DiskMasterAdamW([("weight", p)], self.root / folder, resume=resume,
            chunk_elements=5, identity=identity)
        return p, optimizer

    def update(self, p, optimizer, metadata=None):
        # All three RNG sources affect subsequent updates.
        p.grad = torch.randn_like(p) * (random.random() + float(np.random.random()))
        optimizer.step(0.001, metadata=metadata)
        optimizer.zero_grad()

    def test_resume_exact_parameters_moments_rng_and_metadata(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                folder = str(dtype)
                p, optimizer = self.make(folder=folder, dtype=dtype)
                self.update(p, optimizer, {"next_group": 4})
                self.update(p, optimizer, {"next_group": 8})
                checkpoint_hashes = {k: file_sha256(v) for k, v in optimizer.paths.items()}
                rng = optimizer._runtime()
                # Save expected next update without publishing over the checkpoint.
                copy_dir = self.root / (folder + "-copy")
                shutil.copytree(optimizer.directory, copy_dir)
                self.update(p, optimizer)
                expected = p.detach().clone()
                expected_files = {k: file_sha256(v) for k, v in optimizer.paths.items()}
                torch.rand(31)
                if torch.cuda.is_available():
                    torch.rand(31, device="cuda")
                random.random()
                np.random.random()
                resumed_p, resumed = self.make(folder=folder + "-copy", resume=True, dtype=dtype)
                self.assertEqual(resumed.step_count, 2)
                self.assertEqual(resumed.metadata, {"next_group": 8})
                self.assertEqual(checkpoint_hashes, {k: file_sha256(v) for k, v in resumed.paths.items()})
                torch.testing.assert_close(torch.get_rng_state(), rng["torch_cpu"], rtol=0, atol=0)
                for actual, expected_rng in zip(torch.cuda.get_rng_state_all(), rng["torch_cuda"]):
                    torch.testing.assert_close(actual, expected_rng, rtol=0, atol=0)
                self.update(resumed_p, resumed)
                torch.testing.assert_close(resumed_p, expected, rtol=0, atol=0)
                self.assertEqual(expected_files, {k: file_sha256(v) for k, v in resumed.paths.items()})
                self.assertEqual(len(resumed._generations()), 2)

    def test_interrupted_chunk_write_preserves_last_commit(self):
        p, optimizer = self.make()
        self.update(p, optimizer)
        saved = p.detach().clone()
        original = DiskMasterAdamW._append
        calls = 0

        def fail_during_write(*args):
            nonlocal calls
            calls += 1
            if calls == 5:
                raise KeyboardInterrupt("simulated power loss during state write")
            return original(*args)

        with patch.object(DiskMasterAdamW, "_append", staticmethod(fail_during_write)):
            with self.assertRaises(KeyboardInterrupt):
                self.update(p, optimizer)
        rp, resumed = self.make(resume=True)
        self.assertEqual(resumed.step_count, 1)
        torch.testing.assert_close(rp, saved, rtol=0, atol=0)

    def test_missing_commit_and_corrupt_newest_fall_back(self):
        p, optimizer = self.make()
        self.update(p, optimizer)
        saved = p.detach().clone()
        self.update(p, optimizer)
        with optimizer.paths["moment1"].open("r+b") as handle:
            handle.write(b"BAD!")
        optimizer._candidate(3)  # cut before a completion marker exists
        rp, resumed = self.make(resume=True)
        self.assertEqual(resumed.step_count, 1)
        torch.testing.assert_close(rp, saved, rtol=0, atol=0)

    def test_cut_before_commit_marker(self):
        p, optimizer = self.make()
        saved = p.detach().clone()
        with patch.dict(globals(), {"durable_json": lambda *args: (_ for _ in ()).throw(OSError("power cut"))}):
            with self.assertRaises(OSError):
                self.update(p, optimizer)
        rp, resumed = self.make(resume=True)
        self.assertEqual(resumed.step_count, 0)
        torch.testing.assert_close(rp, saved, rtol=0, atol=0)

    def test_configuration_mismatch_and_nonfinite_rejected(self):
        p, optimizer = self.make()
        with self.assertRaisesRegex(ValueError, "No valid compatible"):
            self.make(resume=True, identity="different-data")
        p.grad = torch.full_like(p, float("nan"))
        with self.assertRaises(FloatingPointError):
            optimizer.step(0.001)
        self.assertEqual(optimizer.step_count, 0)
        self.assertEqual(len(optimizer._generations()), 1)

    def test_buffers_restored(self):
        p = torch.nn.Parameter(torch.ones(2))
        buf = torch.tensor([9.])
        opt = DiskMasterAdamW([("w", p)], self.root / "buffers", buffers=[("b", buf)])
        buf.fill_(17.)
        p.grad = torch.ones_like(p)
        opt.step(0.1)
        buf.zero_()
        DiskMasterAdamW([("w", p)], self.root / "buffers", buffers=[("b", buf)], resume=True)
        self.assertEqual(buf.item(), 17.)

    def test_completed_evaluation_reused_and_hash_checked(self):
        tree = ast.parse("".join(NB["cells"][11]["source"]))
        definition = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate_stage")
        stage = self.root / "full_bf16"
        stage.mkdir()
        frame = pd.DataFrame([{"stage": "full_bf16", "question": "fixture", "answer_nll": 1.25}])
        frame.to_json(stage / "per_question.jsonl", orient="records", lines=True)
        reference = {"training_sha256": "training", "reference_weights_sha256": "full", "nll": [1.25]}
        json_write(self.root / "frozen_full_reference.json", reference)
        json_write(stage / "metadata.json", {"status": "complete", "evaluation_sha256": "eval",
            "model_path": str(self.root / "model"), "quantization": None,
            "artifact_hashes": {"per_question.jsonl": file_sha256(stage / "per_question.jsonl")},
            "reference_sha256": digest(reference)})
        ns = dict(globals(), RUN_DIR=self.root, STAGE_RESULTS={}, STAGE_META={}, EVAL_DIGEST="eval",
            TRAIN_DIGEST="training", MODEL_HASHES={"full": "full"},
            load_model=lambda *args, **kwargs: self.fail("Cached evaluation must not reload the model"))
        exec(compile(ast.Module(body=[definition], type_ignores=[]), "evaluation", "exec"), ns)
        actual = ns["evaluate_stage"]("full_bf16", self.root / "model", cache_reference=True)
        pd.testing.assert_frame_equal(actual, frame)
        ns["STAGE_RESULTS"].clear()
        with (stage / "per_question.jsonl").open("a") as handle:
            handle.write("changed")
        with self.assertRaisesRegex(AssertionError, "Cached evaluation changed"):
            ns["evaluate_stage"]("full_bf16", self.root / "model", cache_reference=True)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA integration check")
    def test_actual_training_loop_interruption_resume_and_final_export(self):
        from safetensors.torch import save_file, load_file

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(11, 8)
                self.head = torch.nn.Linear(8, 11, bias=False)
                from types import SimpleNamespace
                self.config = SimpleNamespace(attention_dropout=0, use_cache=True)

            def forward(self, input_ids, attention_mask=None, use_cache=False):
                from types import SimpleNamespace
                return SimpleNamespace(logits=self.head(self.emb(input_ids)))

            def gradient_checkpointing_enable(self, **kwargs):
                pass

            def gradient_checkpointing_disable(self):
                pass

            def save_pretrained(self, directory, **kwargs):
                folder = Path(directory)
                folder.mkdir()
                save_file(self.state_dict(), str(folder / "model.safetensors"))

        class Tokenizer:
            def save_pretrained(self, directory):
                json_write(Path(directory) / "tokenizer_config.json", {})

        initial = Tiny().state_dict()

        def load_model(*args, **kwargs):
            model = Tiny().cuda()
            model.load_state_dict(initial)
            return model

        def forward_nll(model, example):
            ids = torch.tensor([example["input_ids"]], device="cuda")
            labels = torch.tensor([example["labels"]], device="cuda")
            logits = model(ids).logits[:, :-1].float()
            return F.cross_entropy(logits.reshape(-1, 11), labels[:, 1:].reshape(-1),
                ignore_index=-100, reduction="sum").reshape(1), 2

        @torch.inference_mode()
        def score_one(model, example):
            nll, n = forward_nll(model, example)
            return nll.item(), n, nll.item()/n

        def npo_from_sequence_nll(current, reference, beta):
            return 2 / beta * F.softplus(beta * (reference-current)).mean()

        data = [{"input_ids": [1, i % 9 + 1, (i+1) % 9 + 1], "labels": [-100, i % 9 + 1, (i+1) % 9 + 1]} for i in range(8)]
        settings = dict(REUSE_NPO_PATH=None, RESUME_RUN_PATH=None, TRAIN_DIGEST="tiny-data-v2",
            TRAIN_IDENTITY={"fixture": True}, DEVICE=torch.device("cuda"), TRAIN_FORGET=data,
            TRAIN_RETAIN=data, RETAIN_PAIRS=list(range(8)), MODEL_HASHES={"full": "fixture"},
            NOMINAL_EPOCHS=2, TRAIN_GROUP=2, ACCUM_GROUPS=2, WARMUP_STEPS=1, SAMPLER_SEED=1001,
            SEED=42, LR=1e-3, BETA=0.1, NPO_COEFF=1., RETAIN_COEFF=1., OBJECTIVE="NPO+RT",
            GRADIENT_CHECKPOINTING=True, ADAM_BETAS=(0.9, 0.999), ADAM_EPS=1e-8, WEIGHT_DECAY=0.01,
            CPU_CHUNK_ELEMENTS=13, OPTIMIZER_CPU_THREADS=1, ENVIRONMENT={"packages": {"torch": torch.__version__}},
            FULL_DIR=self.root / "full", load_model=load_model, forward_nll=forward_nll,
            score_one=score_one, npo_from_sequence_nll=npo_from_sequence_nll,
            release_memory=lambda: None, tokenizer=Tokenizer())
        refs_model = load_model()
        cached = {"training_sha256": "tiny-data-v2", "reference_weights_sha256": "fixture",
                  "nll": [score_one(refs_model, example)[0] for example in data]}
        del refs_model
        ns = dict(globals(), **settings)
        code = "".join(NB["cells"][17]["source"]).split("training_receipt = train_npo()")[0]
        exec(code, ns)
        outputs = []
        for name in ("uninterrupted", "resumed", "export_resumed"):
            run = self.root / name
            run.mkdir()
            json_write(run / "frozen_full_reference.json", cached)
            ns.update(RUN_DIR=run, NPO_DIR=run / "npo_model", RESUME_RUN_PATH=None)
            if name == "resumed":
                original = DiskMasterAdamW.step

                def fail_third(opt, *args, **kwargs):
                    if opt.step_count == 2:
                        raise KeyboardInterrupt("simulated interruption")
                    return original(opt, *args, **kwargs)

                with patch.object(DiskMasterAdamW, "step", fail_third):
                    with self.assertRaises(KeyboardInterrupt):
                        ns["train_npo"]()
                # Simulate stray log text from work after the last committed update.
                with (run / "training_log.jsonl").open("a") as handle:
                    handle.write("incomplete log tail")
                ns["RESUME_RUN_PATH"] = str(run)
            if name == "export_resumed":
                def interrupted_export(model, directory, **kwargs):
                    Path(directory).mkdir()
                    (Path(directory) / "partial.bin").write_bytes(b"partial model export")
                    raise KeyboardInterrupt("cut during final model export")
                with patch.object(Tiny, "save_pretrained", interrupted_export):
                    with self.assertRaises(KeyboardInterrupt):
                        ns["train_npo"]()
                ns["RESUME_RUN_PATH"] = str(run)
            receipt = ns["train_npo"]()
            self.assertEqual(receipt["updates"], 4)
            log = pd.read_json(run / "training_log.jsonl", lines=True)
            self.assertEqual(log["update"].tolist(), [1, 2, 3, 4])
            outputs.append(load_file(str(run / "npo_model/model.safetensors")))
            # Completed model reuse should not enter the training loop.
            ns["RESUME_RUN_PATH"] = str(run)
            self.assertEqual(ns["train_npo"](), receipt)
        for output in outputs[1:]:
            for key in outputs[0]:
                torch.testing.assert_close(outputs[0][key], output[key], rtol=0, atol=0)


if __name__ == "__main__":
    for i, cell in enumerate(NB["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"cell_{i}")
    unittest.main(verbosity=2)
