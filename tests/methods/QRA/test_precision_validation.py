import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.reproductions.QRA.author_clusters import author_mapping, validate_author_mapping
from experiments.reproductions.QRA.validate_runtime import (
    expected_modules,
    make_observations,
    run_sequential_workers,
)
from experiments.reproductions.QRA.validation_contract import expected_recipe
from fq_ssiuu.evaluation.certificate import generate_report
from fq_ssiuu.evaluation.content_identity import oracle_manifest
from fq_ssiuu.evaluation.derived_summary import utility_summary
from fq_ssiuu.evaluation.precision_artifact import build_artifact
from scripts.mother_runner import load_recipe
from tests.methods.QRA.fixtures import group


def test_c1_exact_recipe_runs_only_forward_validation(tmp_path):
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(expected_recipe()))
    assert load_recipe(path) == expected_recipe()
    for field, value in [
        ("backward", True),
        ("optimizer_steps", 1),
        ("evaluation_precisions", ["nf4"]),
        ("sample_indices", {"forget05_perturbed": [0, 1]}),
        ("overlay_lock", "requirements/kaggle-overlay.validated.lock"),
    ]:
        recipe = expected_recipe()
        recipe[field] = value
        path.write_text(json.dumps(recipe))
        with pytest.raises(ValueError):
            load_recipe(path)


def test_full_pinned_llama_inventory_is_exact():
    names = expected_modules({"model_type": "llama", "num_hidden_layers": 16})
    assert len(names) == 112 and len(set(names)) == 112
    assert "model.layers.0.self_attn.q_proj" in names
    assert "model.layers.15.mlp.down_proj" in names
    assert "lm_head" not in names
    with pytest.raises(ValueError):
        expected_modules({"model_type": "other"})


def test_sequential_process_exit_releases_retained_objects_before_next_worker(tmp_path):
    # A global retaining a model-like object deliberately survives garbage
    # collection. Process exit, unlike del/empty_cache, ends its lifetime.
    program = "import gc,weakref; C=type('Model',(),{}); held=C(); ref=weakref.ref(held); gc.collect(); assert ref() is held"
    records = run_sequential_workers(
        [(p, [sys.executable, "-c", program]) for p in ("fp16", "int8", "nf4")], tmp_path, 10
    )
    assert len({r["pid"] for r in records}) == 3
    assert all(r["process_reaped"] and r["returncode"] == 0 for r in records)
    assert all(a["finished_unix"] <= b["started_unix"] for a, b in zip(records, records[1:]))


def test_worker_failure_blocks_later_precision_and_preserves_status(tmp_path):
    commands = [
        ("fp16", [sys.executable, "-c", "raise SystemExit(7)"]),
        ("nf4", [sys.executable, "-c", "raise AssertionError('must not run')"]),
    ]
    with pytest.raises(RuntimeError, match="fp16"):
        run_sequential_workers(commands, tmp_path, 10)
    assert json.loads((tmp_path / "fp16_execution.json").read_text())["returncode"] == 7
    assert not (tmp_path / "nf4_execution.json").exists()


@pytest.mark.parametrize("precision", ["int8", "nf4"])
def test_actual_quantized_state_capture_without_cuda_dependency(precision):
    class QuantState:
        quant_type = "nf4"
        blocksize = 64

        def as_dict(self, packed=False):
            return {
                "absmax": torch.tensor([2.0]),
                "quant_map": torch.linspace(-1, 1, 16),
                "quant_type": "nf4",
                "blocksize": 64,
            }

    class Packed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.tensor([1, 2], dtype=torch.int8 if precision == "int8" else torch.uint8),
                requires_grad=False,
            )
            self.weight.quant_state = QuantState()
            self.compute_dtype = torch.float16
            self.state = SimpleNamespace(
                SCB=torch.tensor([2.0]), threshold=6.0, has_fp16_weights=False
            )

    Packed.__module__ = "bitsandbytes.nn.modules"
    Packed.__name__ = "Linear8bitLt" if precision == "int8" else "Linear4bit"
    model = torch.nn.ModuleDict({"proj": Packed()})
    fixture = group()["artifacts"][precision]
    result = build_artifact(
        model,
        precision=precision,
        checkpoint_sha256="a" * 64,
        versions={"torch": "fixture-1", "transformers": "fixture-1", "bitsandbytes": "fixture-1"},
        quantizer_config=fixture["quantizer_config"],
        expected_quantized_modules=["proj"],
    )
    assert result["quantized_modules"][0]["weight"]["bytes"] == 2
    assert (
        result["state_tensors"][0]["sha256"] == result["quantized_modules"][0]["weight"]["sha256"]
    )


def test_real_report_assembly_from_measurements_is_valid_and_null_decision():
    from experiments.reproductions.QRA.validate_runtime import evaluator_identity

    spec = SimpleNamespace(
        tokenizer_id="fixture/tokenizer",
        tokenizer_revision="b" * 40,
        dataset_id="fixture/tofu",
        dataset_revision="c" * 40,
        sha256="4" * 64,
    )
    rows = []
    for split, prefix in [
        ("forget05_perturbed", None),
        ("retain_perturbed", "retain"),
        ("real_authors_perturbed", "real_authors"),
        ("world_facts_perturbed", "world_facts"),
    ]:
        row = {
            "sample_id": split + "/0",
            "source_split": split,
            "source_record_sha256": "9" * 64,
            "tensors_sha256": "b" * 64,
            "generation": {"decoding_sha256": "5" * 64},
            "answer_log_probability": -1.0,
            "rouge_l_recall": 0.5,
            "exact_match": 0.0,
            "tofu_truth_ratio": 0.2,
            "openunlearning_truth_ratio": 0.1,
        }
        if prefix:
            row["utility_components"] = {
                prefix + "_" + m: 0.5
                for m in ("probability", "rouge_l_recall", "truth_ratio_score")
            }
        rows.append(row)
    oracle = oracle_manifest(
        checkpoint_sha256="6" * 64,
        model_revision="d" * 40,
        tokenizer_revision=spec.tokenizer_revision,
        dataset_revision=spec.dataset_revision,
        preprocessing_sha256=spec.sha256,
        evaluator_sha256=evaluator_identity(),
        records=[copy.deepcopy(rows[0])],
    )
    artifacts = {p: v["artifact_manifest"] for p, v in group()["artifacts"].items()}
    g, obs = make_observations(
        {p: copy.deepcopy(rows) for p in artifacts},
        artifacts,
        oracle,
        {
            "target": {
                "checkpoint_sha256": "a" * 64,
                "model_id": "fixture/model",
                "revision": "a" * 40,
            }
        },
        spec,
        "test-c1",
    )
    report = generate_report("test-c1", "c1", obs, g, bootstrap={"seed": 23, "resamples": 20})
    assert report["report_status"] == "VALID_DESCRIPTIVE_RECORD"
    assert (
        len(obs) == 48 and report["attack_detected"] is None and report["durability_pass"] is None
    )
    assert (
        utility_summary([o for o in obs if o.precision == "fp16" and o.kind == "utility_score"], g)[
            "value"
        ]
        == 0.5
    )


def test_author_mapping_uses_explicit_content_and_rejects_ambiguity():
    rows = [
        {"question": "A question", "answer": "Ada Example writes poems."},
        {"question": "Who is Ben Example?", "answer": "A writer."},
    ]
    kwargs = dict(
        dataset_id="fixture",
        dataset_revision="a" * 40,
        split="forget",
        source_file_sha256="b" * 64,
        authors=["Ada Example", "Ben Example"],
    )
    manifest = author_mapping(rows, **kwargs)
    assert [r["author_id"] for r in manifest["assignments"]] == ["Ada Example", "Ben Example"]
    validate_author_mapping(manifest, rows, "b" * 64)
    with pytest.raises(ValueError):
        validate_author_mapping(manifest, list(reversed(rows)), "b" * 64)
    with pytest.raises(ValueError):
        author_mapping([{"question": "Ada Example and Ben Example", "answer": "yes"}], **kwargs)
    with pytest.raises(ValueError):
        author_mapping([{"question": "Who?", "answer": "Unknown"}], **kwargs)
    # The committed pinned mapping contains 10 independently identified authors x 20.
    pinned = json.loads(Path("experiments/reproductions/QRA/forget05_authors_v1.json").read_text())
    assert len(pinned["assignments"]) == 200 and set(pinned["counts"].values()) == {20}
    assert len(pinned["counts"]) == 10
