import copy
from dataclasses import replace

import pytest
import torch

from fq_ssiuu.evaluation.content_identity import evaluation_hash, validate_oracle
from fq_ssiuu.evaluation.derived_summary import (
    UTILITY_COMPONENTS,
    utility_summary,
    validate_utility_summary,
)
from fq_ssiuu.evaluation.identity import digest
from fq_ssiuu.evaluation.precision_artifact import (
    build_artifact,
    tensor_identity,
    validate_artifact,
)
from fq_ssiuu.evaluation.qra_pipeline import compare_observations
from fq_ssiuu.evaluation.statistics import (
    clustered_precision_bootstrap,
    crossed_seed_author_bootstrap,
)
from tests.methods.QRA.fixtures import group, observation, oracle


def test_clustered_comparison_requires_bound_author_evidence():
    g = group(("answer_log_probability",))
    spec = {"method": "paired-author-cluster-bootstrap-v1", "seed": 23, "resamples": 100}
    with pytest.raises(ValueError, match="author evidence"):
        compare_observations(
            observation(manifest=g), observation(precision="nf4", manifest=g), g, bootstrap=spec
        )
    g["author_evidence"] = {
        "forget05": {
            "dataset_id": "fixture/tofu",
            "dataset_revision": "c" * 40,
            "split": "forget05",
            "assignments": [
                {
                    "sample_id": f"sample-{i}",
                    "author_id": f"author-{i}",
                    "source_record_sha256": "9" * 64,
                }
                for i in range(2)
            ],
        }
    }
    result = compare_observations(
        observation(manifest=g),
        observation(precision="nf4", manifest=g, values=(-2.0, -1.0)),
        g,
        bootstrap=spec,
    )
    assert result["paired_comparison"]["mean_change"] == 1.0
    assert result["paired_comparison"]["confidence_interval"] == [1.0, 1.0]
    assert result["attack_detected"] is None


def test_tensor_identity_hashes_raw_bytes_and_fp16_model_state():
    import hashlib

    tensor = torch.tensor([1.0, 2.0], dtype=torch.float16)
    identity = tensor_identity(tensor)
    assert identity["sha256"] == hashlib.sha256(b"\x00\x3c\x00\x40").hexdigest()
    model = torch.nn.Linear(2, 1, bias=False).half()
    model.weight.data.copy_(tensor[None])
    args = dict(
        precision="fp16",
        checkpoint_sha256="a" * 64,
        versions={"torch": "2.10", "transformers": "5.0", "bitsandbytes": "0.46.1"},
        quantizer_config={"dtype": "float16", "quantization": None},
        expected_quantized_modules=[],
    )
    manifest = build_artifact(model, **args)
    assert manifest["state_tensors"][0]["sha256"] == identity["sha256"]
    model.weight.data[0, 0] = 2
    assert digest(build_artifact(model, **args)) != digest(manifest)


@pytest.mark.parametrize("precision", ["int8", "nf4"])
def test_quantized_label_cannot_hide_plain_fp16_model(precision):
    with pytest.raises(ValueError, match="Partial"):
        build_artifact(
            torch.nn.Linear(2, 2).half(),
            precision=precision,
            checkpoint_sha256="a" * 64,
            versions={"torch": "1", "transformers": "1", "bitsandbytes": "1"},
            quantizer_config={},
            expected_quantized_modules=["proj"],
        )


def test_partial_or_wrong_quantized_inventory_fails():
    original = group()["artifacts"]["nf4"]["artifact_manifest"]
    for mutation in (
        lambda m: m["quantized_modules"].clear(),
        lambda m: m["quantized_modules"][0]["state"].update(quant_type="fp4"),
        lambda m: m["quantized_modules"][0]["weight"].update(dtype="torch.float16"),
    ):
        changed = copy.deepcopy(original)
        mutation(changed)
        with pytest.raises(ValueError):
            validate_artifact(changed)


@pytest.mark.parametrize(
    "field,value",
    [
        ("value", -2.4),
        ("evaluation_sha256", "f" * 64),
        ("decoding_sha256", "f" * 64),
        ("evaluator_sha256", "e" * 64),
        ("run_id", "other"),
    ],
)
def test_evaluation_hash_is_recomputed(field, value):
    obs = observation()
    with pytest.raises(ValueError):
        replace(obs, **{field: value}).validate()


def test_rehash_does_not_make_wrong_arithmetic_mean_valid():
    obs = replace(observation(), value=-1.0)
    obs = replace(obs, evaluation_sha256=evaluation_hash(obs))
    with pytest.raises(ValueError, match="Aggregate"):
        obs.validate()


def test_oracle_raw_and_semantic_binding():
    raw = oracle()
    raw["raw_records"][0]["tofu_truth_ratio"] += 0.1
    with pytest.raises(ValueError, match="raw content"):
        validate_oracle(raw)
    g = group(("ks_p_value",))
    obs = observation("ks_p_value", manifest=g, value=1 / 3)
    obs.validate(g)
    wrong = replace(obs, value=0.9)
    wrong = replace(wrong, evaluation_sha256=evaluation_hash(wrong))
    with pytest.raises(ValueError, match="KS diagnostic"):
        wrong.validate(g)
    del g["oracle_evidence"]
    with pytest.raises(ValueError, match="Oracle raw evidence"):
        obs.validate(g)


def test_derived_utility_harmonic_mean_and_component_manifest():
    g = group(UTILITY_COMPONENTS)
    components = [
        observation(m, values=(0.5, 0.5) if i == 0 else (1.0, 1.0), manifest=g)
        for i, m in enumerate(UTILITY_COMPONENTS)
    ]
    summary = utility_summary(components, g)
    assert summary["value"] == pytest.approx(0.9)  # 9 / (2+8)
    assert len(summary["components"]) == 9
    validate_utility_summary(summary, components, g)
    changed = copy.deepcopy(summary)
    changed["value"] = sum(o.value for o in components) / 9
    with pytest.raises(ValueError):
        validate_utility_summary(changed, components, g)
    with pytest.raises(ValueError):
        utility_summary(components[:-1], g)
    with pytest.raises(ValueError):
        utility_summary(components[:-1] + [components[0]], g)
    zero = [
        observation(m, values=(0.0, 0.0) if i == 0 else (1.0, 1.0), manifest=g)
        for i, m in enumerate(UTILITY_COMPONENTS)
    ]
    assert utility_summary(zero, g)["value"] == 0


def test_clustered_estimand_equal_authors_not_row_weighted():
    kwargs = dict(cluster_manifest_sha256="a" * 64, seed=23, resamples=1000)
    result = clustered_precision_bootstrap(
        ["a0", "a1", "b0"], ["a", "a", "b"], [0, 0, 0], [1, 1, 3], **kwargs
    )
    assert result["author_mean_changes"] == [1.0, 3.0]
    assert result["mean_change"] == 2  # row mean would be 5/3
    assert result["confidence_interval"] == [1.0, 3.0]
    assert result == clustered_precision_bootstrap(
        ["a0", "a1", "b0"], ["a", "a", "b"], [0, 0, 0], [1, 1, 3], **kwargs
    )
    assert result["attack_detected"] is None
    with pytest.raises(ValueError, match="two author"):
        clustered_precision_bootstrap(["a0"], ["a"], [0], [1], **kwargs)


def test_crossed_seed_author_uncertainty_is_deterministic_and_complete():
    kwargs = dict(cluster_manifest_sha256="a" * 64, seed=23, resamples=1000)
    changes = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
    result = crossed_seed_author_bootstrap([23, 42, 101], ["a", "b"], changes, **kwargs)
    assert result["mean_change"] == 1 and result["confidence_interval"] == [1.0, 1.0]
    assert result == crossed_seed_author_bootstrap([23, 42, 101], ["a", "b"], changes, **kwargs)
    with pytest.raises(ValueError):
        crossed_seed_author_bootstrap([23, 42], ["a", "b"], changes[:2], **kwargs)
    with pytest.raises(ValueError):
        crossed_seed_author_bootstrap([23, 42, 101], ["a", "b"], changes[:2], **kwargs)


def test_crossed_resampling_preserves_shared_author_dimension():
    import numpy as np

    matrix = np.array([[0.0, 2.0], [1.0, 3.0], [2.0, 4.0]])
    rng = np.random.Generator(np.random.PCG64(5))
    draws = []
    for _ in range(20):
        seeds = rng.integers(0, 3, 3)
        authors = rng.integers(0, 2, 2)
        draws.append(sum(matrix[s, a] for s in seeds for a in authors) / 6)
    result = crossed_seed_author_bootstrap(
        [23, 42, 101],
        ["a", "b"],
        matrix.tolist(),
        cluster_manifest_sha256="a" * 64,
        seed=5,
        resamples=20,
    )
    assert result["confidence_interval"] == pytest.approx(np.quantile(draws, [0.025, 0.975]))
