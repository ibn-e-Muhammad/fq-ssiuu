import json, math, hashlib, tarfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd()))
from fq_ssiuu.evaluation.identity import digest, file_digest
from experiments.reproductions.NPO.validation_contract import expected_recipe
from experiments.reproductions.QRA.metrics import (
    implementation_hash,
    text_metrics,
    truth_ratios,
    forget_quality,
    model_utility,
    utility_components,
)
from scripts.validate_og001_artifacts import validate_setup_artifacts

root = Path("handbook/evidence/work_unit_b/20260922_attempt1_318e523a")
runid = "20260922-101703-ce77656bee84"
run = root / "raw/runs" / runid
sealed = root / "raw/runs" / (runid + "-sealed")
manifest = json.loads((sealed / "ARCHIVE_MANIFEST.json").read_text())
assert file_digest(sealed / "artifacts.tar") == manifest["archive"]["sha256"]
with tarfile.open(sealed / "artifacts.tar") as tar:
    assert set(tar.getnames()) == {x["path"] for x in manifest["artifacts"]}
    for item in manifest["artifacts"]:
        data = tar.extractfile(item["path"]).read()
        assert hashlib.sha256(data).hexdigest() == item["sha256"] == file_digest(run / item["path"])
        assert len(data) == item["size_bytes"]
read = lambda name: json.loads((run / name).read_text())
provenance = read("source_provenance.json")
source = json.loads((root / "SOURCE_MANIFEST.json").read_text())
assert provenance["source_origin_sha"] == "318e523ac302f9f80f2c74f7c712bff08ff96a8e"
assert provenance["source_manifest"] == source
assert provenance["source_manifest_sha256"] == file_digest(root / "SOURCE_MANIFEST.json")
staging = json.loads((root / "staging_receipt.json").read_text())
assert staging["staged_notebook_sha256"] == file_digest(root / "submitted_notebook.ipynb")
assert read("recipe.json") == expected_recipe()
assert read("run_result.json")["scientific_experiment_executed"] is False
assert read("method_validation_execution.json")["returncode"] == 0
assert read("run_result.json")["status"] == "PASS"
env = validate_setup_artifacts(
    run / "environment",
    provenance["snapshot_commit_sha"],
    Path("requirements/kaggle-native-baseline.json"),
)
assert env["status"] == "PASS"
result = read("method_validation/validation_result.json")
assert result["run_id"] == runid and result["evaluator_sha256"] == implementation_hash()
assert (
    result["peak_allocated_bytes"] < 16_000_000_000
    and result["peak_reserved_bytes"] < 16_000_000_000
)
assert (
    result["visible_devices"] == "0" and result["device_count"] == 1 and result["gpu"] == "Tesla T4"
)
assert (
    result["optimizer_steps"] == result["backward_calls"] == 0 and result["model_training"] is False
)
assert (
    result["attack_detected"]
    is result["durability_pass"]
    is result["precision_comparisons"]
    is None
)
pins = json.loads(Path("experiments/reproductions/NPO/inputs_v1.json").read_text())
cfg = read("method_validation/reference_config.json")
assert cfg["candidate"] == cfg["reference"] == pins["target"]
assert (
    cfg["dtype"] == "float16"
    and cfg["device"] == "cuda:0"
    and cfg["load_in_4bit"] is cfg["load_in_8bit"] is False
)
records = []
components = {}
for split in expected_recipe()["sample_indices"]:
    r = read("method_validation/" + split + ".json")
    records.append(r)
    assert digest(r["source_record"]) == r["source_record_sha256"]
    assert digest(r["tensors"]) == r["tensors_sha256"]
    assert digest(r["generation"]["decoding_spec"]) == r["generation"]["decoding_sha256"]
    measured = text_metrics(r["source_record"]["answer"], r["generation"]["text"])
    assert all(measured[k] == r[k] for k in ["rouge_l_recall", "exact_match"])
    tr = truth_ratios(r["paraphrase_mean_nll"], r["perturbation_mean_nlls"])
    assert all(
        math.isclose(tr[k], r[k], rel_tol=1e-12)
        for k in ["tofu_truth_ratio", "openunlearning_truth_ratio"]
    )
    data = {
        "retain_perturbed": "retain",
        "real_authors_perturbed": "real_authors",
        "world_facts_perturbed": "world_facts",
    }.get(split)
    if data:
        computed = utility_components(
            -r["answer_log_probability"],
            r["perturbation_mean_nlls"],
            r["rouge_l_recall"],
            dataset=data,
            paraphrase_mean_nll=r["paraphrase_mean_nll"],
        )
        assert computed == r["utility_components"]
        components.update(computed)
assert digest(records) == result["raw_records_sha256"]
assert digest([r["sample_id"] for r in records]) == result["sample_ids_sha256"]
assert model_utility(components) == result["utility"]
oracle = read("method_validation/retain95_oracle.json")
assert oracle["identity"]["revision"] == pins["retain_only_oracle"]["revision"]
assert (
    oracle["files"]["model.safetensors"]["sha256"]
    == pins["retain_only_oracle"]["checkpoint_sha256"]
)
assert (
    oracle["record"]["sample_id"] == records[0]["sample_id"]
    and oracle["record"]["tensors_sha256"] == records[0]["tensors_sha256"]
)
assert (
    forget_quality([records[0]["tofu_truth_ratio"]], [oracle["record"]["tofu_truth_ratio"]])
    == result["ks_diagnostic"]
)
for record, npo in zip(records, read("method_validation/npo_forward.json")["records"], strict=True):
    assert record["sample_id"] == npo["sample_id"]
    tensors = {
        k: {"dtype": v["dtype"], "shape": [1] + v["shape"], "values": [v["values"]]}
        for k, v in record["tensors"].items()
    }
    assert npo["input_hashes"] == [digest(tensors)] * 2
    assert npo["candidate_sequence_nll"] == npo["reference_sequence_nll"]
    assert npo["gradient_graph"] is False
    assert math.isclose(npo["loss"], 2 * math.log(2) / 0.1, rel_tol=1e-6)
    assert math.isclose(
        -record["answer_log_probability"],
        npo["candidate_sequence_nll"][0] / record["adapter"]["scored_token_count"],
        rel_tol=1e-6,
        abs_tol=1e-7,
    )
for item in read("method_validation/OUTPUT_MANIFEST.json")["artifacts"]:
    assert file_digest(run / "method_validation" / item["path"]) == item["sha256"]
verdict = {
    "status": "PASS_WITHIN_BOUNDED_FORWARD_SCOPE",
    "run_id": runid,
    "origin_commit": provenance["source_origin_sha"],
    "source_tree_sha": provenance["source_tree_sha"],
    "snapshot_commit_sha": provenance["snapshot_commit_sha"],
    "orchestration_commit": provenance["orchestration_commit"],
    "notebook_sha256": staging["staged_notebook_sha256"],
    "verified_archive_members": len(manifest["artifacts"]),
    "matched_examples": len(records),
    "metric_recomputation": "PASS",
    "environment": env["pip_consistency_state"],
    "peak_allocated_bytes": result["peak_allocated_bytes"],
    "peak_reserved_bytes": result["peak_reserved_bytes"],
    "optimizer_steps": 0,
    "backward_calls": 0,
    "npo_reference_cache": "NOT_IMPLEMENTED",
    "runtime_precision": "fp16",
    "int8_nf4_model_evaluations": False,
    "attack_detected": None,
    "durability_pass": None,
    "supported_claims": result["supported_claims"],
    "unsupported_claims": result["unsupported_claims"],
}
# The independently derived values are printed; immutable inputs are never edited.
print(json.dumps(verdict))
