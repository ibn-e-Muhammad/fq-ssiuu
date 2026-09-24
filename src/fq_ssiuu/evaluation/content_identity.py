"""Content-derived evaluation identities and explicit oracle evidence bindings."""

from dataclasses import asdict, is_dataclass

from fq_ssiuu.evaluation.identity import digest, require_hash


def evaluation_hash(observation):
    data = asdict(observation) if is_dataclass(observation) else dict(observation)
    # Include the entire observation except its own digest. This binds both raw
    # values and their reported aggregate, identities, definition and run linkage.
    data.pop("evaluation_sha256", None)
    return digest({"schema": "evaluation-content-v2", "observation": data})


def oracle_manifest(
    *,
    checkpoint_sha256,
    model_revision,
    tokenizer_revision,
    dataset_revision,
    preprocessing_sha256,
    evaluator_sha256,
    records,
):
    result = {
        "schema_version": "fixed-fp16-oracle-v1",
        "precision": "fp16",
        "checkpoint_sha256": checkpoint_sha256,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "dataset_revision": dataset_revision,
        "preprocessing_sha256": preprocessing_sha256,
        "evaluator_sha256": evaluator_sha256,
        "raw_records": records,
        "raw_records_sha256": digest(records),
        "sample_ids_sha256": digest([r["sample_id"] for r in records]),
    }
    validate_oracle(result)
    return result


def validate_oracle(manifest):
    if (
        manifest.get("schema_version") != "fixed-fp16-oracle-v1"
        or manifest.get("precision") != "fp16"
    ):
        raise ValueError("Only explicit fixed FP16 oracle evidence is accepted")
    for name in ("checkpoint_sha256", "preprocessing_sha256", "evaluator_sha256"):
        require_hash(manifest[name])
    for name in ("model_revision", "tokenizer_revision", "dataset_revision"):
        require_hash(manifest[name], 40)
    records = manifest["raw_records"]
    ids = [r["sample_id"] for r in records]
    if not ids or len(ids) != len(set(ids)) or any(not isinstance(x, str) or not x for x in ids):
        raise ValueError("Ordered unique oracle samples required")
    if (
        digest(records) != manifest["raw_records_sha256"]
        or digest(ids) != manifest["sample_ids_sha256"]
    ):
        raise ValueError("Oracle raw content differs")
    from fq_ssiuu.evaluation.statistics import finite_vector

    finite_vector([r["tofu_truth_ratio"] for r in records], nonnegative=True)
    for r in records:
        require_hash(r["source_record_sha256"])
        require_hash(r["tensors_sha256"])
