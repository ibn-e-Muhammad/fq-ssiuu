"""Execute hash-pinned author pairing, sampler and metric functions on CPU fixtures."""

import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.stats import hmean, ks_2samp
from torch.utils.data import DataLoader

from experiments.reproductions.common import digest, file_digest
from experiments.reproductions.NPO.preflight import SOURCE, verify_files
from experiments.reproductions.NPO.source_metrics import (
    SOURCE_TASKS,
    source_forget_quality,
    source_summary,
)
from experiments.reproductions.NPO.training_data import SourcePairDataset, source_batch_plan


def extract(path, name, namespace, class_name=None):
    nodes = ast.parse(Path(path).read_bytes()).body
    if class_name:
        nodes = next(n.body for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def validate(upstream, runtime_sources):
    upstream, runtime_sources = Path(upstream), Path(runtime_sources)
    source = json.loads(SOURCE.read_text())
    verified = verify_files(upstream, source["files"])
    runtime_pin = Path(__file__).with_name("runtime_sources_v1.json")
    runtime = json.loads(runtime_pin.read_text())
    runtime_verified = verify_files(runtime_sources, runtime["files"])
    clipping_defaults = {}
    for name in runtime["files"]:
        if name.startswith("deepspeed-v"):
            tree = ast.parse((runtime_sources / name).read_bytes())
            assignment = next(
                n
                for n in tree.body
                if isinstance(n, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and t.id == "GRADIENT_CLIPPING_DEFAULT"
                    for t in n.targets
                )
            )
            clipping_defaults[name] = ast.literal_eval(assignment.value)
    if not clipping_defaults or any(value != 0.0 for value in clipping_defaults.values()):
        raise ValueError("Historical DeepSpeed gradient clipping assumption differs")
    config = json.loads((upstream / "TOFU/config/ds_config.json").read_text())
    if "gradient_clipping" in config:
        raise ValueError("Author clipping configuration changed")
    fill = extract(
        runtime_sources / "transformers/deepspeed.py", "fill_match", {}, "HfTrainerDeepSpeedConfig"
    )
    holder = SimpleNamespace(find_config_node=lambda key: (config, key), mismatches=[])
    fill(holder, "gradient_clipping", 1.0, "max_grad_norm")
    if "gradient_clipping" in config or holder.mismatches:
        raise ValueError("Trainer filled a missing clipping field unexpectedly")
    distribution = SimpleNamespace(FSDP="fsdp", DEEPSPEED="ds", XLA="xla")
    clip = extract(
        runtime_sources / "accelerate/accelerator.py",
        "clip_grad_norm_",
        {"DistributedType": distribution},
        "Accelerator",
    )
    weight = torch.nn.Parameter(torch.ones(2))
    weight.grad = torch.full_like(weight, 10.0)
    clip(SimpleNamespace(distributed_type="ds"), [weight], 1.0)
    if not torch.equal(weight.grad, torch.full_like(weight, 10.0)):
        raise ValueError("Accelerate applies additional gradient clipping")
    ns = {"torch": torch, "np": np, "hmean": hmean, "ks_2samp": ks_2samp}
    # Execute the original __getitem__, replacing only tokenization so that row
    # identity and post-call RNG state are independently observable.
    ns["convert_raw_data_to_model_format"] = lambda tokenizer, length, q, a, config: (q, a)
    getitem = extract(upstream / "TOFU/data_module.py", "__getitem__", ns, "TextForgetDatasetQA")
    forget = [{"question": f"f{i}", "answer": f"fa{i}"} for i in range(200)]
    retain = [{"question": f"r{i}", "answer": f"ra{i}"} for i in range(3800)]
    source_ds = SimpleNamespace(
        split1="forget",
        split2="retain",
        forget_data=forget,
        retain_data=retain,
        tokenizer=None,
        max_length=500,
        model_configs={},
    )
    local = SourcePairDataset(forget, retain)
    pairs = []
    for i in range(200):
        original = getitem(source_ds, i)
        rng = torch.get_rng_state()
        actual = local[i]
        if original != [
            (actual["forget"]["question"], actual["forget"]["answer"]),
            (actual["retain"]["question"], actual["retain"]["answer"]),
        ]:
            raise ValueError("Source forget/retain pairing differs")
        if not torch.equal(rng, torch.get_rng_state()):
            raise ValueError("Source post-pairing RNG differs")
        pairs.append([i, actual["retain_index"]])
    # Execute the actual upstream loader method: no reimplementation-as-oracle.
    ns.update(
        DataLoader=DataLoader, is_datasets_available=lambda: False, seed_worker=lambda _: None
    )
    loader_fn = extract(
        upstream / "TOFU/dataloader.py", "get_train_dataloader", ns, "CustomTrainerForgetting"
    )
    trainer = SimpleNamespace(
        train_dataset=list(range(200)),
        data_collator=lambda x: x,
        _train_batch_size=4,
        _get_collator_with_removed_columns=lambda x, **_: x,
        args=SimpleNamespace(
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            dataloader_persistent_workers=False,
            dataloader_drop_last=False,
        ),
        seed=1001,
        state=SimpleNamespace(global_step=0),
        accelerator=SimpleNamespace(prepare=lambda x: x),
    )
    loader = loader_fn(trainer)
    expected = [batch for _ in range(10) for batch in loader][:496]
    plan = source_batch_plan(200, 4, 8, 62, 1001)
    if expected != [row["forget_indices"] for row in plan["batches"]]:
        raise ValueError("Source DataLoader order differs")
    # Independent random fixed records cover nontrivial harmonic aggregation,
    # unequal truth ratios, zeros, and the source's KS implementation.
    utility_fn = extract(upstream / "TOFU/utils.py", "get_model_utility", ns)
    quality_fn = extract(upstream / "TOFU/utils.py", "get_forget_quality", ns)
    generator = np.random.default_rng(1001)
    cases = []
    for zero in (False, True):
        records = {
            k: {
                "avg_gt_loss": generator.uniform(0.1, 6, 8).tolist(),
                "rougeL_recall": ([0.0] * 8 if zero else generator.uniform(0, 1, 8).tolist()),
                "avg_paraphrased_loss": generator.uniform(0.1, 6, 8).tolist(),
                "average_perturb_loss": generator.uniform(0.1, 6, (8, 5)).tolist(),
                "kl_divergence": generator.uniform(0, 2, 8).tolist(),
            }
            for k in SOURCE_TASKS
        }
        expected_metrics = utility_fn(records)
        actual_metrics = source_summary(records)["metrics"]
        if set(expected_metrics) != set(actual_metrics):
            raise ValueError("Source metric inventory differs")
        errors = {k: abs(float(expected_metrics[k]) - actual_metrics[k]) for k in expected_metrics}
        if max(errors.values()) > 1e-12:
            raise ValueError("Source utility numerical mismatch")
        oracle = {"eval_log_forget.json": records["eval_log.json"]}
        expected_quality, _ = quality_fn(records, oracle)
        actual_quality = source_forget_quality(
            records["eval_log_forget.json"], oracle["eval_log_forget.json"]
        )
        if (
            actual_quality["ks_p_value"] != expected_quality["Forget Quality"]
            or actual_quality["ks_statistic"] != expected_quality["KS Test Forget"]
        ):
            raise ValueError("Source KS diagnostic differs")
        cases.append(
            {
                "zero_case": zero,
                "max_error": max(errors.values()),
                "input_sha256": digest(records),
                "utility": actual_metrics["Model Utility"],
            }
        )
    return {
        "status": "PASS_CPU_SOURCE_CONFORMANCE_ONLY",
        "source_commit": source["upstream"]["commit"],
        "source_files": verified,
        "runtime_source_files": runtime_verified,
        "runtime_registry_sha256": file_digest(runtime_pin),
        "historical_clipping_defaults": clipping_defaults,
        "effective_gradient_clipping": 0.0,
        "pairing_rows": 200,
        "pairs_sha256": digest(pairs),
        "microbatches_compared": 496,
        "plan_sha256": plan["sha256"],
        "metric_cases": cases,
        "kaggle_submissions": 0,
        "model_weights_loaded": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--runtime-sources", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    result = {"status": "FAIL", "kaggle_submissions": 0}
    try:
        result = validate(args.upstream, args.runtime_sources)
    except Exception as error:
        result["error_type"] = type(error).__name__
        raise
    finally:
        with output.open("x") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
