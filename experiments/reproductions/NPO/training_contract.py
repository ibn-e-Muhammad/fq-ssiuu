"""Closed D1 configuration. No project QRA thresholds or substitute optimizers."""

import json
from pathlib import Path

from experiments.reproductions.common import file_digest
from experiments.reproductions.NPO.preflight import source_schedule

TRAIN = {
    "schema": "npo-d1-training-v1",
    "classification": "bounded-source-faithful-1b-adapted-reproduction-pilot",
    "variant": "npo_grad_diff",
    "beta": 0.1,
    "npo_coeff": 1.0,
    "retain_coeff": 1.0,
    "trainable_parameters": "all",
    "reference": "original-full-unquantized-frozen-fp16",
    "micro_batch": 4,
    "accumulation": 8,
    "world_size": 1,
    "max_length": 500,
    "nominal_epochs": 10,
    "max_steps": 62,
    "warmup_steps": 6,
    "eval_interval": 6,
    "optimizer": "paged_adamw_32bit",
    "learning_rate": 1e-5,
    "weight_decay": 0.01,
    "betas": [0.9, 0.999],
    "epsilon": 1e-8,
    "scheduler": "linear",
    "gradient_clipping": 0.0,
    "gradient_checkpointing": True,
    "precision": "float16",
    "attention": "sdpa",
    "zero_stage": 3,
    "optimizer_offload": "none",
    "parameter_offload": "none",
    "data_seed": 1001,
    "trainer_seed": 42,
    "checkpoint_selection": "final-source-stopping-step",
    "memory_limit_bytes": 16000000000,
}

EVALUATION = {
    "schema": "npo-d1-evaluation-v1",
    "max_length": 200,
    "selection": "first-min-300-per-component",
    "primary_batch": 30,
    "base_batch": 7,
    "generation_seed_per_batch": 0,
    "generation": "source-sampled-pinned-llama3-defaults",
    "source_truth_ratio": "npo-author-geometric-1207bbe-v1",
    "additional_diagnostic": "tofu-paper-arithmetic-v1",
    "oracle": "pinned-retain95-fp16",
    "kl_reference": "original-full-fp16-all-positions",
    "final_evaluation": True,
    "attack_detected": None,
    "durability_pass": None,
}


def read_configs(train_path, eval_path, expected_train_hash, expected_eval_hash):
    values = []
    for path, sha, expected in (
        (train_path, expected_train_hash, TRAIN),
        (eval_path, expected_eval_hash, EVALUATION),
    ):
        if file_digest(path) != sha:
            raise ValueError("Frozen D1 config bytes changed")
        value = json.loads(Path(path).read_text())
        # JSON equality alone treats True as 1; use canonical serialization to
        # reject type drift as well as unknown/unapproved configuration keys.
        if json.dumps(value, sort_keys=True) != json.dumps(expected, sort_keys=True):
            raise ValueError("D1 config changes an accepted source/resource setting")
        values.append(value)
    schedule = source_schedule(200, TRAIN["nominal_epochs"], 4, 8, 1)
    if schedule["max_steps"] != TRAIN["max_steps"] or schedule["warmup_steps"] != 6:
        raise ValueError("D1 schedule does not match author floor semantics")
    return tuple(values)


def deepspeed_config(hidden_size):
    if type(hidden_size) is not int or hidden_size <= 0:
        raise ValueError("Positive model hidden dimension required")
    return {
        "train_batch_size": 32,
        "train_micro_batch_size_per_gpu": 4,
        "gradient_accumulation_steps": 8,
        "gradient_clipping": 0.0,
        "fp16": {"enabled": True},
        "bf16": {"enabled": False},
        "zero_allow_untested_optimizer": True,
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "none", "pin_memory": True},
            "offload_param": {"device": "none", "pin_memory": True},
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1000000000,
            "reduce_bucket_size": hidden_size * hidden_size,
            "stage3_prefetch_bucket_size": int(0.9 * hidden_size * hidden_size),
            "stage3_param_persistence_threshold": 10 * hidden_size,
            "stage3_max_live_parameters": 1000000000,
            "stage3_max_reuse_distance": 1000000000,
            "stage3_gather_16bit_weights_on_model_save": True,
        },
    }


def linear_schedule(step):
    """Publication-era Transformers linear warmup/decay, including initial LR 0."""
    if type(step) is not int or step < 0:
        raise ValueError("Nonnegative optimizer step required")
    return step / 6 if step < 6 else max(0.0, (62 - step) / 56)
