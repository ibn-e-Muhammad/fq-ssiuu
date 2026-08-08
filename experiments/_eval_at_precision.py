#!/usr/bin/env python3
"""
Evaluate an unlearned model at a specific precision using OpenUnlearning's TOFU evaluator.

This script is called by qra_pipeline.py via subprocess. It:
  1. Loads the merged model at the requested precision (FP16 / INT8 / NF4)
  2. Builds the TOFU evaluation config via Hydra compose API
  3. Runs the evaluator directly (no DDP, single GPU)
  4. Saves TOFU_EVAL.json and TOFU_SUMMARY.json to the specified output directory

Usage:
    python _eval_at_precision.py \
        --precision nf4 \
        --checkpoint_path /path/to/merged_model \
        --open_unlearning_path /path/to/open-unlearning \
        --retain_logs_path saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json \
        --output_dir /tmp/eval_nf4 \
        --forget_split forget05

Precision options:
    fp16  — Full 16-bit precision (baseline)
    int8  — bitsandbytes 8-bit quantisation
    nf4   — bitsandbytes NormalFloat4 quantisation (the QRA attack surface)

NOTE: This script must be run from the open-unlearning repo root directory,
or --open_unlearning_path must point to a valid clone with src/ and configs/.
"""

import argparse
import json
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
)
logger = logging.getLogger("eval_at_precision")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate model at a specific precision using OpenUnlearning TOFU."
    )
    parser.add_argument(
        "--precision",
        required=True,
        choices=["fp16", "int8", "nf4"],
        help="Precision to evaluate at.",
    )
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        help="Path to the merged model checkpoint directory.",
    )
    parser.add_argument(
        "--open_unlearning_path",
        required=True,
        help="Path to the open-unlearning repo clone.",
    )
    parser.add_argument(
        "--retain_logs_path",
        required=True,
        help="Path to retain model TOFU_EVAL.json (relative to open-unlearning root).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save evaluation results.",
    )
    parser.add_argument(
        "--forget_split",
        default="forget05",
        help="TOFU forget split (default: forget05).",
    )
    parser.add_argument(
        "--holdout_split",
        default="holdout10",
        help="TOFU holdout split (default: holdout10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    return parser.parse_args()


def load_model_at_precision(checkpoint_path, precision):
    """Load a model from disk at the specified precision.

    Each call creates a FRESH model from disk. Never reuse or re-quantise
    an already-loaded model.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_kwargs = {
        "pretrained_model_name_or_path": checkpoint_path,
        "attn_implementation": "sdpa",
        "device_map": "auto",
        "torch_dtype": torch.float16,
    }

    if precision == "fp16":
        logger.info("Loading at FP16 (baseline precision)...")
    elif precision == "int8":
        from transformers import BitsAndBytesConfig

        logger.info("Loading at INT8 (bitsandbytes 8-bit)...")
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif precision == "nf4":
        from transformers import BitsAndBytesConfig

        logger.info("Loading at NF4 (bitsandbytes 4-bit NormalFloat)...")
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        raise ValueError(f"Unknown precision: {precision}")

    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token: %s", tokenizer.pad_token)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info("Model loaded: %.1fM parameters", param_count / 1e6)

    if torch.cuda.is_available():
        vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
        logger.info("VRAM after model load: %.2f GB", vram_gb)

    return model, tokenizer


def build_eval_config(open_unlearning_path, forget_split, holdout_split,
                      retain_logs_path, output_dir):
    """Build the TOFU evaluation config using Hydra compose API."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    # Clear any existing Hydra state
    GlobalHydra.instance().clear()

    config_dir = os.path.abspath(os.path.join(open_unlearning_path, "configs"))
    if not os.path.isdir(config_dir):
        raise FileNotFoundError(f"Configs directory not found: {config_dir}")

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="eval.yaml",
            overrides=[
                "experiment=eval/tofu/default",
                "model=Llama-3.2-1B-Instruct",
                f"forget_split={forget_split}",
                f"holdout_split={holdout_split}",
                f"retain_logs_path={retain_logs_path}",
                "task_name=_QRA_TEMP",  # We override output_dir directly
            ],
        )

    return cfg


def disable_use_cache(d):
    from omegaconf import DictConfig
    for k, v in d.items():
        if isinstance(v, DictConfig):
            if k == "generation_args" and "use_cache" in v:
                v.use_cache = False
            else:
                disable_use_cache(v)

def run_evaluation(model, tokenizer, cfg, output_dir):
    """Run the TOFU evaluator on a pre-loaded model."""
    from evals import get_evaluators
    from omegaconf import open_dict

    os.makedirs(output_dir, exist_ok=True)

    # Disable use_cache globally to prevent bitsandbytes generation hangs
    with open_dict(cfg):
        disable_use_cache(cfg)

    # Get evaluators from config
    eval_cfgs = cfg.eval
    evaluators = get_evaluators(eval_cfgs)

    results = {}
    for name, evaluator in evaluators.items():
        logger.info("Running evaluator: %s", name)
        # Call evaluate with explicit output_dir to control where files are saved
        summary = evaluator.evaluate(
            model=model,
            output_dir=output_dir,
            overwrite=True,
            tokenizer=tokenizer,
            template_args=cfg.model.template_args,
        )
        results[name] = summary
        logger.info("Evaluator %s complete. Summary: %s", name, summary)

    return results


def main():
    args = parse_args()

    # Set environment
    os.environ["TRANSFORMERS_NO_FLASH_ATTN"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Add OpenUnlearning src to path
    ou_src = os.path.join(args.open_unlearning_path, "src")
    if ou_src not in sys.path:
        sys.path.insert(0, ou_src)
    logger.info("Added to PYTHONPATH: %s", ou_src)

    # Validate retain_logs_path
    retain_logs_abs = os.path.join(args.open_unlearning_path, args.retain_logs_path)
    if not os.path.exists(retain_logs_abs):
        raise FileNotFoundError(
            f"Retain logs not found at: {retain_logs_abs}\n"
            f"Run 'python setup_data.py --eval_logs' in the open-unlearning root first.\n"
            f"This file is REQUIRED for forget_quality computation."
        )
    logger.info("Retain logs validated: %s", retain_logs_abs)

    # Validate checkpoint
    ckpt_safetensors = os.path.join(args.checkpoint_path, "model.safetensors")
    if not os.path.exists(ckpt_safetensors):
        raise FileNotFoundError(
            f"Checkpoint model.safetensors not found at: {ckpt_safetensors}"
        )

    logger.info("=" * 60)
    logger.info("EVALUATION AT %s PRECISION", args.precision.upper())
    logger.info("  Checkpoint: %s", args.checkpoint_path)
    logger.info("  Forget split: %s", args.forget_split)
    logger.info("  Retain logs: %s", args.retain_logs_path)
    logger.info("  Output dir: %s", args.output_dir)
    logger.info("=" * 60)

    t0 = time.time()

    # Step 1: Load model at specified precision
    model, tokenizer = load_model_at_precision(args.checkpoint_path, args.precision)

    # Step 2: Build evaluation config
    # We need to chdir to open-unlearning for relative paths in the config to resolve
    original_cwd = os.getcwd()
    os.chdir(args.open_unlearning_path)
    logger.info("Changed CWD to: %s", os.getcwd())

    cfg = build_eval_config(
        args.open_unlearning_path,
        args.forget_split,
        args.holdout_split,
        args.retain_logs_path,
        args.output_dir,
    )

    # Step 3: Seed
    try:
        from trainer.utils import seed_everything
        seed_everything(args.seed)
    except ImportError:
        import random
        import numpy as np
        import torch
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Step 4: Run evaluation
    results = run_evaluation(model, tokenizer, cfg, args.output_dir)

    # Step 5: Clean up
    os.chdir(original_cwd)
    elapsed = time.time() - t0

    # Step 6: Load and report the summary
    summary_path = os.path.join(args.output_dir, "TOFU_SUMMARY.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        logger.info("=" * 60)
        logger.info("RESULTS AT %s:", args.precision.upper())
        for k, v in sorted(summary.items()):
            logger.info("  %s: %s", k, f"{v:.6f}" if isinstance(v, float) else v)
        logger.info("Evaluation time: %.1f seconds", elapsed)
        logger.info("=" * 60)

        # Write a machine-readable result marker
        marker = {
            "precision": args.precision,
            "summary": summary,
            "elapsed_seconds": elapsed,
            "checkpoint_path": args.checkpoint_path,
        }
        marker_path = os.path.join(args.output_dir, f"_result_{args.precision}.json")
        with open(marker_path, "w") as f:
            json.dump(marker, f, indent=2)
        logger.info("Result marker saved to: %s", marker_path)
    else:
        logger.error("TOFU_SUMMARY.json not found at %s!", summary_path)
        logger.error("Evaluation may have failed. Check output above.")
        sys.exit(1)

    # Clean up GPU
    del model
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    logger.info("Evaluation at %s complete.", args.precision.upper())


if __name__ == "__main__":
    main()
