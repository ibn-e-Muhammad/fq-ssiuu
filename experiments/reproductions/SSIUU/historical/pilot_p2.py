#!/usr/bin/env python3
"""
Pilot P2: SSIUU Attribution Feasibility Test
==============================================

EVB Table 5.2 Specification:
    Objective: Determine whether captum's LayerIntegratedGradients can run
    on Llama-3.2-1B within 16 GB VRAM (single T4 GPU).

This is a MEMORY FEASIBILITY test, not a research result. It measures:
    - Peak VRAM usage before and during IG computation
    - Time per attribution batch
    - Whether OOM occurs

WHY FP16, NOT NF4:
    Integrated Gradients requires gradient computation through model weights
    (∂output/∂input via autograd). bitsandbytes NF4 quantised weights are stored
    as integers and are NOT differentiable — torch.autograd cannot backpropagate
    through them. Therefore, attribution MUST run on the full-precision (FP16)
    model. On T4 GPUs, FP16 is used instead of BF16 (CC 7.5 limitation).

FALLBACK:
    If P2 fails (OOM), the training loop uses GradientNormAttribution from
    src/fq_ssiuu/attribution/ssiuu_reg.py. This fallback is already implemented
    and requires no additional code changes.

Acceptance Criteria:
    Peak VRAM < 14 GB  → PASS (use captum IG n_steps=20 for SSIUU)
    Peak VRAM 14-16 GB → MARGINAL (reduce to n_steps=10 or gradient-norm fallback)
    OOM                → FAIL (use GradientNormAttribution)

Usage:
    python experiments/pilot_p2_attribution.py \
        --hf_checkpoint Xafire/GATE1-NPO-Baseline \
        --drive_output "/content/drive/My Drive/fq-ssiuu-research/results/pilot_p2"

Author: FQ-SSIUU Research Team
Date: 2026-08-08
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
)
logger = logging.getLogger("pilot_p2")

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

TARGET_LAYER_IDX = 14  # Middle of 28-layer Llama-3.2-1B
IG_N_STEPS = 20        # Fast approximation
IG_BATCH_SIZE = 1      # Memory constraint
MAX_SEQ_LEN = 128      # Truncated sequence length for IG
NUM_EXAMPLES = 2       # Number of forget-set examples to test
VRAM_TOTAL_GB = 15.0   # T4 total VRAM


def parse_args():
    parser = argparse.ArgumentParser(description="Pilot P2: SSIUU Attribution Feasibility")
    parser.add_argument(
        "--hf_checkpoint",
        default="Xafire/GATE1-NPO-Baseline",
        help="HuggingFace checkpoint ID.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default="",
        help="Local checkpoint path. If empty, downloads from HF Hub.",
    )
    parser.add_argument(
        "--drive_output",
        default="/tmp/pilot_p2_results",
        help="Directory to save results.",
    )
    parser.add_argument(
        "--hf_token",
        default="",
        help="HuggingFace token.",
    )
    return parser.parse_args()


def get_forget_examples():
    """Load a small batch of TOFU forget05 examples."""
    from datasets import load_dataset

    ds = load_dataset("locuslab/TOFU", "forget05", split="train")
    examples = []
    for i in range(min(NUM_EXAMPLES, len(ds))):
        q = ds[i]["question"]
        a = ds[i]["answer"]
        examples.append(f"Question: {q}\nAnswer: {a}")

    return examples


def measure_vram():
    """Measure current and peak GPU VRAM usage in GB."""
    import torch

    if not torch.cuda.is_available():
        return 0.0, 0.0

    current = torch.cuda.memory_allocated() / (1024**3)
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    return current, peak


def run_ig_test(model, tokenizer, examples):
    """Run Integrated Gradients on the target layer and measure resources."""
    import torch
    import captum.attr

    device = next(model.parameters()).device

    # Tokenise examples
    inputs = tokenizer(
        examples,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LEN,
    ).to(device)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    logger.info("Input shape: %s", input_ids.shape)

    # Get the target layer
    target_layer = model.model.layers[TARGET_LAYER_IDX]
    logger.info("Target layer: model.model.layers[%d] (%s)", TARGET_LAYER_IDX, type(target_layer).__name__)

    # Reset peak VRAM counter
    torch.cuda.reset_peak_memory_stats()
    vram_before_current, _ = measure_vram()



    logger.info("VRAM before IG: %.2f GB", vram_before_current)
    logger.info("Starting IG computation (n_steps=%d, batch_size=%d)...", IG_N_STEPS, IG_BATCH_SIZE)

    t0 = time.time()
    oom_occurred = False
    attribution = None

    try:
        # Run IG
        # We use the input embeddings as the input and zero tensor as baseline
        embeddings = model.model.embed_tokens(input_ids)
        baseline = torch.zeros_like(embeddings)

        # LayerIntegratedGradients on a specific layer
        # We need to use a wrapper that takes embeddings as input
        def embed_forward_func(embeddings):
            # We need to hook into the model to use embeddings instead of input_ids
            # This is a simplified approach: use the model's forward with inputs_embeds
            outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1, :]
            target_ids = input_ids[:, 1:]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            target_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
            # Sum over the sequence dimension (dim=1) to keep the batch dimension
            return target_log_probs.sum(dim=1)

        lig_embed = captum.attr.LayerIntegratedGradients(embed_forward_func, target_layer)

        attribution = lig_embed.attribute(
            embeddings,
            baselines=baseline,
            n_steps=IG_N_STEPS,
            internal_batch_size=IG_BATCH_SIZE,
        )

    except torch.cuda.OutOfMemoryError:
        oom_occurred = True
        logger.error("OOM during IG computation!")
        torch.cuda.empty_cache()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            oom_occurred = True
            logger.error("OOM during IG computation: %s", e)
            torch.cuda.empty_cache()
        else:
            raise

    elapsed = time.time() - t0
    _, vram_peak = measure_vram()

    results = {
        "vram_before_ig_gb": round(vram_before_current, 3),
        "vram_peak_during_ig_gb": round(vram_peak, 3),
        "vram_margin_gb": round(VRAM_TOTAL_GB - vram_peak, 3),
        "elapsed_seconds": round(elapsed, 2),
        "oom_occurred": oom_occurred,
        "n_steps": IG_N_STEPS,
        "batch_size": IG_BATCH_SIZE,
        "target_layer": TARGET_LAYER_IDX,
        "input_shape": list(input_ids.shape),
        "max_seq_len": MAX_SEQ_LEN,
        "num_examples": NUM_EXAMPLES,
    }

    if attribution is not None:
        results["attribution_shape"] = list(attribution.shape)
        results["attribution_mean"] = float(attribution.mean().item())
        results["attribution_max"] = float(attribution.abs().max().item())
        results["attribution_nonzero_frac"] = float(
            (attribution.abs() > 1e-8).float().mean().item()
        )
    else:
        results["attribution_shape"] = None
        results["attribution_mean"] = None
        results["attribution_max"] = None

    # Determine verdict
    if oom_occurred:
        results["verdict"] = "FAIL"
        results["recommended_method"] = "GradientNormAttribution (fallback in ssiuu_reg.py)"
    elif vram_peak < 14.0:
        results["verdict"] = "PASS"
        results["recommended_method"] = f"captum IG (n_steps={IG_N_STEPS})"
    elif vram_peak < 16.0:
        results["verdict"] = "MARGINAL"
        results["recommended_method"] = "captum IG (n_steps=10) or GradientNormAttribution"
    else:
        results["verdict"] = "FAIL"
        results["recommended_method"] = "GradientNormAttribution (fallback in ssiuu_reg.py)"

    return results


def print_verdict(results):
    """Print the P2 verdict in paste-back format."""
    print()
    print("=" * 64)
    print("    PILOT P2: SSIUU ATTRIBUTION FEASIBILITY")
    print("=" * 64)
    print(f"  Model: Llama-3.2-1B-Instruct (FP16)")
    print(f"  Layer: {results['target_layer']} (of 28)")
    print(f"  IG steps: {results['n_steps']}")
    print(f"  Batch size: {results['batch_size']}")
    print(f"  Input shape: {results['input_shape']}")
    print()
    print(f"  VRAM before IG:      {results['vram_before_ig_gb']:.3f} GB")
    print(f"  VRAM peak during IG: {results['vram_peak_during_ig_gb']:.3f} GB "
          f"(margin: {results['vram_margin_gb']:.3f} GB remaining)")
    print(f"  Time for IG:         {results['elapsed_seconds']:.1f} seconds")
    print(f"  OOM occurred:        {'YES ❌' if results['oom_occurred'] else 'NO ✅'}")
    print()

    if results.get("attribution_shape"):
        print(f"  Attribution tensor shape: {results['attribution_shape']}")
        print(f"  Attribution mean: {results['attribution_mean']:.6e}")
        print(f"  Attribution max:  {results['attribution_max']:.6e}")
        if results.get("attribution_nonzero_frac") is not None:
            print(f"  Non-zero fraction: {results['attribution_nonzero_frac']:.4f}")
    else:
        print("  Attribution tensor: NOT COMPUTED (OOM or error)")

    print()
    verdict = results["verdict"]
    if verdict == "PASS":
        print(f"  ✅ VERDICT: PASS")
    elif verdict == "MARGINAL":
        print(f"  ⚠️  VERDICT: MARGINAL")
    else:
        print(f"  ❌ VERDICT: FAIL")

    print(f"  Recommended attribution method: {results['recommended_method']}")
    print("=" * 64)

    print("\n--- RAW JSON ---")
    print(json.dumps(results, indent=2, default=str))


def main():
    args = parse_args()
    hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")

    os.environ["TRANSFORMERS_NO_FLASH_ATTN"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    print("=" * 64)
    print("    PILOT P2: SSIUU ATTRIBUTION FEASIBILITY TEST")
    print("    FQ-SSIUU Project — EVB §5.2")
    print("=" * 64)

    # ── Step 1: Load model at FP16 ───────────────────────────────
    print("\n=== Step 1: Loading Model at FP16 ===")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        ckpt_path = args.checkpoint_path
    else:
        ckpt_path = "/tmp/p2_checkpoint"
        os.makedirs(ckpt_path, exist_ok=True)
        safetensors = os.path.join(ckpt_path, "model.safetensors")
        if not os.path.exists(safetensors):
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=args.hf_checkpoint,
                local_dir=ckpt_path,
                token=hf_token,
            )
        else:
            logger.info("Checkpoint already present at %s", ckpt_path)

    logger.info("Loading model at FP16 with gradient support...")
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Enable gradient computation (required for IG)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(True)

    vram_current, vram_peak = measure_vram()
    logger.info("Model loaded. VRAM: %.2f GB current, %.2f GB peak", vram_current, vram_peak)

    # ── Step 2: Load forget examples ─────────────────────────────
    print("\n=== Step 2: Loading Forget Examples ===")
    examples = get_forget_examples()
    for i, ex in enumerate(examples):
        logger.info("Example %d: %s...", i, ex[:80])

    # ── Step 3: Run IG test ──────────────────────────────────────
    print("\n=== Step 3: Running Integrated Gradients ===")
    results = run_ig_test(model, tokenizer, examples)

    # ── Step 4: Save results ─────────────────────────────────────
    print("\n=== Step 4: Saving Results ===")
    result_path = "/tmp/pilot_p2_results.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    os.makedirs(args.drive_output, exist_ok=True)
    drive_result = os.path.join(args.drive_output, "pilot_p2_results.json")
    import shutil
    shutil.copy(result_path, drive_result)
    logger.info("Results saved to: %s", drive_result)

    # ── Step 5: Print verdict ────────────────────────────────────
    print_verdict(results)

    # Clean up
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit("Historical pilot disabled: RG-005 and method gates block corrected runs. Use the Mother runner after conformance acceptance.")
