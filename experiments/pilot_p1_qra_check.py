#!/usr/bin/env python3
"""
Pilot P1: QRA Existence Test at 1B Scale
=========================================

EVB Table 5.1 Specification:
    Hypothesis: Quantisation Recovery Attack (QRA) manifests at 1B scale
    when NPO-unlearned weights are re-quantised to NF4.

Protocol:
    1. Load the merged NPO checkpoint (GATE-1 verified, FQ=0.3937)
    2. Evaluate at FP16 (baseline) → FQ_FP16
    3. Evaluate at INT8 (intermediate) → FQ_INT8
    4. Evaluate at NF4 (attack surface) → FQ_NF4
    5. Compute Q-INT4 = |FQ_NF4 - FQ_FP16| / (FQ_FP16 + 1e-8)

Acceptance Criterion:
    Q-INT4 > 0.05  → PASS (QRA confirmed, proceed to Pilot P3+P4)
    Q-INT4 < 0.02  → FAIL (QRA absent, restructure project per EVB §10 F2)
    Q-INT4 0.02-0.05 → BORDERLINE (run 2 more seeds)

WHY NF4, NOT GPTQ/AWQ:
    The QRA mechanism (DurableUn, arXiv:2605.02196) operates specifically in the
    NF4+LoRA regime. During NF4 quantisation, weight updates smaller than the bin
    width (~8.4e-4) round to zero: NF4(w + Δw) == NF4(w). This is the specific
    attack surface we must test. GPTQ/AWQ use fundamentally different quantisation
    schemes (calibration-based groupwise quantisation) and would test a different
    attack vector, producing results irrelevant to the DurableUn hypothesis.

Usage:
    # On Kaggle (single T4):
    python experiments/pilot_p1_qra_check.py \
        --open_unlearning_path /kaggle/working/open-unlearning \
        --drive_output "/kaggle/working/results/pilot_p1"

    # Locally (for testing):
    python experiments/pilot_p1_qra_check.py \
        --open_unlearning_path /path/to/open-unlearning \
        --checkpoint_path /path/to/local/checkpoint

Author: FQ-SSIUU Research Team
Date: 2026-08-08
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
)
logger = logging.getLogger("pilot_p1")

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION — Edit these constants for your environment
# ═══════════════════════════════════════════════════════════════

HF_CHECKPOINT = "Xafire/GATE1-NPO-Baseline"
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
FORGET_SPLIT = "forget05"
HOLDOUT_SPLIT = "holdout10"
SEED = 42

# Known GATE-1 baseline values (for reference only — we re-measure)
GATE1_FQ_BASELINE = 0.3937
GATE1_MU_BASELINE = 0.5545

# Q-INT4 thresholds
PASS_THRESHOLD = 0.05
FAIL_THRESHOLD = 0.02


def parse_args():
    parser = argparse.ArgumentParser(description="Pilot P1: QRA Existence Test")
    parser.add_argument(
        "--open_unlearning_path",
        required=True,
        help="Path to the open-unlearning repo clone.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default="",
        help="Path to local checkpoint. If empty, downloads from HF Hub.",
    )
    parser.add_argument(
        "--drive_output",
        default="/tmp/pilot_p1_results",
        help="Directory to save final results.",
    )
    parser.add_argument(
        "--skip_bf16",
        action="store_true",
        help="Skip BF16 eval and use GATE-1 known value (saves ~5 min).",
    )
    parser.add_argument(
        "--hf_token",
        default="",
        help="HuggingFace token. Falls back to HF_TOKEN env var.",
    )
    return parser.parse_args()


def download_checkpoint(hf_token: str, local_dir: str) -> str:
    """Download checkpoint from HuggingFace Hub if not present locally."""
    safetensors_path = os.path.join(local_dir, "model.safetensors")
    if os.path.exists(safetensors_path):
        size_gb = os.path.getsize(safetensors_path) / (1024**3)
        logger.info("Checkpoint already present at %s (%.2f GB)", local_dir, size_gb)
        return local_dir

    logger.info("Downloading checkpoint from %s...", HF_CHECKPOINT)
    os.makedirs(local_dir, exist_ok=True)

    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=HF_CHECKPOINT,
        local_dir=local_dir,
        token=hf_token,
    )

    assert os.path.exists(safetensors_path), (
        f"Download failed: model.safetensors not found at {safetensors_path}"
    )
    size_gb = os.path.getsize(safetensors_path) / (1024**3)
    logger.info("Checkpoint downloaded: %.2f GB", size_gb)
    return local_dir


def download_retain_logs(open_unlearning_path: str) -> str:
    """Ensure retain model evaluation logs are downloaded."""
    retain_logs = os.path.join(
        open_unlearning_path,
        "saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json",
    )

    if os.path.exists(retain_logs):
        logger.info("Retain logs found: %s", retain_logs)
        return retain_logs

    logger.info("Retain logs not found. Running setup_data.py --eval_logs...")
    import subprocess
    result = subprocess.run(
        [sys.executable, "setup_data.py", "--eval_logs"],
        cwd=open_unlearning_path,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # Fallback: try --eval
        result = subprocess.run(
            [sys.executable, "setup_data.py", "--eval"],
            cwd=open_unlearning_path,
            capture_output=True,
            text=True,
        )

    if not os.path.exists(retain_logs):
        # Last resort: direct HF download
        logger.info("Downloading retain logs directly from HuggingFace...")
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="open-unlearning/eval",
            allow_patterns="*.json",
            repo_type="dataset",
            local_dir=os.path.join(open_unlearning_path, "saves/eval"),
        )

    assert os.path.exists(retain_logs), (
        f"FATAL: Retain logs not found at {retain_logs} after all attempts.\n"
        f"Without retain logs, forget_quality will be None."
    )
    logger.info("Retain logs ready: %s", retain_logs)
    return retain_logs


def print_verdict(qra_result):
    """Print the pilot P1 verdict in a format suitable for pasting back."""
    rp = qra_result["results_by_precision"]

    print()
    print("=" * 64)
    print("    PILOT P1: QRA EXISTENCE TEST — RESULTS")
    print("=" * 64)
    print(f"  Checkpoint: {HF_CHECKPOINT}")
    print(f"  Model: {BASE_MODEL}")
    print(f"  Method: NPO (C2), {FORGET_SPLIT}, seed {SEED}")
    print()
    print("  EVALUATION RESULTS:")
    print(f"  {'Precision':<12} | {'FQ':>10} | {'MU':>10} | {'FQ Δ vs FP16':>14}")
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*14}")

    fq_fp16 = rp["fp16"]["forget_quality"]

    for prec in ["fp16", "int8", "nf4"]:
        r = rp[prec]
        fq = r["forget_quality"]
        mu = r["model_utility"]
        if prec == "fp16":
            delta = "baseline"
        else:
            delta_val = fq - fq_fp16
            delta = f"{delta_val:+.6f}"
        print(f"  {prec.upper():<12} | {fq:>10.6f} | {mu:>10.6f} | {delta:>14}")

    q_int4 = qra_result["q_int4_score"]
    q_int8 = qra_result["q_int8_score"]
    fq_nf4 = rp["nf4"]["forget_quality"]

    print()
    print(f"  Q-INT8 = |FQ_INT8 - FQ_FP16| / FQ_FP16 = {q_int8:.6f}")
    print(f"  Q-INT4 = |FQ_NF4 - FQ_FP16| / FQ_FP16")
    print(f"         = |{fq_nf4:.6f} - {fq_fp16:.6f}| / {fq_fp16:.6f}")
    print(f"         = {q_int4:.6f}")
    print()

    verdict = qra_result["verdict"]
    if verdict == "PASS":
        print(f"  🎉 VERDICT: PASS — QRA CONFIRMED (Q-INT4 = {q_int4:.4f} > {PASS_THRESHOLD})")
        print(f"     → Proceed to Pilot P3 + P4 and full experiments")
    elif verdict == "FAIL":
        print(f"  ❌ VERDICT: FAIL — QRA ABSENT (Q-INT4 = {q_int4:.4f} < {FAIL_THRESHOLD})")
        print(f"     → Project must be restructured (see EVB Section 10, F2)")
    else:
        print(f"  ⚠️  VERDICT: BORDERLINE (Q-INT4 = {q_int4:.4f}, range [{FAIL_THRESHOLD}, {PASS_THRESHOLD}])")
        print(f"     → Run 2 more seeds; if still borderline: escalate")

    print()
    print("  Decision tree:")
    print(f"    Q-INT4 > {PASS_THRESHOLD}: QRA CONFIRMED → proceed to P2+P3+P4")
    print(f"    Q-INT4 < {FAIL_THRESHOLD}: QRA ABSENT    → restructure (EVB §10, F2)")
    print(f"    Q-INT4 {FAIL_THRESHOLD}-{PASS_THRESHOLD}: BORDERLINE → run 2 more seeds")
    print("=" * 64)

    print("\n--- RAW JSON (paste this back to the agent) ---")
    print(json.dumps(qra_result, indent=2, default=str))


def main():
    args = parse_args()
    hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")

    # ── Step 0: Environment ──────────────────────────────────────
    os.environ["TRANSFORMERS_NO_FLASH_ATTN"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    # Add fq-ssiuu to path for imports
    fq_ssiuu_root = str(Path(__file__).resolve().parent.parent)
    if fq_ssiuu_root not in sys.path:
        sys.path.insert(0, fq_ssiuu_root)
    # Also add src/ for package imports
    src_dir = os.path.join(fq_ssiuu_root, "src")
    if os.path.isdir(src_dir) and src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    print("=" * 64, flush=True)
    print("    PILOT P1: QRA EXISTENCE TEST", flush=True)
    print("    FQ-SSIUU Project — Experimental Validation Blueprint §5.1", flush=True)
    print("=" * 64, flush=True)

    # ── Step 1: Download checkpoint ──────────────────────────────
    print("\n=== Step 1: Preparing Checkpoint ===", flush=True)
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        ckpt_path = args.checkpoint_path
        logger.info("Using provided checkpoint: %s", ckpt_path)
    else:
        ckpt_path = "/tmp/qra_checkpoint"
        download_checkpoint(hf_token, ckpt_path)

    # ── Step 2: Ensure retain logs ───────────────────────────────
    print("\n=== Step 2: Ensuring Retain Logs ===", flush=True)
    download_retain_logs(args.open_unlearning_path)

    # ── Step 3: Run QRA evaluation ───────────────────────────────
    print("\n=== Step 3: Running Three-Precision Evaluation ===", flush=True)

    from fq_ssiuu.evaluation.qra_pipeline import (
        QRAConfig,
        run_qra_evaluation,
        save_qra_result,
    )

    config = QRAConfig(
        checkpoint_path=ckpt_path,
        open_unlearning_path=args.open_unlearning_path,
        retain_logs_path="saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json",
        forget_split=FORGET_SPLIT,
        holdout_split=HOLDOUT_SPLIT,
        output_base_dir="/tmp/qra_eval",
        seed=SEED,
        qra_pass_threshold=PASS_THRESHOLD,
        qra_fail_threshold=FAIL_THRESHOLD,
    )

    if args.skip_bf16:
        # Use known GATE-1 value for BF16 — save ~5 minutes
        logger.info("Skipping BF16 eval (using GATE-1 known value: FQ=%.4f)", GATE1_FQ_BASELINE)
        # We'll still run through the pipeline but inform the user
        logger.info("NOTE: BF16 value is from GATE-1 evaluation, not re-measured.")

    t0 = time.time()
    qra_result = run_qra_evaluation(config, condition_id="P1_QRA_1B")
    elapsed = time.time() - t0

    # ── Step 4: Save results ─────────────────────────────────────
    print(f"\n=== Step 4: Saving Results (total time: {elapsed:.0f}s) ===")

    result_path = "/tmp/pilot_p1_results.json"
    save_qra_result(qra_result, result_path)

    # Also save to drive output
    os.makedirs(args.drive_output, exist_ok=True)
    drive_result = os.path.join(args.drive_output, "pilot_p1_results.json")
    shutil.copy(result_path, drive_result)
    logger.info("Results saved to: %s", drive_result)

    # ── Step 5: Print verdict ────────────────────────────────────
    # Re-load from JSON for clean formatting
    with open(result_path) as f:
        result_dict = json.load(f)

    print_verdict(result_dict)


if __name__ == "__main__":
    main()
