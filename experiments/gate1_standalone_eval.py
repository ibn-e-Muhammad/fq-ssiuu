#!/usr/bin/env python3
"""
GATE-1 Standalone Evaluation Script
====================================

Evaluates the NPO baseline checkpoint (GATE1-C2-NPO-SEED42) using
OpenUnlearning's eval.py with the correct retain_logs_path to compute
forget_quality, model_utility, and all TOFU metrics.

MUST run on a SINGLE GPU (no accelerate launch, no DDP).
OpenUnlearning's evaluator is incompatible with multi-GPU mode.

Usage on Kaggle (as a notebook cell):
    Copy the contents of this script into a Kaggle notebook cell and run.
    Ensure Internet access is enabled in the notebook settings.

Usage as standalone script:
    cd /kaggle/working/open-unlearning
    CUDA_VISIBLE_DEVICES=0 python ../fq-ssiuu/experiments/gate1_standalone_eval.py

Environment Requirements:
    - Single T4 GPU (15 GB VRAM)
    - Python 3.12
    - HuggingFace token with access to meta-llama/Llama-3.2-1B-Instruct

Author: FQ-SSIUU Research Team
Date: August 2026
"""

import os
import sys
import json
import subprocess
import shutil
from pathlib import Path

# ============================================================
# CONFIGURATION — Edit these paths if your setup differs
# ============================================================

# Where open-unlearning is cloned
OPEN_UNLEARNING_DIR = "/kaggle/working/open-unlearning"

# Where the fq-ssiuu repo is cloned
FQ_SSIUU_DIR = "/kaggle/working/fq-ssiuu"

# Where the NPO checkpoint lives (upload the ZIP and extract here)
CHECKPOINT_DIR = "/kaggle/working/checkpoints/c2_npo/seed42"

# HuggingFace token
HF_TOKEN = "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"

# Evaluation output directory
EVAL_OUTPUT_DIR = "/kaggle/working/eval_output/GATE1_STANDALONE"

# Task name for Hydra
TASK_NAME = "GATE1_STANDALONE_EVAL"

# TOFU evaluation config
FORGET_SPLIT = "forget05"
HOLDOUT_SPLIT = "holdout10"
RETAIN_SPLIT = "retain90"  # Corresponds to forget05

# Retain logs path (relative to open-unlearning repo root)
RETAIN_LOGS_RELPATH = f"saves/eval/tofu_Llama-3.2-1B-Instruct_{RETAIN_SPLIT}/TOFU_EVAL.json"


def run_cmd(cmd, cwd=None, check=True):
    """Run a shell command and print output."""
    print(f"\n>>> {cmd}")
    result = subprocess.run(
        cmd, shell=True, cwd=cwd, capture_output=True, text=True
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")
    return result


def step_1_setup_repos():
    """Clone repositories and apply patches."""
    print("=" * 60)
    print("STEP 1: Setting up repositories")
    print("=" * 60)

    # Clone open-unlearning
    if not Path(OPEN_UNLEARNING_DIR).exists():
        run_cmd(
            "git clone https://github.com/ibn-e-Muhammad/open-unlearning.git "
            f"{OPEN_UNLEARNING_DIR}"
        )
    run_cmd(
        "git fetch origin && git checkout fq-ssiuu-integration",
        cwd=OPEN_UNLEARNING_DIR,
    )

    # Clone fq-ssiuu (for this script and configs)
    if not Path(FQ_SSIUU_DIR).exists():
        run_cmd(
            f"git clone https://github.com/ibn-e-Muhammad/fq-ssiuu.git {FQ_SSIUU_DIR}"
        )
    run_cmd(
        "git fetch origin && git checkout gate/1-baseline-repro",
        cwd=FQ_SSIUU_DIR,
    )


def step_2_patch_dependencies():
    """Apply all known dependency patches."""
    print("=" * 60)
    print("STEP 2: Patching dependencies")
    print("=" * 60)

    # Remove toxic version pins
    run_cmd(f"sed -i '/torch==/d' {OPEN_UNLEARNING_DIR}/requirements.txt")
    run_cmd(f"sed -i '/torchvision==/d' {OPEN_UNLEARNING_DIR}/requirements.txt")
    run_cmd(f"sed -i '/bitsandbytes==/d' {OPEN_UNLEARNING_DIR}/requirements.txt")

    # Fix VLM import bug
    run_cmd(
        f"sed -i 's/hf_vlms/huggingface/g' {OPEN_UNLEARNING_DIR}/src/evals/lm_eval.py"
    )

    # Install
    run_cmd(f"pip install -e {OPEN_UNLEARNING_DIR} -q")
    run_cmd("pip install lm-eval==0.4.4 bitsandbytes==0.45.5 -q")
    run_cmd("pip uninstall timm -y -q 2>/dev/null || true")


def step_3_download_retain_logs():
    """Run setup_data.py --eval to download retain model evaluation logs."""
    print("=" * 60)
    print("STEP 3: Downloading retain model evaluation logs")
    print("=" * 60)

    retain_logs_path = Path(OPEN_UNLEARNING_DIR) / RETAIN_LOGS_RELPATH
    if retain_logs_path.exists():
        print(f"Retain logs already exist at: {retain_logs_path}")
        return

    # Run setup_data.py to download eval logs
    run_cmd(
        "python scripts/setup_data.py --eval",
        cwd=OPEN_UNLEARNING_DIR,
    )

    # Verify
    if not retain_logs_path.exists():
        raise FileNotFoundError(
            f"Retain logs not found at {retain_logs_path} after setup_data.py. "
            f"Check if the file was downloaded to a different location."
        )
    print(f"✅ Retain logs verified at: {retain_logs_path}")


def step_4_verify_checkpoint():
    """Verify the NPO checkpoint exists and has the expected files."""
    print("=" * 60)
    print("STEP 4: Verifying checkpoint")
    print("=" * 60)

    required_files = ["config.json", "model.safetensors", "tokenizer.json"]
    ckpt_path = Path(CHECKPOINT_DIR)

    if not ckpt_path.exists():
        # Try to find the checkpoint from a ZIP
        zip_path = Path("/kaggle/working/NPO_Baseline_Final.zip")
        if zip_path.exists():
            print(f"Found ZIP at {zip_path}, extracting to {CHECKPOINT_DIR}...")
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            run_cmd(f"unzip -o {zip_path} -d {CHECKPOINT_DIR}")
        else:
            # Try HuggingFace Hub
            print("Checkpoint not found locally. Attempting download from HuggingFace Hub...")
            run_cmd(
                f"huggingface-cli download ibn-e-Muhammad/GATE1-NPO-Baseline "
                f"--local-dir {CHECKPOINT_DIR} --token {HF_TOKEN}"
            )

    for f in required_files:
        fpath = ckpt_path / f
        if not fpath.exists():
            raise FileNotFoundError(f"Missing checkpoint file: {fpath}")
        size_mb = fpath.stat().st_size / (1024 * 1024)
        print(f"  ✅ {f} ({size_mb:.1f} MB)")

    print(f"✅ Checkpoint verified at: {CHECKPOINT_DIR}")


def step_5_run_evaluation():
    """Run OpenUnlearning eval.py with correct configuration."""
    print("=" * 60)
    print("STEP 5: Running standalone evaluation")
    print("=" * 60)

    # Set environment
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["TRANSFORMERS_NO_FLASH_ATTN"] = "1"
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "offline"
    os.environ["PYTHONPATH"] = f"{OPEN_UNLEARNING_DIR}/src:{os.environ.get('PYTHONPATH', '')}"

    # Ensure single GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Build the evaluation command
    eval_cmd = (
        f"python src/eval.py "
        f"--config-name=eval.yaml "
        f"experiment=eval/tofu/default "
        f"model=Llama-3.2-1B-Instruct "
        f"model.model_args.pretrained_model_name_or_path={CHECKPOINT_DIR} "
        f'model.model_args.attn_implementation="sdpa" '
        f'model.model_args.torch_dtype="float16" '
        f"forget_split={FORGET_SPLIT} "
        f"holdout_split={HOLDOUT_SPLIT} "
        f"retain_logs_path={RETAIN_LOGS_RELPATH} "
        f"task_name={TASK_NAME} "
    )

    print(f"Running evaluation command:")
    print(f"  cd {OPEN_UNLEARNING_DIR}")
    print(f"  CUDA_VISIBLE_DEVICES=0 {eval_cmd}")

    result = run_cmd(eval_cmd, cwd=OPEN_UNLEARNING_DIR, check=False)

    if result.returncode != 0:
        print(f"\n❌ Evaluation exited with code {result.returncode}")
        print("Check the output above for errors.")
        return False

    print("\n✅ Evaluation completed successfully!")
    return True


def step_6_extract_verdict():
    """Find and parse the evaluation output, print GATE-1 verdict."""
    print("=" * 60)
    print("STEP 6: Extracting evaluation results")
    print("=" * 60)

    # Search for TOFU_SUMMARY.json in the output directory
    ou_dir = Path(OPEN_UNLEARNING_DIR)
    summary_candidates = list(ou_dir.glob("**/TOFU_SUMMARY.json"))

    # Also check the default saves/eval/ path
    default_path = ou_dir / f"saves/eval/{TASK_NAME}/TOFU_SUMMARY.json"
    if default_path.exists() and default_path not in summary_candidates:
        summary_candidates.append(default_path)

    if not summary_candidates:
        print("❌ TOFU_SUMMARY.json not found!")
        print("Looking for any TOFU_EVAL.json files...")
        eval_candidates = list(ou_dir.glob("**/TOFU_EVAL.json"))
        for ec in eval_candidates:
            print(f"  Found: {ec}")
        print("\nPlease locate the evaluation output and parse it manually.")
        return

    # Use the most recently modified summary file
    summary_path = max(summary_candidates, key=lambda p: p.stat().st_mtime)
    print(f"Found evaluation summary at: {summary_path}")

    with open(summary_path, "r") as f:
        summary = json.load(f)

    # Also try to load the detailed eval file
    eval_path = summary_path.parent / "TOFU_EVAL.json"
    eval_data = None
    if eval_path.exists():
        with open(eval_path, "r") as f:
            eval_data = json.load(f)

    # Extract key metrics
    forget_quality = summary.get("forget_quality", "NOT FOUND")
    model_utility = summary.get("model_utility", "NOT FOUND")
    forget_truth_ratio = summary.get("forget_truth_ratio", "NOT FOUND")
    forget_qa_prob = summary.get("forget_Q_A_Prob", "NOT FOUND")
    forget_qa_rouge = summary.get("forget_Q_A_ROUGE", "NOT FOUND")
    extraction_strength = summary.get("extraction_strength", "NOT FOUND")
    privleak = summary.get("privleak", "NOT FOUND")

    # Print all metrics
    print("\n" + "=" * 60)
    print("       GATE-1 EVALUATION RESULTS")
    print("=" * 60)
    print(f"\n  Checkpoint: {CHECKPOINT_DIR}")
    print(f"  Forget split: {FORGET_SPLIT}")
    print(f"  Retain logs: {RETAIN_LOGS_RELPATH}")
    print()

    print("  ┌─────────────────────────┬───────────────┬───────────┬──────────┐")
    print("  │ Metric                  │ Measured      │ Target    │ Verdict  │")
    print("  ├─────────────────────────┼───────────────┼───────────┼──────────┤")

    # forget_quality verdict
    fq_verdict = "PENDING"
    if isinstance(forget_quality, (int, float)) and forget_quality is not None:
        if 0.09 <= forget_quality <= 0.19:
            fq_verdict = "✅ PASS"
        elif forget_quality > 0:
            fq_verdict = "⚠️ COND"
        else:
            fq_verdict = "❌ FAIL"
    fq_str = f"{forget_quality:.4f}" if isinstance(forget_quality, (int, float)) else str(forget_quality)
    print(f"  │ forget_quality          │ {fq_str:>13} │ 0.14±0.05 │ {fq_verdict:>8} │")

    # model_utility verdict
    mu_verdict = "PENDING"
    if isinstance(model_utility, (int, float)) and model_utility is not None:
        if 0.40 <= model_utility <= 0.50:
            mu_verdict = "✅ PASS"
        elif model_utility > 0:
            mu_verdict = "⚠️ COND"
        else:
            mu_verdict = "❌ FAIL"
    mu_str = f"{model_utility:.4f}" if isinstance(model_utility, (int, float)) else str(model_utility)
    print(f"  │ model_utility           │ {mu_str:>13} │ 0.45±0.05 │ {mu_verdict:>8} │")

    # forget_truth_ratio (reference only)
    ftr_str = f"{forget_truth_ratio:.4f}" if isinstance(forget_truth_ratio, (int, float)) else str(forget_truth_ratio)
    print(f"  │ forget_truth_ratio      │ {ftr_str:>13} │ 0.70 ref  │      REF │")

    print("  ├─────────────────────────┼───────────────┼───────────┼──────────┤")

    # Additional metrics
    fqp_str = f"{forget_qa_prob:.4f}" if isinstance(forget_qa_prob, (int, float)) else str(forget_qa_prob)
    print(f"  │ forget_Q_A_Prob         │ {fqp_str:>13} │     —     │      REF │")

    fqr_str = f"{forget_qa_rouge:.4f}" if isinstance(forget_qa_rouge, (int, float)) else str(forget_qa_rouge)
    print(f"  │ forget_Q_A_ROUGE        │ {fqr_str:>13} │     —     │      REF │")

    es_str = f"{extraction_strength:.4f}" if isinstance(extraction_strength, (int, float)) else str(extraction_strength)
    print(f"  │ extraction_strength     │ {es_str:>13} │     —     │      REF │")

    pl_str = f"{privleak:.4f}" if isinstance(privleak, (int, float)) else str(privleak)
    print(f"  │ privleak                │ {pl_str:>13} │     —     │      REF │")

    print("  └─────────────────────────┴───────────────┴───────────┴──────────┘")

    # Overall verdict
    print("\n" + "=" * 60)
    if fq_verdict == "✅ PASS" and mu_verdict == "✅ PASS":
        print("  🎉 GATE-1 VERDICT: PASS")
        print("  Proceed to Pilot P1 (QRA existence test)")
    elif fq_verdict == "❌ FAIL" or mu_verdict == "❌ FAIL":
        print("  ❌ GATE-1 VERDICT: FAIL")
        print("  Training did not produce a valid unlearned model. Investigate and retrain.")
    elif "COND" in fq_verdict or "COND" in mu_verdict:
        print("  ⚠️  GATE-1 VERDICT: CONDITIONAL PASS")
        print("  Results deviate from reference but unlearning occurred.")
        print("  Note: reference was run on 2×L40s; our setup used 2×T4 with 8.64/10 epochs.")
        print("  Proceed to Pilot P1 with this baseline documented.")
    else:
        print("  ⏳ GATE-1 VERDICT: PENDING")
        print("  Some metrics could not be extracted. Check evaluation output above.")
    print("=" * 60)

    # Print raw summary for copy-paste
    print("\n--- RAW SUMMARY JSON (for documentation) ---")
    print(json.dumps(summary, indent=2, default=str))

    return summary


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  GATE-1 STANDALONE EVALUATION")
    print("  Experiment: GATE1-C2-NPO-SEED42")
    print("  Date: August 2026")
    print("=" * 60)

    try:
        step_1_setup_repos()
        step_2_patch_dependencies()
        step_3_download_retain_logs()
        step_4_verify_checkpoint()
        if step_5_run_evaluation():
            step_6_extract_verdict()
        else:
            print("\n⚠️  Evaluation failed. Check errors above.")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
