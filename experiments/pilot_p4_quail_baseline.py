#!/usr/bin/env python3
"""
Pilot P4: QUAIL Baseline — Condition C8
========================================

EVB Table 5.1 Specification:
    Hypothesis: QUAIL logit-space hinge loss (Mishra & Mehreen 2026,
    arXiv:2601.15538) produces an unlearning signal that survives NF4
    quantisation better than NPO (C2), because QUAIL explicitly targets
    the NF4 bin-width margin in output space.

Protocol:
    1. Load BASE_MODEL_ID (meta-llama/Llama-3.2-1B-Instruct) at NF4 with
       fresh LoRA adapters — NO pre-trained NPO checkpoint.
    2. Load HF_BASE_MODEL (the TOFU fine-tuned model) as a frozen reference
       for the QUAIL hinge comparison.
    3. Train on TOFU forget05 for 10 epochs using compute_quail_loss().
    4. After training, merge LoRA adapter into base model.
    5. Run three-precision QRA evaluation on the merged model.
    6. Compare Q-INT4 with NPO baseline (0.2149 from P1).

WHY THIS IS CONDITION C8 (not C2):
    P1 (C2) uses NPO loss on an already-unlearned checkpoint.
    P4 (C8) uses QUAIL loss starting from the fine-tuned base.
    Both target QRA robustness, but via different mechanisms:
      - NPO: probability-space divergence from reference model
      - QUAIL: logit-space hinge ensuring deviation > NF4 bin margin
    Comparing Q-INT4 scores tells us which loss is inherently more
    quantisation-robust BEFORE any FQ-SSIUU weight-space intervention.

WHY WE START FROM THE FINE-TUNED BASE:
    For a fair comparison, QUAIL must start from the same pre-unlearning
    state as NPO. The NPO baseline was trained on the fine-tuned model
    (open-unlearning/tofu_Llama-3.2-1B-Instruct_full). QUAIL does the
    same: it takes that fine-tuned model and applies unlearning with the
    QUAIL loss. Starting from an NPO checkpoint would conflate the two
    unlearning signals.

QUAIL OPERATES ON LOGITS — NO FQ-QAT NEEDED:
    Unlike FQ-SSIUU which operates on weight space and requires fake-quant
    wrappers, QUAIL's hinge loss is computed entirely in logit (output)
    space: max(0, margin - |logit_θ - logit_ref|). This means no
    FakeQuantLoRALinear wrapping is needed for the QUAIL baseline.

Usage:
    # On Kaggle (single T4):
    python experiments/pilot_p4_quail_baseline.py \\
        --open_unlearning_path /kaggle/working/open-unlearning \\
        --drive_output /kaggle/working/results/pilot_p4

    # Dry run (2 steps, no QRA):
    python experiments/pilot_p4_quail_baseline.py \\
        --open_unlearning_path /kaggle/working/open-unlearning \\
        --dry-run

Author: FQ-SSIUU Research Team
Date: 2026-08-09
"""

import argparse
import csv
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
logger = logging.getLogger("pilot_p4")

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION — Constants matching EVB specification
# ═══════════════════════════════════════════════════════════════

FORGET_SPLIT = "forget05"
N_EPOCHS = 10
LR = 1e-5
SEED = 42
NF4_BIN_WIDTH = 8.4e-4
GRADIENT_CLIP = 1.0
CHECKPOINT_DIR = "/kaggle/working/pilot_p4_quail_checkpoints"
OUTPUT_DIR = "/kaggle/working/results/pilot_p4"
HF_BASE_MODEL = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
BASE_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

# NPO baseline Q-INT4 from Pilot P1 (condition C2) — our comparison target
NPO_BASELINE_Q_INT4 = 0.2149

# Effective batch size via gradient accumulation
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 32
EFFECTIVE_BATCH_SIZE = PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS  # 32

# Tokenizer max length for TOFU dataset
MAX_SEQ_LEN = 512


def parse_args():
    """Parse command-line arguments for P4 QUAIL baseline experiment."""
    parser = argparse.ArgumentParser(
        description="Pilot P4: QUAIL Baseline (Condition C8)"
    )
    parser.add_argument(
        "--open_unlearning_path",
        required=True,
        help="Path to the open-unlearning repo clone (needed for QRA eval).",
    )
    parser.add_argument(
        "--drive_output",
        default=OUTPUT_DIR,
        help="Directory to save final results and merged model.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run only 2 steps of epoch 1, skip QRA evaluation.",
    )
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility across torch, numpy, and random."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"  Random seed set to {seed}", flush=True)


def load_tofu_forget_dataset(tokenizer, max_length: int = MAX_SEQ_LEN):
    """Load and tokenize the TOFU forget05 dataset.

    The TOFU dataset contains question-answer pairs about fictional authors.
    The forget split contains the data we want the model to 'unlearn'.

    Each sample is tokenized as: "<question> <answer>" with the question
    tokens masked (-100) so the loss is only computed on the answer.

    Args:
        tokenizer: The HuggingFace tokenizer.
        max_length: Maximum sequence length for tokenization.

    Returns:
        torch.utils.data.DataLoader: DataLoader with batch_size=1.
    """
    import torch
    from datasets import load_dataset

    print(f"  Loading TOFU dataset: locuslab/TOFU, split={FORGET_SPLIT}", flush=True)
    ds = load_dataset("locuslab/TOFU", FORGET_SPLIT)

    # The dataset has a single split called 'train' within the forget05 config
    if isinstance(ds, dict):
        ds = ds["train"]

    print(f"  Dataset size: {len(ds)} samples", flush=True)

    # Tokenize each sample: question + answer, mask question tokens
    tokenized_samples = []
    for sample in ds:
        question = sample["question"]
        answer = sample["answer"]

        # Tokenize question and answer separately to find boundary
        q_tokens = tokenizer(
            question,
            add_special_tokens=True,
            return_tensors="pt",
        )
        full_text = question + " " + answer
        full_tokens = tokenizer(
            full_text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            add_special_tokens=True,
            return_tensors="pt",
        )

        input_ids = full_tokens["input_ids"].squeeze(0)  # [T]
        attention_mask = full_tokens["attention_mask"].squeeze(0)  # [T]

        # Create labels: mask question tokens with -100
        labels = input_ids.clone()
        q_len = q_tokens["input_ids"].shape[1]
        labels[:q_len] = -100
        # Also mask padding tokens
        labels[attention_mask == 0] = -100

        tokenized_samples.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        })

    # Build a simple dataset
    class TOFUDataset(torch.utils.data.Dataset):
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx]

    dataset = TOFUDataset(tokenized_samples)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=PER_DEVICE_BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )

    print(
        f"  DataLoader ready: {len(dataloader)} batches/epoch "
        f"(batch_size={PER_DEVICE_BATCH_SIZE}, grad_accum={GRADIENT_ACCUMULATION_STEPS}, "
        f"effective_batch={EFFECTIVE_BATCH_SIZE})",
        flush=True,
    )
    return dataloader


def setup_csv_logger(output_dir: str) -> str:
    """Create and initialise the training log CSV file.

    Args:
        output_dir: Directory to write the CSV file.

    Returns:
        str: Absolute path to the CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "training_log.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "epoch", "loss", "gradient_norm"])
    print(f"  Training log CSV: {csv_path}", flush=True)
    return csv_path


def log_step(csv_path: str, step: int, epoch: int, loss: float, grad_norm: float):
    """Append a single training step to the CSV log.

    Args:
        csv_path: Path to the training log CSV.
        step: Global step number.
        epoch: Current epoch (1-indexed).
        loss: Loss value for this step.
        grad_norm: Gradient norm before clipping.
    """
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([step, epoch, f"{loss:.8f}", f"{grad_norm:.8f}"])


def train_quail(
    model,
    ref_model,
    dataloader,
    optimizer,
    quail_config,
    csv_path: str,
    dry_run: bool = False,
):
    """Run the QUAIL training loop.

    For each forget batch:
      1. Forward pass through the model to get model_logits
      2. Forward pass through the frozen reference to get ref_logits
      3. Compute QUAIL hinge loss
      4. Backward, clip gradients, optimizer step (with gradient accumulation)

    Args:
        model: The LoRA-wrapped NF4 model (trainable).
        ref_model: The frozen reference model (no gradients).
        dataloader: DataLoader yielding forget05 batches.
        optimizer: torch.optim.AdamW optimizer.
        quail_config: QUAILConfig for the hinge loss.
        csv_path: Path to CSV log file.
        dry_run: If True, run only 2 steps of epoch 1.

    Returns:
        int: Total number of training steps completed.
    """
    import torch

    from fq_ssiuu.unlearning.quail_loss import compute_quail_loss

    model.train()
    ref_model.eval()

    global_step = 0
    total_epochs = 1 if dry_run else N_EPOCHS
    max_steps_per_epoch = 2 if dry_run else float("inf")

    print(f"\n{'='*60}", flush=True)
    print(f"  QUAIL TRAINING LOOP", flush=True)
    print(f"  Epochs: {total_epochs}, LR: {LR}, Gradient Clip: {GRADIENT_CLIP}", flush=True)
    print(f"  QUAIL margin: {quail_config.margin} (NF4 bin width / 2)", flush=True)
    print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS} steps", flush=True)
    if dry_run:
        print(f"  *** DRY RUN: 2 steps only ***", flush=True)
    print(f"{'='*60}", flush=True)

    for epoch in range(1, total_epochs + 1):
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_start = time.time()

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader):
            if epoch_steps >= max_steps_per_epoch:
                break

            # Move batch to GPU
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)

            # Forward pass through trainable model
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                model_outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                model_logits = model_outputs.logits  # [B, T, V]

                # Forward pass through frozen reference (no gradients)
                with torch.no_grad():
                    ref_outputs = ref_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    ref_logits = ref_outputs.logits  # [B, T, V]

                # Compute QUAIL hinge loss
                loss = compute_quail_loss(
                    model_logits=model_logits,
                    ref_logits=ref_logits,
                    labels=labels,
                    config=quail_config,
                )

                # Scale loss for gradient accumulation
                loss = loss / GRADIENT_ACCUMULATION_STEPS

            # Backward pass
            loss.backward()

            # Step on accumulation boundary or last batch
            is_accum_boundary = (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0
            is_last_batch = (batch_idx + 1) == len(dataloader)

            if is_accum_boundary or is_last_batch:
                # Compute gradient norm before clipping (for logging)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=GRADIENT_CLIP
                )

                optimizer.step()
                optimizer.zero_grad()

                global_step += 1
                # Recover unscaled loss for logging
                step_loss = loss.item() * GRADIENT_ACCUMULATION_STEPS
                epoch_loss += step_loss
                epoch_steps += 1

                # Log to CSV
                log_step(csv_path, global_step, epoch, step_loss, grad_norm.item())

                if global_step % 5 == 0 or global_step <= 3 or dry_run:
                    print(
                        f"  [Epoch {epoch}/{total_epochs}] "
                        f"Step {global_step} | "
                        f"Loss: {step_loss:.6f} | "
                        f"Grad norm: {grad_norm:.4f}",
                        flush=True,
                    )

        # End of epoch summary
        epoch_elapsed = time.time() - epoch_start
        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        print(
            f"\n  Epoch {epoch}/{total_epochs} complete: "
            f"avg_loss={avg_epoch_loss:.6f}, "
            f"steps={epoch_steps}, "
            f"time={epoch_elapsed:.1f}s",
            flush=True,
        )

        # Save checkpoint after every epoch
        ckpt_dir = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}")
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        print(f"  Checkpoint saved: {ckpt_dir}", flush=True)

    print(f"\n  Training complete. Total steps: {global_step}", flush=True)
    return global_step


def print_verdict(qra_result_dict: dict, q_int4: float):
    """Print the P4 verdict comparing QUAIL vs NPO baseline.

    Args:
        qra_result_dict: The QRA result loaded from JSON.
        q_int4: The Q-INT4 score from this experiment.
    """
    rp = qra_result_dict["results_by_precision"]

    print(flush=True)
    print("=" * 64, flush=True)
    print("    PILOT P4: QUAIL BASELINE (C8) — RESULTS", flush=True)
    print("=" * 64, flush=True)
    print(f"  Base Model: {BASE_MODEL_ID}", flush=True)
    print(f"  Reference Model: {HF_BASE_MODEL}", flush=True)
    print(f"  Method: QUAIL (C8), {FORGET_SPLIT}, seed {SEED}", flush=True)
    print(f"  Training: {N_EPOCHS} epochs, LR={LR}, margin={NF4_BIN_WIDTH/2}", flush=True)
    print(flush=True)

    # Per-precision results table
    print("  EVALUATION RESULTS:", flush=True)
    print(
        f"  {'Precision':<12} | {'FQ':>10} | {'MU':>10} | {'FQ Δ vs FP16':>14}",
        flush=True,
    )
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*14}", flush=True)

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
        print(f"  {prec.upper():<12} | {fq:>10.6f} | {mu:>10.6f} | {delta:>14}", flush=True)

    # Q-INT4 comparison
    print(flush=True)
    print(f"  QUAIL Q-INT4: {q_int4:.6f}", flush=True)
    print(f"  NPO  Q-INT4: {NPO_BASELINE_Q_INT4:.6f} (from Pilot P1, condition C2)", flush=True)
    print(flush=True)

    # Verdict
    if q_int4 < NPO_BASELINE_Q_INT4:
        pct = (1 - q_int4 / NPO_BASELINE_Q_INT4) * 100
        print(
            f"  ✅ QUAIL Q-INT4 ({q_int4:.4f}) < NPO Q-INT4 ({NPO_BASELINE_Q_INT4:.4f})",
            flush=True,
        )
        print(
            f"     QUAIL is {pct:.1f}% MORE quantisation-robust than NPO.",
            flush=True,
        )
        print(
            f"     → QUAIL unlearning signal better survives NF4 quantisation.",
            flush=True,
        )
    elif q_int4 > NPO_BASELINE_Q_INT4:
        pct = (q_int4 / NPO_BASELINE_Q_INT4 - 1) * 100
        print(
            f"  ⚠️  QUAIL Q-INT4 ({q_int4:.4f}) > NPO Q-INT4 ({NPO_BASELINE_Q_INT4:.4f})",
            flush=True,
        )
        print(
            f"     QUAIL is {pct:.1f}% LESS quantisation-robust than NPO.",
            flush=True,
        )
        print(
            f"     → NPO unlearning is more durable; QUAIL logit-space approach",
            flush=True,
        )
        print(
            f"       does not translate to better QRA resistance at this scale.",
            flush=True,
        )
    else:
        print(
            f"  ≈ QUAIL Q-INT4 ({q_int4:.4f}) ≈ NPO Q-INT4 ({NPO_BASELINE_Q_INT4:.4f})",
            flush=True,
        )
        print(f"     No significant difference between methods.", flush=True)

    print(flush=True)
    print("  Interpretation guide:", flush=True)
    print("    Lower Q-INT4  → more quantisation-robust (unlearning survives NF4)", flush=True)
    print("    Higher Q-INT4 → less quantisation-robust (NF4 reverses unlearning)", flush=True)
    print("=" * 64, flush=True)


def main():
    args = parse_args()

    # ── Step 0: Environment setup ────────────────────────────────
    os.environ["TRANSFORMERS_NO_FLASH_ATTN"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Add fq-ssiuu to path for package imports
    fq_ssiuu_root = str(Path(__file__).resolve().parent.parent)
    if fq_ssiuu_root not in sys.path:
        sys.path.insert(0, fq_ssiuu_root)
    src_dir = os.path.join(fq_ssiuu_root, "src")
    if os.path.isdir(src_dir) and src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    print("=" * 64, flush=True)
    print("    PILOT P4: QUAIL BASELINE (Condition C8)", flush=True)
    print("    FQ-SSIUU Project — Experimental Validation Blueprint §5.1", flush=True)
    print("=" * 64, flush=True)
    print(f"  Dry run: {args.dry_run}", flush=True)
    print(f"  Output: {args.drive_output}", flush=True)
    print(f"  QUAIL margin: {NF4_BIN_WIDTH / 2} (NF4 bin width {NF4_BIN_WIDTH} / 2)", flush=True)
    print(f"  NPO baseline Q-INT4: {NPO_BASELINE_Q_INT4} (from P1, for comparison)", flush=True)

    set_seed(SEED)

    # ── Step 1: Load models ──────────────────────────────────────
    print("\n=== Step 1: Loading Models ===", flush=True)

    import torch

    from fq_ssiuu.models.loader import (
        get_reference_model,
        load_base_model,
        merge_adapter,
        ModelConfig,
        wrap_with_lora,
    )

    # 1a. Load base model at NF4 and wrap with FRESH LoRA adapters
    #     We use BASE_MODEL_ID (the pre-trained Llama) so that the LoRA
    #     adapters start from scratch — a fair comparison with NPO.
    print(f"  Loading base model ({BASE_MODEL_ID}) at NF4...", flush=True)
    base_config = ModelConfig(base_model_id=BASE_MODEL_ID)
    base_model = load_base_model(config=base_config)

    print("  Wrapping with fresh LoRA adapters...", flush=True)
    model = wrap_with_lora(base_model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"  LoRA model ready: {trainable:,} trainable / {total:,} total "
        f"({100.0 * trainable / total:.2f}%)",
        flush=True,
    )

    # 1b. Load reference model (the TOFU fine-tuned base) — frozen
    #     The QUAIL hinge compares current model logits vs this reference.
    #     Using the fine-tuned base means the hinge pushes logits AWAY from
    #     the model that has memorised the forget set.
    print(f"  Loading frozen reference model ({HF_BASE_MODEL})...", flush=True)
    ref_model = get_reference_model(HF_BASE_MODEL, config=base_config)
    print("  Reference model loaded and frozen.", flush=True)

    if torch.cuda.is_available():
        mem_gb = torch.cuda.memory_allocated() / (1024**3)
        print(f"  GPU memory after loading both models: {mem_gb:.2f} GiB", flush=True)

    # ── Step 2: Load tokenizer and dataset ───────────────────────
    print("\n=== Step 2: Loading TOFU forget05 Dataset ===", flush=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        token=os.environ.get("HF_TOKEN", None),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("  Set pad_token = eos_token", flush=True)

    dataloader = load_tofu_forget_dataset(tokenizer, max_length=MAX_SEQ_LEN)

    # ── Step 3: Setup optimizer and QUAIL config ─────────────────
    print("\n=== Step 3: Setup Optimizer & QUAIL Config ===", flush=True)

    from fq_ssiuu.unlearning.quail_loss import QUAILConfig

    quail_config = QUAILConfig(
        nf4_bin_width=NF4_BIN_WIDTH,
        margin=NF4_BIN_WIDTH / 2,
    )
    print(f"  QUAIL config: bin_width={quail_config.nf4_bin_width}, "
          f"margin={quail_config.margin}", flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
    )
    print(f"  Optimizer: AdamW, LR={LR}", flush=True)

    csv_path = setup_csv_logger(args.drive_output)

    # ── Step 4: Training loop ────────────────────────────────────
    print("\n=== Step 4: QUAIL Training Loop ===", flush=True)
    t0_train = time.time()

    total_steps = train_quail(
        model=model,
        ref_model=ref_model,
        dataloader=dataloader,
        optimizer=optimizer,
        quail_config=quail_config,
        csv_path=csv_path,
        dry_run=args.dry_run,
    )

    train_elapsed = time.time() - t0_train
    print(f"\n  Training time: {train_elapsed:.1f}s ({total_steps} steps)", flush=True)

    # ── Step 5: Save final LoRA adapter ──────────────────────────
    print("\n=== Step 5: Saving Final LoRA Adapter ===", flush=True)
    final_adapter_dir = os.path.join(CHECKPOINT_DIR, "final_adapter")
    os.makedirs(final_adapter_dir, exist_ok=True)
    model.save_pretrained(final_adapter_dir)
    print(f"  Final adapter saved: {final_adapter_dir}", flush=True)

    # Free training model and reference model to reclaim VRAM for merging
    del model
    del ref_model
    del optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("  Training models freed from GPU.", flush=True)

    # ── Step 6: Merge adapter & run QRA evaluation ───────────────
    if args.dry_run:
        print("\n=== Step 6: SKIPPED (dry-run mode) ===", flush=True)
        print("  Dry run complete. Skipping merge and QRA evaluation.", flush=True)
        print(f"  Training log: {csv_path}", flush=True)
        print(f"  Checkpoints: {CHECKPOINT_DIR}", flush=True)

        # Save a dry-run summary
        summary = {
            "condition": "C8_QUAIL",
            "dry_run": True,
            "total_steps": total_steps,
            "training_time_seconds": train_elapsed,
            "csv_log": csv_path,
            "checkpoint_dir": CHECKPOINT_DIR,
        }
        summary_path = os.path.join(args.drive_output, "pilot_p4_dry_run_summary.json")
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Dry-run summary: {summary_path}", flush=True)
        return

    print("\n=== Step 6a: Merging LoRA Adapter into Base Model ===", flush=True)
    t0_merge = time.time()

    # merge_adapter loads base at BF16, applies PEFT adapter, merges
    merged_model = merge_adapter(
        peft_model_path=final_adapter_dir,
        base_model_id=BASE_MODEL_ID,
    )

    # Save merged model to disk for QRA evaluation
    merged_model_dir = os.path.join(args.drive_output, "merged_model")
    os.makedirs(merged_model_dir, exist_ok=True)
    merged_model.save_pretrained(merged_model_dir)

    # Also save the tokenizer alongside the merged model
    tokenizer.save_pretrained(merged_model_dir)

    merge_elapsed = time.time() - t0_merge
    print(f"  Merged model saved: {merged_model_dir} ({merge_elapsed:.1f}s)", flush=True)

    # Free merged model before QRA evaluation
    del merged_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("  Merged model freed from GPU.", flush=True)

    # ── Step 6b: Run three-precision QRA evaluation ──────────────
    print("\n=== Step 6b: Running Three-Precision QRA Evaluation ===", flush=True)

    from fq_ssiuu.evaluation.qra_pipeline import (
        QRAConfig,
        run_qra_evaluation,
        save_qra_result,
    )

    qra_config = QRAConfig(
        checkpoint_path=merged_model_dir,
        open_unlearning_path=args.open_unlearning_path,
        retain_logs_path="saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json",
        forget_split=FORGET_SPLIT,
        holdout_split="holdout10",
        output_base_dir=os.path.join(args.drive_output, "qra_eval"),
        seed=SEED,
    )

    t0_eval = time.time()
    qra_result = run_qra_evaluation(qra_config, condition_id="P4_QUAIL_C8")
    eval_elapsed = time.time() - t0_eval
    print(f"\n  QRA evaluation time: {eval_elapsed:.1f}s", flush=True)

    # Save QRA result
    result_path = os.path.join(args.drive_output, "pilot_p4_qra_result.json")
    save_qra_result(qra_result, result_path)

    # Also copy to a standard location
    os.makedirs(args.drive_output, exist_ok=True)
    shutil.copy(result_path, os.path.join(args.drive_output, "pilot_p4_results.json"))

    # Save full experiment metadata
    experiment_meta = {
        "condition": "C8_QUAIL",
        "base_model_id": BASE_MODEL_ID,
        "reference_model": HF_BASE_MODEL,
        "forget_split": FORGET_SPLIT,
        "n_epochs": N_EPOCHS,
        "lr": LR,
        "seed": SEED,
        "nf4_bin_width": NF4_BIN_WIDTH,
        "gradient_clip": GRADIENT_CLIP,
        "per_device_batch_size": PER_DEVICE_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "total_training_steps": total_steps,
        "training_time_seconds": train_elapsed,
        "merge_time_seconds": merge_elapsed,
        "eval_time_seconds": eval_elapsed,
        "q_int4_score": qra_result.q_int4_score,
        "q_int8_score": qra_result.q_int8_score,
        "verdict": qra_result.verdict,
        "npo_baseline_q_int4": NPO_BASELINE_Q_INT4,
        "csv_log": csv_path,
    }
    meta_path = os.path.join(args.drive_output, "pilot_p4_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(experiment_meta, f, indent=2)
    print(f"  Experiment metadata: {meta_path}", flush=True)

    # ── Step 7: Print verdict ────────────────────────────────────
    print("\n=== Step 7: Verdict ===", flush=True)

    # Reload from JSON for clean formatting
    with open(result_path) as f:
        result_dict = json.load(f)

    print_verdict(result_dict, qra_result.q_int4_score)

    # Final raw JSON for pasting
    print("\n--- RAW JSON (paste this back to the agent) ---", flush=True)
    print(json.dumps(result_dict, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
