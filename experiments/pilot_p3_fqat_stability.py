#!/usr/bin/env python3
"""
Pilot P3: FQ-QAT + NPO Training Stability Test
=================================================

EVB Table 5.3 Specification:
    Objective: Verify that FQ-QAT wrapped LoRA + NPO unlearning training is
    numerically stable on Llama-3.2-1B-Instruct for 5 epochs on the TOFU
    forget01 split (40 examples).

This is a STABILITY TEST, not a publishable result. It checks:
    - Loss does not diverge (no NaN/Inf, no > 100 explosion)
    - Gradient norms remain bounded (< 5.0 consistently)
    - Loss decreases monotonically across epoch means
    - Model coherence is preserved (no gibberish generation)

WHY THIS TEST EXISTS:
    FQ-QAT inserts fake-quantisation (QDQ + STE) into the LoRA adapter
    forward pass. The STE introduces discontinuities in the loss landscape.
    Combined with NPO's bounded loss, there is a risk that:
    1. Gradients vanish through STE when adapter deltas are small
    2. NPO's softplus saturates, producing zero gradients
    3. The combination causes training to stall or oscillate
    This pilot verifies none of these pathologies occur.

ARCHITECTURE:
    - Base model: Llama-3.2-1B-Instruct at NF4 (frozen)
    - LoRA adapters: loaded from Xafire/GATE1-NPO-Baseline
    - FQ-QAT: wraps LoRA layers via wrap_model_with_fq_lora()
    - Reference model: open-unlearning/tofu_Llama-3.2-1B-Instruct_full (frozen)
    - Loss: NPO (compute_npo_loss) with β=0.1
    - Optimiser: AdamW (lr=1e-5, grad clip=1.0)

Stability Verdict:
    PASS:     Loss decreases monotonically per epoch mean; epoch-mean grad
              norm < 1.0; no coherence failures.
    FAIL:     Any epoch loss increase > 3x; grad norm > 5.0 consistently;
              coherence fails > 2 epochs.

Usage:
    python experiments/pilot_p3_fqat_stability.py \\
        --open_unlearning_path /path/to/open-unlearning \\
        --drive_output "/content/drive/My Drive/fq-ssiuu-research/results/pilot_p3"

    # Quick pipeline validation (2 steps only):
    python experiments/pilot_p3_fqat_stability.py \\
        --open_unlearning_path /path/to/open-unlearning \\
        --dry-run

Author: FQ-SSIUU Research Team
Date: 2026-08-09
"""

import argparse
import csv
import gc
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION CONSTANTS
# ═══════════════════════════════════════════════════════════════

FORGET_SPLIT = "forget01"       # 40 examples — smallest split for speed
N_EPOCHS = 5
LR = 1e-5
SEED = 42
GRADIENT_CLIP = 1.0
BETA = 0.1                     # NPO temperature parameter
CHECKPOINT_DIR = "/kaggle/working/pilot_p3_checkpoints"
HF_CHECKPOINT = "Xafire/GATE1-NPO-Baseline"
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
FINE_TUNED_MODEL = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"  # reference

# Training constraints for T4 GPU (15.6 GB VRAM)
BATCH_SIZE = 1                  # Per-device, limited by VRAM
GRAD_ACCUM_STEPS = 4            # Effective batch = 4
MAX_SEQ_LEN = 256               # Truncated for VRAM budget

# Coherence check
COHERENCE_PROMPT = (
    "The capital of France is Paris. Paris is known for the Eiffel Tower, "
    "which was built in 1889. The tower stands at"
)
COHERENCE_MAX_TOKENS = 50
COHERENCE_REPETITION_THRESHOLD = 0.50  # Fail if any token > 50% of output

# Stability thresholds
LOSS_DIVERGENCE_THRESHOLD = 100.0      # Emergency stop
LOSS_INCREASE_FACTOR = 3.0             # Epoch mean increase factor → FAIL
GRAD_NORM_PASS_THRESHOLD = 1.0         # Epoch mean < 1.0 → PASS
GRAD_NORM_FAIL_THRESHOLD = 5.0         # Consistent > 5.0 → FAIL
MAX_COHERENCE_FAILURES = 2             # More than 2 → FAIL


# ═══════════════════════════════════════════════════════════════
# ARGUMENT PARSING
# ═══════════════════════════════════════════════════════════════

def parse_args():
    """Parse command-line arguments for the P3 stability test.

    Returns:
        argparse.Namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Pilot P3: FQ-QAT + NPO Training Stability Test"
    )
    parser.add_argument(
        "--open_unlearning_path",
        required=True,
        help="Path to open-unlearning repository root (required for sys.path).",
    )
    parser.add_argument(
        "--drive_output",
        default="/tmp/pilot_p3_results",
        help="Directory to save results JSON and CSV log.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run only 2 steps of epoch 1 for pipeline validation.",
    )
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def set_seed(seed: int):
    """Set random seeds for reproducibility across torch, numpy, and Python."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[SEED] Random seed set to {seed}", flush=True)


def measure_vram():
    """Measure current and peak GPU VRAM usage in GB.

    Returns:
        Tuple of (current_gb, peak_gb). Returns (0.0, 0.0) if no CUDA.
    """
    import torch

    if not torch.cuda.is_available():
        return 0.0, 0.0
    current = torch.cuda.memory_allocated() / (1024 ** 3)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    return current, peak


def compute_grad_norm(model):
    """Compute the total L2 gradient norm across all trainable parameters.

    Only includes parameters that have .requires_grad=True and a non-None
    .grad attribute. Returns 0.0 if no gradients exist yet.

    Args:
        model: The training model (PeftModel wrapped with FQ-QAT).

    Returns:
        float: Total L2 gradient norm.
    """
    import torch

    total_norm_sq = 0.0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            total_norm_sq += p.grad.data.float().norm().item() ** 2
    return total_norm_sq ** 0.5


def check_coherence(model, tokenizer, device):
    """Generate text from a retain-domain prompt and check for gibberish.

    Gibberish is detected by checking if any single token constitutes more
    than COHERENCE_REPETITION_THRESHOLD (50%) of the generated output. This
    catches degenerate repetition (e.g., "the the the the the...").

    Args:
        model: The model to evaluate (should be in eval mode temporarily).
        tokenizer: The tokenizer for encoding/decoding.
        device: CUDA device for generation.

    Returns:
        Tuple of (passed: bool, generated_text: str, details: dict).
    """
    import torch

    model.eval()
    inputs = tokenizer(
        COHERENCE_PROMPT,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN,
    ).to(device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model.generate(
                **inputs,
                max_new_tokens=COHERENCE_MAX_TOKENS,
                do_sample=False,           # Greedy for deterministic check
                temperature=None,          # temperature must be None if do_sample=False
                top_p=None,                # top_p must be None if do_sample=False
                pad_token_id=tokenizer.pad_token_id,
            )

    # Decode only the generated portion (exclude the prompt)
    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0, prompt_len:].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Check for degenerate repetition: if any single token > 50% of output
    if len(generated_ids) == 0:
        return False, "", {"reason": "empty_generation", "token_counts": {}}

    token_counts = Counter(generated_ids)
    most_common_token, most_common_count = token_counts.most_common(1)[0]
    repetition_ratio = most_common_count / len(generated_ids)

    passed = repetition_ratio < COHERENCE_REPETITION_THRESHOLD
    details = {
        "n_tokens_generated": len(generated_ids),
        "most_common_token_id": most_common_token,
        "most_common_token_str": tokenizer.decode([most_common_token]),
        "repetition_ratio": round(repetition_ratio, 4),
        "threshold": COHERENCE_REPETITION_THRESHOLD,
        "passed": passed,
    }

    model.train()
    return passed, generated_text, details


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_forget_dataset(tokenizer):
    """Load the TOFU forget01 dataset and tokenize for causal LM training.

    The forget01 split contains 40 question-answer pairs about fictitious
    authors. Each example is formatted as "Question: ... Answer: ..." and
    tokenized with labels set to the input_ids (shifted internally by the
    model for causal LM loss, but NPO handles its own label alignment).

    Labels use -100 masking for padding tokens so they don't contribute
    to the NPO loss computation.

    Args:
        tokenizer: The tokenizer to use for encoding.

    Returns:
        List of dicts, each with 'input_ids', 'attention_mask', 'labels'
        as tensors on CPU.
    """
    import torch
    from datasets import load_dataset

    print(f"[DATA] Loading TOFU '{FORGET_SPLIT}' split...", flush=True)
    ds = load_dataset("locuslab/TOFU", FORGET_SPLIT, split="train")
    print(f"[DATA] Loaded {len(ds)} examples from {FORGET_SPLIT}", flush=True)

    tokenized = []
    for i, example in enumerate(ds):
        text = f"Question: {example['question']}\nAnswer: {example['answer']}"

        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding="max_length",
        )

        input_ids = encoded["input_ids"].squeeze(0)       # [T]
        attention_mask = encoded["attention_mask"].squeeze(0)  # [T]

        # Labels = input_ids, with -100 at padding positions
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        tokenized.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        })

    print(f"[DATA] Tokenized {len(tokenized)} examples "
          f"(max_seq_len={MAX_SEQ_LEN})", flush=True)
    return tokenized


# ═══════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════

def load_training_model():
    """Load the training model: NF4 base + fresh LoRA + FQ-QAT wrapping.

    Pipeline:
        1. load_base_model() → NF4 quantised finetuned base model
        2. get_peft_model() → attach fresh LoRA adapter
        3. wrap_model_with_fq_lora() → apply fake-quantisation to LoRA layers

    Returns:
        Tuple of (model, tokenizer, device).
    """
    import torch
    from peft import PeftModel, LoraConfig, get_peft_model, TaskType
    from transformers import AutoTokenizer

    from fq_ssiuu.models.loader import ModelConfig, load_base_model
    from fq_ssiuu.quantisation.fq_hooks import wrap_model_with_fq_lora

    # Step 1: Load NF4-quantised base model (finetuned)
    print("[MODEL] Loading NF4 base model...", flush=True)
    config = ModelConfig(base_model_id=FINE_TUNED_MODEL)
    base_model = load_base_model(config)

    vram_cur, vram_peak = measure_vram()
    print(f"[MODEL] Base model loaded. VRAM: {vram_cur:.2f} GB "
          f"(peak: {vram_peak:.2f} GB)", flush=True)

    # Step 2: Attach fresh LoRA adapter
    print(f"[MODEL] Initializing fresh LoRA adapter...", flush=True)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(base_model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] LoRA attached. Trainable: {trainable:,} / {total:,} "
          f"({100.0 * trainable / total:.2f}%)", flush=True)

    # Step 3: Wrap LoRA layers with FQ-QAT
    print("[MODEL] Applying FQ-QAT wrapping to LoRA layers...", flush=True)
    model, n_wrapped = wrap_model_with_fq_lora(model)
    print(f"[MODEL] FQ-QAT wrapped {n_wrapped} LoRA layers", flush=True)

    vram_cur, vram_peak = measure_vram()
    print(f"[MODEL] Training model ready. VRAM: {vram_cur:.2f} GB "
          f"(peak: {vram_peak:.2f} GB)", flush=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = next(model.parameters()).device
    model.train()

    return model, tokenizer, device


def load_reference_model():
    """Load the frozen reference model for NPO loss computation.

    The reference model is the fine-tuned model BEFORE unlearning
    (open-unlearning/tofu_Llama-3.2-1B-Instruct_full). It is loaded
    at NF4 quantisation and fully frozen (no gradients).

    Returns:
        The frozen reference model.
    """
    from fq_ssiuu.models.loader import get_reference_model

    print(f"[REF] Loading frozen reference model: {FINE_TUNED_MODEL}", flush=True)
    ref_model = get_reference_model(FINE_TUNED_MODEL)

    vram_cur, vram_peak = measure_vram()
    print(f"[REF] Reference model loaded. VRAM: {vram_cur:.2f} GB "
          f"(peak: {vram_peak:.2f} GB)", flush=True)

    return ref_model


def download_checkpoint(hf_checkpoint: str, target_dir: str):
    """Download LoRA adapter checkpoint from HuggingFace Hub.

    Args:
        hf_checkpoint: HuggingFace repo ID (e.g., 'Xafire/GATE1-NPO-Baseline').
        target_dir: Local directory to download into.

    Returns:
        str: Path to the downloaded checkpoint directory.
    """
    from huggingface_hub import snapshot_download

    os.makedirs(target_dir, exist_ok=True)
    # Check if already downloaded (adapter_config.json is the sentinel file)
    sentinel = os.path.join(target_dir, "adapter_config.json")
    if os.path.exists(sentinel):
        print(f"[CKPT] Checkpoint already present at {target_dir}", flush=True)
        return target_dir

    print(f"[CKPT] Downloading {hf_checkpoint} to {target_dir}...", flush=True)
    hf_token = os.environ.get("HF_TOKEN", "")
    snapshot_download(
        repo_id=hf_checkpoint,
        local_dir=target_dir,
        token=hf_token if hf_token else None,
    )
    print(f"[CKPT] Download complete: {target_dir}", flush=True)
    return target_dir


# ═══════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════

def train(
    model,
    ref_model,
    tokenizer,
    dataset,
    device,
    dry_run: bool = False,
):
    """Run the FQ-QAT + NPO training loop with stability monitoring.

    For each step:
        1. Forward pass through both training and reference models
        2. Compute NPO loss (bounded by design)
        3. Backward pass with gradient accumulation
        4. Log gradient norm and loss to CSV
        5. Clip gradients and step optimizer every GRAD_ACCUM_STEPS

    Emergency stops if loss > 100 (divergence detected).

    At the end of each epoch:
        - Run coherence check (generate text, detect gibberish)
        - Save LoRA adapter checkpoint

    Args:
        model: FQ-QAT wrapped training model.
        ref_model: Frozen reference model for NPO.
        tokenizer: Tokenizer for coherence check.
        dataset: List of tokenized examples.
        device: CUDA device.
        dry_run: If True, run only 2 steps of epoch 1.

    Returns:
        Dict with training metrics and stability verdict.
    """
    import torch
    from torch.optim import AdamW

    from fq_ssiuu.unlearning.npo_loss import compute_npo_loss

    # ── Setup ────────────────────────────────────────────────
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=0.0,  # No weight decay for stability test
    )

    # CSV log for step-level metrics
    csv_path = "/tmp/pilot_p3_training_log.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "step", "loss", "gradient_norm", "vram_gb"])

    # Tracking accumulators
    all_step_losses = []          # Every step's loss value
    epoch_mean_losses = []        # Mean loss per epoch
    epoch_mean_grad_norms = []    # Mean grad norm per epoch
    coherence_results = []        # Per-epoch coherence check results
    diverged = False              # Emergency stop flag
    divergence_step = None

    n_epochs = 1 if dry_run else N_EPOCHS
    max_steps_per_epoch = 2 if dry_run else len(dataset)

    print(f"\n{'=' * 64}", flush=True)
    print(f"    TRAINING: {'DRY RUN (2 steps)' if dry_run else f'{n_epochs} epochs'}", flush=True)
    print(f"    Dataset: {len(dataset)} examples, batch_size={BATCH_SIZE}", flush=True)
    print(f"    Grad accum: {GRAD_ACCUM_STEPS}, Effective batch: "
          f"{BATCH_SIZE * GRAD_ACCUM_STEPS}", flush=True)
    print(f"    LR: {LR}, Beta: {BETA}, Grad clip: {GRADIENT_CLIP}", flush=True)
    print(f"{'=' * 64}\n", flush=True)

    global_step = 0
    t_start = time.time()

    for epoch in range(n_epochs):
        epoch_losses = []
        epoch_grad_norms = []
        model.train()

        print(f"\n--- Epoch {epoch + 1}/{n_epochs} ---", flush=True)

        for step_in_epoch, example in enumerate(dataset):
            if step_in_epoch >= max_steps_per_epoch:
                break

            global_step += 1

            # Move batch to device — shape: [1, T] (batch_size=1)
            input_ids = example["input_ids"].unsqueeze(0).to(device)
            attention_mask = example["attention_mask"].unsqueeze(0).to(device)
            labels = example["labels"].unsqueeze(0).to(device)

            # ── Forward pass with mixed precision ────────────
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # Training model forward
                model_outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                model_logits = model_outputs.logits  # [1, T, V]

                # Reference model forward (no gradient)
                with torch.no_grad():
                    ref_outputs = ref_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    ref_logits = ref_outputs.logits   # [1, T, V]

                # Compute NPO loss
                loss = compute_npo_loss(
                    model_logits=model_logits,
                    ref_logits=ref_logits,
                    labels=labels,
                    beta=BETA,
                )

                # Scale loss for gradient accumulation
                loss_scaled = loss / GRAD_ACCUM_STEPS

            # ── Backward ─────────────────────────────────────
            loss_scaled.backward()

            loss_val = loss.item()
            epoch_losses.append(loss_val)
            all_step_losses.append(loss_val)

            # ── Check for divergence ─────────────────────────
            if loss_val > LOSS_DIVERGENCE_THRESHOLD or not (
                loss_val == loss_val  # NaN check: NaN != NaN
            ):
                print(f"[EMERGENCY] Loss diverged at epoch {epoch + 1}, "
                      f"step {step_in_epoch + 1}: loss={loss_val:.4f}", flush=True)
                diverged = True
                divergence_step = global_step
                break

            # ── Gradient accumulation step ────────────────────
            if (step_in_epoch + 1) % GRAD_ACCUM_STEPS == 0 or \
               (step_in_epoch + 1) == max_steps_per_epoch:
                # Compute grad norm before clipping
                grad_norm = compute_grad_norm(model)
                epoch_grad_norms.append(grad_norm)

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    GRADIENT_CLIP,
                )

                optimizer.step()
                optimizer.zero_grad()

                # Log to CSV
                vram_cur, _ = measure_vram()
                csv_writer.writerow([
                    epoch + 1,
                    global_step,
                    f"{loss_val:.6f}",
                    f"{grad_norm:.6f}",
                    f"{vram_cur:.3f}",
                ])
                csv_file.flush()

                print(f"  [Step {global_step:4d}] epoch={epoch + 1} "
                      f"step={step_in_epoch + 1:3d}/{max_steps_per_epoch} "
                      f"loss={loss_val:.4f} grad_norm={grad_norm:.4f} "
                      f"vram={vram_cur:.2f}GB", flush=True)

            else:
                # Log step even without optimizer step
                vram_cur, _ = measure_vram()
                csv_writer.writerow([
                    epoch + 1,
                    global_step,
                    f"{loss_val:.6f}",
                    "",  # No grad norm between accum steps
                    f"{vram_cur:.3f}",
                ])

        if diverged:
            break

        # ── Epoch summary ────────────────────────────────────
        epoch_mean_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        epoch_mean_grad = (
            sum(epoch_grad_norms) / len(epoch_grad_norms)
            if epoch_grad_norms else 0.0
        )
        epoch_mean_losses.append(epoch_mean_loss)
        epoch_mean_grad_norms.append(epoch_mean_grad)

        print(f"\n  [Epoch {epoch + 1} Summary] "
              f"mean_loss={epoch_mean_loss:.4f} "
              f"mean_grad_norm={epoch_mean_grad:.4f}", flush=True)

        # ── Coherence check ──────────────────────────────────
        print(f"  [Epoch {epoch + 1}] Running coherence check...", flush=True)
        coh_passed, coh_text, coh_details = check_coherence(
            model, tokenizer, device
        )
        coherence_results.append({
            "epoch": epoch + 1,
            "passed": coh_passed,
            "generated_text": coh_text[:200],  # Truncate for JSON
            "details": coh_details,
        })
        coh_status = "✅ PASS" if coh_passed else "❌ FAIL"
        print(f"  [Epoch {epoch + 1}] Coherence: {coh_status}", flush=True)
        print(f"  [Epoch {epoch + 1}] Generated: \"{coh_text[:100]}...\"", flush=True)

        # ── Save checkpoint ──────────────────────────────────
        ckpt_dir = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch + 1}")
        os.makedirs(ckpt_dir, exist_ok=True)
        try:
            model.save_pretrained(ckpt_dir)
            print(f"  [Epoch {epoch + 1}] Checkpoint saved to: {ckpt_dir}", flush=True)
        except Exception as e:
            print(f"  [Epoch {epoch + 1}] WARNING: Checkpoint save failed: {e}", flush=True)

    csv_file.close()
    elapsed = time.time() - t_start

    # ── Compute stability verdict ────────────────────────────
    verdict, verdict_reasons = compute_verdict(
        epoch_mean_losses=epoch_mean_losses,
        epoch_mean_grad_norms=epoch_mean_grad_norms,
        coherence_results=coherence_results,
        diverged=diverged,
        divergence_step=divergence_step,
    )

    results = {
        "pilot": "P3",
        "test": "FQ-QAT + NPO Training Stability",
        "dry_run": dry_run,
        "base_model": BASE_MODEL,
        "fine_tuned_model": FINE_TUNED_MODEL,
        "hf_checkpoint": HF_CHECKPOINT,
        "forget_split": FORGET_SPLIT,
        "n_examples": len(dataset),
        "n_epochs": n_epochs,
        "lr": LR,
        "beta": BETA,
        "gradient_clip": GRADIENT_CLIP,
        "batch_size": BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "max_seq_len": MAX_SEQ_LEN,
        "total_steps": global_step,
        "elapsed_seconds": round(elapsed, 2),
        "epoch_mean_losses": [round(x, 6) for x in epoch_mean_losses],
        "epoch_mean_grad_norms": [round(x, 6) for x in epoch_mean_grad_norms],
        "all_step_losses": [round(x, 6) for x in all_step_losses],
        "coherence_results": coherence_results,
        "diverged": diverged,
        "divergence_step": divergence_step,
        "verdict": verdict,
        "verdict_reasons": verdict_reasons,
        "csv_log_path": csv_path,
    }

    vram_cur, vram_peak = measure_vram()
    results["vram_peak_gb"] = round(vram_peak, 3)
    results["vram_final_gb"] = round(vram_cur, 3)

    return results


def compute_verdict(
    epoch_mean_losses,
    epoch_mean_grad_norms,
    coherence_results,
    diverged,
    divergence_step,
):
    """Determine the PASS/FAIL stability verdict from training metrics.

    PASS criteria (all must hold):
        - Loss decreases monotonically across epoch means
        - All epoch-mean gradient norms < 1.0
        - No coherence failures

    FAIL criteria (any one triggers FAIL):
        - Training diverged (loss > 100 or NaN)
        - Any epoch-over-epoch loss increase > 3x
        - Gradient norm > 5.0 consistently (majority of epochs)
        - Coherence failures > 2 epochs

    Args:
        epoch_mean_losses: List of mean loss per epoch.
        epoch_mean_grad_norms: List of mean gradient norm per epoch.
        coherence_results: List of coherence check dicts per epoch.
        diverged: Whether emergency divergence stop was triggered.
        divergence_step: Step at which divergence occurred.

    Returns:
        Tuple of (verdict: str, reasons: list[str]).
    """
    reasons = []
    is_fail = False

    # Check 1: Divergence
    if diverged:
        is_fail = True
        reasons.append(
            f"FAIL: Training diverged at step {divergence_step} "
            f"(loss > {LOSS_DIVERGENCE_THRESHOLD} or NaN)"
        )

    # Check 2: Loss monotonicity and 3x increase
    if len(epoch_mean_losses) >= 2:
        loss_monotonic = True
        for i in range(1, len(epoch_mean_losses)):
            prev = epoch_mean_losses[i - 1]
            curr = epoch_mean_losses[i]
            if curr > prev:
                loss_monotonic = False
                if prev > 0 and curr / prev > LOSS_INCREASE_FACTOR:
                    is_fail = True
                    reasons.append(
                        f"FAIL: Epoch {i + 1} loss ({curr:.4f}) increased "
                        f"> {LOSS_INCREASE_FACTOR}x over epoch {i} ({prev:.4f})"
                    )
        if loss_monotonic:
            reasons.append("PASS: Loss decreases monotonically across epochs")
        elif not is_fail:
            reasons.append(
                "WARN: Loss not strictly monotonic, but increase < 3x"
            )

    # Check 3: Gradient norms
    if epoch_mean_grad_norms:
        all_below_pass = all(g < GRAD_NORM_PASS_THRESHOLD for g in epoch_mean_grad_norms)
        high_grad_epochs = sum(
            1 for g in epoch_mean_grad_norms if g > GRAD_NORM_FAIL_THRESHOLD
        )

        if all_below_pass:
            reasons.append(
                f"PASS: All epoch-mean gradient norms < {GRAD_NORM_PASS_THRESHOLD}"
            )
        elif high_grad_epochs > len(epoch_mean_grad_norms) / 2:
            is_fail = True
            reasons.append(
                f"FAIL: Gradient norm > {GRAD_NORM_FAIL_THRESHOLD} in "
                f"{high_grad_epochs}/{len(epoch_mean_grad_norms)} epochs "
                f"(consistently high)"
            )
        else:
            reasons.append(
                f"WARN: Some epoch grad norms above {GRAD_NORM_PASS_THRESHOLD}, "
                f"but not consistently > {GRAD_NORM_FAIL_THRESHOLD}"
            )

    # Check 4: Coherence
    n_coherence_fails = sum(
        1 for r in coherence_results if not r["passed"]
    )
    if n_coherence_fails == 0:
        reasons.append("PASS: All coherence checks passed")
    elif n_coherence_fails <= MAX_COHERENCE_FAILURES:
        reasons.append(
            f"WARN: {n_coherence_fails} coherence failure(s) "
            f"(≤ {MAX_COHERENCE_FAILURES} threshold)"
        )
    else:
        is_fail = True
        reasons.append(
            f"FAIL: {n_coherence_fails} coherence failures "
            f"(> {MAX_COHERENCE_FAILURES} threshold)"
        )

    verdict = "FAIL" if is_fail else "PASS"
    return verdict, reasons


# ═══════════════════════════════════════════════════════════════
# VERDICT DISPLAY
# ═══════════════════════════════════════════════════════════════

def print_verdict(results):
    """Print a formatted verdict block suitable for copy-paste into reports.

    Args:
        results: Dict of training results from train().
    """
    verdict = results["verdict"]
    icon = "✅" if verdict == "PASS" else "❌"

    print(flush=True)
    print("=" * 64, flush=True)
    print("    PILOT P3: FQ-QAT + NPO TRAINING STABILITY", flush=True)
    print("=" * 64, flush=True)
    print(f"  Base model:     {results['base_model']}", flush=True)
    print(f"  Reference:      {results['fine_tuned_model']}", flush=True)
    print(f"  LoRA adapter:   {results['hf_checkpoint']}", flush=True)
    print(f"  Forget split:   {results['forget_split']} "
          f"({results['n_examples']} examples)", flush=True)
    print(f"  Dry run:        {results['dry_run']}", flush=True)
    print(flush=True)
    print(f"  Epochs:         {results['n_epochs']}", flush=True)
    print(f"  Total steps:    {results['total_steps']}", flush=True)
    print(f"  Elapsed:        {results['elapsed_seconds']:.1f}s", flush=True)
    print(f"  VRAM peak:      {results['vram_peak_gb']:.3f} GB", flush=True)
    print(flush=True)

    # Epoch-level metrics table
    print("  Epoch │ Mean Loss   │ Mean Grad Norm │ Coherence", flush=True)
    print("  ──────┼─────────────┼────────────────┼──────────", flush=True)
    for i in range(len(results["epoch_mean_losses"])):
        loss_str = f"{results['epoch_mean_losses'][i]:.6f}"
        grad_str = (
            f"{results['epoch_mean_grad_norms'][i]:.6f}"
            if i < len(results["epoch_mean_grad_norms"]) else "N/A"
        )
        coh_str = (
            "✅" if results["coherence_results"][i]["passed"] else "❌"
        ) if i < len(results["coherence_results"]) else "N/A"
        print(f"  {i + 1:5d} │ {loss_str:>11s} │ {grad_str:>14s} │ {coh_str}",
              flush=True)
    print(flush=True)

    # Verdict reasons
    print("  Stability Analysis:", flush=True)
    for reason in results["verdict_reasons"]:
        prefix = "    ✅" if reason.startswith("PASS") else (
            "    ❌" if reason.startswith("FAIL") else "    ⚠️ "
        )
        print(f"{prefix} {reason}", flush=True)
    print(flush=True)

    print(f"  {icon} VERDICT: {verdict}", flush=True)
    if results["diverged"]:
        print(f"  ⚠️  Diverged at step {results['divergence_step']}", flush=True)
    print("=" * 64, flush=True)

    # Raw JSON for programmatic consumption
    print("\n--- RAW JSON ---", flush=True)
    print(json.dumps(results, indent=2, default=str), flush=True)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    """Main entry point for Pilot P3 stability test."""
    args = parse_args()

    # Environment setup
    os.environ["TRANSFORMERS_NO_FLASH_ATTN"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Add open-unlearning to sys.path if provided
    if args.open_unlearning_path:
        ou_path = os.path.abspath(args.open_unlearning_path)
        if ou_path not in sys.path:
            sys.path.insert(0, ou_path)
            print(f"[PATH] Added to sys.path: {ou_path}", flush=True)

    # Add project root to sys.path (for fq_ssiuu imports)
    project_root = str(Path(__file__).resolve().parent.parent)
    src_path = os.path.join(project_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
        print(f"[PATH] Added to sys.path: {src_path}", flush=True)

    print("=" * 64, flush=True)
    print("    PILOT P3: FQ-QAT + NPO TRAINING STABILITY TEST", flush=True)
    print("    FQ-SSIUU Project — EVB §5.3", flush=True)
    print(f"    Mode: {'DRY RUN (2 steps)' if args.dry_run else 'FULL (5 epochs)'}", flush=True)
    print("=" * 64, flush=True)

    set_seed(SEED)

    # ── Step 2: Load training model ──────────────────────────
    print("\n=== Step 2: Loading Training Model (NF4 + LoRA + FQ-QAT) ===", flush=True)
    model, tokenizer, device = load_training_model()

    # ── Step 3: Load reference model ─────────────────────────
    print("\n=== Step 3: Loading Reference Model (Frozen) ===", flush=True)
    ref_model = load_reference_model()

    # ── Step 4: Load forget dataset ──────────────────────────
    print("\n=== Step 4: Loading TOFU Forget Dataset ===", flush=True)
    dataset = load_forget_dataset(tokenizer)

    # ── Step 5: Training loop ────────────────────────────────
    print("\n=== Step 5: Training Loop ===", flush=True)
    import torch
    torch.cuda.reset_peak_memory_stats()

    results = train(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        device=device,
        dry_run=args.dry_run,
    )

    # ── Step 6: Save results ─────────────────────────────────
    print("\n=== Step 6: Saving Results ===", flush=True)

    # Save to /tmp first
    result_path = "/tmp/pilot_p3_results.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[SAVE] Results saved to: {result_path}", flush=True)

    # Copy to drive output
    os.makedirs(args.drive_output, exist_ok=True)
    drive_result = os.path.join(args.drive_output, "pilot_p3_results.json")
    shutil.copy(result_path, drive_result)
    print(f"[SAVE] Results copied to: {drive_result}", flush=True)

    # Copy CSV log
    csv_src = results.get("csv_log_path", "/tmp/pilot_p3_training_log.csv")
    if os.path.exists(csv_src):
        csv_dst = os.path.join(args.drive_output, "pilot_p3_training_log.csv")
        shutil.copy(csv_src, csv_dst)
        print(f"[SAVE] Training log copied to: {csv_dst}", flush=True)

    # ── Step 7: Print verdict ────────────────────────────────
    print("\n=== Step 7: Verdict ===", flush=True)
    print_verdict(results)

    # Clean up
    del model, ref_model
    gc.collect()
    torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    main()
