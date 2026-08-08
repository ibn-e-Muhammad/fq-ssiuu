"""
src/fq_ssiuu/evaluation/qra_pipeline.py

Three-precision Quantisation Recovery Attack (QRA) evaluation pipeline.

This module implements the core evaluation protocol that determines whether
the QRA manifests at a given model scale — the central test of the FQ-SSIUU
research hypothesis.

Reference: DurableUn, arXiv:2605.02196 — first formalisation of QRA.

THE QRA MECHANISM:
    During NF4 quantisation, each weight w is mapped to the nearest of 16
    codebook values. If the unlearning update dw is smaller than the NF4
    bin width (~8.4e-4), then:
        NF4(w + dw) == NF4(w)
    The quantised model is identical to the pre-unlearning model. The
    "forgotten" knowledge is fully recovered at INT4 precision. This is
    the Quantisation Recovery Attack (QRA).

THE THREE-PRECISION PROTOCOL:
    1. Evaluate merged model at FP16 -> measure FQ_FP16 (baseline)
    2. Evaluate merged model at INT8 -> measure FQ_INT8 (intermediate)
    3. Evaluate merged model at NF4  -> measure FQ_NF4  (attack surface)
    4. Compute: Q-INT4 = |FQ_NF4 - FQ_FP16| / (FQ_FP16 + eps)
    5. Verdict: Q-INT4 > 0.05 = PASS (QRA confirmed)

    Each evaluation loads the model FRESH from disk at the target precision.
    Models are never re-quantised from an already-loaded checkpoint.

IMPORTANT: This module calls experiments/_eval_at_precision.py via subprocess.
    That helper script uses Hydra compose API to build the OpenUnlearning
    evaluation config and runs the TOFU evaluator directly on a pre-loaded
    quantised model. This avoids Hydra CLI limitations with BitsAndBytesConfig.
"""

import json
import logging
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# The precisions to evaluate, in order
PRECISIONS: List[str] = ["fp16", "int8", "nf4"]


@dataclass
class QRAConfig:
    """Configuration for the QRA evaluation pipeline.

    Attributes:
        checkpoint_path: Path to the merged model checkpoint.
        open_unlearning_path: Path to the open-unlearning repo clone.
        retain_logs_path: Relative path (from open-unlearning root) to retain TOFU_EVAL.json.
        forget_split: TOFU forget split (default: forget05).
        holdout_split: TOFU holdout split (default: holdout10).
        output_base_dir: Base directory for per-precision evaluation output.
        seed: Random seed.
        qra_pass_threshold: Q-INT4 above this = PASS (QRA confirmed).
        qra_fail_threshold: Q-INT4 below this = FAIL (QRA absent).
    """

    checkpoint_path: str = ""
    open_unlearning_path: str = ""
    retain_logs_path: str = (
        "saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json"
    )
    forget_split: str = "forget05"
    holdout_split: str = "holdout10"
    output_base_dir: str = "/tmp/qra_eval"
    seed: int = 42
    qra_pass_threshold: float = 0.05
    qra_fail_threshold: float = 0.02


@dataclass
class PrecisionResult:
    """Evaluation results at a specific precision level.

    Attributes:
        precision: The precision level ('fp16', 'int8', 'nf4').
        forget_quality: Forget Quality metric (KS p-value, higher = better unlearning).
        model_utility: Model Utility metric (higher = better utility preservation).
        all_metrics: Complete metric dict from TOFU_SUMMARY.json.
        elapsed_seconds: Time taken for this evaluation.
        evaluation_timestamp: ISO 8601 timestamp.
    """

    precision: str
    forget_quality: float
    model_utility: float
    all_metrics: Dict = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    evaluation_timestamp: str = ""


@dataclass
class QRAResult:
    """Complete QRA evaluation result across all precisions.

    Attributes:
        condition_id: Experimental condition identifier.
        seed: Random seed used.
        results_by_precision: Dict mapping precision name to PrecisionResult.
        q_int4_score: |FQ_NF4 - FQ_FP16| / (FQ_FP16 + eps).
        q_int8_score: |FQ_INT8 - FQ_FP16| / (FQ_FP16 + eps) (intermediate check).
        verdict: 'PASS', 'FAIL', or 'BORDERLINE'.
        certified: True if Q-INT4 > pass_threshold (QRA confirmed).
        notes: Free-text notes.
    """

    condition_id: str
    seed: int
    results_by_precision: Dict[str, PrecisionResult] = field(default_factory=dict)
    q_int4_score: float = 0.0
    q_int8_score: float = 0.0
    verdict: str = "PENDING"
    certified: bool = False
    notes: str = ""


def _clear_gpu_memory():
    """Clear GPU memory between evaluations."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("GPU memory cleared.")
    except ImportError:
        pass


def _find_eval_script() -> str:
    """Locate the _eval_at_precision.py helper script."""
    candidates = [
        # When running from the fq-ssiuu repo root
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "experiments", "_eval_at_precision.py"),
        # When installed as a package
        os.path.join(os.getcwd(), "experiments", "_eval_at_precision.py"),
        # Absolute fallback
        "/kaggle/working/fq-ssiuu/experiments/_eval_at_precision.py",
    ]
    for c in candidates:
        abspath = os.path.abspath(c)
        if os.path.isfile(abspath):
            return abspath

    raise FileNotFoundError(
        "Cannot find experiments/_eval_at_precision.py. "
        "Searched: " + ", ".join(candidates)
    )


def evaluate_at_precision(config: QRAConfig, precision: str) -> PrecisionResult:
    """Evaluate the model at a specific precision via subprocess.

    This function calls _eval_at_precision.py as a subprocess. That script
    loads the model at the requested precision, runs the OpenUnlearning TOFU
    evaluator, and saves TOFU_SUMMARY.json.

    Args:
        config: QRA configuration with all paths set.
        precision: One of 'fp16', 'int8', 'nf4'.

    Returns:
        PrecisionResult with metrics extracted from TOFU_SUMMARY.json.

    Raises:
        RuntimeError: If evaluation subprocess fails.
        FileNotFoundError: If required files are missing.
    """
    if precision not in PRECISIONS:
        raise ValueError(f"Invalid precision '{precision}'. Must be one of {PRECISIONS}")

    # Validate retain_logs_path before starting
    retain_abs = os.path.join(config.open_unlearning_path, config.retain_logs_path)
    if not os.path.exists(retain_abs):
        raise FileNotFoundError(
            f"Retain logs not found at: {retain_abs}\n"
            f"Run 'python setup_data.py --eval_logs' in the open-unlearning root.\n"
            f"Without retain logs, forget_quality will be None (the GATE-1 bug)."
        )

    output_dir = os.path.join(config.output_base_dir, precision)
    os.makedirs(output_dir, exist_ok=True)

    eval_script = _find_eval_script()
    logger.info("Using eval helper: %s", eval_script)

    # Build subprocess command
    cmd = [
        sys.executable,
        eval_script,
        "--precision", precision,
        "--checkpoint_path", config.checkpoint_path,
        "--open_unlearning_path", config.open_unlearning_path,
        "--retain_logs_path", config.retain_logs_path,
        "--output_dir", output_dir,
        "--forget_split", config.forget_split,
        "--holdout_split", config.holdout_split,
        "--seed", str(config.seed),
    ]

    logger.info("Running evaluation at %s precision...", precision.upper())
    logger.info("Command: %s", " ".join(cmd))

    # Run subprocess and stream output in real-time
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
    env["TRANSFORMERS_NO_FLASH_ATTN"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONUNBUFFERED"] = "1"  # Ensure real-time output

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=config.open_unlearning_path,
        bufsize=1,
    )

    for line in iter(process.stdout.readline, ""):
        # Use print instead of logger to bypass any logger buffering
        print(f"[{precision.upper()}] {line.rstrip()}", flush=True)

    process.wait()

    if process.returncode != 0:
        raise RuntimeError(
            f"Evaluation at {precision} failed with exit code {process.returncode}.\n"
            f"Check logs above for details."
        )

    # Parse results from TOFU_SUMMARY.json
    summary_path = os.path.join(output_dir, "TOFU_SUMMARY.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"TOFU_SUMMARY.json not found at {summary_path} after evaluation.\n"
            f"The evaluator may have failed silently."
        )

    with open(summary_path) as f:
        summary = json.load(f)

    # Also parse the result marker if available
    marker_path = os.path.join(output_dir, f"_result_{precision}.json")
    elapsed = 0.0
    if os.path.exists(marker_path):
        with open(marker_path) as f:
            marker = json.load(f)
        elapsed = marker.get("elapsed_seconds", 0.0)

    fq = summary.get("forget_quality")
    mu = summary.get("model_utility")

    if fq is None:
        raise RuntimeError(
            f"forget_quality is None at {precision}. "
            "This means retain_logs_path was not set correctly. "
            "This is the GATE-1 bug — must be fixed before continuing."
        )

    timestamp = datetime.now(timezone.utc).isoformat()

    result = PrecisionResult(
        precision=precision,
        forget_quality=fq,
        model_utility=mu if mu is not None else 0.0,
        all_metrics=summary,
        elapsed_seconds=elapsed,
        evaluation_timestamp=timestamp,
    )

    logger.info(
        "%s evaluation complete: FQ=%.6f, MU=%.6f (%.1f sec)",
        precision.upper(), result.forget_quality, result.model_utility, elapsed,
    )

    return result


def compute_q_score(fq_target: float, fq_baseline: float) -> float:
    """Compute the Q score: relative FQ change from baseline.

    Q = |FQ_target - FQ_baseline| / (FQ_baseline + eps)

    A high Q score means significant FQ degradation at the target precision,
    indicating the QRA is working (unlearning is being reversed by quantisation).
    """
    eps = 1e-8
    return abs(fq_target - fq_baseline) / (abs(fq_baseline) + eps)


def run_qra_evaluation(config: QRAConfig, condition_id: str = "P1_QRA") -> QRAResult:
    """Run the full three-precision QRA evaluation pipeline.

    Evaluates the model at FP16 (baseline), INT8 (intermediate), and NF4
    (attack surface) sequentially. Computes Q-INT4 and issues verdict.

    Args:
        config: Fully configured QRAConfig with all paths.
        condition_id: Identifier for this experimental condition.

    Returns:
        QRAResult with all metrics and verdict.
    """
    logger.info("=" * 60)
    logger.info("STARTING QRA EVALUATION PIPELINE")
    logger.info("  Condition: %s", condition_id)
    logger.info("  Checkpoint: %s", config.checkpoint_path)
    logger.info("  Forget split: %s", config.forget_split)
    logger.info("  Precisions: %s", PRECISIONS)
    logger.info("=" * 60)

    results = {}

    for precision in PRECISIONS:
        logger.info("\n--- Evaluating at %s ---", precision.upper())
        try:
            result = evaluate_at_precision(config, precision)
            results[precision] = result
        except Exception as e:
            logger.error("Evaluation at %s FAILED: %s", precision, e)
            raise

        # Clear GPU memory between evaluations
        _clear_gpu_memory()

    # Compute Q scores
    fq_fp16 = results["fp16"].forget_quality
    fq_int8 = results["int8"].forget_quality
    fq_nf4 = results["nf4"].forget_quality

    q_int8 = compute_q_score(fq_int8, fq_fp16)
    q_int4 = compute_q_score(fq_nf4, fq_fp16)

    # Determine verdict
    if q_int4 > config.qra_pass_threshold:
        verdict = "PASS"
        certified = True
    elif q_int4 < config.qra_fail_threshold:
        verdict = "FAIL"
        certified = False
    else:
        verdict = "BORDERLINE"
        certified = False

    notes = (
        f"FQ_FP16={fq_fp16:.6f}, FQ_INT8={fq_int8:.6f}, FQ_NF4={fq_nf4:.6f}. "
        f"Q-INT8={q_int8:.6f}, Q-INT4={q_int4:.6f}. "
        f"Thresholds: pass>{config.qra_pass_threshold}, fail<{config.qra_fail_threshold}."
    )

    qra_result = QRAResult(
        condition_id=condition_id,
        seed=config.seed,
        results_by_precision=results,
        q_int4_score=q_int4,
        q_int8_score=q_int8,
        verdict=verdict,
        certified=certified,
        notes=notes,
    )

    logger.info("=" * 60)
    logger.info("QRA EVALUATION COMPLETE")
    logger.info("  Q-INT4 = %.6f  (threshold: >%.2f for PASS)", q_int4, config.qra_pass_threshold)
    logger.info("  Q-INT8 = %.6f", q_int8)
    logger.info("  VERDICT: %s", verdict)
    logger.info("=" * 60)

    return qra_result


def save_qra_result(result: QRAResult, output_path: str) -> None:
    """Save a QRA result to a JSON file.

    Args:
        result: The QRA evaluation result to save.
        output_path: File path for the JSON output.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    result_dict = {
        "condition_id": result.condition_id,
        "seed": result.seed,
        "q_int4_score": result.q_int4_score,
        "q_int8_score": result.q_int8_score,
        "verdict": result.verdict,
        "certified": result.certified,
        "notes": result.notes,
        "results_by_precision": {
            k: asdict(v) for k, v in result.results_by_precision.items()
        },
    }

    with open(output_path, "w") as f:
        json.dump(result_dict, f, indent=2)

    logger.info("QRA result saved to '%s'", output_path)


# ─── Self-Test ───────────────────────────────────────────────────

def test_q_int4_formula():
    """Verify Q-INT4 formula on known values.

    Scenario: FQ_BF16 = 0.39 (strong unlearning at full precision),
    FQ_NF4 = 0.05 (unlearning mostly reversed by quantisation).
    Expected Q-INT4 ≈ 0.872 (87% relative degradation → QRA confirmed).
    """
    fq_bf16 = 0.39
    fq_nf4 = 0.05
    q = compute_q_score(fq_nf4, fq_bf16)

    expected = abs(0.05 - 0.39) / (0.39 + 1e-8)
    assert abs(q - expected) < 1e-6, f"Q-INT4 formula error: got {q}, expected {expected}"

    # Also verify the PASS threshold
    assert q > 0.05, f"Q-INT4 {q} should be > 0.05 (PASS threshold)"

    # Verify the scenario where QRA is absent (FQ preserved)
    fq_nf4_no_qra = 0.38
    q_no_qra = compute_q_score(fq_nf4_no_qra, fq_bf16)
    assert q_no_qra < 0.05, f"Q-INT4 {q_no_qra} should be < 0.05 when QRA is absent"

    # Verify edge case: zero baseline
    q_zero = compute_q_score(0.1, 0.0)
    assert q_zero > 0, "Q score with zero baseline should still be positive"

    print(f"  Q-INT4 formula test 1: {q:.6f} ≈ {expected:.6f} ✓")
    print(f"  Q-INT4 formula test 2 (no QRA): {q_no_qra:.6f} < 0.05 ✓")
    print(f"  Q-INT4 formula test 3 (zero baseline): {q_zero:.6f} > 0 ✓")
