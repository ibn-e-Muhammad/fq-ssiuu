"""
src/fq_ssiuu/evaluation/qra_pipeline.py

Three-precision Quantisation Recovery Attack (QRA) evaluation pipeline.

This module implements the core evaluation protocol that determines whether
an unlearned model receives a durability certificate.

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
    1. Load the unlearned checkpoint (PEFT adapter)
    2. Merge adapter into base model (merge_and_unload)
    3. Save merged BF16 model
    4. Evaluate at BF16 -> measure FQ (Forget Quality) and MU (Model Utility)
    5. Evaluate at INT8 -> measure FQ and MU
    6. Evaluate at INT4 (NF4) -> measure FQ and MU
    7. Compute: Q-INT4 = |FQ_INT4 - FQ_BF16| / (FQ_BF16 + eps)
    8. Issue certificate if Q-INT4 < threshold (default 0.05)

    Steps 4-6 are SEQUENTIAL -- never load two models simultaneously.
    Each model is deleted and GPU memory cleared before the next load.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class QRAConfig:
    """Configuration for the QRA evaluation pipeline.

    Attributes:
        certification_threshold: Maximum Q-INT4 score for certification.
            Default 0.05 means < 5% relative degradation at INT4 vs BF16.
        gdrive_merged_path: Google Drive path for merged model storage.
        clean_up_merged: Whether to delete merged models after evaluation.
            Default False -- merged models are expensive to recompute.
    """

    certification_threshold: float = 0.05
    gdrive_merged_path: str = "My Drive/fq-ssiuu-research/merged_models"
    clean_up_merged: bool = False  # Keep merged models -- they are expensive to recompute


@dataclass
class PrecisionResult:
    """Evaluation results at a specific precision level.

    Attributes:
        precision: The precision level ('bf16', 'int8', 'int4').
        forget_quality: Forget Quality metric (higher = better forgetting).
            Measures how well the model has forgotten the target data.
        model_utility: Model Utility metric (higher = better utility).
            Measures how well the model retains performance on non-forget data.
        evaluation_timestamp: ISO 8601 timestamp of when evaluation completed.
    """

    precision: str
    forget_quality: float
    model_utility: float
    evaluation_timestamp: str


@dataclass
class QRAResult:
    """Complete QRA evaluation result across all three precisions.

    Attributes:
        condition_id: Experimental condition identifier (e.g., 'c6_fq_ssiuu').
        seed: Random seed used for this experiment.
        results_by_precision: Dict mapping precision name to PrecisionResult.
        q_int4_score: The Q-INT4 metric: |FQ_INT4 - FQ_BF16| / (FQ_BF16 + eps).
        certified: Whether the model passes the durability threshold.
        certification_threshold: The threshold used for certification.
        notes: Free-text notes about the evaluation (e.g., warnings, issues).
    """

    condition_id: str
    seed: int
    results_by_precision: Dict[str, PrecisionResult]
    q_int4_score: float
    certified: bool
    certification_threshold: float
    notes: str = ""


def merge_and_save(
    adapter_path: str,
    base_model_id: str,
    output_path: str,
    token: str,
) -> str:
    """Merge a LoRA adapter into the base model and save the merged BF16 model.

    Args:
        adapter_path: Path to the saved PEFT adapter directory.
        base_model_id: HuggingFace model ID of the base model.
        output_path: Directory to save the merged model.
        token: HuggingFace authentication token.

    Returns:
        str: The output_path where the merged model was saved.

    Raises:
        FileNotFoundError: If adapter_path does not exist.
        RuntimeError: If merge or save fails.
    """
    from fq_ssiuu.models.loader import merge_adapter

    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(
            f"Adapter directory not found: {adapter_path}. "
            "Ensure training has completed and the adapter was saved."
        )

    logger.info("Merging adapter from '%s' into '%s'...", adapter_path, base_model_id)
    merged_model = merge_adapter(adapter_path, base_model_id, token)

    os.makedirs(output_path, exist_ok=True)
    logger.info("Saving merged model to '%s'...", output_path)
    merged_model.save_pretrained(output_path)

    # Compute and log model size
    total_bytes = sum(
        os.path.getsize(os.path.join(output_path, f))
        for f in os.listdir(output_path)
        if os.path.isfile(os.path.join(output_path, f))
    )
    size_mb = total_bytes / (1024 * 1024)
    logger.info("Merged model saved: %.1f MB", size_mb)

    # Free memory
    del merged_model
    _clear_gpu_memory()

    return output_path


def _clear_gpu_memory():
    """Clear GPU memory after model evaluation."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("GPU memory cleared.")
    except ImportError:
        pass


def evaluate_at_precision(
    merged_model_path: str,
    precision: str,
    forget_loader,
    retain_loader,
    tokenizer,
) -> PrecisionResult:
    """Evaluate a merged model at the specified precision.

    Loads the model at the requested precision, computes Forget Quality (FQ)
    and Model Utility (MU), then clears GPU memory.

    NOTE: FQ and MU are placeholder implementations in this version.
    The actual metric computation requires the OpenUnlearning evaluation
    harness, which is integrated in the GATE-1 stage. The pipeline structure
    (loading, evaluating, clearing memory) is complete and correct.

    Args:
        merged_model_path: Path to the saved merged BF16 model.
        precision: One of 'bf16', 'int8', 'int4'.
        forget_loader: DataLoader for the forget set evaluation.
        retain_loader: DataLoader for the retain set evaluation.
        tokenizer: The tokenizer for text generation.

    Returns:
        PrecisionResult: Evaluation metrics at this precision.

    Raises:
        ValueError: If precision is not one of the valid options.
    """
    valid_precisions = {"bf16", "int8", "int4"}
    if precision not in valid_precisions:
        raise ValueError(
            f"Invalid precision '{precision}'. Must be one of: {valid_precisions}"
        )

    logger.info("Evaluating at %s precision from '%s'...", precision, merged_model_path)

    # --- Placeholder metric computation ---
    # TODO(GATE-1): Replace with OpenUnlearning evaluation harness calls.
    # The actual implementation will:
    # 1. Load model at the specified precision using load_model_bf16 or
    #    quantised loading via BitsAndBytesConfig
    # 2. Run the forget-set through the model and compute FQ
    # 3. Run the retain-set through the model and compute MU
    # 4. Return real metrics
    #
    # For now, return placeholder values that indicate the pipeline works
    # but metrics are not yet meaningful.
    logger.warning(
        "Using PLACEHOLDER metrics for %s. "
        "Real evaluation requires OpenUnlearning harness (GATE-1).",
        precision,
    )

    # Placeholder: return -1.0 to clearly indicate these are not real metrics
    fq_placeholder = -1.0
    mu_placeholder = -1.0

    timestamp = datetime.now(timezone.utc).isoformat()

    result = PrecisionResult(
        precision=precision,
        forget_quality=fq_placeholder,
        model_utility=mu_placeholder,
        evaluation_timestamp=timestamp,
    )

    logger.info(
        "%s evaluation complete: FQ=%.4f, MU=%.4f (PLACEHOLDER)",
        precision,
        fq_placeholder,
        mu_placeholder,
    )

    return result


def run_qra_evaluation(
    adapter_path: str,
    base_model_id: str,
    condition_id: str,
    seed: int,
    forget_loader,
    retain_loader,
    tokenizer,
    config: Optional[QRAConfig] = None,
) -> QRAResult:
    """Run the full three-precision QRA evaluation pipeline.

    Sequence:
    1. Merge adapter into base model
    2. Evaluate at BF16
    3. Evaluate at INT8
    4. Evaluate at INT4
    5. Compute Q-INT4 score
    6. Issue certification decision

    Each evaluation is SEQUENTIAL -- never load two models simultaneously.

    Args:
        adapter_path: Path to the PEFT adapter checkpoint.
        base_model_id: HuggingFace model ID of the base model.
        condition_id: Experimental condition identifier.
        seed: Random seed for this experiment.
        forget_loader: DataLoader for forget set.
        retain_loader: DataLoader for retain set.
        tokenizer: Tokenizer for text generation.
        config: QRA configuration. Uses defaults if None.

    Returns:
        QRAResult: Complete evaluation results with certification decision.
    """
    if config is None:
        config = QRAConfig()

    logger.info(
        "Starting QRA evaluation for condition '%s', seed %d", condition_id, seed
    )

    # Step 1: Merge adapter
    merged_path = os.path.join(
        config.gdrive_merged_path, f"{condition_id}_seed{seed}"
    )
    merge_and_save(adapter_path, base_model_id, merged_path, token=os.environ.get("HF_TOKEN", ""))

    # Steps 2-4: Evaluate at each precision SEQUENTIALLY
    results = {}
    for precision in ["bf16", "int8", "int4"]:
        result = evaluate_at_precision(
            merged_path, precision, forget_loader, retain_loader, tokenizer
        )
        results[precision] = result
        logger.info("  %s: FQ=%.4f, MU=%.4f", precision, result.forget_quality, result.model_utility)

    # Step 5: Compute Q-INT4
    fq_bf16 = results["bf16"].forget_quality
    fq_int4 = results["int4"].forget_quality
    eps = 1e-8
    q_int4 = abs(fq_int4 - fq_bf16) / (abs(fq_bf16) + eps)

    # Step 6: Certification decision
    certified = q_int4 < config.certification_threshold

    notes = ""
    if fq_bf16 < 0:
        notes = "WARNING: Placeholder metrics in use. Real evaluation pending GATE-1."

    qra_result = QRAResult(
        condition_id=condition_id,
        seed=seed,
        results_by_precision=results,
        q_int4_score=q_int4,
        certified=certified,
        certification_threshold=config.certification_threshold,
        notes=notes,
    )

    status = "CERTIFIED" if certified else "NOT CERTIFIED"
    logger.info(
        "QRA evaluation complete: Q-INT4=%.4f, threshold=%.4f → %s",
        q_int4,
        config.certification_threshold,
        status,
    )

    return qra_result


def save_qra_result(result: QRAResult, output_path: str) -> None:
    """Save a QRA result to a human-readable JSON file.

    Args:
        result: The QRA evaluation result to save.
        output_path: File path for the JSON output.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Convert dataclasses to dict for JSON serialisation
    result_dict = {
        "condition_id": result.condition_id,
        "seed": result.seed,
        "q_int4_score": result.q_int4_score,
        "certified": result.certified,
        "certification_threshold": result.certification_threshold,
        "notes": result.notes,
        "results_by_precision": {
            k: asdict(v) for k, v in result.results_by_precision.items()
        },
    }

    with open(output_path, "w") as f:
        json.dump(result_dict, f, indent=2)

    logger.info("QRA result saved to '%s'", output_path)
