"""
src/fq_ssiuu/evaluation/certificate.py

Unlearning Durability Certificate generator.

This module generates, saves, loads, and displays the JSON durability
certificate -- the primary product artefact of the FQ-SSIUU project.

A durability certificate attests that a specific unlearned model produces
QRA-robust forgetting at INT4 precision, as measured by the Q-INT4 metric.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Import QRAResult for the generate_certificate function
from fq_ssiuu.evaluation.qra_pipeline import QRAResult


_GDPR_DISCLAIMER = (
    "This certificate indicates technical QRA robustness at INT4 precision "
    "under the specified evaluation protocol. It does not constitute a legal "
    "determination of GDPR Article 17 compliance. Legal compliance requires "
    "assessment by qualified legal counsel."
)


@dataclass
class DurabilityCertificate:
    """JSON durability certificate for an unlearned model.

    Attributes:
        model_id: The base model HuggingFace identifier.
        condition_id: Experimental condition (e.g., 'c6_fq_ssiuu').
        seed: Random seed used for this experiment.
        date_generated: ISO 8601 timestamp of certificate generation.
        forget_dataset: Name of the forget dataset (e.g., 'tofu').
        forget_split: Forget split identifier (e.g., 'forget05').
        metrics_bf16: Dict of metrics at BF16 precision.
        metrics_int8: Dict of metrics at INT8 precision.
        metrics_int4: Dict of metrics at INT4 precision.
        q_int4_score: The Q-INT4 metric value.
        certification_threshold: The threshold used for certification.
        certified: Whether the model passes the durability threshold.
        gdpr_disclaimer: Legal disclaimer about GDPR compliance.
        method: The unlearning method used (e.g., 'FQ-SSIUU').
        method_config: Dict of method-specific configuration parameters.
        notes: Free-text notes.
    """

    model_id: str
    condition_id: str
    seed: int
    date_generated: str
    forget_dataset: str
    forget_split: str
    metrics_bf16: dict
    metrics_int8: dict
    metrics_int4: dict
    q_int4_score: float
    certification_threshold: float
    certified: bool
    gdpr_disclaimer: str = field(default=_GDPR_DISCLAIMER)
    method: str = ""
    method_config: dict = field(default_factory=dict)
    notes: str = ""


def generate_certificate(
    qra_result: QRAResult,
    model_id: str,
    forget_dataset: str,
    forget_split: str,
    method: str,
    method_config: dict,
) -> DurabilityCertificate:
    """Generate a durability certificate from QRA evaluation results.

    Args:
        qra_result: The complete QRA evaluation result.
        model_id: Base model HuggingFace identifier.
        forget_dataset: Name of the forget dataset.
        forget_split: Forget split identifier.
        method: Unlearning method name.
        method_config: Method-specific configuration parameters.

    Returns:
        DurabilityCertificate: The generated certificate.
    """
    def _precision_to_dict(precision: str) -> dict:
        """Extract metrics dict for a precision level."""
        if precision in qra_result.results_by_precision:
            pr = qra_result.results_by_precision[precision]
            return {
                "forget_quality": pr.forget_quality,
                "model_utility": pr.model_utility,
                "evaluation_timestamp": pr.evaluation_timestamp,
            }
        return {}

    cert = DurabilityCertificate(
        model_id=model_id,
        condition_id=qra_result.condition_id,
        seed=qra_result.seed,
        date_generated=datetime.now(timezone.utc).isoformat(),
        forget_dataset=forget_dataset,
        forget_split=forget_split,
        metrics_bf16=_precision_to_dict("bf16"),
        metrics_int8=_precision_to_dict("int8"),
        metrics_int4=_precision_to_dict("int4"),
        q_int4_score=qra_result.q_int4_score,
        certification_threshold=qra_result.certification_threshold,
        certified=qra_result.certified,
        method=method,
        method_config=method_config,
        notes=qra_result.notes,
    )

    logger.info(
        "Certificate generated for %s (seed %d): %s",
        cert.condition_id,
        cert.seed,
        "CERTIFIED" if cert.certified else "NOT CERTIFIED",
    )

    return cert


def save_certificate(cert: DurabilityCertificate, output_path: str) -> None:
    """Save a durability certificate to a JSON file.

    Args:
        cert: The certificate to save.
        output_path: File path for the JSON output (indent=2 for readability).

    Raises:
        OSError: If the file cannot be written.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    cert_dict = asdict(cert)

    with open(output_path, "w") as f:
        json.dump(cert_dict, f, indent=2)

    logger.info("Certificate saved to '%s'", output_path)


def load_certificate(cert_path: str) -> DurabilityCertificate:
    """Load a durability certificate from a JSON file.

    Args:
        cert_path: Path to the certificate JSON file.

    Returns:
        DurabilityCertificate: The loaded certificate.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    if not os.path.isfile(cert_path):
        raise FileNotFoundError(f"Certificate file not found: {cert_path}")

    with open(cert_path, "r") as f:
        data = json.load(f)

    cert = DurabilityCertificate(**data)

    logger.info(
        "Certificate loaded from '%s': %s",
        cert_path,
        "CERTIFIED" if cert.certified else "NOT CERTIFIED",
    )

    return cert


def print_certificate_summary(cert: DurabilityCertificate) -> None:
    """Print a human-readable summary of a durability certificate.

    Args:
        cert: The certificate to summarise.
    """
    status_icon = "✅" if cert.certified else "❌"
    status_text = "CERTIFIED" if cert.certified else "NOT CERTIFIED"

    print("=" * 60)
    print(f"  UNLEARNING DURABILITY CERTIFICATE  {status_icon}")
    print("=" * 60)
    print(f"  Status:       {status_text}")
    print(f"  Model:        {cert.model_id}")
    print(f"  Condition:    {cert.condition_id}")
    print(f"  Method:       {cert.method}")
    print(f"  Seed:         {cert.seed}")
    print(f"  Date:         {cert.date_generated}")
    print(f"  Dataset:      {cert.forget_dataset} / {cert.forget_split}")
    print("-" * 60)
    print("  Metrics by Precision:")
    for label, metrics in [
        ("BF16", cert.metrics_bf16),
        ("INT8", cert.metrics_int8),
        ("INT4", cert.metrics_int4),
    ]:
        fq = metrics.get("forget_quality", "N/A")
        mu = metrics.get("model_utility", "N/A")
        fq_str = f"{fq:.4f}" if isinstance(fq, float) else str(fq)
        mu_str = f"{mu:.4f}" if isinstance(mu, float) else str(mu)
        print(f"    {label:>4s}: FQ={fq_str}, MU={mu_str}")
    print("-" * 60)
    print(f"  Q-INT4:       {cert.q_int4_score:.6f}")
    print(f"  Threshold:    {cert.certification_threshold:.4f}")
    print("-" * 60)
    print(f"  {cert.gdpr_disclaimer}")
    if cert.notes:
        print(f"  Notes: {cert.notes}")
    print("=" * 60)
