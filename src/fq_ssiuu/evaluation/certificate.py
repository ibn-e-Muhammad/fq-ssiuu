"""
src/fq_ssiuu/evaluation/certificate.py

Empirical Durability Report generator.

This module generates, saves, loads, and displays the JSON empirical durability
report -- a structured record of precision-sensitivity measurements for an
unlearned model.

A durability report records whether the unlearned model's metrics change
substantially across precision levels (FP16, INT8, NF4). It does NOT constitute
a formal certificate or mathematical guarantee of unlearning robustness.

Schema versioning:
    v1 (legacy): Fields 'q_int4_score' and 'certified'. Pilots P1-P4 use this.
    v2 (current): Fields 'relative_precision_sensitivity' and 'threshold_met'.
        Legacy fields preserved as deprecated aliases when loading v1 artifacts.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

from fq_ssiuu.evaluation.qra_pipeline import QRAResult

logger = logging.getLogger(__name__)

_GDPR_DISCLAIMER = (
    "This empirical durability report records precision-sensitivity measurements "
    "under the specified evaluation protocol. It does not constitute a formal "
    "certificate, mathematical guarantee, or legal determination of GDPR "
    "Article 17 compliance. Legal compliance requires assessment by qualified "
    "legal counsel."
)


# Schema version for newly generated reports.
_SCHEMA_VERSION = "v2"


@dataclass
class DurabilityCertificate:
    """Empirical durability report for an unlearned model.

    Schema v2 fields (preferred):
        relative_precision_sensitivity: |FQ_NF4 - FQ_FP16| / FQ_FP16.
            Measures how much Forget Quality changes under NF4 quantisation.
            This is an unsigned relative change, not a directional recovery metric.
        threshold_met: Whether relative_precision_sensitivity < threshold.

    Schema v1 fields (deprecated, preserved for loading historical artifacts):
        q_int4_score: Alias for relative_precision_sensitivity.
        certified: Alias for threshold_met.

    Attributes:
        model_id: The base model HuggingFace identifier.
        condition_id: Experimental condition (e.g., 'c6_fq_ssiuu').
        seed: Random seed used for this experiment.
        date_generated: ISO 8601 timestamp of report generation.
        forget_dataset: Name of the forget dataset (e.g., 'tofu').
        forget_split: Forget split identifier (e.g., 'forget05').
        metrics_fp16: Dict of metrics at FP16 precision.
        metrics_int8: Dict of metrics at INT8 precision.
        metrics_nf4: Dict of metrics at NF4 precision.
        relative_precision_sensitivity: |FQ_NF4 - FQ_FP16| / FQ_FP16.
        certification_threshold: The threshold used for comparison.
        threshold_met: Whether sensitivity < threshold.
        schema_version: 'v1' for legacy, 'v2' for current.
        gdpr_disclaimer: Legal disclaimer.
        method: The unlearning method used.
        method_config: Dict of method-specific configuration parameters.
        notes: Free-text notes.
    """

    model_id: str
    condition_id: str
    seed: int
    date_generated: str
    forget_dataset: str
    forget_split: str
    metrics_fp16: dict
    metrics_int8: dict
    metrics_nf4: dict
    relative_precision_sensitivity: float
    certification_threshold: float
    threshold_met: bool
    schema_version: str = field(default=_SCHEMA_VERSION)
    gdpr_disclaimer: str = field(default=_GDPR_DISCLAIMER)
    method: str = ""
    method_config: dict = field(default_factory=dict)
    notes: str = ""

    # --- Deprecated aliases for v1 compatibility ---
    @property
    def q_int4_score(self) -> float:
        """Deprecated alias for relative_precision_sensitivity."""
        return self.relative_precision_sensitivity

    @property
    def certified(self) -> bool:
        """Deprecated alias for threshold_met."""
        return self.threshold_met


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
        date_generated=datetime.now(UTC).isoformat(),
        forget_dataset=forget_dataset,
        forget_split=forget_split,
        metrics_fp16=_precision_to_dict("fp16"),
        metrics_int8=_precision_to_dict("int8"),
        metrics_nf4=_precision_to_dict("nf4"),
        relative_precision_sensitivity=qra_result.q_int4_score,
        certification_threshold=qra_result.certification_threshold,
        threshold_met=qra_result.certified,
        method=method,
        method_config=method_config,
        notes=qra_result.notes,
    )

    logger.info(
        "Report generated for %s (seed %d): %s",
        cert.condition_id,
        cert.seed,
        "THRESHOLD MET" if cert.threshold_met else "THRESHOLD NOT MET",
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
    # Also include deprecated aliases for readers that expect v1 fields
    cert_dict["q_int4_score"] = cert.relative_precision_sensitivity
    cert_dict["certified"] = cert.threshold_met
    cert_dict["_deprecation_notice"] = (
        "Fields 'q_int4_score' and 'certified' are deprecated v1 aliases. "
        "Use 'relative_precision_sensitivity' and 'threshold_met' (v2)."
    )

    with open(output_path, "w") as f:
        json.dump(cert_dict, f, indent=2)

    logger.info("Report saved to '%s'", output_path)


def load_certificate(cert_path: str) -> DurabilityCertificate:
    """Load a durability report from a JSON file.

    Supports both v1 (legacy) and v2 (current) schema formats.
    v1 artifacts with 'q_int4_score'/'certified' are automatically
    migrated to v2 field names on load.

    Args:
        cert_path: Path to the report JSON file.

    Returns:
        DurabilityCertificate: The loaded report.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    if not os.path.isfile(cert_path):
        raise FileNotFoundError(f"Report file not found: {cert_path}")

    with open(cert_path) as f:
        data = json.load(f)

    # v1 → v2 migration: translate legacy field names
    if "q_int4_score" in data and "relative_precision_sensitivity" not in data:
        data["relative_precision_sensitivity"] = data.pop("q_int4_score")
        data["schema_version"] = "v1"
    if "certified" in data and "threshold_met" not in data:
        data["threshold_met"] = data.pop("certified")

    # Remove fields that are properties, not constructor args
    data.pop("q_int4_score", None)
    data.pop("certified", None)
    data.pop("_deprecation_notice", None)

    cert = DurabilityCertificate(**data)

    logger.info(
        "Report loaded from '%s' (schema %s): %s",
        cert_path,
        cert.schema_version,
        "THRESHOLD MET" if cert.threshold_met else "THRESHOLD NOT MET",
    )

    return cert


def print_certificate_summary(cert: DurabilityCertificate) -> None:
    """Print a human-readable summary of an empirical durability report.

    Args:
        cert: The report to summarise.
    """
    status_icon = "✅" if cert.threshold_met else "❌"
    status_text = "THRESHOLD MET" if cert.threshold_met else "THRESHOLD NOT MET"

    print("=" * 60)
    print(f"  EMPIRICAL DURABILITY REPORT  {status_icon}")
    print("=" * 60)
    print(f"  Status:       {status_text}")
    print(f"  Schema:       {cert.schema_version}")
    print(f"  Model:        {cert.model_id}")
    print(f"  Condition:    {cert.condition_id}")
    print(f"  Method:       {cert.method}")
    print(f"  Seed:         {cert.seed}")
    print(f"  Date:         {cert.date_generated}")
    print(f"  Dataset:      {cert.forget_dataset} / {cert.forget_split}")
    print("-" * 60)
    print("  Metrics by Precision:")
    for label, metrics in [
        ("FP16", cert.metrics_fp16),
        ("INT8", cert.metrics_int8),
        ("NF4", cert.metrics_nf4),
    ]:
        fq = metrics.get("forget_quality", "N/A")
        mu = metrics.get("model_utility", "N/A")
        fq_str = f"{fq:.4f}" if isinstance(fq, float) else str(fq)
        mu_str = f"{mu:.4f}" if isinstance(mu, float) else str(mu)
        print(f"    {label:>4s}: FQ={fq_str}, MU={mu_str}")
    print("-" * 60)
    print(f"  Precision Sensitivity: {cert.relative_precision_sensitivity:.6f}")
    print(f"  Threshold:             {cert.certification_threshold:.4f}")
    print("-" * 60)
    print(f"  {cert.gdpr_disclaimer}")
    if cert.notes:
        print(f"  Notes: {cert.notes}")
    print("=" * 60)
