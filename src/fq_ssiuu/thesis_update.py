"""
src/fq_ssiuu/thesis_update.py

Thesis update packet generator.

Generates structured update packets from validated experiment runs. Each
packet documents what thesis sections are affected, what evidence was
produced, and what claims are supported or unsupported.

This module does NOT edit the thesis directly. It produces JSON packets
that the author uses to update their thesis document.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ThesisUpdatePacket:
    """A structured update packet for thesis maintenance.

    Attributes:
        run_id: The experiment run ID that produced this update.
        condition_id: Experimental condition.
        git_commit: Git commit hash of the code that produced results.
        timestamp: ISO 8601 generation timestamp.
        affected_sections: List of thesis section identifiers.
        metrics: Dict of key metric names to values.
        evidence_paths: List of file paths to evidence artifacts.
        claims_supported: List of claims the evidence supports.
        claims_unsupported: List of claims the evidence does NOT support.
        limitations: List of limitations or caveats.
        corrections_applied: List of correction IDs applied before this run.
        notes: Free-text notes.
    """
    run_id: str
    condition_id: str
    git_commit: str
    timestamp: str = ""
    affected_sections: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    evidence_paths: List[str] = field(default_factory=list)
    claims_supported: List[str] = field(default_factory=list)
    claims_unsupported: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    corrections_applied: List[str] = field(default_factory=list)
    notes: str = ""


def generate_thesis_packet(
    run_id: str,
    condition_id: str,
    git_commit: str,
    metrics: Optional[Dict[str, float]] = None,
    evidence_paths: Optional[List[str]] = None,
    affected_sections: Optional[List[str]] = None,
    claims_supported: Optional[List[str]] = None,
    claims_unsupported: Optional[List[str]] = None,
    limitations: Optional[List[str]] = None,
    corrections_applied: Optional[List[str]] = None,
    notes: str = "",
) -> ThesisUpdatePacket:
    """Generate a thesis update packet from run results.

    Args:
        run_id: The experiment run ID.
        condition_id: Experimental condition identifier.
        git_commit: Git commit hash.
        metrics: Key metrics from the run.
        evidence_paths: Paths to evidence files.
        affected_sections: Thesis sections affected.
        claims_supported: Claims the evidence supports.
        claims_unsupported: Claims the evidence does NOT support.
        limitations: Caveats and limitations.
        corrections_applied: Correction IDs applied.
        notes: Free-text notes.

    Returns:
        ThesisUpdatePacket with all fields populated.
    """
    packet = ThesisUpdatePacket(
        run_id=run_id,
        condition_id=condition_id,
        git_commit=git_commit,
        timestamp=datetime.now(timezone.utc).isoformat(),
        affected_sections=affected_sections or [],
        metrics=metrics or {},
        evidence_paths=evidence_paths or [],
        claims_supported=claims_supported or [],
        claims_unsupported=claims_unsupported or [],
        limitations=limitations or [],
        corrections_applied=corrections_applied or [],
        notes=notes,
    )

    logger.info(
        "Thesis update packet generated for run %s (condition: %s)",
        run_id,
        condition_id,
    )

    return packet


def save_thesis_packet(
    packet: ThesisUpdatePacket,
    output_path: str,
) -> str:
    """Save a thesis update packet to JSON.

    Args:
        packet: The packet to save.
        output_path: File path for the JSON output.

    Returns:
        The output path.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(asdict(packet), f, indent=2)

    logger.info("Thesis update packet saved to '%s'", output_path)
    return output_path


def load_thesis_packet(packet_path: str) -> ThesisUpdatePacket:
    """Load a thesis update packet from JSON.

    Args:
        packet_path: Path to the packet JSON file.

    Returns:
        ThesisUpdatePacket loaded from the file.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.isfile(packet_path):
        raise FileNotFoundError(f"Thesis packet not found: {packet_path}")

    with open(packet_path, "r") as f:
        data = json.load(f)

    return ThesisUpdatePacket(**data)
