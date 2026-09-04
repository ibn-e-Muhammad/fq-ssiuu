"""
src/fq_ssiuu/provenance.py

Run provenance tracking for reproducible experiment records.

Each experiment run produces an immutable provenance record containing:
    - Collision-resistant run ID (UUID4 + timestamp)
    - Environment manifest (via environment.py)
    - Configuration snapshot
    - Seeds used
    - Start/end timestamps
    - Exit status (completed, failed, aborted)
    - SHA-256 checksums of output artifacts

Design constraints:
    - Provenance records are append-only: once written, never overwritten.
    - Each run gets a unique directory: results/<condition>/<run_id>/
    - Failure states are explicitly recorded (not silently dropped).
    - No experiment metrics or results are stored in provenance records.
      Provenance tracks HOW a run was executed, not WHAT it produced.
"""

import hashlib
import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RunContext:
    """Immutable provenance record for a single experiment run.

    Attributes:
        run_id: Collision-resistant unique identifier (UUID4 + timestamp prefix).
        condition_id: Experimental condition (e.g., 'c2_npo', 'c6_fq_ssiuu').
        seed: Random seed used for this run.
        start_time: ISO 8601 UTC timestamp when run began.
        end_time: ISO 8601 UTC timestamp when run ended (set on completion/failure).
        status: One of 'running', 'completed', 'failed', 'aborted'.
        git_commit: Git HEAD commit hash at run start.
        git_dirty: Whether working tree had uncommitted changes.
        git_branch: Git branch name at run start.
        config_snapshot: Complete configuration dict (serialisable).
        environment_manifest: Dict from environment.capture_manifest().
        artifact_checksums: Dict mapping relative path -> SHA-256 hex digest.
        failure_reason: Error message if status='failed'.
        notes: Free-text notes.
    """
    run_id: str
    condition_id: str
    seed: int
    start_time: str
    end_time: str = ""
    status: str = "running"
    git_commit: str = ""
    git_dirty: bool = False
    git_branch: str = ""
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    environment_manifest: Dict[str, Any] = field(default_factory=dict)
    artifact_checksums: Dict[str, str] = field(default_factory=dict)
    failure_reason: str = ""
    notes: str = ""


def generate_run_id() -> str:
    """Generate a collision-resistant run ID.

    Format: YYYYMMDD-HHMMSS-UUID4_short
    Example: 20260903-120000-a1b2c3d4

    Returns:
        Unique run identifier string.
    """
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]
    return f"{timestamp}-{short_uuid}"


def create_run(
    condition_id: str,
    seed: int,
    config: Optional[Dict[str, Any]] = None,
    repo_path: str = ".",
) -> RunContext:
    """Create a new run context and write the initial provenance record.

    Args:
        condition_id: Experimental condition identifier.
        seed: Random seed for this run.
        config: Configuration dict to snapshot (must be JSON-serialisable).
        repo_path: Path to the repository root (for git state capture).

    Returns:
        RunContext with 'running' status.
    """
    run_id = generate_run_id()
    start_time = datetime.now(timezone.utc).isoformat()

    # Capture environment manifest
    env_manifest = {}
    try:
        from fq_ssiuu.environment import capture_manifest
        manifest = capture_manifest(repo_path)
        env_manifest = asdict(manifest) if hasattr(manifest, '__dataclass_fields__') else {}
    except Exception as e:
        logger.warning("Could not capture environment manifest: %s", e)

    ctx = RunContext(
        run_id=run_id,
        condition_id=condition_id,
        seed=seed,
        start_time=start_time,
        status="running",
        config_snapshot=config or {},
        environment_manifest=env_manifest,
        git_commit=env_manifest.get("git_commit", ""),
        git_dirty=env_manifest.get("git_dirty", False),
        git_branch=env_manifest.get("git_branch", ""),
    )

    logger.info(
        "Run created: %s (condition=%s, seed=%d)",
        run_id,
        condition_id,
        seed,
    )

    return ctx


def complete_run(
    ctx: RunContext,
    output_dir: str,
) -> RunContext:
    """Mark a run as completed and compute artifact checksums.

    Args:
        ctx: The run context to complete.
        output_dir: Directory containing output artifacts to checksum.

    Returns:
        Updated RunContext with 'completed' status and checksums.

    Raises:
        ValueError: If the run is not in 'running' status.
    """
    if ctx.status != "running":
        raise ValueError(
            f"Cannot complete run {ctx.run_id}: status is '{ctx.status}', "
            f"expected 'running'."
        )

    ctx.end_time = datetime.now(timezone.utc).isoformat()
    ctx.status = "completed"

    # Compute checksums for all files in output_dir
    if os.path.isdir(output_dir):
        for root, _, files in os.walk(output_dir):
            for fname in sorted(files):
                fpath = os.path.join(root, fname)
                rel_path = os.path.relpath(fpath, output_dir)
                ctx.artifact_checksums[rel_path] = _sha256_file(fpath)

    logger.info(
        "Run completed: %s (%d artifacts checksummed)",
        ctx.run_id,
        len(ctx.artifact_checksums),
    )

    return ctx


def fail_run(
    ctx: RunContext,
    reason: str,
) -> RunContext:
    """Mark a run as failed with an error reason.

    Args:
        ctx: The run context to mark as failed.
        reason: Human-readable failure description.

    Returns:
        Updated RunContext with 'failed' status.
    """
    ctx.end_time = datetime.now(timezone.utc).isoformat()
    ctx.status = "failed"
    ctx.failure_reason = reason

    logger.error("Run failed: %s — %s", ctx.run_id, reason)

    return ctx


def save_provenance(
    ctx: RunContext,
    output_path: str,
    allow_overwrite: bool = False,
) -> str:
    """Save a provenance record to JSON.

    Args:
        ctx: The run context to save.
        output_path: File path for the JSON output.
        allow_overwrite: If False (default), raises if file exists.

    Returns:
        The output path.

    Raises:
        FileExistsError: If output_path exists and allow_overwrite is False.
    """
    if not allow_overwrite and os.path.exists(output_path):
        raise FileExistsError(
            f"Provenance record already exists: {output_path}. "
            f"Provenance records are immutable. Use allow_overwrite=True "
            f"only for updating status (running -> completed/failed)."
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(asdict(ctx), f, indent=2, default=str)

    logger.info("Provenance record saved to '%s'", output_path)
    return output_path


def load_provenance(provenance_path: str) -> RunContext:
    """Load a provenance record from JSON.

    Args:
        provenance_path: Path to the provenance JSON file.

    Returns:
        RunContext loaded from the file.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.isfile(provenance_path):
        raise FileNotFoundError(
            f"Provenance record not found: {provenance_path}"
        )

    with open(provenance_path, "r") as f:
        data = json.load(f)

    return RunContext(**data)


def _sha256_file(filepath: str) -> str:
    """Compute SHA-256 hex digest of a file.

    Args:
        filepath: Path to the file.

    Returns:
        Hex digest string.
    """
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
