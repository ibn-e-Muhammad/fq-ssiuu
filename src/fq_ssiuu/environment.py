"""
src/fq_ssiuu/environment.py

Environment preflight checks and manifest capture for reproducibility.

Provides two main functions:
    - preflight_check(): Validates that the runtime environment meets
      minimum requirements for FQ-SSIUU experiments.
    - capture_manifest(): Captures a complete environment snapshot for
      inclusion in run provenance records.

Designed to run both locally (CPU-only) and on Kaggle (with GPU).
"""

import importlib.metadata
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Minimum requirements
MIN_PYTHON_VERSION = (3, 10)
CRITICAL_PACKAGES = [
    "torch",
    "transformers",
    "peft",
    "bitsandbytes",
    "datasets",
    "captum",
]
MIN_DISK_GB = 5.0


@dataclass
class PreflightResult:
    """Result of a preflight check.

    Attributes:
        check_name: Human-readable name of the check.
        passed: Whether the check passed.
        message: Descriptive message (pass or fail reason).
        severity: 'error' (blocks execution), 'warning' (degraded), 'info'.
    """
    check_name: str
    passed: bool
    message: str
    severity: str = "error"  # error, warning, info


@dataclass
class EnvironmentManifest:
    """Complete environment snapshot for reproducibility.

    Attributes:
        timestamp: ISO 8601 capture time.
        python_version: Full Python version string.
        platform_info: OS and architecture details.
        packages: Dict of installed package name -> version.
        cuda_available: Whether CUDA is available.
        cuda_version: CUDA version string (if available).
        gpu_info: List of GPU descriptions (if available).
        gpu_memory_gb: List of GPU memory in GB (if available).
        driver_version: NVIDIA driver version (if available).
        git_commit: Current git HEAD commit hash.
        git_dirty: Whether the working tree has uncommitted changes.
        git_branch: Current git branch name.
        disk_free_gb: Free disk space in GB at working directory.
        env_vars_set: List of relevant env var names that are set (NOT values).
    """
    timestamp: str = ""
    python_version: str = ""
    platform_info: str = ""
    packages: Dict[str, str] = field(default_factory=dict)
    cuda_available: bool = False
    cuda_version: str = ""
    gpu_info: List[str] = field(default_factory=list)
    gpu_memory_gb: List[float] = field(default_factory=list)
    driver_version: str = ""
    git_commit: str = ""
    git_dirty: bool = False
    git_branch: str = ""
    disk_free_gb: float = 0.0
    env_vars_set: List[str] = field(default_factory=list)


def _check_python_version() -> PreflightResult:
    """Check Python version meets minimum requirement."""
    current = sys.version_info[:2]
    if current >= MIN_PYTHON_VERSION:
        return PreflightResult(
            "Python version",
            True,
            f"Python {current[0]}.{current[1]} >= {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}",
        )
    return PreflightResult(
        "Python version",
        False,
        f"Python {current[0]}.{current[1]} < required {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}",
    )


def _check_critical_packages() -> List[PreflightResult]:
    """Check that all critical packages are importable."""
    results = []
    for pkg in CRITICAL_PACKAGES:
        try:
            version = importlib.metadata.version(pkg)
            results.append(PreflightResult(
                f"Package: {pkg}",
                True,
                f"{pkg}=={version}",
                severity="info",
            ))
        except importlib.metadata.PackageNotFoundError:
            results.append(PreflightResult(
                f"Package: {pkg}",
                False,
                f"{pkg} not installed",
                severity="error",
            ))
    return results


def _check_cuda() -> PreflightResult:
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            mem_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            return PreflightResult(
                "CUDA",
                True,
                f"CUDA available: {device_name} ({mem_gb:.1f} GB)",
                severity="info",
            )
        return PreflightResult(
            "CUDA",
            False,
            "CUDA not available (CPU-only mode)",
            severity="warning",
        )
    except ImportError:
        return PreflightResult(
            "CUDA",
            False,
            "PyTorch not installed — cannot check CUDA",
            severity="error",
        )


def _check_disk_space(path: str = ".") -> PreflightResult:
    """Check available disk space."""
    try:
        usage = shutil.disk_usage(path)
        free_gb = usage.free / (1024**3)
        if free_gb >= MIN_DISK_GB:
            return PreflightResult(
                "Disk space",
                True,
                f"{free_gb:.1f} GB free >= {MIN_DISK_GB} GB minimum",
            )
        return PreflightResult(
            "Disk space",
            False,
            f"{free_gb:.1f} GB free < {MIN_DISK_GB} GB minimum",
        )
    except Exception as e:
        return PreflightResult(
            "Disk space",
            False,
            f"Cannot check disk space: {e}",
            severity="warning",
        )


def _check_git_state(repo_path: str = ".") -> PreflightResult:
    """Check git repository state."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=repo_path,
            timeout=10,
        )
        if result.returncode == 0:
            commit = result.stdout.strip()[:12]
            return PreflightResult(
                "Git state",
                True,
                f"Git repository detected, HEAD={commit}",
                severity="info",
            )
        return PreflightResult(
            "Git state",
            False,
            "Not a git repository",
            severity="warning",
        )
    except Exception as e:
        return PreflightResult(
            "Git state",
            False,
            f"Cannot check git state: {e}",
            severity="warning",
        )


def preflight_check(repo_path: str = ".") -> Tuple[bool, List[PreflightResult]]:
    """Run all preflight checks.

    Args:
        repo_path: Path to the repository root.

    Returns:
        Tuple of (all_passed, list of PreflightResult).
        all_passed is False if any 'error'-severity check failed.
    """
    results: List[PreflightResult] = []

    results.append(_check_python_version())
    results.extend(_check_critical_packages())
    results.append(_check_cuda())
    results.append(_check_disk_space(repo_path))
    results.append(_check_git_state(repo_path))

    all_passed = all(
        r.passed for r in results if r.severity == "error"
    )

    return all_passed, results


def capture_manifest(repo_path: str = ".") -> EnvironmentManifest:
    """Capture a complete environment manifest for reproducibility.

    Args:
        repo_path: Path to the repository root.

    Returns:
        EnvironmentManifest with all fields populated.
    """
    manifest = EnvironmentManifest()
    manifest.timestamp = datetime.now(timezone.utc).isoformat()
    manifest.python_version = sys.version
    manifest.platform_info = f"{platform.system()} {platform.release()} {platform.machine()}"

    # Installed packages
    try:
        for dist in importlib.metadata.distributions():
            name = dist.metadata["Name"]
            version = dist.metadata["Version"]
            if name and version:
                manifest.packages[name] = version
    except Exception:
        logger.warning("Could not enumerate installed packages.")

    # CUDA / GPU info
    try:
        import torch
        manifest.cuda_available = torch.cuda.is_available()
        if manifest.cuda_available:
            manifest.cuda_version = torch.version.cuda or ""
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                manifest.gpu_info.append(props.name)
                manifest.gpu_memory_gb.append(
                    round(props.total_mem / (1024**3), 2)
                )
            # Driver version
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0:
                    manifest.driver_version = result.stdout.strip().split("\n")[0]
            except Exception:
                pass
    except ImportError:
        pass

    # Git state
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=repo_path, timeout=10,
        )
        if commit_result.returncode == 0:
            manifest.git_commit = commit_result.stdout.strip()

        dirty_result = subprocess.run(
            ["git", "diff", "--quiet"],
            capture_output=True, cwd=repo_path, timeout=10,
        )
        manifest.git_dirty = dirty_result.returncode != 0

        branch_result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, cwd=repo_path, timeout=10,
        )
        if branch_result.returncode == 0:
            manifest.git_branch = branch_result.stdout.strip()
    except Exception:
        pass

    # Disk space
    try:
        usage = shutil.disk_usage(repo_path)
        manifest.disk_free_gb = round(usage.free / (1024**3), 2)
    except Exception:
        pass

    # Environment variable presence (names only, NOT values)
    relevant_vars = [
        "HF_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY",
        "GDRIVE_SERVICE_ACCOUNT_JSON", "GDRIVE_FOLDER_ID",
        "CUDA_VISIBLE_DEVICES", "TRANSFORMERS_CACHE",
        "HF_HOME", "BNB_CUDA_VERSION",
    ]
    manifest.env_vars_set = [v for v in relevant_vars if v in os.environ]

    return manifest


def save_manifest(manifest: EnvironmentManifest, output_path: str) -> None:
    """Save an environment manifest to JSON.

    Args:
        manifest: The manifest to save.
        output_path: File path for the JSON output.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(asdict(manifest), f, indent=2)
    logger.info("Environment manifest saved to '%s'", output_path)


def preflight_main() -> None:
    """CLI entry point for preflight checks.

    Prints results and exits with code 0 (all passed) or 1 (failures).
    """
    print("=" * 60)
    print("  FQ-SSIUU Environment Preflight Check")
    print("=" * 60)

    all_passed, results = preflight_check()

    for r in results:
        icon = "✅" if r.passed else ("⚠️" if r.severity == "warning" else "❌")
        print(f"  {icon} [{r.severity.upper():>7s}] {r.check_name}: {r.message}")

    print("=" * 60)
    if all_passed:
        print("  PREFLIGHT: ALL CRITICAL CHECKS PASSED")
        sys.exit(0)
    else:
        print("  PREFLIGHT: CRITICAL CHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    preflight_main()
