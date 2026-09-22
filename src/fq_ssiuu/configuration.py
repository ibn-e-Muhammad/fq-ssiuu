"""Fail-closed loader for versioned accepted configurations, never historical defaults."""

from pathlib import Path

import yaml


def load_config(path):
    path = Path(path)
    if "historical" in path.parts:
        raise ValueError("Historical configurations cannot execute")
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict) or value.get("method_status") != "ACCEPTED_FOR_VALIDATION":
        raise ValueError("Configuration is historical, blocked, or has no accepted schema status")
    if value.get("schema_version") != "1.0":
        raise ValueError("Unknown configuration schema")
    return value
