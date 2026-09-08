#!/usr/bin/env python3
"""
scripts/research_gate.py

Scientific gate diagnostic reporter.

Checks the status of all research gates (scientific blockers) by reading
handbook/research_gates.yaml without promoting statuses from implementation
heuristics, then reporting which experimental conditions are blocked.

Usage:
    python scripts/research_gate.py
    python scripts/research_gate.py --json  # machine-readable output
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass
class Gate:
    id: str
    title: str
    category: str
    status: str
    blocks_training: bool
    blocks_evaluation: bool
    blocked_conditions: list[str]
    dependencies: list[str]
    source_citations: list[str]
    verification_command: str
    required_evidence: str
    validating_commit: str
    validation_date: str
    rationale: str


def load_gates(yaml_path: Path) -> list[Gate]:
    if not yaml_path.exists():
        print(f"Error: {yaml_path} not found.")
        sys.exit(1)
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or not isinstance(data.get("gates"), list) or not data["gates"]:
        raise ValueError("Research gate registry must contain a non-empty gates list")
    gates = []
    seen = set()
    for g in data["gates"]:
        if g.get("id") in seen:
            raise ValueError("Duplicate research gate ID")
        seen.add(g.get("id"))
        gates.append(Gate(**g))
    return gates


def evaluate_gate(gate: Gate) -> Gate:
    """Report the authoritative record; code heuristics cannot resolve a gate.

    Method acceptance requires the recorded mathematical, runtime and evidence
    criteria. Source text, quantiser metadata and one scalar hyperparameter are
    insufficient to promote or otherwise rewrite this status.
    """
    if gate.status not in {"BLOCKED", "OPEN", "RESOLVED"}:
        raise ValueError(f"Invalid research gate status: {gate.id}")
    return gate


def run_all_gates(yaml_path: Path) -> list[Gate]:
    gates = load_gates(yaml_path)
    evaluated_gates = []
    for g in gates:
        evaluated_gates.append(evaluate_gate(g))
    return evaluated_gates


def is_blocking(gate: Gate) -> bool:
    evaluate_gate(gate)
    return gate.status == "BLOCKED" or (
        gate.status != "RESOLVED" and (gate.blocks_training or gate.blocks_evaluation)
    )


def print_gate_report(gates: list[Gate]) -> None:
    blocked = [g for g in gates if g.status == "BLOCKED"]
    resolved = [g for g in gates if g.status == "RESOLVED"]
    open_gates = [g for g in gates if g.status == "OPEN"]

    print("=" * 70)
    print("  FQ-SSIUU RESEARCH GATE REPORT")
    print("=" * 70)

    for g in gates:
        if g.status == "BLOCKED":
            icon = "❌"
        elif g.status == "RESOLVED":
            icon = "✅"
        else:
            icon = "⚠️"

        print(f"\n  {icon} {g.id}: {g.title}")
        print(f"     Status: {g.status}")
        print(f"     Category: {g.category}")
        if g.blocked_conditions:
            print(f"     Blocks: {', '.join(g.blocked_conditions)}")
        print(f"     Rationale: {g.rationale}")

    print("\n" + "=" * 70)
    print(f"  SUMMARY: {len(blocked)} BLOCKED, {len(resolved)} RESOLVED, {len(open_gates)} OPEN")

    unresolved_blockers = [g for g in gates if is_blocking(g)]
    if unresolved_blockers:
        all_blocked = set()
        for g in unresolved_blockers:
            all_blocked.update(g.blocked_conditions)
        print(f"  BLOCKED CONDITIONS: {', '.join(sorted(all_blocked))}")
        print("  UNRESOLVED BLOCKING GATES: " + ", ".join(g.id for g in unresolved_blockers))
        print("\n  ❌ EXPERIMENT PHASE CANNOT PROCEED for blocked conditions.")
    else:
        print(
            "\n  Research gates resolved or non-blocking; operational and protocol approval still apply."
        )

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="FQ-SSIUU Research Gate Diagnostic Reporter")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON.")
    args = parser.parse_args()

    yaml_path = Path(__file__).parent.parent / "handbook" / "research_gates.yaml"
    gates = run_all_gates(yaml_path)

    if args.json:
        output = {
            "gates": [asdict(g) for g in gates],
            "blocked_count": sum(1 for g in gates if g.status == "BLOCKED"),
            "unresolved_blocking_count": sum(is_blocking(g) for g in gates),
            "all_clear": not any(is_blocking(g) for g in gates),
        }
        print(json.dumps(output, indent=2))
    else:
        print_gate_report(gates)

    if any(is_blocking(g) for g in gates):
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
