#!/usr/bin/env python3
"""
scripts/research_gate.py

Scientific gate diagnostic reporter.

Checks the status of all research gates (scientific blockers) by reading
handbook/research_gates.yaml and executing dynamic checks where applicable,
then reporting which experimental conditions are blocked.

Usage:
    python scripts/research_gate.py
    python scripts/research_gate.py --json  # machine-readable output
"""

import argparse
import json
import sys
from pathlib import Path
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Any
import yaml


@dataclass
class Gate:
    id: str
    title: str
    category: str
    status: str
    blocks_training: bool
    blocks_evaluation: bool
    blocked_conditions: List[str]
    dependencies: List[str]
    source_citations: List[str]
    verification_command: str
    required_evidence: str
    validating_commit: str
    validation_date: str
    rationale: str

def load_gates(yaml_path: Path) -> List[Gate]:
    if not yaml_path.exists():
        print(f"Error: {yaml_path} not found.")
        sys.exit(1)
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    gates = []
    for g in data.get("gates", []):
        gates.append(Gate(**g))
    return gates


def evaluate_gate(gate: Gate) -> Gate:
    """Dynamically evaluate whether the gate condition holds and update status."""
    if gate.id == "RG-001":
        try:
            import inspect
            from fq_ssiuu.attribution.ssiuu_reg import compute_negative_attribution
            source = inspect.getsource(compute_negative_attribution)
            has_detach = ".detach()" in source
            gate.status = "BLOCKED" if has_detach else "RESOLVED"
        except Exception:
            gate.status = "BLOCKED"
            
    elif gate.id == "RG-002":
        try:
            from fq_ssiuu.quantisation.fake_quant import QUANTISER_META
            is_deployment_equivalent = QUANTISER_META.get("deployment_equivalent", False)
            gate.status = "RESOLVED" if is_deployment_equivalent else "BLOCKED"
        except Exception:
            gate.status = "BLOCKED"
            
    elif gate.id == "RG-003":
        try:
            from fq_ssiuu.unlearning.quail_loss import QUAILConfig
            config = QUAILConfig()
            margin_matches = abs(config.margin - 0.5) < 0.01
            gate.status = "RESOLVED" if margin_matches else "BLOCKED"
        except Exception:
            gate.status = "BLOCKED"
            
    elif gate.id == "RG-004":
        # Evaluated manually via documentation review, currently OPEN
        if gate.status != "RESOLVED":
            gate.status = "OPEN"
    
    return gate


def run_all_gates(yaml_path: Path) -> List[Gate]:
    gates = load_gates(yaml_path)
    evaluated_gates = []
    for g in gates:
        evaluated_gates.append(evaluate_gate(g))
    return evaluated_gates


def print_gate_report(gates: List[Gate]) -> None:
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

    if blocked:
        all_blocked = set()
        for g in blocked:
            all_blocked.update(g.blocked_conditions)
        print(f"  BLOCKED CONDITIONS: {', '.join(sorted(all_blocked))}")
        print("\n  ❌ EXPERIMENT PHASE CANNOT PROCEED for blocked conditions.")
    else:
        print("\n  ✅ ALL GATES RESOLVED (or non-blocking) — experiment phase can proceed.")

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
            "all_clear": all(g.status != "BLOCKED" for g in gates),
        }
        print(json.dumps(output, indent=2))
    else:
        print_gate_report(gates)

    if any(g.status == "BLOCKED" for g in gates):
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
