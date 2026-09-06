"""Gate reporting cannot replace evidence-backed decisions with code heuristics."""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.research_gate import evaluate_gate, load_gates, main, run_all_gates

ROOT = Path(__file__).resolve().parents[1]


class TestResearchGateAuthority(unittest.TestCase):
    def test_open_blocking_flags_control_json_and_exit_status(self):
        gate = load_gates(ROOT / "handbook/research_gates.yaml")[0]
        gate.status = "OPEN"
        for training, evaluation, expected in [
            (False, False, 0),
            (True, False, 1),
            (False, True, 1),
        ]:
            with self.subTest(training=training, evaluation=evaluation):
                gate.blocks_training, gate.blocks_evaluation = training, evaluation
                output = io.StringIO()
                with (
                    patch("scripts.research_gate.run_all_gates", return_value=[gate]),
                    patch("sys.argv", ["research_gate", "--json"]),
                    contextlib.redirect_stdout(output),
                    self.assertRaises(SystemExit) as exit_result,
                ):
                    main()
                self.assertEqual(exit_result.exception.code, expected)
                self.assertEqual(json.loads(output.getvalue())["all_clear"], expected == 0)

    def test_blocked_science_gates_are_not_promoted_or_imported(self):
        gates = load_gates(ROOT / "handbook/research_gates.yaml")
        for gate in gates:
            if gate.id in {"RG-001", "RG-002", "RG-003"}:
                original = copy.deepcopy(gate)
                with patch(
                    "builtins.__import__",
                    side_effect=AssertionError("No scientific imports allowed"),
                ):
                    observed = evaluate_gate(gate)
                self.assertEqual(observed, original)
                self.assertEqual(observed.status, "BLOCKED")

    def test_documentation_gate_keeps_its_authoritative_status(self):
        gate = next(
            g for g in load_gates(ROOT / "handbook/research_gates.yaml") if g.id == "RG-004"
        )
        self.assertEqual(evaluate_gate(gate).status, gate.status)

    def test_invalid_status_fails_closed(self):
        gate = load_gates(ROOT / "handbook/research_gates.yaml")[0]
        gate.status = "UNKNOWN"
        with self.assertRaises(ValueError):
            evaluate_gate(gate)

    def test_empty_registry_cannot_report_all_clear(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "gates.yaml")
            path.write_text("gates: []\n")
            with self.assertRaises(ValueError):
                run_all_gates(path)
