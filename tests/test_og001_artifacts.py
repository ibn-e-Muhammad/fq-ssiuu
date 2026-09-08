import json
import tempfile
import unittest
from pathlib import Path

from scripts.validate_og001_artifacts import validate_setup_artifacts
from tests.og001_fixtures import refresh_manifest, write_success_artifacts

SHA = "a" * 40


class TestOG001Evidence(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "evidence"
        write_success_artifacts(self.output, SHA)

    def mutate(self, filename, callback):
        path = self.output / filename
        value = json.loads(path.read_text())
        callback(value)
        path.write_text(json.dumps(value))
        refresh_manifest(self.output, SHA)

    def test_success_requires_explicit_dirty_state_and_verified_evidence(self):
        result = validate_setup_artifacts(self.output, SHA)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["pip_consistency_state"], "BASELINE_DIRTY_NO_REGRESSION")

    def test_forged_consistency_pass_is_rejected(self):
        self.mutate("pip_consistency.json", lambda x: x.update(state="CLEAN"))
        with self.assertRaisesRegex(ValueError, "disagrees"):
            validate_setup_artifacts(self.output, SHA)

    def test_missing_hash_coverage_rejected_even_when_all_files_exist(self):
        self.mutate("MANIFEST.json", lambda x: None)
        path = self.output / "MANIFEST.json"
        m = json.loads(path.read_text())
        m["artifacts"] = [x for x in m["artifacts"] if x["path"] != "after_manifest.json"]
        path.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError, "every evidence file"):
            validate_setup_artifacts(self.output, SHA)

    def test_raw_hash_tampering_rejected(self):
        (self.output / "pip_check_before_stdout.log").write_text("tampered")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_setup_artifacts(self.output, SHA)

    def test_runtime_failure_rejected_despite_overall_pass(self):
        self.mutate(
            "kaggle_env_verification.json",
            lambda x: x["checks"]["nf4_quantise_dequantise"].update(status="FAIL"),
        )
        with self.assertRaisesRegex(ValueError, "Substantive runtime"):
            validate_setup_artifacts(self.output, SHA)

    def test_invalid_peak_memory_observation_rejected(self):
        self.mutate(
            "kaggle_env_verification.json",
            lambda x: x["checks"]["peak_memory"]["observed"].update(
                peak_reserved_bytes=16_000_000_000
            ),
        )
        with self.assertRaisesRegex(ValueError, "Peak memory"):
            validate_setup_artifacts(self.output, SHA)

    def test_lock_cannot_omit_observed_install(self):
        self.mutate("overlay_lock.json", lambda x: x.update(lock_entries=[]))
        with self.assertRaisesRegex(ValueError, "lock disagrees"):
            validate_setup_artifacts(self.output, SHA)
