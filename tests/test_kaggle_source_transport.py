"""Offline transport tests using actual Git repositories and bundle round-trips."""

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.package_kaggle_source import (
    BUNDLE_NAME,
    MANIFEST_NAME,
    digest,
    git,
    package_source,
    verify_bundle,
)


class TestSourceTransport(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="test-source-transport-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / "source"
        self.repo.mkdir()
        git(self.repo, "init")
        (self.repo / "old-secret.txt").write_text("hf_" + "A" * 35)
        self.commit("historical content")
        git(self.repo, "rm", "old-secret.txt")
        (self.repo / "source.py").write_bytes(b"print('tracked source')\n")
        self.origin = self.commit("safe source")
        (self.repo / ".env").write_text("untracked fixture")
        (self.repo / "thesis.docx").write_bytes(b"untracked fixture")
        (self.repo / "untracked.py").write_text("excluded")
        self.output = self.root / "bundle"

    def commit(self, message):
        git(self.repo, "add", "--all")
        git(
            self.repo,
            "-c",
            "user.name=Transport Test",
            "-c",
            "user.email=test@invalid",
            "commit",
            "-m",
            message,
        )
        return git(self.repo, "rev-parse", "HEAD").decode().strip()

    def package(self):
        return package_source(self.repo, self.origin, self.output)

    def rewrite_manifest(self, **changes):
        path = self.output / MANIFEST_NAME
        manifest = json.loads(path.read_text())
        manifest.update(changes)
        path.write_text(json.dumps(manifest))

    def test_parentless_tree_identical_bundle_excludes_history_and_untracked_files(self):
        before_refs = git(self.repo, "show-ref")
        before_status = git(self.repo, "status", "--porcelain")
        manifest = self.package()
        self.assertEqual(verify_bundle(self.output, self.origin), manifest)
        self.assertEqual(
            manifest["source_tree_sha"], git(self.repo, "rev-parse", "HEAD^{tree}").decode().strip()
        )
        self.assertEqual(
            manifest["bundle_sha256"],
            hashlib.sha256((self.output / BUNDLE_NAME).read_bytes()).hexdigest(),
        )
        git(
            self.root, "clone", "-b", "kaggle-snapshot", str(self.output / BUNDLE_NAME), "roundtrip"
        )
        clone = self.root / "roundtrip"
        self.assertEqual(git(clone, "rev-list", "--count", "--all").strip(), b"1")
        self.assertEqual(len(git(clone, "rev-list", "--parents", "-n", "1", "HEAD").split()), 1)
        self.assertEqual(git(clone, "ls-files").splitlines(), [b"source.py"])
        self.assertEqual((clone / "source.py").read_bytes(), (self.repo / "source.py").read_bytes())
        self.assertEqual(git(self.repo, "show-ref"), before_refs)
        self.assertEqual(git(self.repo, "status", "--porcelain"), before_status)

    def test_tampered_bundle_is_rejected_before_clone(self):
        self.package()
        with open(self.output / BUNDLE_NAME, "ab") as handle:
            handle.write(b"corruption")
        with self.assertRaisesRegex(ValueError, "hash or size mismatch"):
            verify_bundle(self.output)

    def test_manifest_digest_and_origin_mismatch_rejected(self):
        self.package()
        original_hash = digest(self.output / MANIFEST_NAME)
        self.rewrite_manifest(source_origin_sha="a" * 40)
        with self.assertRaisesRegex(ValueError, "Manifest SHA-256 mismatch"):
            verify_bundle(self.output, expected_manifest_sha256=original_hash)
        with self.assertRaisesRegex(ValueError, "Source origin mismatch"):
            verify_bundle(self.output, expected_origin=self.origin)

    def test_tree_and_head_mismatches_rejected(self):
        manifest = self.package()
        self.rewrite_manifest(source_tree_sha="a" * 40)
        with self.assertRaisesRegex(ValueError, "tree mismatch"):
            verify_bundle(self.output)
        self.rewrite_manifest(
            source_tree_sha=manifest["source_tree_sha"], snapshot_commit_sha="b" * 40
        )
        with self.assertRaisesRegex(ValueError, "snapshot ref"):
            verify_bundle(self.output)

    def test_full_history_bundle_is_rejected_even_with_matching_hash(self):
        self.package()
        full = self.root / "full.bundle"
        git(self.repo, "branch", "kaggle-snapshot", self.origin)
        git(self.repo, "bundle", "create", str(full), "refs/heads/kaggle-snapshot")
        (self.output / BUNDLE_NAME).write_bytes(full.read_bytes())
        self.rewrite_manifest(
            snapshot_commit_sha=self.origin,
            bundle_sha256=digest(full),
            bundle_size_bytes=full.stat().st_size,
        )
        with self.assertRaisesRegex(ValueError, "history is present"):
            verify_bundle(self.output)

    def test_tracked_secret_is_rejected_without_revealing_value(self):
        secret = "ghp_" + "Z" * 36
        (self.repo / "unsafe.txt").write_text(secret)
        git(self.repo, "add", "unsafe.txt")
        git(
            self.repo,
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@invalid",
            "commit",
            "-m",
            "unsafe",
        )
        origin = git(self.repo, "rev-parse", "HEAD").decode().strip()
        with self.assertRaises(ValueError) as caught:
            package_source(self.repo, origin, self.output)
        self.assertIn("github-token", str(caught.exception))
        self.assertNotIn(secret, str(caught.exception))
        self.assertFalse(self.output.exists())

    def test_tracked_forbidden_file_rejected_without_filtering_tree(self):
        git(self.repo, "add", "thesis.docx")
        git(
            self.repo,
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@invalid",
            "commit",
            "-m",
            "document",
        )
        origin = git(self.repo, "rev-parse", "HEAD").decode().strip()
        with self.assertRaisesRegex(ValueError, "forbidden-file"):
            package_source(self.repo, origin, self.output)

    def test_existing_output_is_not_overwritten(self):
        self.package()
        before = (self.output / MANIFEST_NAME).read_bytes()
        with self.assertRaisesRegex(ValueError, "absent or empty"):
            self.package()
        self.assertEqual((self.output / MANIFEST_NAME).read_bytes(), before)

    def test_cli_uses_resolved_full_sha(self):
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "scripts/package_kaggle_source.py",
                "--source-repo",
                str(self.repo),
                "--source-origin-sha",
                self.origin[:12],
                "--output-dir",
                str(self.output),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["source_origin_sha"], self.origin)


class TestNotebookBootstrap(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="test-og001-notebook-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / "source"
        (self.repo / "scripts").mkdir(parents=True)
        (self.repo / "scripts/package_kaggle_source.py").write_bytes(
            Path("scripts/package_kaggle_source.py").read_bytes()
        )
        self.setup_path = self.repo / "scripts/kaggle_setup.py"
        git(self.repo, "init")
        path = Path(
            "experiments/notebooks/kaggle/og001_overlay_validation/OG001_KAGGLE_OVERLAY_VALIDATION.ipynb"
        )
        notebook = json.loads(path.read_text())
        self.namespace = {"__name__": "notebook_test"}
        with patch.dict("os.environ"):
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    exec(compile("".join(cell["source"]), str(path), "exec"), self.namespace)
        self.namespace["ORCHESTRATION_COMMIT"] = "c" * 40
        self.namespace["RUN_INTEGRATION_TESTS"] = False
        self.working = self.root / "working"
        self.input = self.root / "input"

    def make_input(self, setup_code):
        self.setup_path.write_text(setup_code)
        git(self.repo, "add", "--all")
        git(
            self.repo,
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@invalid",
            "commit",
            "-m",
            "fixture",
        )
        origin = git(self.repo, "rev-parse", "HEAD").decode().strip()
        manifest = package_source(self.repo, origin, self.input / "attached-source")
        self.namespace.update(
            SOURCE_ORIGIN_SHA=origin,
            SOURCE_TREE_SHA=manifest["source_tree_sha"],
            SOURCE_MANIFEST_SHA256=digest(self.input / "attached-source" / MANIFEST_NAME),
        )

    def run_notebook(self):
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0"}):
            return self.namespace["run_validation"](self.input, self.working)

    def summary(self):
        return json.loads((self.working / "kernel_summary.json").read_text())

    def test_missing_source_writes_failure_summary_before_raising(self):
        with self.assertRaisesRegex(RuntimeError, "attached source manifest"):
            self.run_notebook()
        self.assertEqual(self.summary()["failure_phase"], "source_validation")
        self.assertIsNone(self.summary()["setup_returncode"])
        self.assertTrue((self.working / "orchestration_failure.log").is_file())

    def test_existing_evidence_is_never_overwritten_on_notebook_rerun(self):
        self.working.mkdir()
        original = b'{"original_run": "preserve-byte-for-byte"}\n'
        path = self.working / "kernel_summary.json"
        path.write_bytes(original)
        with self.assertRaisesRegex(FileExistsError, "evidence already exists"):
            self.run_notebook()
        self.assertEqual(path.read_bytes(), original)
        self.assertFalse((self.working / "orchestration_failure.log").exists())
        self.assertFalse((self.working / "OG001_RUN.json").exists())

    def test_corrupt_bundle_never_reaches_setup(self):
        self.make_input("raise SystemExit(37)\n")
        with open(self.input / "attached-source" / BUNDLE_NAME, "ab") as handle:
            handle.write(b"corruption")
        with self.assertRaisesRegex(RuntimeError, "bundle SHA-256"):
            self.run_notebook()
        self.assertIsNone(self.summary()["setup_returncode"])
        self.assertFalse((self.working / "fq-ssiuu").exists())

    def test_real_setup_failure_preserves_logs_and_return_code(self):
        self.make_input(
            "import sys\nprint('fixture setup stdout')\nprint('fixture causal failure', file=sys.stderr)\nraise SystemExit(37)\n"
        )
        with self.assertRaisesRegex(RuntimeError, "exit 37"):
            self.run_notebook()
        summary = self.summary()
        self.assertEqual(summary["source_validation_status"], "PASS")
        self.assertEqual(summary["setup_returncode"], 37)
        self.assertIn(
            "fixture causal failure", (self.working / "kaggle_runner_stderr.log").read_text()
        )
        self.assertFalse((self.working / "fq-ssiuu").exists())

    def test_zero_exit_without_evidence_is_failure(self):
        self.make_input("pass\n")
        with self.assertRaisesRegex(RuntimeError, "Missing setup evidence"):
            self.run_notebook()
        self.assertEqual(self.summary()["failure_phase"], "evidence_validation")
        self.assertEqual(self.summary()["execution_status"], "FAIL")

    def test_verified_dirty_baseline_success_through_real_bundle_bootstrap(self):
        # The fixture creates synthetic evidence; no GPU or installation occurs.
        for name in (
            "kaggle_setup.py",
            "verify_kaggle_env.py",
            "verify_pip_report.py",
            "pip_consistency.py",
            "validate_og001_artifacts.py",
        ):
            target = self.repo / "scripts" / name
            if name != "kaggle_setup.py":
                target.write_bytes(Path("scripts", name).read_bytes())
        (self.repo / "tests").mkdir()
        (self.repo / "tests/__init__.py").write_text("")
        (self.repo / "tests/og001_fixtures.py").write_bytes(
            Path("tests/og001_fixtures.py").read_bytes()
        )
        (self.repo / "requirements").mkdir()
        (self.repo / "requirements/kaggle-native-baseline.json").write_bytes(
            Path("requirements/kaggle-native-baseline.json").read_bytes()
        )
        # Fixture imports the actual setup helpers under another module name.
        (self.repo / "scripts/setup_helpers.py").write_bytes(
            Path("scripts/kaggle_setup.py").read_bytes()
        )
        fixture = self.repo / "tests/og001_fixtures.py"
        fixture.write_text(
            fixture.read_text().replace(
                "from scripts import kaggle_setup as ks", "from scripts import setup_helpers as ks"
            )
        )
        validation = self.repo / "scripts/validate_og001_artifacts.py"
        validation.write_text(
            validation.read_text().replace(
                "from scripts.kaggle_setup import", "from scripts.setup_helpers import"
            )
        )
        self.make_input(
            "import sys, subprocess\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).resolve().parents[1]))\nfrom tests.og001_fixtures import write_success_artifacts\nsha=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()\nwrite_success_artifacts(sys.argv[sys.argv.index('--output-dir')+1], sha)\n"
        )
        result = self.run_notebook()
        self.assertEqual(result["execution_status"], "PASS")
        self.assertEqual(result["pip_consistency_state"], "BASELINE_DIRTY_NO_REGRESSION")
        self.assertTrue((self.working / "og001_acceptance.json").is_file())


if __name__ == "__main__":
    unittest.main()
