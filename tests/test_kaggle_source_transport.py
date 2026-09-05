"""Offline transport tests using actual Git repositories and bundle round-trips."""

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from scripts.package_kaggle_source import (
    BUNDLE_NAME, MANIFEST_NAME, digest, git, package_source, verify_bundle,
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
        git(self.repo, "-c", "user.name=Transport Test", "-c", "user.email=test@invalid",
            "commit", "-m", message)
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
        self.assertEqual(manifest['source_tree_sha'], git(self.repo, 'rev-parse', 'HEAD^{tree}').decode().strip())
        self.assertEqual(manifest['bundle_sha256'], hashlib.sha256((self.output / BUNDLE_NAME).read_bytes()).hexdigest())
        git(self.root, 'clone', '-b', 'kaggle-snapshot', str(self.output / BUNDLE_NAME), 'roundtrip')
        clone = self.root / 'roundtrip'
        self.assertEqual(git(clone, 'rev-list', '--count', '--all').strip(), b'1')
        self.assertEqual(len(git(clone, 'rev-list', '--parents', '-n', '1', 'HEAD').split()), 1)
        self.assertEqual(git(clone, 'ls-files').splitlines(), [b'source.py'])
        self.assertEqual((clone / 'source.py').read_bytes(), (self.repo / 'source.py').read_bytes())
        self.assertEqual(git(self.repo, 'show-ref'), before_refs)
        self.assertEqual(git(self.repo, 'status', '--porcelain'), before_status)

    def test_tampered_bundle_is_rejected_before_clone(self):
        self.package()
        with open(self.output / BUNDLE_NAME, 'ab') as handle:
            handle.write(b'corruption')
        with self.assertRaisesRegex(ValueError, 'hash or size mismatch'):
            verify_bundle(self.output)

    def test_manifest_digest_and_origin_mismatch_rejected(self):
        self.package()
        original_hash = digest(self.output / MANIFEST_NAME)
        self.rewrite_manifest(source_origin_sha='a' * 40)
        with self.assertRaisesRegex(ValueError, 'Manifest SHA-256 mismatch'):
            verify_bundle(self.output, expected_manifest_sha256=original_hash)
        with self.assertRaisesRegex(ValueError, 'Source origin mismatch'):
            verify_bundle(self.output, expected_origin=self.origin)

    def test_tree_and_head_mismatches_rejected(self):
        manifest = self.package()
        self.rewrite_manifest(source_tree_sha='a' * 40)
        with self.assertRaisesRegex(ValueError, 'tree mismatch'):
            verify_bundle(self.output)
        self.rewrite_manifest(source_tree_sha=manifest['source_tree_sha'], snapshot_commit_sha='b' * 40)
        with self.assertRaisesRegex(ValueError, 'snapshot ref'):
            verify_bundle(self.output)

    def test_full_history_bundle_is_rejected_even_with_matching_hash(self):
        manifest = self.package()
        full = self.root / 'full.bundle'
        git(self.repo, 'branch', 'kaggle-snapshot', self.origin)
        git(self.repo, 'bundle', 'create', str(full), 'refs/heads/kaggle-snapshot')
        (self.output / BUNDLE_NAME).write_bytes(full.read_bytes())
        self.rewrite_manifest(snapshot_commit_sha=self.origin, bundle_sha256=digest(full),
                              bundle_size_bytes=full.stat().st_size)
        with self.assertRaisesRegex(ValueError, 'history is present'):
            verify_bundle(self.output)

    def test_tracked_secret_is_rejected_without_revealing_value(self):
        secret = 'ghp_' + 'Z' * 36
        (self.repo / 'unsafe.txt').write_text(secret)
        git(self.repo, 'add', 'unsafe.txt')
        git(self.repo, '-c', 'user.name=Test', '-c', 'user.email=test@invalid', 'commit', '-m', 'unsafe')
        origin = git(self.repo, 'rev-parse', 'HEAD').decode().strip()
        with self.assertRaises(ValueError) as caught:
            package_source(self.repo, origin, self.output)
        self.assertIn('github-token', str(caught.exception))
        self.assertNotIn(secret, str(caught.exception))
        self.assertFalse(self.output.exists())

    def test_tracked_forbidden_file_rejected_without_filtering_tree(self):
        git(self.repo, 'add', 'thesis.docx')
        git(self.repo, '-c', 'user.name=Test', '-c', 'user.email=test@invalid', 'commit', '-m', 'document')
        origin = git(self.repo, 'rev-parse', 'HEAD').decode().strip()
        with self.assertRaisesRegex(ValueError, 'forbidden-file'):
            package_source(self.repo, origin, self.output)

    def test_existing_output_is_not_overwritten(self):
        self.package()
        before = (self.output / MANIFEST_NAME).read_bytes()
        with self.assertRaisesRegex(ValueError, 'absent or empty'):
            self.package()
        self.assertEqual((self.output / MANIFEST_NAME).read_bytes(), before)

    def test_cli_uses_resolved_full_sha(self):
        import sys
        result = subprocess.run([sys.executable, 'scripts/package_kaggle_source.py',
                                 '--source-repo', str(self.repo), '--source-origin-sha', self.origin[:12],
                                 '--output-dir', str(self.output)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['source_origin_sha'], self.origin)


if __name__ == '__main__':
    unittest.main()
