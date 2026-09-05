# CR-011: Kaggle Overlay Validation Coverage and Evidence Linkage

| Field | Value |
|-------|-------|
| Status | DOCUMENTED — BEHAVIORALLY REPAIRED |
| Date discovered | 2026-09-05 |
| Affected files | `tests/test_kaggle_dependency_policy.py`, `scripts/kaggle_setup.py`, `scripts/verify_kaggle_env.py` |
| Severity | MEDIUM — infrastructure validation integrity |

---

## 1. Summary of Correction

The previous 22-test suite in `tests/test_kaggle_dependency_policy.py` provided static and structural checks but did not behaviorally exercise every production setup failure path. Specifically:

1. Tests inspected the local host machine GPU or Python environment rather than fully mocked execution paths.
2. Production code duplicated logic inside tests instead of invoking real production failure paths.
3. Subprocess failures (`pip freeze`, `git` provenance), missing install reports, verifier status discrepancies (`PARTIAL`), and run ID mismatches were not verified with end-to-end behavioral tests.
4. Environment manifests (`before_manifest.json`, `after_manifest.json`, `post_attempt_manifest.json`) and overlay locks lacked a single unified `run_id` across setup and verifier invocations.

The previous 22-test result was real in executing static checks, but it suffered from insufficient behavioral coverage and incomplete provenance linkage.

---

## 2. Corrective Actions Taken (Stage 1H-D)

1. **Baseline Validation Enforcement**:
   - `scripts/kaggle_setup.py` and `scripts/verify_kaggle_env.py` reuse `validate_baseline` from `scripts/verify_pip_report.py`.
   - Setup halts immediately with exit code 1 and writes `setup_status.json` if the baseline JSON is malformed, empty, or missing required protected stacks.

2. **Unified Run ID Linkage**:
   - `scripts/verify_kaggle_env.py` accepts `--run-id`.
   - `scripts/kaggle_setup.py` passes its generated run ID to `verify_kaggle_env.py` and validates that the verifier report's `run_id`, `git_sha`, and overall status (`PASS`) match.

3. **Complete Environment Snapshots**:
   - Implemented `create_environment_snapshot` to write schema-conforming `before_manifest.json`, `after_manifest.json`, and best-effort `post_attempt_manifest.json` on failure paths.

4. **Fail-Closed Subprocess & Artifact Guard**:
   - Git provenance failures (`UNKNOWN` or non-zero exit) halt setup.
   - Non-zero `pip freeze` exits halt setup.
   - Missing `pip_install_report.json` after pip installation fails setup.
   - Verifier stdout, stderr, and return codes are persisted to `verifier_stdout.log`, `verifier_stderr.log`, and `verification_report.json`.

5. **Production Overlay Lock Generation**:
   - Refactored lock calculation into `generate_overlay_lock(before_pkgs, after_pkgs, baseline, run_id, git_sha)`.
   - Evaluates all non-protected distributions that were added, upgraded, or downgraded (including changed transitive dependencies).
   - Strictly excludes the protected compute stack and unchanged native stability stack.

6. **Behavioral Unit Test Suite**:
   - Fully updated `tests/test_kaggle_dependency_policy.py` with 20 pure behavioral tests using temporary directories and mocks.
   - Proves all 20 specified setup failure paths and lock generation behaviors without depending on host machine hardware or packages.

---

## 3. Scope and Status Limits

- **OG-001** remains `PARTIAL` pending Stage 1I validation on Kaggle.
- **RG-001**, **RG-002**, and **RG-003** remain `BLOCKED`.
- Environment reproducibility has not been claimed or demonstrated on live Kaggle hardware.
