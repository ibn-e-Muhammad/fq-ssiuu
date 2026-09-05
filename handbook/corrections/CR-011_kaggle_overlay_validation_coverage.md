# CR-011: Kaggle Overlay Validation Coverage and Evidence Linkage

| Field | Value |
|-------|-------|
| Status | DOCUMENTED — BEHAVIORALLY REPAIRED |
| Date discovered | 2026-09-05 |
| Affected files | `tests/test_kaggle_dependency_policy.py`, `scripts/kaggle_setup.py`, `scripts/verify_kaggle_env.py` |
| Severity | MEDIUM — infrastructure validation integrity |

---

## 1. Summary of Correction

The previous test suite in `tests/test_kaggle_dependency_policy.py` provided static and structural checks but did not behaviorally exercise every production setup failure path. Specifically:

1. Tests inspected the local host machine GPU or Python environment rather than fully mocked execution paths.
2. Production code duplicated logic inside tests instead of invoking real production failure paths.
3. Subprocess failures (`pip freeze`, `git` provenance), missing install reports, verifier status discrepancies (`PARTIAL`), and run ID mismatches were not verified with end-to-end behavioral tests.
4. Environment manifests (`before_manifest.json`, `after_manifest.json`, `post_attempt_manifest.json`) and overlay locks lacked a single unified `run_id` across setup and verifier invocations.

The previous test result was real in executing static checks, but it suffered from insufficient behavioral coverage and incomplete provenance linkage.

---

## 2. Corrective Actions Taken (Stage 1H-D & Stage 1I Preflight Repair)

1. **PEP 440 Version Delta Comparison**:
   - `scripts/kaggle_setup.py` uses `compare_versions` with `packaging.version.Version` and `InvalidVersion` for PEP 440-aware comparison (`0.9.0 → 0.10.0` upgraded, `1.10.0 → 1.9.0` downgraded, `1.0rc1 → 1.0` upgraded, invalid parseable versions classified as `changed-unordered`).

2. **Protected Package Mutation Guard**:
   - `validate_protected_packages_unchanged` in `scripts/kaggle_setup.py` asserts that no protected compute stack (`torch`, `torchvision`, `torchaudio`, `triton`) or native stability stack (`transformers`, `datasets`, `accelerate`, `peft`, `huggingface-hub`, `numpy`, `scipy`, `pandas`, `omegaconf`, `tokenizers`, `safetensors`) was mutated. Lock generation fails closed with `ValueError` if any protected package changed.

3. **Single Final Environment Snapshot**:
   - `scripts/kaggle_setup.py` executes `pip freeze` exactly once after installation for `after_freeze.log`, `after_manifest.json`, and lock calculation. Requires `after_manifest["snapshot_status"] == "PASS"`.

4. **Verifier Execution Record & Linkage**:
   - Created `verifier_execution.json` recording setup Git SHA, verifier Git SHA, returncode, overall status, log paths, log SHA-256 hashes, run ID linkage verdict, and Git SHA linkage verdict.
   - Implemented `finalize_setup_failure` ensuring all failure artifacts (`setup_status.json`, `post_attempt_manifest.json`, `verifier_execution.json`, `MANIFEST.json`) are persisted on all failure branches.

5. **Production Helper Functions & Behavioral Unit Test Suite**:
   - Refactored `exit_code_for_status(status)`, `run_nf4_smoke(torch, bnb)`, and `run_sdpa_smoke(torch)` in `scripts/verify_kaggle_env.py`.
   - Updated `tests/test_kaggle_dependency_policy.py` with 20 behavioral tests asserting all setup failure paths, version delta classifications, protected stack mutation rejections, snapshot checks, and verifier execution artifact persistence using fakes/mocks without depending on host machine hardware or packages.

---

## 3. Scope and Status Limits

- **OG-001** remains `PARTIAL` pending live single Kaggle T4 validation execution.
- **RG-001**, **RG-002**, and **RG-003** remain `BLOCKED`.
- Environment reproducibility has not been claimed or demonstrated on live Kaggle hardware.
