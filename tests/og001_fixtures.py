"""Synthetic evidence for orchestration tests; never hardware/research evidence."""

import json
from pathlib import Path

from scripts import kaggle_setup as ks
from scripts import pip_consistency as pc
from scripts.verify_kaggle_env import REQUIRED_RUNTIME_CHECKS


def synthetic_inventory(packages):
    return {
        "packages": packages,
        "distributions": [
            {
                "name": name,
                "version": version,
                "location": "/synthetic/site-packages",
                "precedence": 0,
            }
            for name, version in sorted(packages.items())
        ],
        "duplicate_names": [],
    }


def write_success_artifacts(output, sha, baseline_path=None):
    output = Path(output)
    output.mkdir(parents=True)
    run = "synthetic-run"
    baseline_path = baseline_path or Path(ks.REPO_ROOT, "requirements/kaggle-native-baseline.json")
    baseline = json.loads(Path(baseline_path).read_text())
    (output / "native_baseline.json").write_bytes(Path(baseline_path).read_bytes())
    versions = {**baseline["hard_protected_compute_stack"], **baseline["native_stability_stack"]}
    post = {**versions, "bitsandbytes": "0.45.5"}

    def write(name, data):
        ks.atomic_write_json(str(output / name), data)

    dirty = "native-owner 1.0 requires absent-dep, which is not installed.\n"
    records = {}
    for stage in ("before", "after", "verifier"):
        records[stage] = pc.make_record(dirty, "", 1, stage, run, sha)
        write(f"pip_check_{stage}.json", records[stage])
        (output / f"pip_check_{stage}_stdout.log").write_bytes(dirty.encode())
        (output / f"pip_check_{stage}_stderr.log").write_bytes(b"")
    consistency = pc.compare_checks(records["before"], records["after"], ["bitsandbytes"])
    write("pip_consistency.json", consistency)
    report = {"install": [{"metadata": {"name": "bitsandbytes", "version": "0.45.5"}}]}
    for name in ("pip_install_report.json", "pip_dry_run_report.json"):
        write(name, report)
    for name in ("install_verification_report.json", "verification_report.json"):
        write(name, {"run_id": run, "status": "PASS"})
    snapshots = {}
    for stage, packages in (("before", versions), ("after", post)):
        snapshots[stage] = dict(
            run_id=run,
            git_commit=sha,
            git_dirty=False,
            stage=stage,
            snapshot_status="PASS",
            driver_version="580.159.04",
            python_version="3.12.13",
            installed_packages=packages,
        )
        write(
            f"{stage}_distributions.json",
            {
                "schema_version": "1.1",
                "stage": stage,
                "run_id": run,
                "git_sha": sha,
                "source": "importlib.metadata.distributions",
                **synthetic_inventory(packages),
                "freeze_disagreements": [],
                "status": "PASS",
            },
        )
        write(f"{stage}_manifest.json", snapshots[stage])
        (output / f"{stage}_freeze.log").write_text(
            "\n".join(f"{k}=={v}" for k, v in packages.items()) + "\n"
        )
    write(
        "environment_transition.json",
        ks.compare_environment_transition(
            snapshots["before"],
            snapshots["after"],
            versions,
            post,
            baseline,
            report,
            synthetic_inventory(versions)["distributions"],
            synthetic_inventory(post)["distributions"],
        ),
    )
    checks = {name: {"status": "PASS"} for name in REQUIRED_RUNTIME_CHECKS}
    checks["pip_check"]["observed"] = pc.compare_checks(
        records["before"], records["verifier"], ["bitsandbytes"]
    )
    checks["peak_memory"]["observed"] = {
        "device": "cuda:0",
        "peak_allocated_bytes": 1024,
        "peak_reserved_bytes": 2048,
        "budget_bytes": 16_000_000_000,
    }
    checks["cuda_device_count"]["observed"] = 1
    checks["cuda_visible_devices_var"]["observed"] = "0"
    checks["nf4_quantise_dequantise"]["observed"] = {"quant_type": "nf4"}
    write(
        "kaggle_env_verification.json",
        {"run_id": run, "git_sha": sha, "statuses": {"overall": "PASS"}, "checks": checks},
    )
    (output / "verifier_stdout.log").write_text("synthetic runtime boundary\n")
    (output / "verifier_stderr.log").write_bytes(b"")
    ks.create_verifier_execution_record(str(output), run, sha, sha, 0, "PASS", True, True, "PASS")
    hashes = {
        key: ks.file_sha256(str(output / file))
        for key, file in [
            ("pip_dry_run_report", "pip_dry_run_report.json"),
            ("pip_install_report", "pip_install_report.json"),
            ("pip_consistency", "pip_consistency.json"),
            ("environment_transition", "environment_transition.json"),
        ]
    }
    lock, text = ks.generate_overlay_lock(versions, post, baseline, run, sha, hashes)
    write("overlay_lock.json", lock)
    (output / "kaggle-overlay.lock").write_text(text)
    write(
        "setup_status.json",
        {
            "run_id": run,
            "git_sha": sha,
            "overall_status": "PASS",
            "mode": "APPLY",
            "dry_run_exit_code": 0,
            "install_exit_code": 0,
            "editable_exit_code": 0,
            "verifier_exit_code": 0,
            "pip_check_exit_code": 1,
            "pip_consistency_state": "BASELINE_DIRTY_NO_REGRESSION",
        },
    )
    ks.generate_manifest(str(output), run, sha, "PASS", records["before"]["utc_timestamp"])


def refresh_manifest(output, sha):
    ks.generate_manifest(str(output), "synthetic-run", sha, "PASS", "synthetic-start")
