"""Persistent local environment lifecycle; never install while reusing a receipt."""

import hashlib
import json
import os
from datetime import UTC, datetime

CONTRACT_FILES = (
    "requirements/local-windows.in",
    "pyproject.toml",
    "experiments/reproductions/NPO/runtime_sources_v1.json",
    "scripts/check_local_environment.py",
)
RECEIPT_NAME = "fq-environment.json"


def contract(root):
    return {
        "schema": 1,
        "files": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in CONTRACT_FILES
        },
    }


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def ensure_environment(root, raw, env, command, install, mode="auto"):
    """auto: install only if absent; check: never install; repair: explicit reconcile."""
    if mode not in {"auto", "check", "repair"}:
        raise ValueError("Environment mode must be auto, check or repair")
    venv = root / ".venv-local"
    python = venv / "Scripts/python.exe"
    receipt_path = venv / RECEIPT_NAME
    wanted = contract(root)
    existed = venv.exists()
    if mode == "check" and not python.is_file():
        raise RuntimeError("No local environment. Run --setup-only once (environment auto).")
    # The lock also covers read-only verification: no concurrent pip mutation.
    venv.mkdir(exist_ok=True)
    lock = venv / "fq-setup.lock"
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise RuntimeError(
            f"Environment setup/check already locked: {lock}. If a process crashed, "
            "confirm its PID is no longer running before removing this lock."
        ) from error
    try:
        with handle:
            json.dump({"pid": os.getpid(), "started_utc": datetime.now(UTC).isoformat()}, handle)
        previous = None
        if receipt_path.is_file():
            try:
                previous = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (ValueError, UnicodeError) as error:
                if mode != "repair":
                    raise RuntimeError(
                        "Invalid environment receipt; use --setup-only --environment repair"
                    ) from error
        if mode != "repair" and previous and previous.get("contract") != wanted:
            raise RuntimeError(
                "Environment requirements or verifier changed. No packages were installed. "
                "Run --setup-only --environment repair to reconcile explicitly."
            )
        if mode != "repair" and existed and not python.is_file():
            raise RuntimeError(
                "Existing .venv-local has no Python executable (incomplete setup). "
                "Run --setup-only --environment repair."
            )
        action = "REUSED" if previous else "ADOPTED_EXISTING"
        if mode == "repair" or not existed:
            action = "REPAIRED" if existed else "CREATED"
            # Preserve the old receipt in the attempt, then invalidate before mutation.
            if receipt_path.exists():
                (raw / "previous_environment_receipt.json").write_bytes(receipt_path.read_bytes())
                receipt_path.unlink()
            print(f"Environment {action}: installing dependencies into {venv}", flush=True)
            install(raw, env)
        else:
            print(f"Environment {action}: checking {venv}; no pip install or downloads", flush=True)

        # These commands are local checks. Do not invoke pip install even on failure.
        check_env = dict(
            env,
            PIP_DISABLE_PIP_VERSION_CHECK="1",
            PIP_NO_INDEX="1",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            HF_DATASETS_OFFLINE="1",
        )
        command("pip_check", [python, "-m", "pip", "check"], raw, check_env)
        probe = raw / "environment_probe.json"
        command(
            "environment_probe",
            [
                python,
                root / "scripts/check_local_environment.py",
                "--root",
                root,
                "--output",
                probe,
            ],
            raw,
            check_env,
        )
        report = json.loads(probe.read_text(encoding="utf-8"))
        if report.get("status") != "PASS":
            raise RuntimeError("Environment probe failed; see environment_probe.json")
        if previous and mode != "repair":
            for key in ("inventory", "python", "prefix", "project_root"):
                if report[key] != previous.get("verified", {}).get(key):
                    raise RuntimeError(
                        f"Environment drift ({key}); no packages were installed. "
                        "Inspect environment_probe.json and use --setup-only --environment repair."
                    )
        command("inventory", [python, "-m", "pip", "freeze", "--all"], raw, check_env)
        receipt = {
            "contract": wanted,
            "verified": report,
            "verified_utc": datetime.now(UTC).isoformat(),
        }
        save(receipt_path, receipt)
        save(raw / "environment_receipt.json", receipt)
        save(
            raw / "environment_action.json",
            {
                "mode": mode,
                "action": action,
                "package_installation_invoked": action in {"CREATED", "REPAIRED"},
                "interpreter": str(python),
                "receipt": str(receipt_path),
            },
        )
        print(f"Environment ready ({action}): {python}", flush=True)
        return python
    except Exception:
        print(
            "Environment not accepted. Logs are preserved; there is no automatic reinstall. "
            "After diagnosing the error, use --setup-only --environment repair if needed.",
            flush=True,
        )
        raise
    finally:
        lock.unlink(missing_ok=True)
