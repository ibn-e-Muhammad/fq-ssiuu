"""Read-only live display of subprocess logs and existing NPO evidence files."""

import codecs
import json
import os
import signal
import subprocess
import time
from datetime import UTC, datetime


def timestamp():
    return datetime.now(UTC).isoformat()


class EvidenceDisplay:
    """Report observed artifacts, never infer success from a file's existence."""

    def __init__(self, raw):
        self.raw = raw
        self.seen = {}

    def poll(self):
        root = self.raw / "NPO"
        paths = list(root.glob("*.json"))
        paths += list(root.glob("*/RESULT.json"))
        paths += list(root.glob("evaluation_step_*/eval_log_forget.json"))
        paths += list(root.glob("oracle_evaluation/*.json"))
        paths += [root / "training_checkpoint/MANIFEST.json"]
        paths += [root / "traceback.log"]
        paths += [
            root / "training" / name
            for name in ("forward_observations.jsonl", "microbatches.jsonl")
        ]
        for path in sorted(set(paths)):
            if not path.is_file():
                continue
            stat = path.stat()
            marker = (stat.st_mtime_ns, stat.st_size)
            if self.seen.get(path) == marker:
                continue
            self.seen[path] = marker
            relative = path.relative_to(self.raw).as_posix()
            message = f"[{timestamp()}] [NPO evidence observed] {relative} ({stat.st_size} bytes)"
            if path.suffix == ".jsonl":
                with path.open("rb") as handle:
                    handle.seek(max(0, stat.st_size - 16384))
                    lines = handle.read().split(b"\n")
                # Ignore the last fragment until the runtime flushes a complete line.
                if len(lines) >= 2:
                    try:
                        record = json.loads(lines[-2])
                        keys = (
                            "microbatch",
                            "phase",
                            "optimizer_step",
                            "total_loss",
                            "optimizer_boundary",
                            "fp16_overflow",
                        )
                        message += " " + json.dumps({k: record[k] for k in keys if k in record})
                    except (ValueError, UnicodeDecodeError):
                        pass
            elif path.name == "traceback.log":
                message += "\n" + path.read_text(encoding="utf-8", errors="replace")
            print(message, flush=True)
            with (self.raw / "npo.progress.log").open("a", encoding="utf-8") as log:
                log.write(message + "\n")


def stop_child(child):
    """Stop the owned process tree before the caller seals its output directory."""
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(child.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if child.poll() is None:
        child.kill()
    child.wait(timeout=30)


def stream_command(name, argv, raw, env, root, write):
    argv = [str(x) for x in argv]
    record = {"argv": argv, "cwd": str(root), "started_utc": timestamp()}
    print(
        f"\n[{timestamp()}] START {name}\nCommand: {subprocess.list2cmdline(argv)}"
        f"\nLogs: {raw / (name + '.stdout.log')} / {name}.stderr.log",
        flush=True,
    )
    child = None
    observer = EvidenceDisplay(raw) if name == "npo" else None
    env = dict(env, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")

    def active_command(status):
        temporary = raw / "active_command.tmp"
        temporary.write_text(
            json.dumps({"name": name, "status": status, **record}, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(raw / "active_command.json")

    active_command("STARTING")
    try:
        with (
            (raw / f"{name}.stdout.log").open("xb") as out,
            (raw / f"{name}.stderr.log").open("xb") as err,
            (raw / f"{name}.stdout.log").open("rb") as out_read,
            (raw / f"{name}.stderr.log").open("rb") as err_read,
        ):
            decoders = [codecs.getincrementaldecoder("utf-8")("replace") for _ in range(2)]

            def display(final=False):
                for handle, decoder in zip((out_read, err_read), decoders, strict=True):
                    while chunk := handle.read(65536):
                        print(decoder.decode(chunk), end="", flush=True)
                    if final:
                        print(decoder.decode(b"", final=True), end="", flush=True)
                if observer:
                    try:
                        observer.poll()
                    except OSError as error:
                        # Display races or inaccessible evidence must not alter training.
                        print(f"[evidence display warning] {error}", flush=True)

            options = (
                {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                if os.name == "nt"
                else {"start_new_session": True}
            )
            child = subprocess.Popen(argv, cwd=root, env=env, stdout=out, stderr=err, **options)
            try:
                record["pid"] = child.pid
                active_command("RUNNING")
                last_heartbeat = time.monotonic()
                while child.poll() is None:
                    display()
                    if time.monotonic() - last_heartbeat >= 30:
                        print(
                            f"[{timestamp()}] RUNNING {name} (PID {child.pid}); "
                            "waiting for further child output",
                            flush=True,
                        )
                        last_heartbeat = time.monotonic()
                    time.sleep(0.2)
            except BaseException:
                stop_child(child)
                raise
            finally:
                record["returncode"] = child.poll()
                display(final=True)
        if child.returncode:
            raise RuntimeError(
                f"{name} failed with exit code {child.returncode}; "
                f"see {raw / (name + '.stderr.log')} and {raw / 'NPO/traceback.log'}"
            )
    except BaseException as error:
        record["error_type"] = type(error).__name__
        print(f"\n[{timestamp()}] FAILED {name}: {type(error).__name__}: {error}", flush=True)
        raise
    finally:
        record["finished_utc"] = timestamp()
        write(raw / f"{name}.execution.json", record)
        active_command("FAILED" if "error_type" in record else "COMPLETE")
    print(f"\n[{timestamp()}] COMPLETE {name}", flush=True)
