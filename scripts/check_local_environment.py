"""Offline package/import probe, executed by the managed Windows interpreter."""

import argparse
import importlib
import importlib.metadata as metadata
import json
import platform
import re
import sys
import traceback
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname


def normalized(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def verify(root):
    if platform.system() != "Windows" or sys.version_info[:2] not in ((3, 11), (3, 12)):
        raise RuntimeError("Requires native Windows Python 3.11/3.12")
    if Path(sys.prefix).resolve() != (root / ".venv-local").resolve():
        raise RuntimeError("Probe must run inside this checkout's .venv-local")
    inventory = {}
    for dist in metadata.distributions():
        name, version = dist.metadata.get("Name"), dist.version
        if not name or not version:
            raise RuntimeError("Damaged package metadata: missing Name/Version")
        key = normalized(name)
        if key in inventory:
            # Setuptools editable installs may expose both the source egg-info
            # and the installed dist-info for this same project.
            if key == "fq-ssiuu" and inventory[key] == version:
                continue
            raise RuntimeError(f"Duplicate installed distribution: {key}")
        inventory[key] = version
    for line in (root / "requirements/local-windows.in").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split("==")
        name = normalized(parts[0])
        if name not in inventory or (len(parts) == 2 and inventory[name] != parts[1]):
            raise RuntimeError(f"Missing or mismatched requirement: {line}")
    pin = json.loads((root / "experiments/reproductions/NPO/runtime_sources_v1.json").read_text())[
        "deepspeed"
    ]
    if inventory.get("deepspeed") != pin["version"]:
        raise RuntimeError("DeepSpeed version differs from D1 pin")
    direct = json.loads(metadata.distribution("deepspeed").read_text("direct_url.json") or "{}")
    archive = direct.get("archive_info", {})
    source_hash = archive.get("hashes", {}).get("sha256")
    if source_hash is None and archive.get("hash", "").startswith("sha256="):
        source_hash = archive["hash"].split("=", 1)[1]
    if direct.get("url") != pin["url"] or source_hash != pin["sha256"]:
        raise RuntimeError("DeepSpeed installation has no matching pinned-source receipt")
    project = json.loads(metadata.distribution("fq-ssiuu").read_text("direct_url.json") or "{}")
    project_url = urlparse(project.get("url", ""))
    installed_root = Path(url2pathname(project_url.path)).resolve()
    if (
        project_url.scheme != "file"
        or installed_root != root
        or project.get("dir_info", {}).get("editable") is not True
    ):
        raise RuntimeError("Project must be installed editable from this checkout")
    modules = (
        "torch",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "deepspeed",
        "numpy",
        "scipy",
        "safetensors",
        "huggingface_hub",
        "datasets",
        "peft",
        "rouge_score",
        "nltk",
        "pynvml",
        "psutil",
        "hjson",
        "yaml",
        "dotenv",
        "fq_ssiuu",
    )
    for module in modules:
        print(f"Import check: {module}", flush=True)
        importlib.import_module(module)
    import torch
    from bitsandbytes.optim import PagedAdamW32bit
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.version.cuda != "12.8":
        raise RuntimeError("Expected the pinned CUDA 12.8 Torch build")
    # No model loading, optimizer construction or GPU-memory admission gate.
    return {
        "status": "PASS",
        "python": platform.python_version(),
        "prefix": str(Path(sys.prefix).resolve()),
        "project_root": str(root),
        "inventory": dict(sorted(inventory.items())),
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "import_checks": list(modules),
        "symbols": [
            PagedAdamW32bit.__name__,
            AutoModelForCausalLM.__name__,
            AutoTokenizer.__name__,
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify(args.root.resolve())
    except Exception as error:
        traceback.print_exc()
        result = {"status": "FAIL", "error_type": type(error).__name__, "error": str(error)}
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "inventory"}), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
