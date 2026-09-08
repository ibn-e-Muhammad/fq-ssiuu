"""Stage the single Mother notebook with verified, immutable run inputs."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from scripts.package_kaggle_source import verify_bundle


def stage(source, dataset_id, recipe, output):
    root = Path(__file__).resolve().parents[1]
    source, output = Path(source), Path(output)
    receipt_path = Path(str(output) + ".provenance.json")
    if output.exists() or receipt_path.exists():
        raise FileExistsError("Staging output or provenance already exists")
    manifest_path = source / "SOURCE_MANIFEST.json"
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest = verify_bundle(source, expected_manifest_sha256=manifest_hash)
    if Path(recipe).is_absolute() or ".." in Path(recipe).parts:
        raise ValueError("Recipe must be relative to the verified source")
    subprocess.check_call(
        ["git", "cat-file", "-e", manifest["source_origin_sha"] + ":" + recipe],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root
    ).strip():
        raise ValueError("Commit tracked changes before staging")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip()
    path = root / "experiments/notebooks/kaggle/MOTHER_RUNNER.ipynb"
    notebook = json.loads(path.read_text())
    parameters = {
        "source_origin_sha": manifest["source_origin_sha"],
        "source_tree_sha": manifest["source_tree_sha"],
        "source_manifest_sha256": manifest_hash,
        "dataset_id": dataset_id,
        "orchestration_commit": commit,
        "recipe": recipe,
    }
    notebook["cells"][1]["source"] = ("PARAMETERS = " + repr(parameters) + "\n").splitlines(True)
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), path.name, "exec")
    output.mkdir(parents=True, exist_ok=False)
    (output / path.name).write_text(json.dumps(notebook, indent=1) + "\n")
    owner = dataset_id.split("/")[0]
    metadata = {
        "id": f"{owner}/fq-ssiuu-mother-runner",
        "title": "FQ-SSIUU Mother Runner",
        "code_file": path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [dataset_id],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
        "docker_image_pinning_type": "original",
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
    }
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    receipt = {
        "parameters": parameters,
        "committed_notebook_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "staged_notebook_sha256": hashlib.sha256((output / path.name).read_bytes()).hexdigest(),
        "metadata_sha256": hashlib.sha256(
            (output / "kernel-metadata.json").read_bytes()
        ).hexdigest(),
    }
    # Keep staging receipts outside the upload directory.
    with receipt_path.open("x") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--recipe", default="configs/runs/integration_v1.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(stage(args.source, args.dataset_id, args.recipe, args.output), indent=2))
