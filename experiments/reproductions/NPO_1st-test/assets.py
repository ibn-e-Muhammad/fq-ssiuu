"""Fetch public, hash-pinned method-validation inputs; never load credentials."""

import argparse
import json
from pathlib import Path
from urllib.request import urlopen

from experiments.reproductions.common import file_digest

PINS = Path(__file__).with_name("inputs_v1.json")
REPOSITORIES = {
    "npo": "licong-lin/negative-preference-optimization",
    "openunlearning": "locuslab/open-unlearning",
}


def fetch_verified(url, path, expected):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        # Public endpoints only. urllib does not consult dotenv or HF credentials.
        with urlopen(url, timeout=90) as response:
            data = response.read()
        import hashlib

        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("Downloaded bytes differ from pinned SHA256: " + path.name)
        with path.open("xb") as handle:
            handle.write(data)
    if file_digest(path) != expected:
        raise ValueError("Existing asset differs from pinned SHA256: " + path.name)
    return path


def materialize(root, *, upstream=True):
    root = Path(root)
    pins = json.loads(PINS.read_text())
    records = []
    for item in pins["files"]:
        dataset = item["repo"] == pins["dataset"]["id"]
        prefix = "datasets/" if dataset else ""
        url = f"https://huggingface.co/{prefix}{item['repo']}/resolve/{item['revision']}/{item['path']}"
        path = root / ("dataset" if dataset else "tokenizer") / item["path"]
        fetch_verified(url, path, item["sha256"])
        records.append({**item, "verified": True})
    if upstream:
        for name, upstream_pin in pins["upstreams"].items():
            for filename, expected in upstream_pin["files"].items():
                url = f"https://raw.githubusercontent.com/{REPOSITORIES[name]}/{upstream_pin['revision']}/{filename}"
                fetch_verified(url, root / name / filename, expected)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.root), indent=2))
