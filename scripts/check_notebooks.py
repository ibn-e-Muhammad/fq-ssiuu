"""Validate tracked notebooks and compile Python cells without executing them."""

import argparse
import json
import subprocess
from pathlib import Path

import nbformat


def check_notebooks(root):
    root = Path(root)
    names = (
        subprocess.check_output(["git", "ls-files", "-z", "*.ipynb"], cwd=root).decode().split("\0")
    )
    results = []
    for name in filter(None, names):
        entry = {"path": name, "status": "PASS", "python_cells": 0, "transformed_cells": 0}
        try:
            notebook = nbformat.read(root / name, as_version=4)
            nbformat.validate(notebook)
            for i, cell in enumerate(notebook.cells):
                if cell.cell_type != "code":
                    continue
                source = cell.source
                try:
                    compile(source, f"{name}:cell-{i}", "exec")
                except SyntaxError:
                    from IPython.core.inputtransformer2 import TransformerManager

                    source = TransformerManager().transform_cell(source)
                    compile(source, f"{name}:cell-{i}:ipython", "exec")
                    entry["transformed_cells"] += 1
                entry["python_cells"] += 1
        except Exception as exc:
            entry.update(status="FAIL", error_type=type(exc).__name__)
        results.append(entry)
    return {
        "status": "PASS" if results and all(r["status"] == "PASS" for r in results) else "FAIL",
        "notebooks": results,
        "execution_performed": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = check_notebooks(args.root)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "notebook_count": len(result["notebooks"])}))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
