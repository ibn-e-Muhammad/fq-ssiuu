"""Summarise explicit human labels; never classify model answers heuristically."""

import argparse
from pathlib import Path

import pandas as pd

CATEGORIES = {"correct", "partial", "refusal", "gibberish", "fabrication", "substitution", "other"}


def summarize_labels(path, output):
    data = pd.read_csv(path).fillna("")
    for key in ("stage", "split", "id", "category", "reviewer"):
        if key not in data:
            raise ValueError(f"Missing annotation field: {key}")
    labeled = data.loc[data.category.str.strip().ne("")].copy()
    if not set(labeled.category).issubset(CATEGORIES):
        raise ValueError("Unknown category; follow the supplied human audit rubric")
    if (labeled.reviewer.str.strip() == "").any():
        raise ValueError("Every judgment must identify its reviewer")
    totals = data.groupby(["stage", "split"]).size().rename("available")
    counts = labeled.groupby(["stage", "split", "category"]).size().rename("count").reset_index()
    reviewed = labeled.groupby(["stage", "split"]).size().rename("reviewed")
    counts = counts.merge(totals.reset_index(), on=["stage", "split"]).merge(
        reviewed.reset_index(), on=["stage", "split"]
    )
    counts["rate_among_reviewed"] = counts["count"] / counts["reviewed"]
    counts["review_coverage"] = counts["reviewed"] / counts["available"]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts.to_csv(output, index=False)
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(summarize_labels(args.annotations, args.output).to_string(index=False))
