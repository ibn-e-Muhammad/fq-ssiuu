"""Reanalyse immutable second-run CSVs. No training, model loading or inference."""

import argparse
import ast
import importlib.metadata
import json
import platform
from pathlib import Path

import pandas as pd
from npo3.metrics import (
    digest,
    load_existing_stages,
    quantization_comparisons,
    sha256,
    summarize,
    write_json,
)


def run(run_dir, output_dir, repeats=2000):
    run_dir, output_dir = Path(run_dir).resolve(), Path(output_dir).resolve()
    if output_dir == run_dir or run_dir in output_dir.parents:
        raise ValueError("Derived analysis must live outside the immutable original run")
    output_dir.mkdir(parents=True, exist_ok=True)
    stages, inventory = load_existing_stages(run_dir)
    if not all(
        s in stages
        for s in (
            "npo_bf16",
            "retain95_bf16",
            "npo_int8",
            "npo_nf4",
            "retain95_int8",
            "retain95_nf4",
        )
    ):
        raise ValueError("All six candidate/oracle per-question CSVs required")
    # Verify contiguous source order against the saved question manifest before
    # using TOFU's 20-questions-per-author block bootstrap.
    questions_path = run_dir / "evaluation_questions.json"
    blocks = None
    if questions_path.is_file():
        questions = json.loads(questions_path.read_text(encoding="utf-8"))["forget05"]
        before = stages["npo_bf16"].query("split == 'forget05'").sort_values("id")
        if len(questions) == 200 and before.question.tolist() == [r["question"] for r in questions]:
            blocks = {row["question"]: i // 20 for i, row in enumerate(questions)}
    fq, paired = quantization_comparisons(stages, repeats=repeats, author_blocks=blocks)
    fq.to_csv(output_dir / "reanalysis_forget_quality.csv", index=False)
    paired.to_csv(output_dir / "reanalysis_directional_qra.csv", index=False)
    overview, components = [], []
    for name, frame in stages.items():
        summary, parts = summarize(frame, stages["retain95_bf16"])
        overview.append({"stage": name, **summary})
        parts.insert(0, "stage", name)
        components.append(parts)
    pd.DataFrame(overview).to_csv(output_dir / "reanalysis_stage_metrics.csv", index=False)
    pd.concat(components).to_csv(output_dir / "reanalysis_utility_components.csv", index=False)
    termination_rows = []
    for name, frame in stages.items():
        for split, group in frame.groupby("split"):
            sequences = [
                ast.literal_eval(x) if isinstance(x, str) else x for x in group.generated_token_ids
            ]
            termination_rows.append(
                {
                    "stage": name,
                    "split": split,
                    "n": len(group),
                    "mean_generated_tokens": sum(map(len, sequences)) / len(sequences),
                    "legacy_96_token_cap_rate": float(group.generation_hit_limit.mean()),
                    "fraction_containing_source_stop": sum(
                        any(t in (128001, 128008, 128009) for t in seq) for seq in sequences
                    )
                    / len(sequences),
                    "fraction_containing_other_source_stop": sum(
                        any(t in (128001, 128008) for t in seq) for seq in sequences
                    )
                    / len(sequences),
                }
            )
    pd.DataFrame(termination_rows).to_csv(
        output_dir / "generation_termination_audit.csv", index=False
    )
    history = pd.read_csv(run_dir / "training_log.csv")
    plan = json.loads((run_dir / "training_plan.json").read_text())
    receipt = json.loads((run_dir / "npo_model/training_complete.json").read_text())
    unique_retain = len(set(plan["retain_pairs"]))
    verification = {
        "status": "derived-analysis-complete",
        "source_run": str(run_dir),
        "source_run_id": run_dir.name,
        "source_files": inventory,
        "reanalysis_code_sha256": sha256(Path(__file__)),
        "metrics_code_sha256": sha256(Path(__file__).parent / "npo3/metrics.py"),
        "analysis_environment": {
            "python": platform.python_version(),
            "packages": {
                name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy")
            },
        },
        "bootstrap_repeats": repeats,
        "question_bootstrap": "exploratory due to within-author dependence",
        "author_block_bootstrap": "10 contiguous pinned TOFU author blocks"
        if blocks
        else "not available",
        "completed_updates": int(receipt["updates"]),
        "training_log_rows": len(history),
        "unique_retain_rows": unique_retain,
        "available_retain_rows": 3800,
        "training_seconds_from_logged_rows": float(history.seconds.sum()),
        "training_log_timing_limit": "Resume history may omit post-commit optimizer timing fields; not a complete wall-clock estimate",
        "old_interruption_record_preserved": (run_dir / "training_failure.json").is_file(),
        "model_weight_files_available": bool(list((run_dir / "npo_model").glob("*.safetensors"))),
        "model_weight_integrity": "not reverified: supplied archive has receipts, not checkpoint tensors",
        "attack_detected": None,
        "durability_pass": None,
        "missing_study_components": [
            "membership inference with matched nonmembers",
            "FADE",
            "human-coded semantic audit",
            "additional quantizers",
            "relearning",
            "weight/bin analysis",
            "multiple training seeds",
        ],
    }
    write_json(output_dir / "VERIFICATION.json", verification)
    output_hashes = {
        p.name: sha256(p)
        for p in output_dir.iterdir()
        if p.is_file() and p.name != "DERIVED_MANIFEST.json"
    }
    write_json(
        output_dir / "DERIVED_MANIFEST.json",
        {"inputs_sha256": digest(inventory), "outputs": output_hashes},
    )
    print(fq.to_string(index=False))
    print(paired.loc[paired.metric == "source_geometric_truth_ratio_log"].to_string(index=False))
    return verification


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    args = parser.parse_args()
    run(args.run_dir, args.output_dir, args.bootstrap_repeats)
