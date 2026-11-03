"""Complete numeric/answer exports; missing studies stay explicitly unmeasured."""

import html
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .metrics import quantization_comparisons, summarize, write_json


def export_seed_report(folder, stages, config, author_blocks):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    baseline = stages["retain95_bf16"]
    overview, components = [], []
    for name, frame in stages.items():
        summary, parts = summarize(frame, baseline)
        overview.append({"stage": name, **summary})
        parts.insert(0, "stage", name)
        components.append(parts)
    matrix = pd.DataFrame(overview)
    matrix.to_csv(folder / "comparison_matrix.csv", index=False)
    pd.concat(components).to_csv(folder / "utility_components.csv", index=False)
    fq, directional = quantization_comparisons(
        stages, config.bootstrap_repeats, author_blocks=author_blocks
    )
    fq.to_csv(folder / "forget_quality_fixed_and_matched.csv", index=False)
    directional.to_csv(folder / "paired_directional_recovery.csv", index=False)
    frames = []
    for name, frame in stages.items():
        frame = frame.copy()
        frame["stage"] = name
        frames.append(frame)
    long = pd.concat(frames, ignore_index=True)
    long.to_csv(folder / "all_per_question.csv", index=False)
    answers = long.pivot(
        index=["split", "id", "question", "ground_truth"],
        columns="stage",
        values="generated_answer",
    ).reset_index()
    answers.to_csv(folder / "answers_side_by_side.csv", index=False)
    style = "<meta charset='utf-8'><style>body{font:14px system-ui;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:8px;vertical-align:top;white-space:pre-wrap;min-width:150px}th{background:#eef;position:sticky;top:0}</style>"
    (folder / "answers_side_by_side.html").write_text(
        style + answers.to_html(index=False, escape=True), encoding="utf-8"
    )
    if "npo_adapter" in stages:
        before = stages["npo_adapter"].set_index("id").sort_index()
        after = stages["npo_bf16"].set_index("id").sort_index()
        if not before.index.equals(after.index):
            raise ValueError("Adapter/merged evaluation rows differ")
        merge = []
        for split in before.split.unique():
            ids = before.index[before.split == split]
            for metric in (
                "answer_probability",
                "answer_nll",
                "rouge_l_recall",
                "source_geometric_truth_ratio_log",
            ):
                merge.append(
                    {
                        "split": split,
                        "metric": metric,
                        "merged_minus_adapter": float(
                            (after.loc[ids, metric] - before.loc[ids, metric]).mean()
                        ),
                    }
                )
        pd.DataFrame(merge).to_csv(folder / "merge_drift.csv", index=False)
    # Labels are deliberately empty. A human must make semantic judgments.
    annotations = long[
        [
            "stage",
            "split",
            "id",
            "question",
            "ground_truth",
            "generated_answer",
            "generation_hit_limit",
        ]
    ].copy()
    annotations["category"] = ""
    annotations["reviewer"] = ""
    annotations["notes"] = ""
    annotation_path = folder / "human_audit_template.csv"
    if not annotation_path.exists():
        annotations.to_csv(annotation_path, index=False)
    status = {
        "core_evaluation": "measured",
        "geometric_FQ_fixed_and_matched": "measured",
        "arithmetic_model_utility": "measured",
        "directional_truth_ratio_bootstrap": "measured",
        "KL_diagnostic": "measured" if config.evaluate_kl else "not requested",
        "quantization_modes_requested": config.quantization_modes,
        "human_semantic_audit": "awaiting manual labels; no rates inferred from ROUGE",
        "membership_inference": "not measured; answer-token/min-k scores alone are not an MIA without matched nonmembers",
        "FADE": "not measured",
        "second_quantizer_family": "not measured",
        "relearning": "not measured",
        "weight_quantization_bin_analysis": "not measured",
        "attack_detected": None,
        "durability_pass": None,
    }
    write_json(folder / "evaluation_coverage.json", status)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    indexed = matrix.set_index("stage")
    for ax, column in zip(
        axes, ("forget05_answer_probability", "forget05_rouge_l_recall", "model_utility")
    ):
        indexed[column].plot.bar(ax=ax)
        ax.set_title(column)
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(folder / "comparison.png", dpi=160)
    plt.close(fig)
    report = style + "<h1>LoRA NPO+RT / merged-model quantization</h1>"
    report += "<p>Source-style sampled decoding is primary. Inspect adapter-to-merge drift before quantization changes. KS non-rejection does not prove erasure.</p>"
    for title, frame in (
        ("Stage metrics", matrix),
        ("Fixed and matched geometric FQ", fq),
        ("Paired quantization changes", directional),
    ):
        report += f"<h2>{html.escape(title)}</h2>" + frame.round(6).to_html(
            index=False, escape=True
        )
    report += "<p><a href='answers_side_by_side.html'>All model-generated answers</a></p><img src='comparison.png' style='max-width:100%'>"
    report += "<h2>Study coverage</h2>" + pd.Series(status).to_frame("status").to_html(escape=True)
    (folder / "report.html").write_text(report, encoding="utf-8")
    return matrix


def export_cross_seed(root, matrices, expected_seeds):
    frames = []
    for seed, frame in matrices.items():
        frame = frame.copy()
        frame.insert(0, "seed", seed)
        frames.append(frame)
    values = pd.concat(frames, ignore_index=True)
    values.to_csv(Path(root) / "all_seeds_metrics.csv", index=False)
    numeric = [c for c in values.select_dtypes("number").columns if c != "seed"]
    aggregate = values.groupby("stage")[numeric].agg(["mean", "std", "count"])
    aggregate.to_csv(Path(root) / "cross_seed_mean_std.csv")
    write_json(
        Path(root) / "seed_coverage.json",
        {
            "expected_seeds": expected_seeds,
            "completed_seeds": list(matrices),
            "complete": set(matrices) == set(expected_seeds),
            "minimum_three_seeds_met": len(matrices) >= 3,
            "interpretation": "Seed means/spread are descriptive; fewer than three seeds is a pilot and bootstrap rows are not independent training runs",
        },
    )
    return aggregate
