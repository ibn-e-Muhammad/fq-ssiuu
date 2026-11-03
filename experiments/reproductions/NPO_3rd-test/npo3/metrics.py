"""TOFU arithmetic utility, geometric FQ, and paired descriptive recovery.

Geometric preference is log(p_correct / geometric_mean(p_wrong)); larger means
more preference for the target. KS is evaluated on logs, a monotone transform.
No p-value or interval is turned into an automatic QRA verdict.
"""

import ast
import hashlib
import json
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import ks_2samp


def _json_default(value):
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, (Path, Enum)):
        return value.value if isinstance(value, Enum) else str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported metadata type: {type(value).__name__}")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, default=_json_default).encode()
    ).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=_json_default)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def truth_logs(correct_mean_nll, wrong_mean_nll):
    wrong = np.asarray(wrong_mean_nll, dtype=np.float64)
    if (
        wrong.ndim != 1
        or not len(wrong)
        or not np.isfinite(wrong).all()
        or not np.isfinite(correct_mean_nll)
    ):
        raise ValueError("Finite paraphrased-answer and perturbed-answer NLLs required")
    return {
        "arithmetic_truth_ratio_log": float(
            logsumexp(-wrong) - np.log(len(wrong)) + correct_mean_nll
        ),
        "source_geometric_truth_ratio_log": float(wrong.mean() - correct_mean_nll),
    }


def harmonic(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Complete nonnegative utility components required")
    return 0.0 if (values == 0).any() else float(len(values) / np.reciprocal(values).sum())


def enrich(frame):
    """Recompute definitions from saved primitive NLLs, not old summary fields."""
    frame = frame.copy()
    required = {
        "id",
        "split",
        "question",
        "ground_truth",
        "answer_nll",
        "paraphrase_mean_nll",
        "perturbed_mean_nll",
        "rouge_l_recall",
    }
    if not required.issubset(frame):
        raise ValueError(f"Missing per-question fields: {sorted(required - set(frame))}")
    if frame.duplicated(["split", "id"]).any():
        raise ValueError("Duplicate question IDs")
    computed = []
    for row in frame.to_dict("records"):
        wrong = row["perturbed_mean_nll"]
        if isinstance(wrong, str):
            wrong = ast.literal_eval(wrong)
        logs = truth_logs(row["paraphrase_mean_nll"], wrong)
        mean = float(row["answer_nll"])
        if not np.isfinite(mean) or mean < 0:
            raise ValueError("Nonfinite or negative gold-answer NLL")
        probability = float(np.exp(-mean))
        normalized = float(np.exp(-mean - logsumexp([-mean] + [-float(x) for x in wrong])))
        computed.append(
            {
                **logs,
                "answer_probability": probability,
                "utility_probability": probability
                if row["split"] in ("forget05", "retain")
                else normalized,
                "arithmetic_truth_score": float(
                    -np.expm1(min(0.0, logs["arithmetic_truth_ratio_log"]))
                ),
                "source_geometric_truth_score": float(
                    -np.expm1(min(0.0, -logs["source_geometric_truth_ratio_log"]))
                ),
            }
        )
    for col in computed[0]:
        frame[col] = [row[col] for row in computed]
    return frame


def align(frame, reference, split="forget05"):
    a = frame.loc[frame["split"] == split].set_index("id").sort_index()
    b = reference.loc[reference["split"] == split].set_index("id").sort_index()
    if not len(a) or not a.index.equals(b.index):
        raise ValueError("Matched nonempty question IDs required")
    for col in ("question", "ground_truth"):
        if not a[col].equals(b[col]):
            raise ValueError(f"Mismatched {col}")
    return a, b


def forget_quality(candidate, oracle):
    a, b = align(candidate, oracle)
    x = a["source_geometric_truth_ratio_log"].to_numpy(dtype=float)
    y = b["source_geometric_truth_ratio_log"].to_numpy(dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite geometric preferences")
    test = ks_2samp(x, y, alternative="two-sided", method="auto")
    arithmetic = ks_2samp(
        a["arithmetic_truth_ratio_log"], b["arithmetic_truth_ratio_log"], method="auto"
    )
    return {
        "forget_quality": float(test.pvalue),
        "ks_statistic": float(test.statistic),
        "paper_arithmetic_ks_statistic_diagnostic": float(arithmetic.statistic),
        "paper_arithmetic_ks_p_value_diagnostic": float(arithmetic.pvalue),
        "n": len(a),
        "definition": "tofu-geometric-preference-KS-v1",
        "interpretation": "Non-rejection is not proof of equivalence or erasure",
    }


def summarize(frame, oracle=None):
    frame = enrich(frame)
    parts = (
        frame.loc[frame["split"] != "forget05"]
        .groupby("split", sort=True)[
            ["utility_probability", "rouge_l_recall", "arithmetic_truth_score"]
        ]
        .mean()
    )
    if set(parts.index) != {"retain", "real_authors", "world_facts"}:
        raise ValueError("All nine TOFU utility components required")
    record = {
        "model_utility": harmonic(parts.to_numpy().flatten()),
        "utility_definition": "tofu-arithmetic-nine-components-v1",
    }
    for split, group in frame.groupby("split", sort=True):
        record[f"{split}_n"] = len(group)
        for col in (
            "answer_probability",
            "answer_nll",
            "rouge_l_recall",
            "token_f1",
            "exact_match",
            "source_geometric_truth_ratio_log",
            "generation_hit_limit",
            "eos_terminated",
            "kl_full_ref_to_model_all_positions",
        ):
            if col in group:
                record[f"{split}_{col}"] = (
                    float(group[col].mean()) if group[col].notna().any() else None
                )
    if oracle is not None:
        record.update(forget_quality(frame, enrich(oracle)))
    return record, parts.reset_index()


def paired_bootstrap(before, after, repeats=2000, seed=42, clusters=None):
    a, b = np.asarray(before, dtype=float), np.asarray(after, dtype=float)
    if (
        a.ndim != 1
        or a.shape != b.shape
        or not len(a)
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("Aligned finite nonempty observations required")
    d = b - a
    rng = np.random.default_rng(seed)
    estimates = []
    if clusters is not None:
        clusters = np.asarray(clusters)
        if len(clusters) != len(d):
            raise ValueError("Cluster IDs must align with questions")
        unique = np.unique(clusters)
        blocks = [d[clusters == c] for c in unique]
        for _ in range(repeats):
            sample = rng.integers(0, len(blocks), len(blocks))
            estimates.append(float(np.concatenate([blocks[i] for i in sample]).mean()))
        unit = "author-block"
    else:
        for _ in range(repeats):
            estimates.append(float(d[rng.integers(0, len(d), len(d))].mean()))
        unit = "question-exploratory"
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "mean_change": float(d.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "fraction_increased": float((d > 0).mean()),
        "fraction_decreased": float((d < 0).mean()),
        "resampling_unit": unit,
        "bootstrap_repeats": repeats,
        "n": len(d),
    }


def quantization_comparisons(stages, repeats=2000, seed=42, author_blocks=None):
    """Both fixed-BF16 and same-precision oracles; directional paired statistics."""
    stages = {name: enrich(frame) for name, frame in stages.items()}
    fq_rows, changes = [], []
    for name, frame in stages.items():
        precision = name.rsplit("_", 1)[-1]
        for policy, oracle_name in (
            ("fixed_bf16", "retain95_bf16"),
            ("matched_precision", f"retain95_{precision}"),
        ):
            if oracle_name in stages:
                fq_rows.append(
                    {
                        "stage": name,
                        "oracle_policy": policy,
                        "oracle_stage": oracle_name,
                        **forget_quality(frame, stages[oracle_name]),
                    }
                )
    if "npo_bf16" in stages:
        for mode in ("int8", "nf4"):
            name = f"npo_{mode}"
            if name not in stages:
                continue
            after, before = align(stages[name], stages["npo_bf16"])
            clusters = (
                [author_blocks[q] for q in before.question] if author_blocks is not None else None
            )
            for metric in (
                "source_geometric_truth_ratio_log",
                "answer_probability",
                "rouge_l_recall",
                "token_f1",
                "exact_match",
            ):
                for grouping in [None, clusters] if clusters is not None else [None]:
                    result = paired_bootstrap(
                        before[metric], after[metric], repeats, seed, grouping
                    )
                    row = {
                        "quantization": mode,
                        "metric": metric,
                        **result,
                        "before": float(before[metric].mean()),
                        "after": float(after[metric].mean()),
                    }
                    for control in ("full", "retain95"):
                        base, quant = f"{control}_bf16", f"{control}_{mode}"
                        if base in stages and quant in stages:
                            c1, c0 = align(stages[quant], stages[base])
                            if not c1.index.equals(before.index):
                                raise ValueError("Control question IDs differ")
                            delta = float((c1[metric] - c0[metric]).mean())
                            row[f"{control}_change"] = delta
                            row[f"npo_minus_{control}_change"] = row["mean_change"] - delta
                    changes.append(row)
    return pd.DataFrame(fq_rows), pd.DataFrame(changes)


def load_existing_stages(run_dir, require_complete=True):
    run_dir = Path(run_dir)
    stages, inventory = {}, []
    for folder in sorted(run_dir.iterdir()):
        meta_path, csv_path = folder / "metadata.json", folder / "per_question.csv"
        if not folder.is_dir() or not meta_path.is_file() or not csv_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if require_complete and meta.get("status") != "complete":
            raise ValueError(f"Incomplete stage: {folder.name}")
        for filename, expected in meta.get("artifact_hashes", {}).items():
            path = folder / filename
            if not path.is_file() or sha256(path) != expected:
                raise ValueError(f"Evidence hash mismatch: {path}")
            inventory.append({"path": str(path), "sha256": expected, "bytes": path.stat().st_size})
        stages[folder.name] = enrich(pd.read_csv(csv_path))
    if not stages:
        raise ValueError("No completed per-question CSVs found")
    return stages, inventory
