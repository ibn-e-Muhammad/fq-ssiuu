"""D1 source-sized FP16 evaluation. No project QRA decision rules."""

import json
from pathlib import Path

import torch
import torch.nn.functional as functional

from experiments.reproductions.common import digest, file_digest, tensors_digest
from experiments.reproductions.NPO.adapter import encode_example
from experiments.reproductions.NPO.candidate import causal_sequence_nll
from experiments.reproductions.NPO.source_metrics import source_forget_quality, source_summary
from experiments.reproductions.NPO.training_data import fixed_batch
from experiments.reproductions.QRA.metrics import forget_quality, text_metrics, truth_ratios

TASKS = {
    "eval_log.json": ("retain_perturbed", "paraphrased_answer"),
    "eval_real_author_wo_options.json": ("real_authors_perturbed", "answer"),
    "eval_real_world_wo_options.json": ("world_facts_perturbed", "answer"),
    "eval_log_forget.json": ("forget05_perturbed", "paraphrased_answer"),
}


def evaluator_hash():
    directory = Path(__file__).parent
    files = [
        directory / name
        for name in (
            "evaluation_runtime.py",
            "source_metrics.py",
            "adapter.py",
            "candidate.py",
            "training_data.py",
        )
    ]
    files.extend((directory.parent / "common.py", directory.parent / "QRA/metrics.py"))
    return digest({str(path.relative_to(directory.parent)): file_digest(path) for path in files})


def write_new(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def source_kl(current, reference, positions_per_chunk=8):
    """Author diagnostic: KL(ref || current), averaged across ALL 200 positions.

    Both logits are copied to CPU before calculation; chunking avoids a second
    full vocabulary-sized FP32 allocation on the training GPU. This includes
    prompt/padding positions intentionally, exactly as the source diagnostic.
    """
    if (
        current.shape != reference.shape
        or current.ndim != 3
        or not all(current.shape)
        or type(positions_per_chunk) is not int
        or positions_per_chunk < 1
    ):
        raise ValueError("Aligned nonempty source KL logits required")
    if current.device.type != "cpu" or reference.device.type != "cpu":
        raise ValueError("Source KL computation requires CPU copies")
    sums = torch.zeros(current.shape[0], dtype=torch.float64)
    for start in range(0, current.shape[1], positions_per_chunk):
        a = current[:, start : start + positions_per_chunk].float().log_softmax(-1)
        b = reference[:, start : start + positions_per_chunk].float().log_softmax(-1)
        values = functional.kl_div(a, b, reduction="none", log_target=True).sum(-1)
        sums += values.double().sum(-1)
    result = sums / current.shape[1]
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite source KL")
    return result.tolist()


def score_examples(
    model, examples, tokenizer, spec, batch_size, *, reference=None, device="cuda:0", sink=None
):
    records = []
    if model.training or (reference is not None and reference.training):
        raise ValueError("Source evaluation requires eval mode")
    if not examples or type(batch_size) is not int or batch_size < 1:
        raise ValueError("Nonempty examples and positive integer batch size required")
    for offset in range(0, len(examples), batch_size):
        rows = examples[offset : offset + batch_size]
        batch = fixed_batch(rows, max_length=spec.max_length, pad_token_id=tokenizer.pad_token_id)
        tensors = {k: v.to(device) for k, v in batch["tensors"].items()}
        with torch.no_grad():
            logits = model(
                input_ids=tensors["input_ids"], attention_mask=tensors["attention_mask"]
            ).logits
            nll = causal_sequence_nll(logits, tensors["labels"]).cpu()
            cpu_logits = logits.cpu() if reference is not None else None
        del logits
        if not torch.isfinite(nll).all():
            raise ValueError("Nonfinite evaluation NLL")
        kl = [None] * len(rows)
        if reference is not None:
            with torch.no_grad():
                oracle_logits = reference(
                    input_ids=tensors["input_ids"], attention_mask=tensors["attention_mask"]
                ).logits.cpu()
            kl = source_kl(cpu_logits, oracle_logits)
            del cpu_logits, oracle_logits
        if tensors_digest(tensors) != batch["tensors_sha256"]:
            raise ValueError("Evaluation model mutated the input tensors")
        completed = []
        for i, example in enumerate(rows):
            count = int((tensors["labels"][i, 1:] != -100).sum())
            completed.append(
                {
                    "sample_id": example["sample_id"],
                    "source_record_sha256": example["source_record_sha256"],
                    "preprocessing_sha256": example["preprocessing_sha256"],
                    "tensors_sha256": example["tensors_sha256"],
                    "batch_tensors_sha256": batch["tensors_sha256"],
                    "input_ids": example["input_ids"].tolist(),
                    "attention_mask": example["attention_mask"].tolist(),
                    "labels": example["labels"].tolist(),
                    "sequence_nll": float(nll[i]),
                    "scored_tokens": count,
                    "mean_nll": float(nll[i]) / count,
                    "kl_reference_to_candidate_all_positions": kl[i],
                }
            )
        if sink is not None:
            sink(completed)
        records.extend(completed)
        del tensors
    return records


def generate_source_batch(model, tokenizer, examples, source_generation, *, device="cuda:0"):
    from transformers import GenerationConfig

    if model.training or not examples:
        raise ValueError("Generation requires eval mode and nonempty examples")
    # Check the same answer/prompt boundary used for teacher-forced scoring.
    fixed_batch(examples, max_length=200, pad_token_id=tokenizer.pad_token_id)
    prompts = [e["input_ids"][: e["prompt_token_count"]] for e in examples]
    length = max(map(len, prompts))
    if length >= 200:
        raise ValueError("Source generation limit leaves no answer space")
    ids, masks = [], []
    for prompt in prompts:
        padding = length - len(prompt)
        ids.append(
            torch.cat((torch.full((padding,), tokenizer.pad_token_id, dtype=torch.long), prompt))
        )
        masks.append(
            torch.cat(
                (torch.zeros(padding, dtype=torch.long), torch.ones(len(prompt), dtype=torch.long))
            )
        )
    ids, masks = torch.stack(ids).to(device), torch.stack(masks).to(device)
    # D1 uses source sampling, not C1's project greedy diagnostic.
    settings = {
        **source_generation,
        "max_length": 200,
        "max_new_tokens": None,
        "do_sample": True,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
    }
    config = GenerationConfig(**settings)
    target = getattr(model, "module", model)
    previous = target.generation_config
    try:
        target.generation_config = config
        torch.manual_seed(0)  # The author resets this once for each generation batch.
        with torch.no_grad():
            output = target.generate(
                input_ids=ids, attention_mask=masks, generation_config=config, synced_gpus=False
            )
    finally:
        target.generation_config = previous
    if not torch.equal(output[:, :length], ids):
        raise ValueError("Generated sequence lost source prompt prefix")
    specification = {
        "version": "npo-author-sampled-llama3-v1",
        "seed_per_batch": 0,
        "generation_config": config.to_dict(),
        "batch_size": len(examples),
        "padding": "left",
        "cleanup_tokenization_spaces": tokenizer.clean_up_tokenization_spaces,
    }
    records = []
    for i, example in enumerate(examples):
        tokens = output[i, length:].cpu().tolist()
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        expected = tokenizer.decode(
            example["input_ids"][example["prompt_token_count"] :], skip_special_tokens=True
        )
        records.append(
            {
                "sample_id": example["sample_id"],
                "prompt_token_ids": prompts[i].tolist(),
                "generated_token_ids": tokens,
                "generated_text": text,
                "decoded_target": expected,
                "specification": specification,
                "decoding_sha256": digest(specification),
                **text_metrics(expected, text),
            }
        )
    return records


def encoded_variants(rows, split, tokenizer, spec, base_key):
    correct, base, perturbed = [], [], []
    count = None
    for index, row in enumerate(rows):
        wrong = row["perturbed_answer"]
        if not isinstance(wrong, list) or not wrong:
            raise ValueError("Source perturbations required")
        count = len(wrong) if count is None else count
        if len(wrong) != count:
            raise ValueError("Author expects a consistent option count within a split")
        correct.append(encode_example(row, index, split, tokenizer, spec))
        base.append(encode_example({**row, "answer": row[base_key]}, index, split, tokenizer, spec))
        perturbed.extend(
            encode_example(
                {**row, "answer": answer}, index, split + f"/perturb-{option}", tokenizer, spec
            )
            for option, answer in enumerate(wrong)
        )
    return correct, base, perturbed, count


def verified_rows(dataset_dir, split, registry):
    """Read only hash-pinned dataset bytes, in source order, without shuffling."""
    name = split + ".json"
    entries = [
        f
        for f in registry["files"]
        if f["repo"] == registry["dataset"]["id"]
        and f["revision"] == registry["dataset"]["revision"]
        and f["path"] == name
    ]
    path = Path(dataset_dir) / name
    if len(entries) != 1 or file_digest(path) != entries[0]["sha256"]:
        raise ValueError("Evaluation dataset differs from pinned registry")
    if path.stat().st_size != entries[0]["bytes"]:
        raise ValueError("Evaluation dataset size mismatch")
    return [json.loads(line) for line in path.read_text().splitlines()][:300]


def batch_sink(path):
    """Append completed batches; a failed evaluation never discards prior rows."""
    handle = Path(path).open("x")

    def write(records):
        for record in records:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        handle.flush()

    return handle, write


def score_to_file(path, *args, **kwargs):
    handle, sink = batch_sink(path)
    try:
        return score_examples(*args, **kwargs, sink=sink)
    finally:
        handle.close()


def oracle_binding(spec, registry, base, wrong):
    return {
        "oracle": registry["retain_only_oracle"],
        "dataset": registry["dataset"],
        "preprocessing_sha256": spec.sha256,
        "ordered_sample_ids": [x["sample_id"] for x in base],
        "base_tensors_sha256": [x["tensors_sha256"] for x in base],
        "perturbation_tensors_sha256": [x["tensors_sha256"] for x in wrong],
        "evaluator_sha256": evaluator_hash(),
    }


def evaluate_oracle(model, tokenizer, spec, dataset_dir, registry, output, *, device="cuda:0"):
    """Compute the fixed FP16 retain-only oracle once, before training."""
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Retain-only oracle must be frozen and in eval mode")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = verified_rows(dataset_dir, "forget05_perturbed", registry)
    _, base, wrong, options = encoded_variants(
        rows, "forget05_perturbed", tokenizer, spec, "paraphrased_answer"
    )
    base_scores = score_to_file(
        output / "base.jsonl", model, base, tokenizer, spec, 7, device=device
    )
    wrong_scores = score_to_file(
        output / "perturbations.jsonl", model, wrong, tokenizer, spec, 7 * options, device=device
    )
    record = {
        "binding": oracle_binding(spec, registry, base, wrong),
        "base": base_scores,
        "perturbations": wrong_scores,
        "options_per_example": options,
    }
    write_new(output / "ORACLE.json", {**record, "record_sha256": digest(record)})
    return output / "ORACLE.json"


def load_oracle(path, expected_file_sha256, binding, options):
    if file_digest(path) != expected_file_sha256:
        raise ValueError("Oracle evidence bytes changed")
    raw = json.loads(Path(path).read_text())
    recorded = raw.pop("record_sha256")
    if digest(raw) != recorded or raw["binding"] != binding:
        raise ValueError("Oracle content or source/sample/preprocessing identity mismatch")
    if raw["options_per_example"] != options:
        raise ValueError("Oracle perturbation count differs")
    for key, hashes in (
        ("base", binding["base_tensors_sha256"]),
        ("perturbations", binding["perturbation_tensors_sha256"]),
    ):
        if [x["tensors_sha256"] for x in raw[key]] != hashes:
            raise ValueError("Oracle records do not match bound tensors")
        for row in raw[key]:
            tensors = {k: torch.tensor(row[k]) for k in ("input_ids", "attention_mask", "labels")}
            if tensors_digest(tensors) != row["tensors_sha256"]:
                raise ValueError("Oracle tensors changed")
            count = int((tensors["labels"][1:] != -100).sum())
            if count != row["scored_tokens"] or count < 1:
                raise ValueError("Oracle scored-token count mismatch")
            if row["mean_nll"] != row["sequence_nll"] / count:
                raise ValueError("Oracle derived mean differs from sequence NLL")
    if [x["sample_id"] for x in raw["base"]] != binding["ordered_sample_ids"]:
        raise ValueError("Oracle sample order changed")
    return {
        "avg_paraphrased_loss": [x["mean_nll"] for x in raw["base"]],
        "average_perturb_loss": [
            [x["mean_nll"] for x in raw["perturbations"][i : i + options]]
            for i in range(0, len(raw["perturbations"]), options)
        ],
    }


def evaluate_source(
    model,
    reference,
    tokenizer,
    spec,
    dataset_dir,
    source_generation,
    output,
    provenance,
    oracle_path,
    oracle_sha256,
    registry,
    *,
    device="cuda:0",
):
    """Source's first min(300,n) rows, batch 30 / base-and-options batch 7."""
    if spec.max_length != 200 or spec.padding_side != "right":
        raise ValueError("Source evaluation adapter must use length 200 and right padding")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model.eval()
    logs, raw_hashes = {}, {}
    for task, (split, base_key) in TASKS.items():
        rows = verified_rows(dataset_dir, split, registry)
        correct, base, wrong, options = encoded_variants(rows, split, tokenizer, spec, base_key)
        if split == "forget05_perturbed":
            oracle_scores = load_oracle(
                oracle_path, oracle_sha256, oracle_binding(spec, registry, base, wrong), options
            )
        primary = score_to_file(
            output / (split + "_primary.jsonl"),
            model,
            correct,
            tokenizer,
            spec,
            30,
            reference=reference,
            device=device,
        )
        generated = []
        handle, sink = batch_sink(output / (split + "_generation.jsonl"))
        try:
            for offset in range(0, len(rows), 30):
                batch_records = generate_source_batch(
                    model,
                    tokenizer,
                    correct[offset : offset + 30],
                    source_generation,
                    device=device,
                )
                sink(batch_records)
                generated.extend(batch_records)
        finally:
            handle.close()
        base_scores = score_to_file(
            output / (split + "_base.jsonl"), model, base, tokenizer, spec, 7, device=device
        )
        # The source flattens seven rows x options into the model forward.
        wrong_scores = score_to_file(
            output / (split + "_perturbations.jsonl"),
            model,
            wrong,
            tokenizer,
            spec,
            7 * options,
            device=device,
        )
        log = {
            "avg_gt_loss": [x["mean_nll"] for x in primary],
            "rougeL_recall": [x["rouge_l_recall"] for x in generated],
            "avg_paraphrased_loss": [x["mean_nll"] for x in base_scores],
            "average_perturb_loss": [
                [x["mean_nll"] for x in wrong_scores[i : i + options]]
                for i in range(0, len(wrong_scores), options)
            ],
            "kl_divergence": [x["kl_reference_to_candidate_all_positions"] for x in primary],
        }
        raw = {
            "provenance": provenance,
            "split": split,
            "selection": "source-first-min-300",
            "primary": primary,
            "generation": generated,
            "base": base_scores,
            "perturbations": wrong_scores,
            "source_logs": log,
            "evaluator_sha256": evaluator_hash(),
        }
        write_new(output / task, raw)
        raw_hashes[task] = file_digest(output / task)
        logs[task] = log
    source = source_summary(logs)
    source["forget_quality_diagnostic"] = source_forget_quality(
        logs["eval_log_forget.json"], oracle_scores
    )

    # Researcher-selected paper arithmetic diagnostic stays canonical and named
    # separately; source-author geometric summaries remain reproduction outputs.
    def paper_ratios(values):
        return [
            truth_ratios(a, b)["tofu_truth_ratio"]
            for a, b in zip(
                values["avg_paraphrased_loss"], values["average_perturb_loss"], strict=True
            )
        ]

    source["tofu_paper_arithmetic_diagnostic"] = forget_quality(
        paper_ratios(logs["eval_log_forget.json"]), paper_ratios(oracle_scores)
    )
    source["raw_file_sha256"] = raw_hashes
    source["provenance"] = provenance
    source["oracle_file_sha256"] = oracle_sha256
    write_new(output / "SUMMARY.json", source)
    return source
