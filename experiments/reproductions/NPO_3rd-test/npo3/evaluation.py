"""Measured likelihoods, source-style generation, long-output audit and provenance."""

import gc
import importlib.metadata
import json
import math
import random
import re
import time
import unicodedata
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn.functional as functional
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig

from .data import one_batch, prompt_ids
from .metrics import digest, enrich, sha256, summarize, write_json
from .objective import sequence_nll


def release_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def dtype_for(config):
    return getattr(torch, config.dtype)


def model_loader(path, config, quantization=None):
    major = int(importlib.metadata.version("transformers").split(".")[0])
    kwargs = {
        "local_files_only": True,
        "trust_remote_code": False,
        "use_safetensors": True,
        "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa",
        ("dtype" if major >= 5 else "torch_dtype"): dtype_for(config),
    }
    if quantization:
        if not str(config.device).startswith("cuda"):
            raise ValueError("This experiment's bitsandbytes path requires CUDA")
        if quantization == "int8":
            qc = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
        elif quantization == "nf4":
            qc = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype_for(config),
                bnb_4bit_use_double_quant=True,
            )
        else:
            raise ValueError(quantization)
        kwargs.update(quantization_config=qc, device_map={"": str(config.device)})
    model = AutoModelForCausalLM.from_pretrained(str(path), **kwargs)
    if quantization is None:
        model.to(config.device)
    model.requires_grad_(False).eval()
    if quantization:
        import bitsandbytes as bnb

        kind = bnb.nn.Linear8bitLt if quantization == "int8" else bnb.nn.Linear4bit
        count = sum(isinstance(m, kind) for m in model.modules())
        loaded = getattr(
            model, "is_loaded_in_8bit" if quantization == "int8" else "is_loaded_in_4bit", False
        )
        if not count or not loaded:
            raise ValueError("Actual quantization was not established")
    return model


def tensor_fingerprint(model, skip_adapters=False):
    """Hash packed weights and quantization statistics without a whole CPU copy."""
    records = []
    for name, tensor in sorted(model.state_dict().items()):
        if skip_adapters and "lora_" in name:
            continue
        if not isinstance(tensor, torch.Tensor):
            records.append({"name": name, "metadata": str(tensor)})
            continue
        h = __import__("hashlib").sha256()
        flat = tensor.detach().reshape(-1)
        for start in range(0, flat.numel(), 1_048_576):
            raw = (
                flat[start : start + 1_048_576]
                .cpu()
                .contiguous()
                .view(torch.uint8)
                .numpy()
                .tobytes()
            )
            h.update(raw)
        records.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256": h.hexdigest(),
            }
        )
    return {"sha256": digest(records), "tensors": records}


def model_metadata(model, quantization):
    qc = getattr(model.config, "quantization_config", None)
    if qc is not None and hasattr(qc, "to_dict"):
        qc = qc.to_dict()
    modules = []
    if quantization:
        import bitsandbytes as bnb

        kind = bnb.nn.Linear8bitLt if quantization == "int8" else bnb.nn.Linear4bit
        modules = [name for name, layer in model.named_modules() if isinstance(layer, kind)]
    return {
        "quantization": quantization,
        "quantization_config": qc,
        "quantized_linear_modules": modules,
        "quantized_module_count": len(modules),
        "model_memory_bytes": model.get_memory_footprint(),
        "packed_state": tensor_fingerprint(model),
    }


@contextmanager
def preserve_rng():
    cpu, cuda = (
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )
    python, numpy = random.getstate(), np.random.get_state()
    try:
        yield
    finally:
        random.setstate(python)
        np.random.set_state(numpy)
        torch.set_rng_state(cpu)
        if cuda:
            torch.cuda.set_rng_state_all(cuda)


def text_scores(gold, generated):
    def normalize(text):
        return " ".join(unicodedata.normalize("NFKC", text).casefold().split())

    a, b = re.findall(r"\w+", normalize(gold)), re.findall(r"\w+", normalize(generated))
    common = sum((Counter(a) & Counter(b)).values())
    return {
        "rouge_l_recall": float(
            rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            .score(gold, generated)["rougeL"]
            .recall
        ),
        "token_f1": 2 * common / (len(a) + len(b)) if a and b else 0.0,
        "exact_match": float(normalize(gold) == normalize(generated)),
    }


@torch.no_grad()
def score_example(model, example, tokenizer, config, reference=None):
    inputs, labels = one_batch(
        example, config.device, tokenizer.pad_token_id, config.evaluation_max_length
    )
    logits = model(**inputs, use_cache=False).logits
    nll = sequence_nll(logits, labels)
    mask = labels[:, 1:] != -100
    targets = labels[:, 1:][mask].unsqueeze(1)
    tokens = (
        functional.log_softmax(logits[:, :-1][mask].float(), dim=-1).gather(1, targets).flatten()
    )
    values = tokens.cpu().double().tolist()
    kl = None
    if reference is model:
        kl = 0.0
    elif reference is not None:
        ref_logits = reference(**inputs, use_cache=False).logits
        total = 0.0
        for start in range(0, logits.shape[1], 8):
            current = logits[:, start : start + 8].float().log_softmax(-1)
            frozen = ref_logits[:, start : start + 8].float().log_softmax(-1)
            total += float(
                functional.kl_div(current, frozen, log_target=True, reduction="none")
                .sum(-1)
                .double()
                .sum()
            )
        kl = total / logits.shape[1]
        del ref_logits
    count, total = int(mask.sum()), float(nll.item())
    if not math.isfinite(total) or (kl is not None and not math.isfinite(kl)):
        raise FloatingPointError("Nonfinite evaluation score")
    k = max(1, math.ceil(len(values) * 0.2))
    return {
        "sequence_nll": total,
        "scored_tokens": count,
        "answer_nll": total / count,
        "token_log_probabilities": values,
        "min_k20_log_probability": float(np.mean(sorted(values)[:k])),
        "kl_full_ref_to_model_all_positions": kl,
    }


@torch.no_grad()
def generate_batch(model, rows, tokenizer, config, profile):
    prompts = [prompt_ids(tokenizer, p["question"], config) for p in rows]
    width = max(map(len, prompts))
    ids = torch.full(
        (len(rows), width), tokenizer.pad_token_id, dtype=torch.long, device=config.device
    )
    masks = torch.zeros_like(ids)
    for i, prompt in enumerate(prompts):
        ids[i, -len(prompt) :] = torch.tensor(prompt, device=config.device)
        masks[i, -len(prompt) :] = 1
    eos = {tokenizer.eos_token_id}
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot, int) and eot != tokenizer.unk_token_id:
        eos.add(eot)
    if profile != "legacy_greedy96":
        eos.update(config.generation_eos_ids)
    kwargs = dict(
        num_beams=1,
        eos_token_id=sorted(eos),
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        use_cache=True,
        repetition_penalty=1.0,
    )
    if profile == "source_sampled":
        if width >= config.evaluation_max_length:
            raise ValueError("Source generation limit leaves no answer space")
        kwargs.update(
            do_sample=True,
            temperature=config.source_temperature,
            top_p=config.source_top_p,
            max_length=config.evaluation_max_length,
        )
        available = config.evaluation_max_length - width
    else:
        available = (
            config.legacy_greedy_tokens
            if profile == "legacy_greedy96"
            else config.long_audit_tokens
        )
        kwargs.update(do_sample=False, max_new_tokens=available)
    generation = GenerationConfig(**kwargs)
    old = model.generation_config
    try:
        model.generation_config = generation
        torch.manual_seed(0)  # Author's generation seed per batch, isolated from training RNG.
        outputs = model.generate(input_ids=ids, attention_mask=masks, generation_config=generation)
    finally:
        model.generation_config = old
    if not torch.equal(outputs[:, :width], ids):
        raise ValueError("Generation lost the prompt prefix")
    result = []
    for output in outputs[:, width:].cpu().tolist():
        end = next((i for i, token in enumerate(output) if token in eos), None)
        tokens = output[: end + 1] if end is not None else output
        text = tokenizer.decode(
            tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        result.append(
            {
                "generated_answer": text,
                "generated_token_ids": tokens,
                "generated_token_count": len(tokens),
                "eos_terminated": end is not None,
                "generation_hit_limit": end is None and len(tokens) >= available,
                "generation_profile": profile,
                "generation_prompt_width": width,
                "generation_available_tokens": available,
                "generation_eos_token_ids": sorted(eos),
            }
        )
    return result


def _write_frame(folder, frame, oracle=None):
    frame.to_csv(folder / "per_question.csv", index=False, encoding="utf-8-sig")
    matched_oracle = oracle.loc[oracle.id.isin(frame.id)] if oracle is not None else None
    summary, components = summarize(frame, matched_oracle)
    write_json(folder / "summary.json", summary)
    components.to_csv(folder / "utility_components.csv", index=False)
    columns = [
        c
        for c in (
            "answer_nll",
            "answer_probability",
            "rouge_l_recall",
            "token_f1",
            "exact_match",
            "source_geometric_truth_ratio_log",
            "eos_terminated",
            "generation_hit_limit",
            "kl_full_ref_to_model_all_positions",
        )
        if c in frame and frame[c].notna().any()
    ]
    frame.groupby("split")[columns].mean().to_csv(folder / "metrics.csv")
    return summary


def evaluate(
    model,
    tokenizer,
    packs,
    config,
    folder,
    binding,
    reference=None,
    oracle=None,
    profiles=("source_sampled",),
    shared_likelihoods=None,
):
    """Reuse only complete identity-bound stages; partial files remain intact."""
    folder = Path(folder)
    identity = digest(
        {
            "binding": binding,
            "packs": packs,
            "profiles": list(profiles),
            "generation": {
                k: getattr(config, k)
                for k in (
                    "generation_batch_size",
                    "source_temperature",
                    "source_top_p",
                    "evaluation_max_length",
                    "legacy_greedy_tokens",
                    "long_audit_tokens",
                    "long_audit_per_split",
                )
            },
            "evaluator_sha256": sha256(Path(__file__)),
            "metrics_sha256": sha256(Path(__file__).with_name("metrics.py")),
        }
    )
    marker = folder / "metadata.json"
    if marker.is_file():
        old = json.loads(marker.read_text())
        if old.get("status") == "complete":
            if old["identity"] != identity:
                raise ValueError(f"Cached evaluation identity differs: {folder}")
            for name, expected in old["artifact_hashes"].items():
                if sha256(folder / name) != expected:
                    raise ValueError(f"Cached evaluation bytes changed: {folder / name}")
            return pd.read_json(folder / "per_question.jsonl", lines=True)
    if folder.exists():
        folder.rename(folder.with_name(folder.name + f"_incomplete_{time.time_ns()}"))
    folder.mkdir(parents=True)
    previous_mode = model.training
    model.eval()
    started = time.perf_counter()
    metadata = {
        "status": "running",
        "identity": identity,
        "binding": binding,
        "profiles": list(profiles),
        "likelihood_source": "shared" if shared_likelihoods else "measured",
    }
    if str(config.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(config.device)
        free, total = torch.cuda.mem_get_info(config.device)
        metadata.update(device_free_bytes_before=free, device_total_bytes=total)
    write_json(marker, metadata)
    primary = []
    try:
        with preserve_rng():
            scores = shared_likelihoods if shared_likelihoods is not None else {}
            if shared_likelihoods is None:
                with (folder / "likelihoods.jsonl").open("x", encoding="utf-8") as sink:
                    for split, examples in packs.items():
                        print(f"Scoring {folder.name}: {split} ({len(examples)})", flush=True)
                        for p in examples:
                            gold = score_example(model, p["gold"], tokenizer, config, reference)
                            base = (
                                gold["answer_nll"]
                                if p["base"] == p["gold"]
                                else score_example(model, p["base"], tokenizer, config)[
                                    "answer_nll"
                                ]
                            )
                            wrong = [
                                score_example(model, e, tokenizer, config)["answer_nll"]
                                for e in p["wrong"]
                            ]
                            record = {
                                **{
                                    k: p[k]
                                    for k in (
                                        "id",
                                        "split",
                                        "source_index",
                                        "question",
                                        "ground_truth",
                                    )
                                },
                                **gold,
                                "paraphrase_mean_nll": base,
                                "perturbed_mean_nll": wrong,
                            }
                            if "author_block" in p:
                                record["author_block"] = p["author_block"]
                            scores[p["id"]] = record
                            sink.write(
                                json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                            )
                            sink.flush()
            for profile in profiles:
                destination = folder if profile == "source_sampled" else folder / profile
                destination.mkdir(exist_ok=True)
                rows = []
                with (destination / "per_question.jsonl").open("x", encoding="utf-8") as sink:
                    for split, examples in packs.items():
                        selected = examples
                        if profile == "long_greedy":
                            rng = np.random.default_rng(config.evaluation_seed)
                            ix = sorted(
                                rng.choice(
                                    len(examples),
                                    min(len(examples), config.long_audit_per_split),
                                    replace=False,
                                )
                            )
                            selected = [examples[int(i)] for i in ix]
                        print(
                            f"Generating {folder.name}/{profile}: {split} ({len(selected)})",
                            flush=True,
                        )
                        batch_size = (
                            1 if profile == "legacy_greedy96" else config.generation_batch_size
                        )
                        for start in range(0, len(selected), batch_size):
                            batch = selected[start : start + batch_size]
                            generated = generate_batch(model, batch, tokenizer, config, profile)
                            for pack, response in zip(batch, generated):
                                row = {
                                    **scores[pack["id"]],
                                    **response,
                                    **text_scores(
                                        pack["ground_truth"], response["generated_answer"]
                                    ),
                                }
                                rows.append(row)
                                sink.write(
                                    json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                                )
                            sink.flush()
                frame = enrich(pd.DataFrame(rows))
                summary = _write_frame(destination, frame, oracle)
                print(
                    json.dumps(
                        {
                            "profile": profile,
                            "model_utility": summary["model_utility"],
                            "forget_quality": summary.get("forget_quality"),
                        }
                    ),
                    flush=True,
                )
                if profile == "source_sampled":
                    primary = frame
        outputs = [p for p in folder.rglob("*") if p.is_file() and p.name != "metadata.json"]
        metadata.update(
            status="complete",
            seconds=time.perf_counter() - started,
            process_ram_bytes=psutil.Process().memory_info().rss,
            artifact_hashes={str(p.relative_to(folder)): sha256(p) for p in outputs},
        )
        if str(config.device).startswith("cuda"):
            torch.cuda.synchronize(config.device)
            metadata.update(
                peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(config.device),
                peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(config.device),
            )
        write_json(marker, metadata)
        return primary
    except BaseException as exc:
        write_json(
            folder / "failure.json", {"error": repr(exc), "seconds": time.perf_counter() - started}
        )
        raise
    finally:
        model.train(previous_mode)
