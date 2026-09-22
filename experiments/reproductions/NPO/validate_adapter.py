"""Execute immutable upstream preprocessing against pinned real TOFU/tokenizer bytes.

This is a local CPU input-boundary check, never a model run. Sources are fetched
separately by assets.py and verified before selected actual ASTs execute.
"""

import argparse
import ast
import json
import logging
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Union

import numpy as np
import scipy as sc
import torch
import transformers
import yaml

from experiments.reproductions.NPO.adapter import AdapterSpec, collate_examples, encode_example
from experiments.reproductions.NPO.assets import PINS
from experiments.reproductions.QRA.metrics import model_utility, truth_ratios, utility_components
from fq_ssiuu.evaluation.identity import file_digest, tensor_record, tensors_digest


def upstream_nodes(path, expected, names, namespace):
    if file_digest(path) != expected:
        raise ValueError("Upstream source hash mismatch")
    nodes = [
        node
        for node in ast.parse(Path(path).read_bytes()).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    if {node.name for node in nodes} != set(names):
        raise ValueError("Required upstream definitions absent")
    for node in nodes:
        node.decorator_list = []  # Execute metric body, not framework registration.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def specification(pins, **kwargs):
    target, dataset = pins["target"], pins["dataset"]
    return AdapterSpec(
        target["model_id"],
        target["revision"],
        target["tokenizer_id"],
        target["tokenizer_revision"],
        dataset["id"],
        dataset["revision"],
        **kwargs,
    )


class UpstreamTokenizerCompatibility:
    """Preserve the upstream Transformers 4 return type on Transformers 5.

    Tokenization itself and the executed upstream function are unchanged.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    def apply_chat_template(self, *args, **kwargs):
        if kwargs.get("tokenize", True):
            kwargs["return_dict"] = False
        return self.tokenizer.apply_chat_template(*args, **kwargs)


def validate(root):
    root = Path(root)
    pins = json.loads(PINS.read_text())
    for item in pins["files"]:
        directory = "dataset" if item["repo"] == pins["dataset"]["id"] else "tokenizer"
        if file_digest(root / directory / item["path"]) != item["sha256"]:
            raise ValueError("Input asset hash mismatch")
    for name, pin in pins["upstreams"].items():
        for path, sha in pin["files"].items():
            if file_digest(root / name / path) != sha:
                raise ValueError("Upstream hash mismatch")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        root / "tokenizer", local_files_only=True
    )
    namespace = dict(
        torch=torch,
        transformers=transformers,
        Dict=dict,
        Any=Any,
        List=list,
        Union=Union,
        Sequence=Sequence,
        IGNORE_INDEX=-100,
        logger=logging.getLogger(__name__),
        np=np,
        sc=sc,
    )
    ou = pins["upstreams"]["openunlearning"]["files"]
    for path, names in [
        ("src/data/utils.py", ["preprocess_chat_instance"]),
        ("src/data/collators.py", ["DataCollatorForSupervisedDataset"]),
        ("src/evals/metrics/utils.py", ["aggregate_to_1D"]),
        ("src/evals/metrics/memorization.py", ["truth_ratio", "probability_w_options"]),
        ("src/evals/metrics/utility.py", ["hm_aggregate"]),
    ]:
        upstream_nodes(root / "openunlearning" / path, ou[path], names, namespace)
    template = yaml.safe_load(
        (root / "openunlearning/configs/model/Llama-3.2-1B-Instruct.yaml").read_text()
    )["template_args"]
    observations = []
    batches = []
    for split in (pins["dataset"]["forget"], pins["dataset"]["retain"]):
        rows = [
            json.loads(line)
            for line in (root / "dataset" / f"{split}.json").read_text().splitlines()
        ][:2]
        examples = []
        upstream = []
        for index, row in enumerate(rows):
            example = encode_example(row, index, split, tokenizer, specification(pins))
            actual = namespace["preprocess_chat_instance"](
                UpstreamTokenizerCompatibility(tokenizer),
                template,
                [row["question"]],
                [row["answer"]],
                512,
            )
            for key in ("input_ids", "attention_mask", "labels"):
                torch.testing.assert_close(example[key], actual[key], rtol=0, atol=0)
            examples.append(example)
            upstream.append(actual)
            observations.append(
                {
                    key: (
                        tensor_record({"value": value})["value"]
                        if isinstance(value, torch.Tensor)
                        else value
                    )
                    for key, value in example.items()
                }
            )
        for side in ("left", "right"):
            side_spec = specification(pins, padding_side=side)
            project = collate_examples(
                [encode_example(row, i, split, tokenizer, side_spec) for i, row in enumerate(rows)],
                tokenizer,
                side_spec,
            )
            other = namespace["DataCollatorForSupervisedDataset"](tokenizer, padding_side=side)(
                upstream
            )
            for key in ("input_ids", "labels"):
                torch.testing.assert_close(project["tensors"][key], other[key], rtol=0, atol=0)
            changed = project["tensors"]["attention_mask"].bool() != other["attention_mask"]
            assert torch.all(project["tensors"]["input_ids"][changed] == tokenizer.pad_token_id)
            assert torch.all(project["tensors"]["attention_mask"][changed] == 1)
            batches.append(
                {
                    "split": split,
                    "padding_side": side,
                    "project_sha256": project["tensors_sha256"],
                    "upstream_sha256": tensors_digest(other),
                    "real_eos_attention_positions_corrected": int(changed.sum()),
                }
            )
        boundary = examples[0]["prompt_token_count"]
        try:
            encode_example(
                rows[0], 0, split, tokenizer, replace(specification(pins), max_length=boundary)
            )
        except ValueError:
            pass
        else:
            raise AssertionError("Fully truncated answer accepted")
    # Execute the author adapter too. Its older tagged prompt is a distinct
    # preprocessing method; record the actual output, never claim parity.
    author_path = "TOFU/data_module.py"
    upstream_nodes(
        root / "npo" / author_path,
        pins["upstreams"]["npo"]["files"][author_path],
        ["convert_raw_data_to_model_format"],
        namespace,
    )
    tagged = {
        "question_start_tag": template["user_start_tag"],
        "question_end_tag": template["user_end_tag"],
        "answer_tag": template["asst_start_tag"],
    }
    ids, labels, attention = namespace["convert_raw_data_to_model_format"](
        tokenizer, 512, rows[0]["question"], rows[0]["answer"], tagged
    )
    author = {
        "identity": "author-tagged-adapter-with-translated-llama3-tags",
        "parity_claim": False,
        "reason": "Different prompt wrapping and EOS/padding semantics; OU chat is the selected target adapter",
        "tensors": {
            k: tensor_record({"value": v})["value"]
            for k, v in {"input_ids": ids, "labels": labels, "attention_mask": attention}.items()
        },
    }
    correct, wrong = -np.log(0.5), [-np.log(0.01), -np.log(0.81)]
    pre = {
        "correct": {"value_by_index": {"0": {"avg_loss": correct, "prob": 0.5}}},
        "wrong": {"value_by_index": {"0": {"avg_loss": wrong, "prob": [0.01, 0.81]}}},
    }
    diagnostic = namespace["truth_ratio"](None, pre_compute=pre, aggregator="closer_to_1_better")[
        "value_by_index"
    ]["0"]["score"]
    measured = truth_ratios(correct, wrong)
    np.testing.assert_allclose(measured["openunlearning_truth_ratio"], diagnostic, rtol=1e-14)
    option = namespace["probability_w_options"](None, pre_compute=pre)["agg_value"]
    project_option = utility_components(correct, wrong, 0.5, dataset="real_authors")[
        "real_authors_probability"
    ]
    np.testing.assert_allclose(project_option, option, rtol=1e-14)
    components = {
        f"{data}_{metric}": 0.5
        for data in ("retain", "real_authors", "world_facts")
        for metric in ("probability", "rouge_l_recall", "truth_ratio_score")
    }
    components["retain_probability"] = 0.25
    harmonic = namespace["hm_aggregate"](
        None, pre_compute={k: {"agg_value": v} for k, v in components.items()}
    )["agg_value"]
    np.testing.assert_allclose(model_utility(components)["value"], harmonic, rtol=1e-14)
    return {
        "status": "PASS",
        "scope": "Real pinned tokenizer/data and actual upstream preprocessing; synthetic metric parity; no model weights, backward, optimizer or experiment",
        "compatibility": "Explicit return_dict=False preserves upstream Transformers 4 chat-template return type on Transformers 5",
        "pins": pins,
        "examples": observations,
        "collations": batches,
        "author_adapter": author,
        "metrics": {
            "canonical_paper": measured["tofu_truth_ratio"],
            "upstream_geometric": float(diagnostic),
            "upstream_option_probability": float(option),
            "upstream_harmonic_mean": float(harmonic),
        },
        "versions": {"transformers": transformers.__version__, "torch": torch.__version__},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = validate(args.root)
    with Path(args.output).open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(
        json.dumps(
            {"status": result["status"], "examples": len(result["examples"]), "output": args.output}
        )
    )
