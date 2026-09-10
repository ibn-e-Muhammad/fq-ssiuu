"""Compare NPO values/gradients with the actual pinned author functions, offline.

Supply a separately fetched author checkout; no upstream code is redistributed.
Only hash-verified function ASTs execute, avoiding the DeepSpeed trainer launcher.
"""

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

AUTHOR_COMMIT = "1207bbe9a1abf502bc2d7b43a0a467d9524b4036"
AUTHOR_FILES = {
    "TOFU/data_module.py": "8e076f18c20b663dbad9862a5d97c25e91fe4f9475bed77e597dbc2b349abf31",
    "TOFU/dataloader.py": "a692b709d01d05204a81e40c38b1f6e64faac9286a662d9d7cf9e4cc4dcf6c69",
}


def author_functions(checkout):
    import torch

    parsed = {}
    for name, expected in AUTHOR_FILES.items():
        data = (Path(checkout) / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("Author source differs from the pinned file hash")
        parsed[name] = ast.parse(data, filename=name)
    helper = next(
        node
        for node in parsed["TOFU/data_module.py"].body
        if isinstance(node, ast.FunctionDef) and node.name == "get_batch_loss"
    )
    trainer = next(
        node
        for node in parsed["TOFU/dataloader.py"].body
        if isinstance(node, ast.ClassDef) and node.name == "CustomTrainerForgetting"
    )
    objective = next(
        node
        for node in trainer.body
        if isinstance(node, ast.FunctionDef) and node.name == "compute_loss"
    )
    namespace = {"torch": torch, "nn": torch.nn, "F": torch.nn.functional}
    for function in (helper, objective):
        module = ast.Module(body=[function], type_ignores=[])
        exec(compile(module, "hash_verified_author_function", "exec"), namespace)
    return namespace["get_batch_loss"], namespace["compute_loss"]


OPENUNLEARNING_COMMIT = "4ad738aaf60f6a4385f6e2506d01da99e76c31f3"
OPENUNLEARNING_FILES = {
    "src/trainer/utils.py": "55ac9337111ffe1be0be0b4ab4529ae1471dc19ecda2dee366ac9e10748bf776",
    "src/trainer/unlearn/npo.py": "9115e1af39b1992b72855cbcf69f1f49eb83a03f00182d12143e392abe37040e",
    "src/trainer/unlearn/grad_diff.py": "6fe2ae02f62664a5a43a270c7b30e758daba5d703e4408adb67c3c5323f872ea",
    "configs/trainer/NPO.yaml": "ff5c7313f6f3b6666300ace3fd4b1862caeb0a56ea0a511275cb91c2320264e8",
}


def openunlearning_functions(checkout):
    import torch

    parsed = {}
    for name, digest in OPENUNLEARNING_FILES.items():
        data = (Path(checkout) / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("OpenUnlearning source differs from pinned hash")
        if name.endswith(".py"):
            parsed[name] = ast.parse(data, filename=name)
    namespace = {"torch": torch, "nn": torch.nn, "F": torch.nn.functional}
    selected = [
        node
        for node in parsed["src/trainer/utils.py"].body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"compute_batch_nll", "compute_dpo_loss"}
    ]
    for name, class_name, method in [
        ("src/trainer/unlearn/grad_diff.py", "GradDiff", "compute_retain_loss"),
        ("src/trainer/unlearn/npo.py", "NPO", "compute_loss"),
    ]:
        cls = next(
            n for n in parsed[name].body if isinstance(n, ast.ClassDef) and n.name == class_name
        )
        selected.append(
            next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method)
        )
    for function in selected:
        exec(
            compile(
                ast.Module(body=[function], type_ignores=[]),
                "hash_verified_openunlearning_function",
                "exec",
            ),
            namespace,
        )
    return namespace


def compare_case(
    helper, objective, ou, backend, dtype, variant, beta, forget_weight, retain_weight
):
    from types import MethodType

    import torch

    from experiments.reproductions.NPO.candidate import (
        compute_causal_npo_loss,
        compute_causal_npo_rt_loss,
    )

    generator = torch.Generator().manual_seed(42)
    current, reference, retain = [
        torch.randn(2, 5, 7, generator=generator, dtype=dtype, requires_grad=True) for _ in range(3)
    ]
    labels = torch.tensor([[-100, -100, 1, 2, 3], [-100, 4, 2, -100, -100]])
    forget_ids, retain_ids = torch.zeros_like(labels), torch.ones_like(labels)

    def model(input_ids, labels, attention_mask):
        logits = current if input_ids is forget_ids else retain
        return SimpleNamespace(
            logits=logits, loss=helper(logits, labels).sum() / (labels[:, 1:] != -100).sum()
        )

    def reference_model(*args, **kwargs):
        return SimpleNamespace(logits=reference)

    trainer = SimpleNamespace(
        loss_type=variant,
        ref_policy="fine_tuned",
        beta=beta,
        npo_coeff=forget_weight,
        grad_diff_coeff=retain_weight,
        gamma=forget_weight,
        alpha=retain_weight,
        retain_loss_type="NLL",
        oracle_model=reference_model,
        ref_model=reference_model,
    )
    forget_inputs = {
        "input_ids": forget_ids,
        "labels": labels,
        "attention_mask": torch.ones_like(labels),
    }
    retain_inputs = {**forget_inputs, "input_ids": retain_ids}
    if backend == "author":
        expected = objective(
            trainer, model, (tuple(forget_inputs.values()), tuple(retain_inputs.values()))
        )
    elif variant == "npo":
        expected, _ = ou["compute_dpo_loss"](
            model, reference_model, win_inputs=None, lose_inputs=forget_inputs, beta=beta
        )
    else:
        trainer.compute_retain_loss = MethodType(ou["compute_retain_loss"], trainer)
        expected = ou["compute_loss"](
            trainer, model, {"forget": forget_inputs, "retain": retain_inputs}
        )
    if variant == "npo":
        actual = compute_causal_npo_loss(current, reference, labels, beta)
        parameters = (current,)
    else:
        actual = compute_causal_npo_rt_loss(
            current, reference, labels, retain, labels, beta, forget_weight, retain_weight
        ).total
        parameters = (current, retain)
    tolerance = 1e-12 if dtype == torch.float64 else 2e-6
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    actual_grads = torch.autograd.grad(actual, parameters, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, parameters, retain_graph=True)
    for a, b in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(a, b, rtol=tolerance, atol=tolerance)
    assert torch.autograd.grad(actual, reference, allow_unused=True)[0] is None
    assert torch.autograd.grad(expected, reference, allow_unused=True)[0] is None
    return {
        "reference_implementation": backend,
        "variant": variant,
        "dtype": str(dtype),
        "beta": beta,
        "forget_weight": forget_weight,
        "retain_weight": retain_weight,
        "upstream_loss": float(expected.detach()),
        "project_loss": float(actual.detach()),
        "maximum_gradient_absolute_error": max(
            float((a - b).abs().max()) for a, b in zip(actual_grads, expected_grads, strict=True)
        ),
        "reference_gradient": None,
        "status": "PASS",
    }


def validate(checkout, openunlearning):
    import torch

    helper, objective = author_functions(checkout)
    ou = openunlearning_functions(openunlearning)
    observations = [
        compare_case(helper, objective, ou, backend, dtype, variant, beta, forget, retain)
        for backend in ("author", "openunlearning")
        for dtype in (torch.float64, torch.float32)
        for variant in ("npo", "npo_grad_diff")
        for beta, forget, retain in ((0.1, 1.0, 1.0), (0.3, 0.7, 1.4))
    ]
    return {
        "status": "PASS",
        "author_commit": AUTHOR_COMMIT,
        "source_sha256": AUTHOR_FILES,
        "openunlearning_commit": OPENUNLEARNING_COMMIT,
        "openunlearning_sha256": OPENUNLEARNING_FILES,
        "execution": "Actual hash-verified function ASTs from both pinned upstreams",
        "observations": observations,
        "model_downloaded": False,
        "scope": "CPU objective values/current-retain gradients/reference detachment; no dataset preprocessing, GPU or 1B pilot claim",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--openunlearning", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with Path(args.output).open("x") as handle:
        try:
            result = validate(args.upstream, args.openunlearning)
        except Exception as exc:
            result = {"status": "FAIL", "error_type": type(exc).__name__}
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
