"""Actual CPU ZeRO-3 subgroup equivalence/resume, with a random tiny NPO+RT model.

Small test thresholds force multiple groups. CPU AdamW is only a grouping fixture,
not a substitute for the runtime paged FP32 optimizer or a CUDA acceptance claim.
"""

import argparse
import copy
import json
import os
from pathlib import Path

import torch

from experiments.reproductions.common import digest, file_digest
from experiments.reproductions.NPO.candidate import compute_causal_npo_rt_loss
from experiments.reproductions.NPO.engine import source_scheduler
from experiments.reproductions.NPO.optimizer import parameter_groups
from experiments.reproductions.NPO.training_state import (
    IDENTITY_KEYS,
    capture_rng,
    load_deepspeed_state,
    save_deepspeed_state,
)
from experiments.reproductions.NPO.validate_deepspeed_cpu import TinyCausalModel


def validate(output):
    import deepspeed

    if torch.cuda.is_available():
        raise ValueError("CPU conformance only")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    deepspeed.ops.__compatible_ops__["deepspeed_shm_comm"] = False
    os.environ.update(LOCAL_RANK="0", RANK="0", WORLD_SIZE="1")
    torch.distributed.init_process_group(
        "gloo", init_method="file://" + str(output / "rendezvous"), rank=0, world_size=1
    )
    try:
        torch.manual_seed(42)
        original = TinyCausalModel()
        initial = copy.deepcopy(original.state_dict())
        reference = copy.deepcopy(original).eval().requires_grad_(False)

        def construct(size):
            model = TinyCausalModel()
            model.load_state_dict(initial)
            groups, _ = parameter_groups(model, 0.01)
            optimizer = torch.optim.AdamW(groups, lr=1e-5, betas=(0.9, 0.999), eps=1e-8)
            return deepspeed.initialize(
                model=model,
                optimizer=optimizer,
                lr_scheduler=source_scheduler,
                config={
                    "train_batch_size": 32,
                    "train_micro_batch_size_per_gpu": 4,
                    "gradient_accumulation_steps": 8,
                    "zero_optimization": {"stage": 3, "sub_group_size": size},
                    "zero_allow_untested_optimizer": True,
                    "gradient_clipping": 0.0,
                },
                dist_init_required=False,
            )[0]

        def micro(engine):
            ids, retained = [torch.randint(0, 11, (4, 5)) for _ in range(2)]
            labels, retain_labels = ids.clone(), retained.clone()
            labels[:, :2] = retain_labels[:, :2] = -100
            with torch.no_grad():
                ref = reference(ids)
            loss = compute_causal_npo_rt_loss(
                engine(ids), ref, labels, engine(retained), retain_labels
            ).total
            value = float(loss.detach())
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite NPO+RT loss")
            engine.backward(loss)
            engine.step()
            return value

        def state(engine):
            groups = engine.optimizer.fp32_partitioned_groups_flat
            inner = engine.optimizer.optimizer
            return {
                "masters": torch.cat([p.detach().clone() for p in groups]),
                "moment1": torch.cat([inner.state[p]["exp_avg"].clone() for p in groups]),
                "moment2": torch.cat([inner.state[p]["exp_avg_sq"].clone() for p in groups]),
            }

        cases = []
        for size in (1000000000, 40):
            engine = construct(size)
            subgroup_count = len(engine.optimizer.fp32_partitioned_groups_flat)
            torch.manual_seed(1001)
            before = torch.cat(
                [p.detach().clone() for p in engine.optimizer.fp32_partitioned_groups_flat]
            )
            losses = [micro(engine) for _ in range(64)]
            after = state(engine)
            if not torch.isfinite(after["masters"]).all() or torch.equal(before, after["masters"]):
                raise ValueError("Degenerate optimizer update")
            identities = {
                key: "a" * (40 if key.endswith("revision") or key == "source_commit" else 64)
                for key in IDENTITY_KEYS
            }
            identities["train_config_sha256"] = digest({"cpu_fixture_subgroup_size": size})
            cursor = {
                "optimizer_step": 8,
                "micro_step": 0,
                "epoch": 0,
                "batch_in_epoch": 64,
                "sampler_state": torch.Generator().manual_seed(1001).get_state(),
            }
            checkpoint = output / f"checkpoint_{size}"
            manifest = save_deepspeed_state(checkpoint, engine, cursor, identities)
            expected_losses = [micro(engine) for _ in range(16)]
            expected_state = state(engine)
            expected_rng = capture_rng()
            scheduler = copy.deepcopy(engine.lr_scheduler.state_dict())
            engine.destroy()
            del engine
            engine = construct(size)
            restored = load_deepspeed_state(checkpoint, engine, identities)
            actual_losses = [micro(engine) for _ in range(16)]
            for name, expected in expected_state.items():
                torch.testing.assert_close(expected, state(engine)[name], rtol=0, atol=0)
            if expected_losses != actual_losses or not torch.equal(
                expected_rng["torch"], torch.get_rng_state()
            ):
                raise ValueError("Resume loss/RNG mismatch")
            if (
                scheduler != engine.lr_scheduler.state_dict()
                or restored["optimizer_step"] != 8
                or engine.global_steps != 10
            ):
                raise ValueError("Resume scheduler/cursor mismatch")
            cases.append(
                {
                    "size": size,
                    "count": subgroup_count,
                    "losses": losses,
                    "state": after,
                    "checkpoint_manifest_sha256": file_digest(checkpoint / "MANIFEST.json"),
                    "checkpoint_files": manifest["files"],
                    "resume_loss_error": 0.0,
                }
            )
            engine.destroy()
            del engine
        if cases[1]["count"] <= cases[0]["count"]:
            raise ValueError("Fixture did not split optimizer groups")
        loss_error = max(
            abs(a - b) for a, b in zip(cases[0]["losses"], cases[1]["losses"], strict=True)
        )
        torch.testing.assert_close(
            torch.tensor(cases[0]["losses"]), torch.tensor(cases[1]["losses"]), rtol=0, atol=1e-7
        )
        errors = {}
        for key in cases[0]["state"]:
            a, b = cases[0]["state"][key], cases[1]["state"][key]
            torch.testing.assert_close(a, b, rtol=0, atol=1e-7)
            errors[key] = float((a - b).abs().max())
        for key, value in reference.state_dict().items():
            torch.testing.assert_close(value, initial[key], rtol=0, atol=0)
        if any(p.grad is not None for p in reference.parameters()):
            raise ValueError("Reference gradient leak")
        result = {
            "status": "PASS_CPU_SUBGROUP_EQUIVALENCE_AND_RESUME_ONLY",
            "torch": torch.__version__,
            "deepspeed": deepspeed.__version__,
            "optimizer": "torch.optim.AdamW CPU fixture, not runtime optimizer",
            "compared_microbatches": 64,
            "resume_microbatches": 16,
            "cross_group_loss_max_abs_error": loss_error,
            "cross_group_state_max_abs_error": errors,
            "within_group_resume_bit_identical": True,
            "reference_unchanged": True,
            "nondegenerate_updates": True,
            "cuda_paging_or_one_b_training_validated": False,
            "cases": [{k: v for k, v in c.items() if k != "state"} for c in cases],
        }
        (output / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = validate(args.output)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "cross_group_loss_max_abs_error",
                    "cross_group_state_max_abs_error",
                )
            }
        )
    )
