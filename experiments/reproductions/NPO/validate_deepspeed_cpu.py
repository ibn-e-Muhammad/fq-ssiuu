"""Optional local ZeRO-3 persistence check, using a random tiny CPU model.

This deliberately uses CPU AdamW; it makes no bnb/CUDA/FP16 conformance claim.
The offline CI suite exercises the checkpoint contract without this dependency.
"""

import argparse
import copy
import json
import os
from pathlib import Path

import torch

from experiments.reproductions.common import file_digest
from experiments.reproductions.NPO.candidate import compute_causal_npo_rt_loss
from experiments.reproductions.NPO.training_state import (
    IDENTITY_KEYS,
    capture_rng,
    load_deepspeed_state,
    save_deepspeed_state,
)


class TinyCausalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(11, 4)
        self.head = torch.nn.Linear(4, 11)

    def forward(self, ids):
        return self.head(self.embedding(ids))


def validate(output):
    import deepspeed

    if torch.cuda.is_available():
        raise ValueError("This check must run on CPU")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    # CPU-only test compatibility switch: standard Gloo, not an optional shared
    # memory extension. This switch must never be used by the Kaggle runner.
    deepspeed.ops.__compatible_ops__["deepspeed_shm_comm"] = False
    os.environ.update(LOCAL_RANK="0", RANK="0", WORLD_SIZE="1")
    torch.distributed.init_process_group(
        "gloo", init_method="file://" + str(output / "rendezvous"), rank=0, world_size=1
    )
    try:
        torch.manual_seed(1001)
        initial = TinyCausalModel().state_dict()
        reference = TinyCausalModel().eval().requires_grad_(False)
        reference.load_state_dict(initial)
        config = {
            "train_batch_size": 8,
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 8,
            "zero_optimization": {"stage": 3},
            "zero_allow_untested_optimizer": True,
            "gradient_clipping": 0.0,
        }

        def construct():
            model = TinyCausalModel()
            model.load_state_dict(initial)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda s: min(s / 2, max(0, (8 - s) / 6))
            )
            engine, _, _, _ = deepspeed.initialize(
                model=model,
                optimizer=optimizer,
                lr_scheduler=scheduler,
                config=config,
                dist_init_required=False,
            )
            return engine

        def micro(engine):
            ids = torch.randint(0, 11, (1, 5))
            retained = torch.randint(0, 11, (1, 5))
            labels, retain_labels = ids.clone(), retained.clone()
            labels[:, :2] = retain_labels[:, :2] = -100
            with torch.no_grad():
                ref = reference(ids)
            loss = compute_causal_npo_rt_loss(
                engine(ids), ref, labels, engine(retained), retain_labels
            ).total
            value = float(loss.detach())
            engine.backward(loss)
            engine.step()
            return value

        engine = construct()
        for _ in range(16):
            micro(engine)
        identities = {
            key: "a" * (40 if key.endswith("revision") or key == "source_commit" else 64)
            for key in IDENTITY_KEYS
        }
        cursor = {
            "optimizer_step": 2,
            "micro_step": 0,
            "epoch": 0,
            "batch_in_epoch": 16,
            "sampler_state": torch.Generator().manual_seed(1001).get_state(),
        }
        checkpoint = output / "checkpoint2"
        manifest = save_deepspeed_state(checkpoint, engine, cursor, identities)
        expected = [micro(engine) for _ in range(16)]
        expected_rng = capture_rng()
        masters = [x.detach().clone() for x in engine.optimizer.fp32_partitioned_groups_flat]
        scheduler_state = copy.deepcopy(engine.lr_scheduler.state_dict())
        second = construct()
        restored = load_deepspeed_state(checkpoint, second, identities)
        actual = [micro(second) for _ in range(16)]
        torch.testing.assert_close(torch.tensor(expected), torch.tensor(actual), rtol=0, atol=0)
        for a, b in zip(masters, second.optimizer.fp32_partitioned_groups_flat, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        if not torch.equal(expected_rng["torch"], torch.get_rng_state()):
            raise ValueError("DeepSpeed resume random stream differs")
        if scheduler_state != second.lr_scheduler.state_dict():
            raise ValueError("DeepSpeed scheduler restore differs")
        if restored["optimizer_step"] != 2 or second.global_steps != 4:
            raise ValueError("Wrong restored data/optimization cursor")
        result = {
            "status": "PASS_CPU_ZERO3_RESUME_ONLY",
            "torch": torch.__version__,
            "deepspeed": deepspeed.__version__,
            "optimizer": "torch.optim.AdamW CPU test only",
            "checkpoint_manifest_sha256": file_digest(checkpoint / "MANIFEST.json"),
            "checkpoint_files": manifest["files"],
            "compared_microbatches": 16,
            "loss_max_abs_error": max(abs(a - b) for a, b in zip(expected, actual, strict=True)),
            "masters_bit_identical": True,
            "scheduler_equal": True,
            "rng_equal": True,
            "cuda_or_paged_optimizer_validated": False,
        }
        with (output / "RESULT.json").open("x") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
        return result
    except Exception as error:
        with (output / "FAILURE.json").open("x") as handle:
            json.dump({"status": "FAIL", "error_type": type(error).__name__}, handle)
        raise
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.output)))
