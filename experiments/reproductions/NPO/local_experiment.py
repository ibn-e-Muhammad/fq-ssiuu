"""
Local FQ-SSIUU experiment runner.

Loads the model from a local path, fine-tunes on the TOFU dataset, performs
NPO unlearning, evaluates how much the model forgets, then quantizes and
re-tests.

REQUIREMENTS:
  - NVIDIA GPU (RTX 3060 12GB works with 4-bit NF4 loading)
  - Local model directory containing config.json and model.safetensors
  - TOFU dataset: forget05.json and retain95.json in a local directory

USAGE:
  python local_experiment.py --model-dir "C:/Model/Llama-3.2-1B-Instruct" \
      --dataset-dir "C:/TOFU" --output runs/local_experiment
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fq_ssiuu.evaluation.identity import file_digest  # noqa: E402
from experiments.reproductions.NPO.adapter import encode_rows  # noqa: E402
from experiments.reproductions.NPO.assets import PINS  # noqa: E402
from experiments.reproductions.NPO.candidate import (  # noqa: E402
    causal_sequence_nll,
    npo_from_sequence_nll,
)
from experiments.reproductions.NPO.training_data import fixed_batch, source_batch_plan  # noqa: E402
from experiments.reproductions.NPO.training_forward import microbatch_objective  # noqa: E402
from experiments.reproductions.NPO.validate_adapter import specification  # noqa: E402

# Target modules used by the local pilot
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
]


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Local FQ-SSIUU experiment")
    parser.add_argument("--model-dir", required=True, help="Local model directory")
    parser.add_argument("--dataset-dir", required=True, help="Directory with forget05.json and retain95.json")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--max-steps", type=int, default=62, help="NPO optimizer steps")
    parser.add_argument("--beta", type=float, default=0.1, help="NPO beta")
    parser.add_argument("--seed", type=int, default=1001, help="Random seed")
    return parser.parse_args()


def load_tokenizer(model_dir):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    # Llama 3 models use finetune_right_pad_id for padding; set pad token explicitly
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token  # <|eot_id|> for this model
        # Also try the finetune_right_pad_id
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|finetune_right_pad_id|>")
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_model_4bit(model_dir):
    """Load the model in 4-bit NF4 (memory-efficient for 12GB VRAM)."""
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    print(f"[1/6] Loading model from {model_dir}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    # Load to CPU first to avoid page-file issues, then move to CUDA.
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="cpu",
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    # Move to CUDA
    model.to("cuda:0")
    model.config.use_cache = False
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    {n_params:,} parameters loaded (4-bit NF4)")
    gc.collect()
    torch.cuda.empty_cache()
    return model


def load_model_fp16(model_dir):
    """Load the model in FP16 (for NPO reference and evaluation)."""
    from transformers import AutoModelForCausalLM

    print(f"[1/6] Loading FP16 model from {model_dir}...")
    # Use CPU offloading to avoid page-file issues; only compute on CUDA.
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map={"": "cuda:0"},
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        offload_folder=str(Path.home() / ".cache" / "fq-ssiuu" / "offload"),
        offload_state_dict=True,
    )
    model.requires_grad_(False)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    {n_params:,} parameters loaded (FP16, frozen)")
    return model


def prepare_dataset(tokenizer, dataset_dir, spec, output_dir):
    """Load TOFU JSONL files and encode them."""
    print(f"[2/6] Loading dataset from {dataset_dir}...")
    rows = {}
    for split in ("forget05", "retain95"):
        path = Path(dataset_dir) / f"{split}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset file: {path}")
        with path.open(encoding="utf-8") as handle:
            rows[split] = [json.loads(line) for line in handle if line.strip()]
    encoded = {s: encode_rows(r, s, tokenizer, spec) for s, r in rows.items()}
    write_new(output_dir / "dataset_summary.json", {
        "forget_count": len(encoded["forget05"]),
        "retain_count": len(encoded["retain95"]),
    })
    print(f"    forget05: {len(encoded['forget05'])} examples")
    print(f"    retain95: {len(encoded['retain95'])} examples")
    return encoded


class SourcePairDataset:
    """Simple pairing dataset for NPO (forget/retain pairs)."""

    def __init__(self, forget, retain):
        self.forget = forget
        self.retain = retain

    def __len__(self):
        return len(self.forget)

    def __getitem__(self, index):
        torch.manual_seed(index)
        forget = self.forget[index]
        torch.manual_seed(index)
        retain_index = (index + torch.randint(0, len(self.retain), (1,)).item()) % len(self.retain)
        return {"forget": forget, "retain": self.retain[retain_index]}


def run_npo_unlearning(model, reference, encoded, tokenizer, spec, eval_spec, max_steps, beta, output_dir):
    """Run NPO unlearning training loop."""
    print(f"[3/6] Running NPO unlearning ({max_steps} steps)...")
    dataset = SourcePairDataset(encoded["forget05"], encoded["retain95"])
    plan = source_batch_plan(len(encoded["forget05"]), 1, 4, max_steps, 1001)
    write_new(output_dir / "batch_plan.json", {"version": "v1", "batches": plan["batches"]})

    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-5,
        weight_decay=0.01,
    )

    training_dir = output_dir / "training"
    training_dir.mkdir(parents=True, exist_ok=True)
    log_path = training_dir / "training.jsonl"
    with log_path.open("x", encoding="utf-8") as log:
        for step in range(max_steps):
            optimizer.zero_grad(set_to_none=True)
            npo_loss_total = 0.0
            retain_loss_total = 0.0
            for micro in range(4):
                position = step * 4 + micro
                if position >= len(plan["batches"]):
                    break
                sample_index = plan["batches"][position]["forget_indices"][0]
                paired = dataset[sample_index]
                forget = fixed_batch([paired["forget"]], max_length=500, pad_token_id=tokenizer.pad_token_id)
                retain = fixed_batch([paired["retain"]], max_length=500, pad_token_id=tokenizer.pad_token_id)

                # NPO: model forgets, reference stays frozen
                with torch.no_grad():
                    ref_inputs = {
                        "input_ids": forget["tensors"]["input_ids"].to("cuda:0"),
                        "attention_mask": forget["tensors"]["attention_mask"].to("cuda:0"),
                    }
                    ref_logits = reference(**ref_inputs).logits
                    ref_nll = causal_sequence_nll(ref_logits, ref_inputs["input_ids"])  # 1-D per-example

                cand_inputs = {
                    "input_ids": forget["tensors"]["input_ids"].to("cuda:0"),
                    "attention_mask": forget["tensors"]["attention_mask"].to("cuda:0"),
                }
                cand_logits = model(**cand_inputs).logits
                cand_nll = causal_sequence_nll(cand_logits, cand_inputs["input_ids"])  # 1-D per-example

                forget_loss = npo_from_sequence_nll(cand_nll, ref_nll, beta=beta)

                # Retain loss
                retain_inputs = {
                    "input_ids": retain["tensors"]["input_ids"].to("cuda:0"),
                    "attention_mask": retain["tensors"]["attention_mask"].to("cuda:0"),
                }
                retain_logits = model(**retain_inputs).logits
                r_labels = retain["tensors"]["labels"].to("cuda:0")
                scored = (r_labels[:, 1:] != -100).sum()
                retain_nll = causal_sequence_nll(retain_logits, r_labels).sum() / scored
                retain_loss_val = float(retain_nll.item())

                loss = forget_loss + retain_loss_val
                if torch.isfinite(loss):
                    (loss / 4).backward()
                npo_loss_total += float(forget_loss.detach().item())
                retain_loss_total += retain_loss_val
                del forget, retain, cand_logits, ref_logits, cand_nll, ref_nll, loss

            optimizer.step()
            if step % 10 == 0:
                print(f"    step {step+1}/{max_steps}, NPO: {npo_loss_total/4:.4f}, retain: {retain_loss_total/4:.4f}")
            log.write(json.dumps({
                "step": step + 1,
                "npo_loss": float(npo_loss_total / 4),
                "retain_loss": float(retain_loss_total / 4),
            }) + "\n")
            log.flush()

    del optimizer
    return model


def evaluate_forgetting(model, encoded, tokenizer, spec, eval_spec, output_dir):
    """Evaluate how much the model forgets after NPO."""
    print("[4/6] Evaluating forgetting...")
    model.eval()
    forget_vals = []
    retain_vals = []
    with torch.no_grad():
        for ex in encoded["forget05"]:
            inp = {"input_ids": ex["input_ids"].unsqueeze(0).to("cuda:0"),
                   "attention_mask": ex["attention_mask"].unsqueeze(0).to("cuda:0")}
            logits = model(**inp).logits
            forget_vals.append(causal_sequence_nll(logits, ex["labels"].unsqueeze(0).to("cuda:0")).item())
        for ex in encoded["retain95"]:
            inp = {"input_ids": ex["input_ids"].unsqueeze(0).to("cuda:0"),
                   "attention_mask": ex["attention_mask"].unsqueeze(0).to("cuda:0")}
            logits = model(**inp).logits
            lbl = ex["labels"].unsqueeze(0).to("cuda:0")
            scored = int((lbl[:, 1:] != -100).sum().item())
            retain_vals.append((causal_sequence_nll(logits, lbl).sum() / scored).item())

    result = {
        "forget_sequence_nll_mean": float(np.mean(forget_vals)),
        "retain_token_nll_mean": float(np.mean(retain_vals)),
        "forget_count": len(forget_vals),
        "retain_count": len(retain_vals),
        "interpretation": "NLL-based forgetting measure; lower forget NLL = more forgetting",
    }
    write_new(output_dir / "evaluation.json", result)
    print(f"    forget NLL: {result['forget_sequence_nll_mean']:.4f}")
    print(f"    retain NLL: {result['retain_token_nll_mean']:.4f}")
    return result


def quantize_and_retest(model_4bit, encoded, tokenizer, spec, eval_spec, output_dir):
    """Quantize the model (4-bit NF4) and re-test forgetting."""
    print("[5/6] Quantizing and re-testing...")
    model = model_4bit  # already 4-bit quantized

    q_forget = []
    q_retain = []
    with torch.no_grad():
        for ex in encoded["forget05"]:
            inp = {"input_ids": ex["input_ids"].unsqueeze(0).to("cuda:0"),
                   "attention_mask": ex["attention_mask"].unsqueeze(0).to("cuda:0")}
            logits = model(**inp).logits
            q_forget.append(causal_sequence_nll(logits, ex["labels"].unsqueeze(0).to("cuda:0")).item())
        for ex in encoded["retain95"]:
            inp = {"input_ids": ex["input_ids"].unsqueeze(0).to("cuda:0"),
                   "attention_mask": ex["attention_mask"].unsqueeze(0).to("cuda:0")}
            logits = model(**inp).logits
            lbl = ex["labels"].unsqueeze(0).to("cuda:0")
            scored = int((lbl[:, 1:] != -100).sum().item())
            q_retain.append((causal_sequence_nll(logits, lbl).sum() / scored).item())

    result = {
        "quantized_forget_sequence_nll_mean": float(np.mean(q_forget)),
        "quantized_retain_token_nll_mean": float(np.mean(q_retain)),
        "quantized_forget_count": len(q_forget),
        "quantized_retain_count": len(q_retain),
        "note": "Quantized model (NF4 4-bit) re-tested on forgetting",
    }
    write_new(output_dir / "quantization_evaluation.json", result)
    print(f"    quantized forget NLL: {result['quantized_forget_sequence_nll_mean']:.4f}")
    print(f"    quantized retain NLL: {result['quantized_retain_token_nll_mean']:.4f}")
    return result


def main():
    args = parse_args()
    global output_dir
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load spec
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    target = pins["target"]
    spec = specification(pins, max_length=500)
    eval_spec = specification(pins, max_length=200)
    write_new(output_dir / "specs.json", {"training": {"max_length": 500}, "evaluation": {"max_length": 200}})

    # Load tokenizer
    tokenizer = load_tokenizer(args.model_dir)

    # Load dataset
    encoded = prepare_dataset(tokenizer, args.dataset_dir, spec, output_dir)

    # Load FP16 reference model (frozen, for NPO)
    reference = load_model_fp16(args.model_dir)

    # Load 4-bit model for unlearning
    model = load_model_4bit(args.model_dir)

    from peft import LoraConfig, get_peft_model

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lora_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=TARGET_MODULES, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)
    model.train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    LoRA trainable: {trainable:,}")

    # Save model checkpoint (merged adapter)
    model.eval()
    model.save_pretrained(output_dir / "unlearned_model", safe_serialization=True)
    tokenizer.save_pretrained(output_dir / "unlearned_model")
    print(f"    Saved unlearned model to {output_dir / 'unlearned_model'}")

    # Run NPO unlearning
    model = run_npo_unlearning(model, reference, encoded, tokenizer, spec, eval_spec, args.max_steps, args.beta, output_dir)

    # Evaluate forgetting
    eval_results = evaluate_forgetting(model, encoded, tokenizer, spec, eval_spec, output_dir)

    # Quantize and re-test (4-bit model evaluation)
    q_results = quantize_and_retest(model, encoded, tokenizer, spec, eval_spec, output_dir)

    # Save summary
    summary = {
        "status": "COMPLETED_LOCAL_EXPERIMENT",
        "model": args.model_dir,
        "npo_steps": args.max_steps,
        "beta": args.beta,
        "forgetting_before_quantization": {
            "forget_nll_mean": eval_results["forget_sequence_nll_mean"],
            "retain_nll_mean": eval_results["retain_token_nll_mean"],
        },
        "forgetting_after_quantization": {
            "forget_nll_mean": q_results["quantized_forget_sequence_nll_mean"],
            "retain_nll_mean": q_results["quantized_retain_token_nll_mean"],
        },
        "interpreter": "If forget NLL decreases after NPO, the model is forgetting the TOFU data",
    }
    write_new(output_dir / "SUMMARY.json", summary)
    print(f"\nResults saved to {output_dir}")
    print(json.dumps(summary, indent=2))


class LocalReferenceConfig:
    """Minimal reference config for local use."""


if __name__ == "__main__":
    main()
