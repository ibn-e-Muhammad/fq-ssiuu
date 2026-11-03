"""Versioned third-run choices; LoRA is an explicit change to trainable weights."""

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Config:
    full_path: str = ""
    retain_path: str = ""
    tokenizer_path: str = ""
    data_path: str = ""
    output_root: str = "runs"
    allow_model_download: bool = False
    allow_data_download: bool = False
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    seeds: list = field(default_factory=lambda: [42, 43, 44])
    evaluation_seed: int = 20261103
    beta: float = 0.1
    npo_coefficient: float = 1.0
    retain_coefficient: float = 1.0
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    nominal_epochs: int = 10
    logical_micro_batch: int = 4
    device_micro_batch: int = 4
    accumulation: int = 8
    warmup_steps: int = 6
    evaluation_every_updates: int = 6
    training_max_length: int = 500
    evaluation_max_length: int = 200
    retain_sampling: str = "reshuffled_stream"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_targets: list = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    gradient_checkpointing: bool = True
    evaluation_retain_count: int = 300
    generation_batch_size: int = 30
    source_temperature: float = 0.6
    source_top_p: float = 0.9
    generation_eos_ids: list = field(default_factory=list)
    legacy_greedy_tokens: int = 96
    long_audit_tokens: int = 256
    long_audit_per_split: int = 20
    bootstrap_repeats: int = 2000
    evaluate_kl: bool = True
    quantization_modes: list = field(default_factory=lambda: ["int8", "nf4"])
    quantized_controls: bool = True
    keep_recent_checkpoints: int = 2
    system_prompt: str = "You are a helpful assistant."
    template_date: str = "10 Apr 2025"

    def validate(self):
        numeric = (
            self.beta,
            self.npo_coefficient,
            self.retain_coefficient,
            self.learning_rate,
            self.weight_decay,
            self.lora_dropout,
            self.source_temperature,
            self.source_top_p,
        )
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError("Finite numeric configuration required")
        if self.dtype not in ("bfloat16", "float32"):
            raise ValueError("BF16 for RTX 3060; FP32 is only the CPU fixture path")
        if self.device == "cpu" and self.dtype != "float32":
            raise ValueError("CPU fixtures require float32")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Distinct declared training seeds required")
        if any(type(seed) is not int or seed < 0 for seed in self.seeds):
            raise ValueError("Nonnegative integer seeds required")
        if self.retain_sampling not in ("reshuffled_stream", "source_fixed"):
            raise ValueError("Unknown retain sampling mode")
        if self.beta <= 0 or self.npo_coefficient <= 0 or self.retain_coefficient < 0:
            raise ValueError("Invalid objective coefficients")
        if self.lora_rank <= 0 or self.lora_alpha <= 0 or not 0 <= self.lora_dropout < 1:
            raise ValueError("Invalid LoRA configuration")
        if (
            min(
                self.logical_micro_batch,
                self.device_micro_batch,
                self.accumulation,
                self.evaluation_every_updates,
                self.generation_batch_size,
            )
            <= 0
        ):
            raise ValueError("Positive batch, accumulation and evaluation intervals required")
        if self.device_micro_batch > self.logical_micro_batch:
            raise ValueError("Device microbatch cannot exceed the logical microbatch")
        if self.keep_recent_checkpoints < 2:
            raise ValueError("Retain at least two checkpoints for corruption fallback")
        if any(q not in ("int8", "nf4") for q in self.quantization_modes):
            raise ValueError("Unknown quantization mode")
        if self.training_max_length < self.evaluation_max_length or self.evaluation_max_length <= 1:
            raise ValueError("Invalid sequence limits")
        if (
            min(
                self.legacy_greedy_tokens,
                self.long_audit_tokens,
                self.long_audit_per_split,
                self.bootstrap_repeats,
            )
            <= 0
        ):
            raise ValueError("Positive generation/audit/bootstrap sizes required")
        if (
            self.learning_rate <= 0
            or self.weight_decay < 0
            or not 0 < self.source_top_p <= 1
            or self.source_temperature <= 0
        ):
            raise ValueError("Invalid optimizer or sampling parameters")
        return self

    @property
    def effective_batch(self):
        return self.logical_micro_batch * self.accumulation

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        obj = json.loads(path.read_text(encoding="utf-8"))
        config = cls(**obj).validate()
        for key in ("full_path", "retain_path", "tokenizer_path", "data_path", "output_root"):
            value = getattr(config, key)
            if value and not Path(value).is_absolute():
                setattr(config, key, str((path.parent / value).resolve()))
        return config
