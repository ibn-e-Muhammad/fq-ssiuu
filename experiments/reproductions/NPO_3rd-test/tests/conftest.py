import sys
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from npo3.config import Config


@pytest.fixture
def tiny(tmp_path):
    torch.set_num_threads(1)
    vocab = {
        t: i
        for i, t in enumerate(
            [
                "<pad>",
                "<eot>",
                "<unk>",
                "<system>",
                "<user>",
                "<assistant>",
                "You",
                "are",
                "a",
                "helpful",
                "assistant",
                ".",
                "?",
                "A",
                "B",
                "C",
                "D",
            ]
            + [f"q{i}" for i in range(64)]
        )
    }
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        eos_token="<eot>",
        unk_token="<unk>",
        additional_special_tokens=["<system>", "<user>", "<assistant>"],
    )
    tokenizer.chat_template = "{% for m in messages %}{{ '<' + m['role'] + '>' }} {{ m['content'] }} <eot> {% endfor %}{% if add_generation_prompt %}<assistant> {% endif %}"
    cfg = LlamaConfig(
        vocab_size=len(vocab),
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=128,
        tie_word_embeddings=True,
        attention_dropout=0.0,
        eos_token_id=1,
        pad_token_id=0,
    )
    torch.manual_seed(123)
    state = LlamaForCausalLM(cfg).state_dict()

    def make_model():
        model = LlamaForCausalLM(cfg)
        model.load_state_dict(state)
        return model

    config = Config(
        device="cpu",
        dtype="float32",
        seeds=[42],
        nominal_epochs=2,
        logical_micro_batch=2,
        device_micro_batch=2,
        accumulation=2,
        warmup_steps=1,
        evaluation_every_updates=2,
        training_max_length=64,
        evaluation_max_length=24,
        evaluation_retain_count=3,
        generation_batch_size=2,
        legacy_greedy_tokens=4,
        long_audit_tokens=8,
        long_audit_per_split=2,
        lora_rank=2,
        lora_alpha=4,
        gradient_checkpointing=True,
        quantization_modes=[],
        bootstrap_repeats=100,
        output_root=str(tmp_path / "runs"),
    )

    def row(i):
        return {
            "question": f"q{i} ?",
            "answer": "A",
            "paraphrased_answer": "B",
            "perturbed_answer": ["C", "D"],
        }

    data = {
        "forget05": [row(i) for i in range(8)],
        "retain95": [row(i) for i in range(8, 56)],
        "forget05_perturbed": [row(i) for i in range(8)],
        "retain_perturbed": [row(i) for i in range(8, 12)],
        "real_authors_perturbed": [row(i) for i in range(56, 58)],
        "world_facts_perturbed": [row(i) for i in range(58, 60)],
    }
    return config, tokenizer, data, make_model
