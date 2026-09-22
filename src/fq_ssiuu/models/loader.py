"""
src/fq_ssiuu/models/loader.py

Model loading utilities for FQ-SSIUU.

This module provides all model loading, LoRA wrapping, and adapter merging
functions used throughout the project. Every experiment calls at least one
function from this module.

Supported loading modes:
  - NF4 quantised (4-bit) via BitsAndBytesConfig — used for training (16 GB VRAM)
  - FP16 full-precision — used for QRA evaluation after merge
  - LoRA adapter wrapping via peft — used for parameter-efficient unlearning
  - Adapter merging via merge_and_unload() — used for deployment-precision evaluation

Dependencies (installed via scripts/setup_env.sh):
  - torch >= 2.4.1
  - transformers >= 4.46.0
  - bitsandbytes >= 0.42.0
  - peft >= 0.12.0

Authentication:
  Legacy loader functions use the lazy project secret provider. Missing or
  placeholder credentials raise SecretNotFoundError; no values are logged.
  The NPO candidate has a separate immutable, unquantized reference contract.
"""

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Legacy model-loading configuration; not the immutable NPO reference API.

    Attributes:
        base_model_id: HuggingFace model identifier.
        load_in_4bit: Whether to load with NF4 quantisation.
        bnb_4bit_quant_type: Quantisation type for BitsAndBytesConfig.
        bnb_4bit_compute_dtype: Compute dtype string (converted to torch dtype internally).
        bnb_4bit_use_double_quant: Enable double quantisation for memory savings.
        torch_dtype: Model weight dtype string (converted to torch dtype internally).
        device_map: Device mapping strategy for model parallelism.
    """

    base_model_id: str = "meta-llama/Llama-3.2-1B-Instruct"
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True
    torch_dtype: str = "float16"
    device_map: dict = field(default_factory=lambda: {"": 0})
    # Historical LoRA defaults; accepted reproductions use versioned method specs.
    lora_r: int = 16
    lora_alpha: int = 32
    lora_target_modules: list = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    lora_task_type: str = "CAUSAL_LM"


def _resolve_dtype(dtype_str: str):
    """Convert a string dtype to a torch dtype object.

    Args:
        dtype_str: One of 'bfloat16', 'float16', 'float32'.

    Returns:
        The corresponding torch.dtype.

    Raises:
        ValueError: If dtype_str is not a recognised dtype.
    """
    import torch

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_str not in dtype_map:
        raise ValueError(
            f"Unsupported dtype '{dtype_str}'. Must be one of: {list(dtype_map.keys())}"
        )
    return dtype_map[dtype_str]


def _get_hf_token() -> str:
    """Return HF_TOKEN using project precedence; raise SecretNotFoundError if unusable."""
    from fq_ssiuu.secrets import SecretNotFoundError, get_secret

    try:
        token = get_secret("HF_TOKEN")
        if token == "PENDING":
            raise SecretNotFoundError("Placeholder is not a usable credential")
    except SecretNotFoundError:
        raise SecretNotFoundError(
            "HF_TOKEN is unavailable. Configure it in the process environment, "
            "an enabled Kaggle Secret (individual or FQ_SSIUU_DOTENV), or the ignored "
            "local .env through the project secret loader. See .env.example; "
            "never put credentials in notebook cells or artifacts."
        ) from None
    return token


def load_base_model(config: ModelConfig | None = None):
    """Load the base model with NF4 quantisation via BitsAndBytesConfig.

    This is the primary loading function for training. It loads the model
    in 4-bit NF4 quantisation to fit within 16 GB VRAM.

    Args:
        config: Model configuration. Uses defaults from ModelConfig if None.

    Returns:
        AutoModelForCausalLM: The quantised model on GPU.

    Raises:
        SecretNotFoundError: If the project secret provider cannot resolve HF_TOKEN.
        Other model-loading exceptions propagate from Transformers.
        ValueError: If config contains invalid parameters.
    """
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    if config is None:
        config = ModelConfig()

    token = _get_hf_token()

    logger.info("Loading model '%s' with NF4 quantisation...", config.base_model_id)

    compute_dtype = _resolve_dtype(config.bnb_4bit_compute_dtype)
    model_dtype = _resolve_dtype(config.torch_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_id,
        quantization_config=bnb_config,
        torch_dtype=model_dtype,
        device_map=config.device_map,
        token=token,
        attn_implementation="sdpa",
    )

    # Log memory usage for VRAM budget tracking
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated() / (1024**3)
        mem_reserved = torch.cuda.memory_reserved() / (1024**3)
        logger.info(
            "Model loaded. GPU memory: %.2f GiB allocated, %.2f GiB reserved",
            mem_allocated,
            mem_reserved,
        )
    else:
        logger.info("Model loaded (no CUDA — running on CPU).")

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Total parameters: %s", f"{n_params:,}")

    return model


def load_model_fp16(model_id: str, token: str | None = None):
    """Load a model in full FP16 precision (no quantisation).

    Used for QRA evaluation after adapter merging. The merged model
    is evaluated at FP16, INT8, and INT4 precisions to compute Q-INT4.

    Args:
        model_id: HuggingFace model ID or local path to merged model.
        token: HuggingFace token. If None, uses the lazy project secret provider.

    Returns:
        AutoModelForCausalLM: The FP16 model.

    Raises:
        SecretNotFoundError: If no token was passed and the project provider cannot resolve it.
    """
    import torch
    from transformers import AutoModelForCausalLM

    if token is None:
        token = _get_hf_token()

    logger.info("Loading model '%s' in FP16 precision...", model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map={"": 0},
        token=token,
        attn_implementation="sdpa",
    )

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("FP16 model loaded. Parameters: %s", f"{n_params:,}")

    return model


def wrap_with_lora(model, lora_config_dict: dict | None = None):
    """Apply LoRA adapters to the model for parameter-efficient fine-tuning.

    Uses peft's get_peft_model to wrap the base model with LoRA adapters
    on the attention projection layers (q_proj, v_proj, k_proj, o_proj).

    Args:
        model: The base model (AutoModelForCausalLM), typically NF4-quantised.
        lora_config_dict: Optional dict of LoRA parameters. If None, uses
            ModelConfig defaults matching configs/base.yaml.

    Returns:
        PeftModel: The model with LoRA adapters attached.

    Raises:
        ValueError: If lora_config_dict contains invalid keys.
    """
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    defaults = ModelConfig()
    cfg = lora_config_dict or {}

    lora_config = LoraConfig(
        r=cfg.get("r", defaults.lora_r),
        lora_alpha=cfg.get("lora_alpha", defaults.lora_alpha),
        target_modules=cfg.get("target_modules", defaults.lora_target_modules),
        lora_dropout=cfg.get("lora_dropout", defaults.lora_dropout),
        bias=cfg.get("bias", defaults.lora_bias),
        task_type=cfg.get("task_type", defaults.lora_task_type),
    )

    # Prepare model for k-bit training (casts layer norms to float32, etc.)
    model = prepare_model_for_kbit_training(model)

    peft_model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    logger.info(
        "LoRA applied. Trainable: %s / %s (%.2f%%)",
        f"{trainable:,}",
        f"{total:,}",
        100.0 * trainable / total,
    )

    return peft_model


def get_reference_model(model_id: str, config: ModelConfig | None = None):
    """Retired unpinned reference entry point; never inherit generic NF4 defaults."""
    raise ValueError(
        "Unpinned or quantized NPO references are forbidden. Use the immutable "
        "NPO ReferenceConfig and load_reference in experiments/reproductions/NPO/reference.py."
    )


def merge_adapter(peft_model_path: str, base_model_id: str, token: str | None = None):
    """Merge a trained LoRA adapter into the base model.

    Loads the PEFT adapter from disk, merges it into the base model using
    merge_and_unload(), and returns a plain AutoModelForCausalLM. The merged
    model is suitable for QRA evaluation at different precisions.

    Args:
        peft_model_path: Path to the saved PEFT adapter directory.
        base_model_id: HuggingFace model ID of the base model.
        token: HuggingFace token. Uses the lazy project secret provider if None.

    Returns:
        AutoModelForCausalLM: The merged FP16 model (no adapters).

    Raises:
        FileNotFoundError: If peft_model_path does not exist.
        RuntimeError: If merge fails.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    if token is None:
        token = _get_hf_token()

    if not os.path.isdir(peft_model_path):
        raise FileNotFoundError(
            f"PEFT adapter directory not found: {peft_model_path}. "
            "Ensure the adapter was saved with peft_model.save_pretrained()."
        )

    logger.info("Loading base model '%s' for merge...", base_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map={"": 0},
        token=token,
        attn_implementation="sdpa",
    )

    logger.info("Loading PEFT adapter from '%s'...", peft_model_path)
    peft_model = PeftModel.from_pretrained(base_model, peft_model_path)

    logger.info("Merging adapter into base model...")
    merged_model = peft_model.merge_and_unload()

    n_params = sum(p.numel() for p in merged_model.parameters())
    logger.info("Merge complete. Merged model parameters: %s", f"{n_params:,}")

    return merged_model
