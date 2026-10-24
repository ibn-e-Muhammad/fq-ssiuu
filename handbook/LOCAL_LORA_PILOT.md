# Local NPO+LoRA exploratory pilot

This is an optional, experimental local route for a single CUDA GPU. It does not
change the Kaggle/T4 D1 runner or its frozen configs. It keeps the pinned 1B
checkpoint, TOFU splits, preprocessing boundary, and NPO+retain loss, but changes
full-parameter optimization to LoRA, uses one example per GPU pass with gradient
accumulation, and stores frozen-reference NLLs on CPU. Therefore, it is not a
source-faithful D1 reproduction and cannot establish unlearning effectiveness or
quantisation durability.

## Windows setup

Activate the existing project virtual environment in PowerShell from the
repository root:

```powershell
& "C:\path\to\your\venv\Scripts\Activate.ps1"
```

Check the selected interpreter, CUDA-enabled PyTorch, and GPU:

```powershell
python -c "import sys, torch; print(sys.executable); print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA GPU')"
```

Check that the active venv has the runtime packages:

```powershell
python -m pip show torch transformers peft huggingface-hub numpy
```

If they are already installed, register the checkout in the venv without allowing
pip to replace or upgrade its CUDA-enabled PyTorch:

```powershell
python -m pip install -e . --no-deps
```

If any non-PyTorch runtime dependency is missing, install it into this venv, then
register the project with `--no-deps` as above. Do not install or upgrade PyTorch
blindly; use the official PyTorch selector if you need a CUDA-enabled wheel, and
repeat the CUDA check before continuing.

## Run

From the repository root, use a new output directory for each run:

```powershell
python -m experiments.reproductions.NPO.local_lora_pilot `
  --output results\local_lora_pilot_001
```

The runner downloads the hash-pinned public TOFU files, tokenizer and model
checkpoint on first use. The model weights are several gigabytes; allow enough
disk space and keep the PC online. Re-running with the same cache reuses verified
assets. No Kaggle, DeepSpeed, or Hugging Face token is required for these public
inputs.

Defaults are capped at 62 optimizer steps, 500 tokens, one example per GPU pass,
accumulation 4, LoRA rank 8, and 16 descriptive evaluation examples per split.
The local optimizer uses a constant learning rate, unlike the frozen D1 schedule.
The completed adapter and logs are saved under the new output directory. To first
smoke-test fewer updates, use a different new directory and add `--max-steps 1`.

`RESULT.json`, `evaluation.json`, and `training.jsonl` describe the local run; they
are not scientific pass/fail or recovery/durability verdicts. Preserve the run
settings and logs when comparing experiments. The accepted source-faithful D1
pilot remains blocked by a backward-pass out-of-memory error on a 16 GB Kaggle
T4; this local LoRA route does not resolve that result.
