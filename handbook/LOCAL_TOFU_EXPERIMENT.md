# Local TOFU experiment

Open `Mother_Local_v2.ipynb` at the repository root. The previous notebook is preserved
as `notebooks/Mother_Local_v2.before_local_pipeline.ipynb`.

Use the existing `C:/VENV/fyp/Scripts/python.exe` environment provided by the user.
Do not use the repository's `.venv`. No installation is needed if the existing
environment passes the notebook imports.

```powershell
C:/VENV/fyp/Scripts/python.exe -c "import torch; print(torch.cuda.is_available())"
```

Select `C:/VENV/fyp/Scripts/python.exe` as the notebook kernel in your editor.
Restart any previous notebook kernels holding GPU memory before running all cells.
The notebook checks for at least 6 GiB free GPU memory before loading models.
The local model is loaded only from `C:/Model/Llama-3.2-1B-Instruct`.
TOFU downloads from Hugging Face on first use and each run saves dataset snapshots.
No model access token is needed for the local model or public TOFU dataset.

The stages are full-TOFU SFT with FP16 LoRA, saving and reloading a complete SFT
model, baseline evaluation, NPO plus retain
cross-entropy using cached frozen SFT answer log probabilities, adapter merge,
merged FP16 evaluation, NF4 quantization, and the same evaluation again.
Only LoRA parameters are optimized. This is a local adaptation, not the gated
Kaggle reproduction and not full-parameter training.

Outputs go into a new `runs/local_tofu_npo_*` directory: settings, environment,
dataset snapshots, SFT and NPO adapters, a complete fine-tuned model in
`sft_full_fp16/`, unlearned FP16 weights in `merged_fp16/`, NF4 weights in
`quantized_nf4/`,
training histories, per-question evaluations, comparison.csv and
quantization_paired.csv. Training does not resume automatically after interruption.

The SFT folder contains the full merged model weights, model config, tokenizer,
and `sft_training.json` with dataset identity and completed epoch counts. It loads
directly with `AutoModelForCausalLM.from_pretrained(...)`; the original base and
SFT adapter folders are not needed to load it. NPO reloads this folder as both its
frozen reference and starting policy, then trains a fresh LoRA adapter. The NPO
adapter is merged into this SFT base and saved separately before quantization.
Unlearning never saves over `sft_full_fp16/`.

To reuse a trained checkpoint, set this in notebook section 1, then run all cells:

```python
REUSE_SFT_MODEL = Path(r'C:\MY-FYP\fq-ssiuu\runs\YOUR_RUN\sft_full_fp16')
```

This skips SFT, checks the saved dataset identity, and puts the new unlearning
outputs in a fresh run directory. A smoke checkpoint cannot be reused as a
completed SFT model. The equivalent command-line option is
`--reuse-sft-model runs/YOUR_RUN/sft_full_fp16`.

To execute without a notebook editor, from the repository root:

```powershell
& C:/VENV/fyp/Scripts/python.exe -u -X utf8 scripts/run_local_tofu_notebook.py --smoke
& C:/VENV/fyp/Scripts/python.exe -u -X utf8 scripts/run_local_tofu_notebook.py
```

The first command is only a plumbing check: one training step per stage and two
evaluation rows per split. It cannot demonstrate forgetting. The second uses the
notebook's pilot settings and saves an executed notebook after each cell.
If training and merging have completed but evaluation was interrupted, resume
the saved run with `--resume-eval runs/local_tofu_npo_TIMESTAMP`. Evaluation samples
and settings are restored from that run. `execution_status.json` records current status;
`failure.json`, if present, preserves the most recent failure for diagnosis.

Start with the notebook defaults, then inspect baseline generations. Increase SFT
epochs if it has not learned the forget facts. Evaluation is a pilot with 40 fixed
samples per split; increase EVAL_PER_SPLIT for a larger study. Forget01 contains
40 QA examples from two fictitious authors. Report the sample counts alongside
any result. NLL increasing means the target answers are less likely; inspect
retain utility to distinguish selective forgetting from general model damage.
Improved forget answer accuracy after quantization can indicate recovery.
`SFT_EPOCHS = 3` visits all 4000 training rows three times in shuffled order,
including the last partial accumulation batch if the batch size changes. With
accumulation 8 this is 1500 optimizer steps. The full run is substantially longer
than the smoke test.

No binary deletion claim follows from these metrics. Official TOFU forget quality
requires additional evaluation and a retain-only retrained comparator.

Sources: https://locuslab.github.io/tofu/ ; https://arxiv.org/abs/2404.05868 ;
https://huggingface.co/docs/transformers/quantization/bitsandbytes
