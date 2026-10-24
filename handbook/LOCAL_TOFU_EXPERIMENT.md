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

The stages are full-TOFU SFT with FP16 LoRA, baseline evaluation, NPO plus retain
cross-entropy using cached frozen SFT answer log probabilities, adapter merge,
merged FP16 evaluation, NF4 quantization, and the same evaluation again.
Only LoRA parameters are optimized. This is a local adaptation, not the gated
Kaggle reproduction and not full-parameter training.

Outputs go into a new `runs/local_tofu_npo_*` directory: settings, environment,
dataset snapshots, SFT and NPO adapters, merged FP16 weights, quantized weights,
training histories, per-question evaluations, comparison.csv and
quantization_paired.csv. Training does not resume automatically after interruption.

To execute without a notebook editor, from the repository root:

```powershell
& C:/VENV/fyp/Scripts/python.exe -u -X utf8 scripts/run_local_tofu_notebook.py --smoke
& C:/VENV/fyp/Scripts/python.exe -u -X utf8 scripts/run_local_tofu_notebook.py
```

The first command is only a plumbing check: one training step per stage and two
evaluation rows per split. It cannot demonstrate forgetting. The second uses the
notebook's pilot settings and saves an executed notebook after each cell.
If training and merging have completed but evaluation was interrupted, resume
the saved run with `--resume-eval runs/local_tofu_npo_TIMESTAMP`. Include `--smoke`
only when resuming a smoke run. `execution_status.json` records current status;
`failure.json`, if present, preserves the most recent failure for diagnosis.

Start with the notebook defaults, then inspect baseline generations. Increase SFT
steps if it has not learned the forget facts. Evaluation is a pilot with 40 fixed
samples per split; increase EVAL_PER_SPLIT for a larger study. Forget01 contains
40 QA examples from two fictitious authors. Report the sample counts alongside
any result. NLL increasing means the target answers are less likely; inspect
retain utility to distinguish selective forgetting from general model damage.
Improved forget answer accuracy after quantization can indicate recovery.
The default 1500 SFT steps with accumulation 8 cover all 4000 training rows three
times, in shuffled order. The full run is substantially longer than the smoke test.

No binary deletion claim follows from these metrics. Official TOFU forget quality
requires additional evaluation and a retain-only retrained comparator.

Sources: https://locuslab.github.io/tofu/ ; https://arxiv.org/abs/2404.05868 ;
https://huggingface.co/docs/transformers/quantization/bitsandbytes
