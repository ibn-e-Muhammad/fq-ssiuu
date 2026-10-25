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
training histories, per-question evaluations, and an `interpretation/` report folder.
SFT saves recovery checkpoints every 50 optimizer steps and at the final step.
The newest two complete saves are retained in each run's `sft_checkpoints/` folder.
These contain the LoRA adapter, tokenizer, optimizer, cosine scheduler, random
states, shuffled data order, and next data position. They are different from the
finished, merged `sft_full_fp16/` model. NPO training does not yet have recovery saves.

After a power cut, set these in notebook configuration, restart the kernel and run
the configuration, dataset, helpers, and SFT cells again:

```python
REUSE_SFT_MODEL = None
SFT_RESUME_FROM = Path(r'C:\MY-FYP\fq-ssiuu\runs\YOUR_INTERRUPTED_RUN')
SFT_SAVE_EVERY = 50
SFT_KEEP_CHECKPOINTS = 2
```

The newest intact checkpoint is selected; incomplete writes are ignored and a
damaged latest checkpoint falls back to the previous intact one. An explicit
`sft_checkpoints/step-00000050` path selects only that checkpoint. Resume writes
to a new run and preserves the source run. It also copies the loaded recovery
point into the new run. After another interruption, use that newest run folder.
Keep the original training settings: data, encoded tokens, base-model path,
LoRA settings, epochs, accumulation, learning rate and scheduler must match.
The original base-model files must remain unchanged. The normal save interval
can lose up to 49 completed updates plus in-progress work; a failure during a
save or a corrupt save can require falling back further. Files are flushed and
checksummed before the checkpoint directory is published, but disk failure can
still destroy data. Old runs without recovery checkpoints cannot resume SFT.

To retrain from scratch, set both `SFT_RESUME_FROM` and `REUSE_SFT_MODEL` to `None`
and rerun configuration to create a new run. Existing runs are preserved.
Equivalent CLI options (followed by the rest of the experiment):

```powershell
& C:/VENV/fyp/Scripts/python.exe -u scripts/run_local_tofu_notebook.py --resume-sft runs/YOUR_INTERRUPTED_RUN
& C:/VENV/fyp/Scripts/python.exe -u scripts/run_local_tofu_notebook.py --fresh-sft
```

Notebook edits do not modify a currently executing cell. The recovery mechanism
takes effect only when the updated code is executed. The notebook remains
standalone; `scripts/sft_checkpoint_cell.py` is its editable source template and
`scripts/install_sft_recovery.py` embeds it without changing user training settings.

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
epochs if it has not learned the forget facts. The notebook now defaults to
`FORGET_SPLIT = 'forget05'`, automatically paired with `retain95`. Switching to
`forget01` or `forget10` automatically selects `retain99` or `retain90`.
Every forget question is evaluated by default (`EVAL_FORGET_LIMIT = None`):
40 for forget01, 200 for forget05, or 400 for forget10. Retain and world-facts
utility use fixed samples controlled by `EVAL_PER_SPLIT` (40 by default).
The same questions, reference answers, and decoding settings are used at all
three model stages. NLL increasing means the target answers are less likely; inspect
retain utility to distinguish selective forgetting from general model damage.
Improved forget answer accuracy after quantization can indicate recovery.
`SFT_EPOCHS` visits all 4000 training rows that many times in shuffled order,
including the last partial accumulation batch if the batch size changes. The
user's five-epoch settings, learning rate, scheduler and LoRA configuration are
preserved. The full run is substantially longer than the smoke test.

## Interpretable results

The final notebook cell embeds all report code, so the notebook stays standalone.
It produces `interpretation/report.html` (searchable side-by-side answers),
`RESULTS.md` (plain-language findings), `summary.json`, `stage_comparison.csv`,
and `question_comparison.csv` in the same folder. Reports do not need model loading.
For a completed run:

```powershell
& C:/VENV/fyp/Scripts/python.exe scripts/run_local_tofu_notebook.py --report-only runs/YOUR_RUN
```

Exact-answer recovery is counted only on questions with the sequence:
exact at SFT -> not exact after NPO -> exact after NF4. The denominator is the
number of previously exact answers lost after NPO, not every forget question.
If no answers qualify, the recovery percentage is unavailable, not zero.

The report separately gives how much of the mean NLL increase and F1 decrease
quantization reversed. These are signed score reversals, not percentages of facts
recovered. F1 threshold tables are exploratory word-overlap flags, not semantic
correctness judgments. Check factual details in the generated answers and retained
utility before interpreting the experiment as selective forgetting.

The completed run `local_tofu_npo_20261024_123755_749521` used forget01/retain99,
three SFT epochs, and the earlier LoRA settings. Its report describes that run;
changing the current notebook to forget05 does not relabel historical results.

No binary deletion claim follows from these metrics. Official TOFU forget quality
requires additional evaluation and a retain-only retrained comparator.

Sources: https://locuslab.github.io/tofu/ ; https://arxiv.org/abs/2404.05868 ;
https://huggingface.co/docs/transformers/quantization/bitsandbytes
