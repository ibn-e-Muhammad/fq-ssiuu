# NPO local third run: LoRA NPO+RT and quantization recovery

Researcher decision: 2026-11-03, [DL-024](../../../handbook/decisions/DL-024_project_lora_training.md).
This is a **LoRA adaptation** of NPO+RT with source-grounded evaluation. The
low-rank trainable subspace and fresh retain sampler differ from the original
full-weight author recipe. Neither change is labeled an exact reproduction.
The second run and its negative quantization observations remain intact.

## Run on the existing Windows environment

Use the existing CUDA-enabled environment that completed run 2. Do not reinstall
or downgrade PyTorch/CUDA as part of this experiment. Add PEFT if it is absent:

```powershell
C:\VENV\fyp\Scripts\python.exe -m pip install peft pytest
Set-Location C:\FYP\fq-ssiuu\experiments\reproductions\NPO_3rd-test
C:\VENV\fyp\Scripts\python.exe -m pytest tests -q
C:\VENV\fyp\Scripts\python.exe run_experiment.py --config config.json
```

Reserve at least **12 GiB free on the output SSD** for three merged 1B checkpoints, adapter state and results. The preflight checks native BF16, available VRAM/RAM and disk space.

`requirements.txt` lists compatibility ranges, not a universally validated lock.
For missing dependencies, inspect `python -m pip install --dry-run -r requirements.txt`
before changing the environment; keep its working CUDA PyTorch build.

Edit `config.json` first. `full_path` must point to the downloaded
`open-unlearning/tofu_Llama-3.2-1B-Instruct_full`; `retain_path` points to
`open-unlearning/tofu_Llama-3.2-1B-Instruct_retain95`. Leave `tokenizer_path` empty
to use the tokenizer in the full-model folder. Set `data_path` to the folder
holding the six original pinned TOFU JSONL files (their extension is `.json`).
The archive under run 2 contains receipts but no model tensors: its `Model`
folders are not a substitute for the actual weights on the Windows PC.

`NPO_3RD_RUN.ipynb` launches this same code with the selected notebook kernel and
displays the saved CSV/HTML reports. There is no separately maintained training
implementation inside notebook cells.

Resume an interrupted run with unchanged scientific settings/code/packages:

```powershell
C:\VENV\fyp\Scripts\python.exe run_experiment.py --config config.json --resume runs\RUN_ID
```

Every optimizer update writes a small checksummed adapter/AdamW/RNG checkpoint.
The latest two complete generations are retained; incomplete generations cannot
become authoritative. A failed scheduled evaluation is retried on resume.
Completed identity-bound evaluation outputs are verified before reuse.

## Declared configuration

| Item | Default |
|---|---|
| Target / diagnostic oracle | Pinned TOFU Llama-3.2-1B full / retain95 |
| Forget / available retain | All 200 / all 3,800 source rows |
| Loss | Sequence-sum causal NPO Eq. 3 + token-mean retain CE |
| Beta, coefficients | 0.1, 1.0 + 1.0 |
| Training | Unquantized BF16 backbone; only FP32 LoRA parameters train |
| LoRA | rank 16, alpha 32, dropout 0, bias none; seven attention/MLP projections |
| Optimizer | Ordinary AdamW over adapters; FP32 states; no clipping |
| LR / weight decay | 1e-5 / 0.01; no unregistered LoRA LR tuning |
| Logical / device microbatch | 4 / 4; accumulation 8; effective batch 32 |
| Stop / warmup | 62 updates (9.92 actual forget passes), 6 warmup updates |
| Training seeds | 42, 43, 44, independently initialized and trained |
| Retain sampling | Seeded reshuffled stream without replacement until exhausted |
| Expected retain coverage | 1,984 distinct rows in 62 updates, not all 3,800 |
| Source-sampling control | `retain_sampling="source_fixed"` reproduces the 195-row limitation |
| Frozen reference | Same frozen full backbone with adapters disabled, eval, no_grad |
| Training evaluation | Updates 6,12,...,60,62; actual data-pass fraction is reported |
| Retain diagnostics | Seeded random 300 of eligible 400, frozen across seeds/stages |
| Primary generation | Sampled, total length 200, batch 30, seed 0 per batch |
| Llama-3 generation adapter | temperature 0.6, top-p 0.9, recorded explicitly |
| Legacy generation | Greedy, 96 new tokens, batch 1, every final-stage question |
| Long generation audit | Greedy 256 new tokens, frozen 20 per component |

The original author code uses index-seeded fixed retain pairs. The supplied audit
identifies their limited coverage correctly, but describing that pairing as a
deviation from the pinned code is inaccurate. The new reshuffled sampler is an
explicit project choice, not a silently repaired upstream bug. It draws 1,984
distinct rows because the source stop processes 1,984 pairs; claiming all 3,800
were trained would be incorrect. Changing both LoRA and sampling prevents causal
attribution of any improvement to one change alone. Use the source-fixed option
as a controlled LoRA comparison if that attribution is needed.

## Evaluation and deployment order

1. Evaluate full and retain95 BF16 and their actual INT8/NF4 controls once.
2. For each seed, train LoRA NPO+RT with periodic complete TOFU evaluation.
3. Evaluate the **unmerged adapter model**.
4. Merge adapters into the original unquantized backbone in FP32, verify merge
   probes, cast to BF16, save and reload the deployment checkpoint.
5. Evaluate merged BF16, then independently load INT8 and NF4 from that same
   merged checkpoint. Do not attach a full-precision adapter to a quantized base
   and call it quantization of the unlearned model.
6. Report adapter-to-merge drift separately from BF16-to-quantization changes.

The online reference adds no second training-model copy: disabling LoRA exposes
the unchanged frozen full backbone. The final backbone hash must match its
pretraining-loop hash. Model/reference scoring and gradients are separate.

The unmerged intervention suppresses answers conditionally: disabling/removing
its adapter restores the original base. The evaluated deployment artifact is the
merged checkpoint; no original-base erasure is claimed.

The metrics intentionally follow TOFU's distinct implementations:

- **Forget Quality:** geometric correct-vs-perturbed preference + KS. Both the
  fixed BF16 retain95 oracle and same-precision retain95 oracle are reported.
  Arithmetic-ratio KS remains a separately named paper-text diagnostic.
- **Model Utility:** arithmetic perturbed-probability ratio, transformed TOFU
  truth score, and the harmonic mean of nine probability/ROUGE/truth components.
- **Directional recovery:** paired change in log geometric target preference,
  probability and generated-answer overlap; 2,000 question and author-block
  bootstrap repetitions. Positive preference change points toward recovery;
  p-value changes are not recovery magnitudes.
- **KL:** original full-reference to candidate across all 200 evaluation
  positions, including prompt/padding as in the source diagnostic.
- **Generation:** actual text, token IDs, EOS termination, cap hits, length,
  ROUGE-L recall, token F1 and exact match.

Primary metric source: [TOFU utility/FQ implementation](https://github.com/locuslab/tofu/blob/main/utils.py).
The paper's written arithmetic ratio and the geometric FQ implementation must
remain distinguishable. [NPO Eq. 3](https://arxiv.org/html/2404.05868v2) and
[PEFT LoRA documentation](https://huggingface.co/docs/peft/developer_guides/lora).
Generated text is untrusted content and is HTML-escaped in reports.
The sampled and long-audit profiles honor all three stop IDs from the pinned
full-model generation configuration; legacy greedy preserves the old EOS list.
Every generated row records its exact stop IDs, so this profile difference is
visible rather than silently attributed to the token budget.

## Output files

`runs/RUN_ID/controls/` contains the six shared controls. Each `seed_N/` contains:

- `training_plan.json`, `training_identity.json`, `training_log.csv`, recovery
  `checkpoints/`, final `adapter/`, and `training_complete.json`;
- `epoch_evaluations/update_*/` and `epoch_metrics.csv`;
- `npo_adapter/`, `npo_bf16/`, `npo_int8/`, `npo_nf4/`: per-question JSONL/CSV,
  summaries, nine utility components, model/quantizer identities and packed-state
  hashes; separate `legacy_greedy96/` and `long_greedy/` generation profiles;
- `comparison_matrix.csv`, `forget_quality_fixed_and_matched.csv`,
  `paired_directional_recovery.csv`, `merge_drift.csv`, `comparison.png`,
  `report.html`, `answers_side_by_side.csv/.html`, `all_per_question.csv`;
- `human_audit_template.csv`, [human rubric](HUMAN_AUDIT.md), and
  `evaluation_coverage.json` with measured/missing fields.

The root adds `all_seeds_metrics.csv`, `cross_seed_mean_std.csv` and seed coverage.
Evaluated epoch adapters are also retained with their own hashes, enabling later
checkpoint inspection without preserving every epoch's full optimizer state.
Only a successful full run writes `run_complete.json`. Interrupted/failed runs
stay visible; no checkpoint is selected for deployment by best forget score.
Epoch trajectories permit comparison with paper plots without test-set tuning.

## Reanalyse run 2 without inference

```bash
python reanalyse_second_run.py ../NPO_2nd-test/npo_results/20261031_092340_493700 \
  --output-dir ../../../handbook/evidence/npo_run2/20261103_audit/reanalysis
```

This verifies the 27 recorded stage artifact hashes and produces corrected
geometric matched-precision FQ, paired directional CIs, utility and provenance.
Original files are never rewritten. Author-block intervals use the pinned TOFU
20-question ordering; the historical subset is unchanged. New random evaluation
sampling and sampled decoding mean run-2 and run-3 numbers are not directly paired.

## Acceptance and remaining scope

CPU tests exercise scalar/gradient parity, frozen backbone/reference, microbatch
equivalence, sampler coverage, KS orientation, arithmetic utility, paired CIs,
checkpoint interruption/corruption recovery, merge/reload, full tiny-model
evaluation/report generation and reuse protection. Optional CUDA tests exercise
real LoRA backward and real INT8/NF4 conversion on a tiny model. Neither is a
substitute for measuring the actual 1B Windows run.

Min-k20 answer-token scores are exported for future analysis, but a proper MIA
also needs declared matched nonmembers, AUROC/TPR definitions and leakage checks.
FADE, a second quantizer family, relearning, and quantization-bin mechanisms are
explicitly pending. Human semantic rates require real labels. Three seeds are a
resource-bounded study, not the paper's five-seed replication. No QRA success or
unlearning durability verdict is generated; existing scientific gates stay open.

Run tests before GPU work and preserve their output:

```bash
python -m pytest tests -q --junitxml=validation.xml
```

Record the actual environment instead of treating run-2's package versions or
small CPU checks as proof of compatibility on another PC. LoRA removes the old
full-weight disk optimizer, but complete periodic generation remains substantial
work. The 1B memory/runtime result is pending until measured.

The checked implementation has a [dated validation receipt](../../../handbook/evidence/npo_run3/20261103_cpu_validation/README.md):
238 NPO/LoRA/QRA tests passed, three CUDA-only tests skipped, and the complete
pinned-tokenizer preflight passed. This does not replace the target-PC tests.
