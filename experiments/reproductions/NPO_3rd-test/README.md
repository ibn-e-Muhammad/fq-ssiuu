# NPO local third run — one self-contained Jupyter notebook

Open **[NPO_3RD_RUN.ipynb](NPO_3RD_RUN.ipynb)**. It contains the entire experiment:
configuration, input pins, model loading, NPO+RT, recovery checkpoints, tests,
evaluation, adapter merging, quantization, live output, matrices, plots, actual
answers, human-label summaries and optional second-run reanalysis.

**Only the notebook is needed as experiment code.** Copy it to the Windows PC;
no sibling `.py` files, `config.json`, repository import or subprocess launcher
is required. Downloaded model weights and the six pinned TOFU `.json` files
remain external inputs. The old modular implementation is preserved in Git at
`4d3089f`; it is superseded as the third-run entry point.

## Run it

1. Select the existing FYP CUDA notebook kernel that completed run 2. Keep its
   working PyTorch/CUDA stack. The environment cell lists dependencies; inspect
   installations before changing that environment.
2. Edit **Your settings** near the top: `full_path`, `retain_path`, `data_path`
   and `output_root`. Leave `tokenizer_path` empty to use the tokenizer beside
   the full model's weights. The full path must be the **TOFU full checkpoint**,
   not the original Meta base model.
3. Save the notebook, restart the kernel and **Run All**. Embedded tests run
   first and failures block real work. `RUN_EXPERIMENT=True` starts the complete
   three-seed GPU experiment; set it false for definitions/tests only.
4. Follow logs, the updating loss plot and the stage table beneath the run cell.
   Each seed's final metrics display as it completes. Later cells expose every
   CSV matrix, complete answers and generation-cap statistics.

`ANSWER_SPLIT`, `ANSWER_PROFILE` and `ANSWER_LIMIT` select which saved actual
answers to display. Set `ANSWER_LIMIT=None` for all questions in the selected
split. All text is HTML escaped and displayed without string truncation.
The default shows ten forget questions to keep notebook output manageable.

Reserve at least **12 GiB free on the output SSD**. The preflight checks native
BF16, VRAM, RAM, disk and the quantization backend. The real 1B Windows runtime,
resource use and effectiveness remain pending until measured.

## Resume, logs and identity

After an interruption, set `RESUME_RUN` to the exact printed run directory, save,
restart and Run All with the same scientific settings, core cells and packages.
Do not manually skip/replace training updates. Small adapter/optimizer/RNG
checkpoints are saved every update; the latest two complete generations are kept.
The notebook retries interrupted scheduled evaluation and verifies completed
output hashes before reuse.

The implementation identity hashes **saved core-cell sources**, not a repository
script or notebook outputs. Core cells must be saved before execution; unsaved
scientific edits and changed saved source are rejected. Saving outputs/metadata
alone does not change the identity. A modular run from `4d3089f` has a different
implementation identity and needs that historical code to resume.

Terminal text remains visible in Jupyter and is also copied to
`output_root/notebook_logs/`. Tests write their own per-case logs and validation
receipt under `validation/`. Final run records also include
`notebook_validation.json` and `notebook_session.json`.

The method remains the researcher-authorized class C LoRA adaptation in
[DL-024](../../../handbook/decisions/DL-024_project_lora_training.md).
Consolidation changes packaging/observation, not the declared scientific method.

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

## Optional analysis in the same notebook

The final cells also include:

- **Run-2 reanalysis:** fill `RUN2_INPUT` and `RUN2_DERIVED_OUTPUT`. No training or
  inference is performed. Original hashes are checked and original outputs stay
  untouched. Matched geometric FQ, directional CIs, utility and termination
  tables are regenerated outside the archived run.
- **Human audit:** fill real category/reviewer labels in the generated template,
  then set `HUMAN_LABELS_CSV`. The notebook supplies the rubric and calculates
  rates among reviewed answers with coverage. It never invents semantic labels.

## Validation and remaining scope

[Standalone notebook validation](../../../handbook/evidence/npo_run3/20261103_notebook_validation/README.md)
executes a copy in an empty directory without local code/config files. Its
embedded tests exercise formula/gradient parity, frozen reference, sampling,
logical-batch weighting, exact interruption/resume, corruption fallback, complete
tiny-Llama reports, merge/reload, live views/log capture and source-identity checks.
Actual CUDA LoRA backward/INT8/NF4 is a separately identified optional GPU test.
IPython history registration and all three answer-profile displays are checked.
The notebook's reanalysis reproduces all five existing scientific CSVs exactly.

The older modular acceptance receipt (238 passing NPO/QRA tests, three GPU skips)
remains historical evidence for commit `4d3089f`, not a new 1B run. No real
third-run 1B result is prefilled. Three training seeds, full periodic generation
and actual GPU resource measurements still need execution on the target PC.

Min-k20 token scores alone are not a proper MIA without matched nonmembers.
FADE, another quantizer family, relearning and weight/bin mechanisms remain
explicit later studies. Semantic rates need actual human labels. No attack or
durability verdict is inferred from a KS p-value or unrun study.
