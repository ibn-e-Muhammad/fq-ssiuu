"""Document Experiment 01 and freeze inputs for the NF4-only control."""
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
from statistics import mean

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'runs/local_tofu_npo_20261025_150047_128001'
settings=json.loads((SOURCE/'settings.json').read_text(encoding='utf-8'))
SFT=Path(settings['sft_model_dir'])
training=json.loads((SFT/'sft_training.json').read_text(encoding='utf-8'))
environment=json.loads((SOURCE/'environment.json').read_text(encoding='utf-8'))
nb=json.loads((ROOT/'Mother_Local_v2.ipynb').read_text(encoding='utf-8'))
source=lambda i: ''.join(nb['cells'][i]['source'])
stages=('sft','npo_fp16','npo_merged_nf4')
records={s:list(csv.DictReader((SOURCE/f'eval_{s}.csv').open(encoding='utf-8'))) for s in stages}
expected={(r['split'],r['question']):r['reference'] for r in records['sft']}
for rows in records.values():
    assert len(rows)==len(expected)==280
    assert {(r['split'],r['question']):r['reference'] for r in rows}==expected

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

table=[]
for split in ('forget05','retain95','world_facts'):
    for stage in stages:
        rows=[r for r in records[stage] if r['split']==split]
        table.append(dict(split=split,stage=stage,count=len(rows),**{m:mean(float(r[m]) for r in rows) for m in ('exact_match','token_f1','answer_nll_per_token')}))

readme=f'''# Experiment 01 — TOFU SFT, NPO + retain loss, and NF4

Recorded on 2026-10-25. This document describes completed evaluation run
`{SOURCE.name}`, using the SFT checkpoint from `{SFT.parent.name}`.
The earlier NPO evaluation in the SFT run is a different run and is not mixed into these results.

## Objective and outcome

Test whether quantization makes knowledge suppressed by NPO accessible again.
The desired observation is a specific fact correct after SFT, incorrect after NPO,
and correct again after quantization. Recovery is a hypothesis, not an expected
requirement or a reason to discard a negative result.

SFT learned nearly all evaluated TOFU answers. NPO strongly reduced both forget
and retain metrics, indicating poor selectivity. NF4 worsened mean forget/retain
F1 and likelihood relative to unlearned FP16. Strict exact-answer recovery was
0/197 previously exact forget answers lost after NPO. This does NOT mean zero
factual recovery: individual birthplace/occupation facts returned, and the current
exact-match metric rejects correct paraphrases.

## Model and environment

| Setting | Value |
|---|---|
| Base model | Local Llama-3.2-1B-Instruct |
| Original path | `C:/Model/Llama-3.2-1B-Instruct` |
| SFT checkpoint | `{SFT.as_posix()}` |
| GPU | NVIDIA RTX 3060, 12 GiB |
| Python environment | `C:/VENV/fyp/Scripts/python.exe`, Python 3.13 |
| Base/merged training weights | FP16; LoRA managed by PEFT |
| Model architecture | 16 layers; hidden size 2048; intermediate size 8192; 32 attention heads; 8 KV heads; vocabulary 128256 |
| Attention implementation | SDPA |
| Seed | 42 (shuffle/evaluation); NPO forget sampler 43; retain sampler 44 |
| Maximum encoded training/NLL length | 256 tokens |
| Gradient checkpointing | Enabled, `use_reentrant=False` |
| Training KV cache | Disabled |
| Gradient norm clipping | 1.0 |

Package versions from saved `environment.json`:
```json
{json.dumps(environment,indent=2)}
```

## Dataset and evaluation coverage

- Dataset: `locuslab/TOFU`, cached locally; fictional biographies and QA.
- SFT: full 4,000 rows; 8 complete epochs, 32,000 example visits, 4,000 optimizer steps.
- Forget: `forget05`, 200 questions; retain: `retain95`, 3,800 questions. Disjoint union verified against full.
- Evaluation: all 200 forget questions, 40 seeded retain questions, 40 seeded world-facts questions.
- These exact same 280 questions and references are paired across all three stages.
- Full dataset content SHA-256: `{training['dataset_sha256']}`.
- Fixed evaluation manifest: `eval_samples.json`; dataset snapshots: `dataset_*.jsonl`.
- Retain utility conclusions apply to the sampled 40 questions, not a full retain census.
- World-facts performance is already weak at SFT; no original-base evaluation was saved to isolate SFT's effect on general utility.

## SFT hyperparameters

| Setting | Value |
|---|---|
| Method | Supervised LoRA, then merge into base and save complete model |
| Epochs | 8 |
| Effective examples per optimizer update | 8 |
| Micro-batch | 2 in recovered notebook; not independently recorded in completion metadata |
| Forward/backward passes per update | 4 with micro-batch 2 |
| Optimizer | `torch.optim.AdamW` |
| Learning rate | 0.0002 |
| Scheduler | Cosine with 5% warmup; 200 of 4,000 steps (notebook code) |
| Loss | Mean per-example answer-token NLL; prompt and padding labels masked |
| LoRA rank / alpha / dropout | 32 / 32 / 0.0 |
| Target projections | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Trainable LoRA parameters | 22,544,384 |
| Bias | None |
| Recovery checkpoints | Every 50 optimizer updates, final update also saved; newest two retained |
| Complete model output | `sft_full_fp16/` in SFT run |

Only learning rate is explicitly passed to AdamW. Installed AdamW source confirms
defaults: betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01. These are reconstructed
library defaults, not separately logged optimizer-state measurements from this run.
Other optional optimizer flags were left at library defaults. LoRA configuration is verified from saved adapter config.
Eight epochs/example counts/step counts are verified from `sft_training.json`.
Micro-batching and scheduler details above are reconstructed from notebook code,
not a separately frozen historical training script. NPO creates a fresh adapter
on the already merged SFT model.

## NPO hyperparameters and actual loss

| Setting | Value |
|---|---|
| Starting model and frozen reference | Same complete SFT checkpoint |
| Trainable parameters | Fresh LoRA with rank 32, alpha 32, dropout 0, same seven target projections |
| Optimizer | AdamW, constant learning rate 0.00005 |
| Optimizer updates | 60 |
| Accumulation | 8 forget/retain example pairs per update; micro-batch 1 |
| Beta | 0.1 |
| Retain coefficient | 1.0 |
| Forget sampling | Uniform with replacement, seed 43 |
| Retain sampling | Uniform with replacement, seed 44 |
| Forget draws | 480; reconstructed unique questions 181/200, 19 never directly sampled |
| Reference scores | Frozen SFT summed answer log probabilities, cached before training |
| Scheduler | None for NPO |
| Output | `npo_adapter/`, then merge into SFT and save `merged_fp16/` |

```python
delta = logP_policy_answer - logP_frozen_SFT_answer  # summed over answer tokens
npo_loss = -(2 / beta) * logsigmoid(-beta * delta).mean()
retain_loss = -(retain_logP / retain_answer_token_count).mean()
loss = (npo_loss + retain_coeff * retain_loss) / accumulation
```

The NPO sign/reference match the published objective. Minimizing it decreases
forget-answer likelihood. NPO uses sequence sums while retain loss uses token
averages: coefficient 1 is not evidence of balanced gradients. Poor retain
preservation motivates calibration, but raw loss magnitudes alone do not prove
the gradient imbalance. Logs record the LAST micro-batch every five updates,
not update-averaged losses; do not interpret them as smooth dataset-level curves.
The first NPO loss is approximately 13.86294; this is consistent with identical
initial policy/reference at beta 0.1, not a failure by itself.

## Quantization

- Merge the NPO LoRA adapter into the saved SFT base before quantization.
- bitsandbytes 4-bit NF4, double quantization enabled, FP16 compute, uint8 storage.
- Save `quantized_nf4/`, then reload it before evaluating.
- `quantization_source.json` points to this run's `merged_fp16/`.
- Metadata indicates the intended lineage; the original run did not save content
  hashes linking every model generation. This is not a retrospective proof that
  no file was ever replaced.
- No quantized-SFT-only control, INT8, GPTQ, or AWQ results existed for this run.

## Evaluation settings and metric definitions

- Prompt: tokenizer chat template, one user question, `add_generation_prompt=True`.
- Generation: greedy (`do_sample=False`), at most 128 new tokens, cache enabled.
- EOS and generation defaults come from the saved tokenizer/model; pad token is EOS if absent.
- NLL is answer-only teacher-forced negative log likelihood per token using encoded length 256.
- Exact match lowercases, removes punctuation through word extraction, and normalizes spaces.
- Token F1 is bag-of-words overlap. It is not ROUGE-L or a factual correctness score.
- No semantic judge, fact annotations, retain-only retrained comparator, or official TOFU forget-quality score was used.
- Prompt encoding can truncate at 256 for NLL/training; generation uses the full question prompt.

## Complete results

| Split | Model | N | Exact match | Token F1 | Answer NLL/token (lower = more likely) |
|---|---|---:|---:|---:|---:|
'''
for r in table:
    readme+=f"| {r['split']} | {r['stage']} | {r['count']} | {r['exact_match']:.2%} | {r['token_f1']:.2%} | {r['answer_nll_per_token']:.6f} |\n"
readme+='''
## Paired recovery and factual examples

- Strict forget baseline: 197/200 exact; all 197 cease to be exact after NPO;
  none become exact again after NF4. Conditional strict recovery: 0/197.
- Mean forget F1 falls another 1.75 percentage points after NF4.
- Mean forget NLL rises another 0.501482 after NF4; higher means less likely.
- Exploratory overlap thresholds give 12/166 returns at F1 >= .6, 4/194 at >= .8,
  and 2/196 at >= .9. These thresholds were not preregistered; they are NOT factual recovery rates.
- A correct name after NPO ("The full name of the author is Hina Ameen.") is scored
  non-exact against "The author's name is Hina Ameen." This illustrates a false
  inference of factual forgetting from exact-match failure.
- Birthplace example: SFT says Karachi, Pakistan; NPO says Karachi, India;
  NF4 says "Hina Ameenmith was born in Karachi, Pakistan." The birthplace returns
  while the name is corrupted. This is an exploratory fact-level example.
- Parents example: SFT says father Real Estate Agent / mother Doctor;
  NPO says Construction Manager / Scientist; NF4 says Civil Engineer / Doctor.
  One component returns, while the full answer remains incorrect.
- These examples were selected after inspecting outputs. They do not establish
  the frequency or statistical significance of factual recovery.

## Conclusions and limitations

The run demonstrates strong exact memorization at SFT, broad changes after NPO,
and worse mean forget/retain metrics after NF4. It does not establish clean
selective unlearning, complete deletion, or zero factual recovery. Retain damage,
one training seed, sampled utility evaluation, exact-match false negatives,
unreviewed facts, incomplete direct forget exposure, and the missing quantized-SFT
control limit interpretation. The NPO formula is not shown to be wrong. The current
1B/TOFU/LoRA/NF4 experiment is an adaptation, not an exact reproduction of the paper's
main NEWS/BOOKS setup. We must not tune or redefine metrics solely to manufacture recovery.

## Next experiment

Experiment 02 adds NF4(SFT) with the exact same question manifest and decoding,
without retraining SFT or NPO. It then produces the four-stage comparison and a
stage-blinded factual-review file. After reviewing this control, calibrate NPO
on predeclared development questions using shuffled complete forget passes and
periodic checkpoints. Select unlearning checkpoints using FP16 forget/retain
performance BEFORE examining their quantized outcomes. Mitigation research remains deferred.

## Source artifacts

- `settings.json`, `environment.json`, `npo_history.json`, `reference_logps.json`.
- `npo_adapter/adapter_config.json`, `quantized_nf4/quantization_source.json`.
- `eval_sft.csv`, `eval_npo_fp16.csv`, `eval_npo_merged_nf4.csv`.
- `interpretation/report.html`: searchable side-by-side answers.
- `interpretation/summary.json`: exact numerical summaries and original CSV hashes.
- SFT source run: `sft_full_fp16/sft_training.json`, `sft_adapter/adapter_config.json`, `sft_history.json`.

## Research references

- [NPO objective, Equation 3](https://arxiv.org/html/2404.05868v1#S3).
- [Catastrophic Failure of LLM Unlearning via Quantization](https://arxiv.org/html/2410.16454v3).
- [TOFU benchmark](https://locuslab.github.io/tofu/).

## Evaluation CSV integrity at documentation time

'''
for stage in stages:
    readme+=f"- `eval_{stage}.csv`: `{digest(SOURCE/f'eval_{stage}.csv')}`\n"
(SOURCE/'EXPERIMENT_RESULTS.md').write_text(readme,encoding='utf-8')

stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
output=ROOT/'runs'/f'experiment02_nf4_control_{stamp}'
output.mkdir();(output/'sources').mkdir()
for name in ['settings.json','environment.json','eval_samples.json','dataset_full.jsonl','dataset_forget05.jsonl','dataset_retain95.jsonl','dataset_world_facts.jsonl']+[f'eval_{s}.csv' for s in stages]:
    shutil.copy2(SOURCE/name,output/'sources'/name)
helpers=[]
for i,names in [(8,{'prompt_text','encode_qa','collate','answer_logps'}),(12,{'normalize','f1_score'})]:
    for node in ast.parse(source(i)).body:
        if isinstance(node,ast.FunctionDef) and node.name in names:
            helpers.append(ast.get_source_segment(source(i),node))
(output/'evaluation_helpers.py').write_text('\n\n'.join(helpers)+'\n',encoding='utf-8')
shutil.copy2(ROOT/'Mother_Local_v2.ipynb',output/'sources/Mother_Local_v2_at_preparation.ipynb')
shutil.copy2(ROOT/'scripts/run_nf4_control.py',output/'run_control.py')
protocol=dict(experiment='02: NF4-only SFT control',created_utc=datetime.now(timezone.utc).isoformat(),
    source_run=str(SOURCE),sft_model=str(SFT),seed=settings['seed'],max_length=settings['max_length'],
    eval_new_tokens=settings['eval_new_tokens'],do_sample=False,eval_count=280,
    forget_split=settings['forget_split'],retain_split=settings['retain_split'],
    quantization=dict(bits=4,quant_type='nf4',double_quant=True,compute_dtype='float16',storage='uint8'),
    primary_question='How much does NF4 alone change the learned answers before any unlearning?',
    primary_metrics=['exact_match','token_f1','answer_nll_per_token'],
    factual_review='Blinded manual labels; no automatic conversion of overlap to correctness',
    training='none; same saved SFT and copied previous NPO results',
    negative_results='Retain all outcomes; recovery is not a success criterion')
(output/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
# Hash source weights as well as inputs so a future resume cannot silently mix models.
inputs=list((output/'sources').glob('*'))+[output/'evaluation_helpers.py',output/'protocol.json',output/'run_control.py']+list(SFT.glob('*.json'))+list(SFT.glob('*.safetensors'))+list(SFT.glob('*.jinja'))
manifest={'inputs':{str(p.resolve()):digest(p) for p in inputs},'prepared_from_notebook_sha256':digest(ROOT/'Mother_Local_v2.ipynb')}
(output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
(output/'FACTUAL_REVIEW.md').write_text('''# Factual review protocol

This review is exploratory because prior NPO/NF4 outputs have already been seen.
Do not call it a preregistered or independent evaluation.

The final `factual_review_blinded.csv` contains all 280 questions × four stages in
shuffled order. Stage names are absent. `factual_review_key.json` is stored separately;
do not open it while grading. Model behavior may still reveal the stage.

For each question, identify the facts requested using only the question and gold
reference before judging the prediction. Accept paraphrases and unambiguous aliases.
Labels: `correct`, `partial`, `incorrect`, `unreviewed` (initial state).

- Correct: all requested facts are supported, with no contradictory answer to the question.
- Partial: at least one requested component is correct, but another is wrong or missing.
- Incorrect: no requested component is correctly answered, or the sole requested fact is wrong/refused.
- Unreviewed: no judgment has been made, or ambiguity requires adjudication. Never count as incorrect.
- A name mentioned in the question is not evidence that a different requested fact was remembered.
- Repeating the correct value and also a contradictory value is not fully correct.
- Extra unsupported claims relevant to the requested facts affect the label. Record unrelated hallucinations separately.
- Dates must identify the same date; do not invent a locale when numeric formats are ambiguous.
- Record supported and wrong/missing components explicitly. For partial fact recovery, track the same component at all stages.

After labels are frozen, unblind and count correct(SFT) -> incorrect(NPO) -> correct(NPO NF4).
Report the denominator of eligible facts/questions, coverage, uncertainty, and partial
recoveries separately. Also measure SFT -> SFT NF4 degradation and retain correctness.
Questions or facts already correct after NPO are survival, not recovery. A second
reviewer should adjudicate ambiguous answers. Do not use this training/evaluation set
as an independent held-out test for a later tuned experiment.
''',encoding='utf-8')
(output/'FUTURE_NPO_PLAN.md').write_text('''# Proposed next training experiment — not executed

Proceed after assessing the NF4-only control and factual review. Reuse the strong
saved SFT checkpoint. Preserve Experiment 01 as a negative/ambiguous result.

1. Create and freeze a development/test partition (prefer author-disjoint where
   feasible). Existing questions have been inspected; disclose this rather than
   claiming untouched testing. Freeze fact rubrics before examining new outputs.
2. Replace replacement sampling with complete shuffled forget passes and a cycling
   shuffled retain iterator; log coverage and update-averaged losses.
3. Predeclare a bounded exploratory grid: NPO learning rates 1e-5 and 5e-5,
   retain coefficients 1, 5, and 10; keep beta .1, effective batch 8, and the saved
   LoRA structure fixed. These are candidate values, not established optimal settings.
4. Train up to three complete forget passes and save at every five updates and
   epoch boundary. Evaluate FP16 forget and retain development questions at checkpoints.
5. Before running, fix a retain-utility acceptance threshold (for example no more
   than 5 percentage points of factual accuracy loss relative to SFT) and a
   minimum forget reduction. If no checkpoint qualifies, report failure rather
   than relax thresholds after seeing quantization.
6. Select checkpoints only using FP16 development trade-offs. Quantize selected
   checkpoints plus SFT with the same quantizer, evaluate held-out questions, and
   report all qualifying configurations, including negative recovery results.
7. Repeat promising configurations with additional seeds before drawing a robust conclusion.

Do not change NPO sequence normalization without treating it as a separate method:
token-averaging the NPO log ratio changes beta's effective interpretation. Log
forget/retain gradient norms on a small fixed development batch to diagnose their
balance; comparing raw loss magnitudes is insufficient. Mitigation work stays deferred.
''',encoding='utf-8')
(output/'README.md').write_text(f'''# Experiment 02 — NF4-only SFT control

The full settings and hypotheses are frozen in `protocol.json`. No training occurs.
The original model/evaluation files are preserved. Results appear in `RESULTS.md`
and `comparison.html` when all 280 questions finish. Check `status.json` for progress.

Run or resume using the existing environment:
```powershell
& C:/VENV/fyp/Scripts/python.exe -u "{(output/'run_control.py').as_posix()}" "{output.as_posix()}"
```

Each question is saved atomically under `question_progress/`. Interrupted evaluation
resumes from completed questions. An interrupted quantization save is left in an
`.incomplete_*` directory and is not reused. No GPU process is automatically stopped.

After completion, factual review remains a separate explicit task; blank/unreviewed
labels are not automatically interpreted as incorrect.
''',encoding='utf-8')
notebook=dict(nbformat=4,nbformat_minor=5,metadata=nb['metadata'],cells=[
    dict(cell_type='markdown',metadata={},source=['# Experiment 02 — NF4-only SFT control\n','Run this cell to execute or resume the frozen experiment. It trains no models. See README and protocol in the experiment folder.']),
    dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=[f"from pathlib import Path\nEXPERIMENT_DIR = Path({str(output)!r})\nexec(compile((EXPERIMENT_DIR/'run_control.py').read_text(encoding='utf-8').split('# CLI entry point.')[0], '<frozen-control>', 'exec'))\nrun_experiment(EXPERIMENT_DIR)\n"])
])
(ROOT/'Experiment_02_NF4_Control.ipynb').write_text(json.dumps(notebook,indent=1),encoding='utf-8')
(ROOT/'EXPERIMENTS.md').write_text(f'''# Local TOFU experiments

| Experiment | Purpose | Record |
|---|---|---|
| 01 | Eight-epoch SFT, NPO + retain CE, NF4; inspect forgetting and recovery | [Full settings and results](runs/{SOURCE.name}/EXPERIMENT_RESULTS.md) |
| 02 | Quantize SFT without NPO; complete four-stage comparison | [Protocol and execution](runs/{output.name}/README.md) |

Start Experiment 02 with `Experiment_02_NF4_Control.ipynb` or the command in its README.
Experiment 01 is preserved; the current Mother notebook is not repointed or retrained.
''',encoding='utf-8')
print('Experiment 01 record:',SOURCE/'EXPERIMENT_RESULTS.md')
print('Experiment 02:',output)
