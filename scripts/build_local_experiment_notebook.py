"""Build the local experiment from the existing local prototype without changing it."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if '# Paired recovery report version' in (ROOT/'Mother_Local_v2.ipynb').read_text(encoding='utf-8'):
    raise SystemExit('Mother_Local_v2 now has user-maintained training settings and paired recovery reporting. Edit it directly; this older prototype builder must not overwrite it.')
notebook = json.loads((ROOT / 'FQ_SSIUU_Local_TOFU_NPO_3060.ipynb').read_text(encoding='utf-8'))
cells = notebook['cells']

def set_cell(index, source):
    cells[index]['source'] = source.splitlines(keepends=True)

def source(index):
    return ''.join(cells[index]['source'])

set_cell(0, '''# Local TOFU: fine-tuning → NPO → FP16 evaluation → NF4 evaluation

Uses `C:\\Model\\Llama-3.2-1B-Instruct`. Select `C:\\VENV\\fyp\\Scripts\\python.exe` as the kernel.
Restart the kernel from the old Mother notebook first to release its models.
This is a local LoRA NPO+retain experiment, not the repo's source-faithful reproduction.
Defaults are a pilot: verify baseline memorization before interpreting forgetting.
''')
set_cell(2, '''# Use the existing environment: C:/VENV/fyp/Scripts/python.exe
import sys
print('Notebook interpreter:', sys.executable)
''')
set_cell(1, '## 0 · Verify the existing fyp kernel')
set_cell(7, '## 3 · Tokenizer, answer-only supervision, and FP16 LoRA model')
set_cell(11, '## 5 · Evaluate answer likelihood, exact match, and token F1')
s = source(4).replace(', prepare_model_for_kbit_training', '')
s = s.replace("method='local QLoRA NPO+retain, NOT source-faithful D1'", "method='FP16 LoRA NPO+retain, then merged FP16 to NF4'")
s = s.replace('EVAL_PER_SPLIT = 16', 'EVAL_PER_SPLIT = 40')
s = s.replace('SFT_STEPS = 150    # increase after confirming baseline memorization, e.g. 300-600', 'SFT_STEPS = 1500   # three passes over 4000 rows with accumulation=8; verify memorization')
s = s.replace('EVAL_NEW_TOKENS = 48', 'EVAL_NEW_TOKENS = 128')
s += '''\nfree, total = torch.cuda.mem_get_info()
print(f'Free GPU memory: {free/2**30:.2f} / {total/2**30:.2f} GiB')
assert free > 6 * 2**30, 'Release models in other notebook kernels before training (need at least 6 GiB free).'
import importlib.metadata
(EXPERIMENT_DIR/'environment.json').write_text(json.dumps({p: importlib.metadata.version(p) for p in ['torch','transformers','peft','datasets','bitsandbytes']}, indent=2))
'''
set_cell(4, s)
set_cell(6, source(6) + '''\nassert {key(x) for x in full} == {key(x) for x in forget} | {key(x) for x in retain}
for name, ds in [('full', full), ('forget01', forget), ('retain99', retain), ('world_facts', world)]:
    ds.to_json(str(EXPERIMENT_DIR / f'dataset_{name}.jsonl'))
''')
s = source(8).replace('def load_4bit():', 'def load_fp16():').replace('quantization_config=qcfg,', '')
s = s.replace('lora_dropout=0.05', 'lora_dropout=0.0')
# Avoid materializing a second full vocabulary-sized log-softmax tensor.
start = s.index('    selected=targets.masked_fill')
end = s.index('\nsft_rows=', start)
s = s[:start] + '''    losses=F.cross_entropy(logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
                           targets.reshape(-1), ignore_index=-100, reduction='none')
    return -losses.reshape_as(targets).sum(dim=-1), valid.sum(dim=-1).clamp(min=1)
''' + s[end:]
set_cell(8, s)
s = source(10).replace('load_4bit()', 'load_fp16()')
s = s.replace('policy=prepare_model_for_kbit_training(policy, use_gradient_checkpointing=True)', "policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})\npolicy.enable_input_require_grads()")
s = s.replace('rng=random.Random(SEED)', 'rng=random.Random(SEED)\nsft_order=[]\nsft_position=0')
s = s.replace('        example=sft_rows[rng.randrange(len(sft_rows))]', '''        if sft_position >= len(sft_order):
            sft_order=list(range(len(sft_rows)))
            rng.shuffle(sft_order)
            sft_position=0
        example=sft_rows[sft_order[sft_position]]
        sft_position+=1''')
s = s.replace('        loss.backward()', "        if not torch.isfinite(loss): raise RuntimeError('Nonfinite SFT loss')\n        loss.backward()")
s = s.replace('    optimizer.step();', '    torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0)\n    optimizer.step();')
s += '\ndel loss, logps, token_count, batch; gc.collect(); torch.cuda.empty_cache()\n'
set_cell(10, s)
s = source(12).replace('load_4bit()', 'load_fp16()')
s = s.replace("[forget[i] for i in range(min(EVAL_PER_SPLIT,len(forget)))]", "[forget[i] for i in random.Random(SEED).sample(range(len(forget)), min(EVAL_PER_SPLIT,len(forget)))]")
s = s.replace("[retain[i] for i in range(min(EVAL_PER_SPLIT,len(retain)))]", "[retain[i] for i in random.Random(SEED).sample(range(len(retain)), min(EVAL_PER_SPLIT,len(retain)))]")
s += "\n(EXPERIMENT_DIR/'eval_samples.json').write_text(json.dumps(eval_sets, indent=2))\n"
set_cell(12, s)
set_cell(13, '''**Baseline check:** Inspect `eval_sft.csv` and generations. Increase SFT_STEPS if the forget answers were not learned. Low baseline accuracy makes a forgetting claim inconclusive. Evaluation rows overlap training deliberately to test memorization; these are not held-out generalization scores.''')
s = source(15).replace('load_4bit()', 'load_fp16()')
split = s.index('# Policy starts')
s = s[:split] + '''# Cache the frozen SFT reference scores, then free that model before NPO.
with torch.inference_mode():
    reference_logps = [float(answer_logps(reference, collate([row]))[0].item()) for row in forget_rows]
(EXPERIMENT_DIR/'reference_logps.json').write_text(json.dumps(reference_logps))
del reference; gc.collect(); torch.cuda.empty_cache()

''' + s[split:]
s = s.replace('policy.gradient_checkpointing_enable()', "policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})")
s = s.replace("        bf=collate([forget_rows[rng_forget.randrange(len(forget_rows))]])", "        forget_index=rng_forget.randrange(len(forget_rows))\n        bf=collate([forget_rows[forget_index]])")
s = s.replace('        with torch.no_grad():\n            ref_lp,_=answer_logps(reference,bf)', "        ref_lp=torch.tensor([reference_logps[forget_index]], device='cuda')")
s = s.replace('        loss.backward()', "        if not torch.isfinite(loss): raise RuntimeError('Nonfinite NPO loss')\n        loss.backward()")
s = s.replace('    opt.step();', '    torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0)\n    opt.step();')
s = s.replace('del opt,policy,reference;', 'del opt,policy,loss,pi_lp,retain_lp,npo_loss,retain_loss;')
set_cell(15, s)
set_cell(16, '## 7 · Merge the unlearned adapter and save the FP16 checkpoint')
set_cell(17, source(20))
set_cell(18, '## 8 · Evaluate the exact merged FP16 checkpoint before quantization')
set_cell(19, 'The next stage quantizes this same saved checkpoint. Identical evaluation samples and decoding settings are used throughout.')
set_cell(20, '''unlearned=AutoModelForCausalLM.from_pretrained(str(MERGED_DIR), torch_dtype=torch.float16,
    device_map='cpu', local_files_only=True, attn_implementation='sdpa').to('cuda')
eval_npo=evaluate(unlearned,'npo_fp16')
del unlearned; gc.collect(); torch.cuda.empty_cache()
''')
set_cell(22, source(22).replace("attn_implementation='sdpa')", "attn_implementation='sdpa',local_files_only=True)").replace("eval_quant=evaluate", "quantized.save_pretrained(str(EXPERIMENT_DIR/'quantized_nf4'))\ntokenizer.save_pretrained(str(EXPERIMENT_DIR/'quantized_nf4'))\neval_quant=evaluate"))
set_cell(24, source(24) + '''\npaired=eval_npo.merge(eval_quant,on=['split','question','reference'],suffixes=('_fp16','_nf4'),validate='one_to_one')
paired['nll_change_nf4_minus_fp16']=paired.answer_nll_per_token_nf4-paired.answer_nll_per_token_fp16
paired['f1_change_nf4_minus_fp16']=paired.token_f1_nf4-paired.token_f1_fp16
paired.to_csv(EXPERIMENT_DIR/'quantization_paired.csv',index=False)
print('Quantization changes (negative NLL change on forget can indicate recovery):')
print(paired.groupby('split')[['nll_change_nf4_minus_fp16','f1_change_nf4_minus_fp16']].mean())
''')
set_cell(25, '''## Interpreting the experiment

- First establish that SFT learned the forget answers. Longer training may be necessary.
- Forgetting evidence: forget answer NLL rises and generation accuracy/F1 falls versus SFT.
- Selectivity: retain and world-fact utility should remain stable. Global degradation is not successful selective forgetting.
- Quantization recovery: forget NLL falls or answer F1 rises for NF4 versus the same merged FP16 checkpoint. Inspect paired generations and report magnitudes, not a binary guarantee.
- This pilot uses forget01/retain99 and NPO plus retain cross-entropy. It does not establish deletion or official TOFU forget quality. A stronger study needs retain-only retraining, perturbed-answer truth ratios, author-level uncertainty, more evaluation examples, multiple seeds and extraction attacks.

Sources: [TOFU](https://locuslab.github.io/tofu/), [NPO paper](https://arxiv.org/abs/2404.05868), [NF4 documentation](https://huggingface.co/docs/transformers/quantization/bitsandbytes).
''')
# Save a complete SFT checkpoint, then treat it as the base for a fresh NPO adapter.
set_cell(4, '''import gc, hashlib, json, math, random, re
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model

# Select C:/VENV/fyp/Scripts/python.exe as your notebook kernel.
MODEL_PATH = Path(r'C:\\Model\\Llama-3.2-1B-Instruct')
# To repeat unlearning without fine-tuning again, paste a saved sft_full_fp16 path here.
REUSE_SFT_MODEL = None  # e.g. Path(r'C:\\MY-FYP\\fq-ssiuu\\runs\\YOUR_RUN\\sft_full_fp16')
SEED = 42
MAX_LEN = 256
SFT_EPOCHS = 3          # every row of TOFU full is visited once per epoch
SFT_MAX_STEPS = None   # leave None for complete training; smoke tests override this
NPO_STEPS = 60
GRAD_ACCUM = 8
LR_SFT = 1e-4
LR_NPO = 5e-5
BETA = 0.1
RETAIN_COEFF = 1.0
EVAL_PER_SPLIT = 40
EVAL_NEW_TOKENS = 128

# The optional command-line runner supplies overrides; normal notebook execution needs none.
options = globals().get('LOCAL_RUN_OPTIONS', {})
if options.get('smoke'):
    SFT_MAX_STEPS, NPO_STEPS, GRAD_ACCUM = 1, 1, 1
    EVAL_PER_SPLIT, EVAL_NEW_TOKENS = 2, 16
if options.get('reuse_sft_model'):
    REUSE_SFT_MODEL = Path(options['reuse_sft_model'])
if REUSE_SFT_MODEL is not None:
    REUSE_SFT_MODEL = Path(REUSE_SFT_MODEL).resolve()
    assert (REUSE_SFT_MODEL/'config.json').is_file(), 'Saved full SFT model not found'
else:
    assert MODEL_PATH.is_dir(), f'Local model path not found: {MODEL_PATH}'
assert SFT_EPOCHS > 0 and GRAD_ACCUM > 0
assert SFT_MAX_STEPS is None or SFT_MAX_STEPS > 0
assert torch.cuda.is_available(), 'CUDA GPU required'
print('Torch:', torch.__version__, '| CUDA:', torch.version.cuda, '| GPU:', torch.cuda.get_device_name(0))
free, total = torch.cuda.mem_get_info()
print(f'Free GPU memory: {free/2**30:.2f} / {total/2**30:.2f} GiB')
assert free > 6 * 2**30, 'Release models in other notebook kernels before training (need at least 6 GiB free).'
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
ROOT = Path.cwd()
if options.get('resume_eval'):
    EXPERIMENT_DIR = Path(options['resume_eval']).resolve()
    assert (EXPERIMENT_DIR/'merged_fp16'/'config.json').is_file()
else:
    EXPERIMENT_DIR = ROOT/'runs'/('local_tofu_npo_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=False)
SFT_ADAPTER = EXPERIMENT_DIR/'sft_adapter'
SFT_MODEL_DIR = REUSE_SFT_MODEL or EXPERIMENT_DIR/'sft_full_fp16'
NPO_ADAPTER = EXPERIMENT_DIR/'npo_adapter'
MERGED_DIR = EXPERIMENT_DIR/'merged_fp16'
settings = dict(model_path=str(MODEL_PATH), sft_model_dir=str(SFT_MODEL_DIR),
    method='FP16 LoRA SFT on full TOFU, merged SFT checkpoint, fresh NPO+retain LoRA, NF4',
    seed=SEED, max_length=MAX_LEN, sft_epochs=SFT_EPOCHS, sft_max_steps=SFT_MAX_STEPS,
    npo_steps=NPO_STEPS, accumulation=GRAD_ACCUM, lr_sft=LR_SFT, lr_npo=LR_NPO,
    beta=BETA, retain_coeff=RETAIN_COEFF, eval_per_split=EVAL_PER_SPLIT,
    eval_new_tokens=EVAL_NEW_TOKENS, smoke_only=bool(options.get('smoke')))
if not options.get('resume_eval'):
    (EXPERIMENT_DIR/'settings.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
import importlib.metadata
(EXPERIMENT_DIR/'environment.json').write_text(json.dumps({p: importlib.metadata.version(p) for p in ['torch','transformers','peft','datasets','bitsandbytes']}, indent=2), encoding='utf-8')
print('Experiment:', EXPERIMENT_DIR)
print('Complete fine-tuned model:', SFT_MODEL_DIR)
''')
set_cell(6, source(6) + '''\nfull_dataset_sha256 = hashlib.sha256(json.dumps(list(full), sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()
''')
s = source(8).replace('str(MODEL_PATH), local_files_only=True', 'str(SFT_MODEL_DIR if REUSE_SFT_MODEL else MODEL_PATH), local_files_only=True')
s = s.replace('def load_fp16():', 'def load_fp16(path=MODEL_PATH):')
s = s.replace('str(MODEL_PATH), \n                device_map={\'\':0}', "str(path), \n                device_map='cpu'")
s = s.replace("attn_implementation='sdpa')", "attn_implementation='sdpa').to('cuda')")
set_cell(8, s)
set_cell(9, '''## 4 · Fine-tune on the complete TOFU full split and save a standalone model

Each epoch visits all 4,000 rows in a shuffled order. The SFT adapter is merged into
FP16 weights and saved with its tokenizer in `sft_full_fp16/`. This folder can be
loaded directly by `AutoModelForCausalLM` without the original model or adapter.
To reuse it on another run, set `REUSE_SFT_MODEL` in section 1.
''')
set_cell(10, '''if REUSE_SFT_MODEL is not None:
    training_record = json.loads((SFT_MODEL_DIR/'sft_training.json').read_text(encoding='utf-8'))
    assert training_record['dataset_sha256'] == full_dataset_sha256, 'Saved SFT used different TOFU rows'
    assert training_record['completed_full_epochs'] >= 1, 'This is an incomplete/smoke checkpoint; complete SFT first'
    assert training_record['max_length'] == MAX_LEN, 'Match MAX_LEN to the saved SFT training settings'
    print('Reusing complete SFT checkpoint:', SFT_MODEL_DIR)
else:
    if SFT_MODEL_DIR.exists() or SFT_ADAPTER.exists():
        raise FileExistsError('SFT output already exists. Start a new run or set REUSE_SFT_MODEL.')
    policy=get_peft_model(load_fp16(MODEL_PATH), lora_config)
    policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    policy.enable_input_require_grads()
    policy.config.use_cache=False
    policy.print_trainable_parameters()
    optimizer=torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad],lr=LR_SFT)
    rng=random.Random(SEED)
    step=0; examples_seen=0; completed_full_epochs=0; sft_history=[]
    total_steps=SFT_EPOCHS*math.ceil(len(sft_rows)/GRAD_ACCUM)
    policy.train()
    for epoch in range(SFT_EPOCHS):
        order=list(range(len(sft_rows))); rng.shuffle(order)
        epoch_seen=0
        for start in range(0,len(order),GRAD_ACCUM):
            if SFT_MAX_STEPS is not None and step >= SFT_MAX_STEPS:
                break
            indices=order[start:start+GRAD_ACCUM]
            optimizer.zero_grad(set_to_none=True)
            mean_loss=0.0
            for index in indices:
                batch=collate([sft_rows[index]])
                logps,token_count=answer_logps(policy,batch)
                loss=-(logps/token_count).mean()/len(indices)
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite SFT loss')
                loss.backward()
                mean_loss+=float(loss.detach())
            torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad],1.0,error_if_nonfinite=True)
            optimizer.step()
            step+=1; epoch_seen+=len(indices); examples_seen+=len(indices)
            if step==1 or step%10==0:
                sft_history.append({'step':step,'epoch':epoch+1,'loss_per_token':mean_loss})
                print(f'SFT {step}/{total_steps}, epoch {epoch+1}, loss/token {mean_loss:.4f}')
        if epoch_seen == len(sft_rows): completed_full_epochs+=1
        if SFT_MAX_STEPS is not None and step >= SFT_MAX_STEPS: break
    if SFT_MAX_STEPS is None:
        assert completed_full_epochs == SFT_EPOCHS
        assert examples_seen == SFT_EPOCHS*len(sft_rows)
    policy.save_pretrained(str(SFT_ADAPTER))
    tokenizer.save_pretrained(str(SFT_ADAPTER))
    (EXPERIMENT_DIR/'sft_history.json').write_text(json.dumps(sft_history,indent=2))
    del optimizer, policy, loss, logps, token_count, batch
    gc.collect(); torch.cuda.empty_cache()

    # Persist a complete model before starting any unlearning.
    base_cpu=AutoModelForCausalLM.from_pretrained(str(MODEL_PATH),torch_dtype=torch.float16,
        device_map='cpu',local_files_only=True)
    sft_merged=PeftModel.from_pretrained(base_cpu,str(SFT_ADAPTER),is_trainable=False).merge_and_unload(safe_merge=True)
    sft_merged.config.use_cache=True
    sft_merged.save_pretrained(str(SFT_MODEL_DIR),safe_serialization=True,max_shard_size='2GB')
    tokenizer.save_pretrained(str(SFT_MODEL_DIR))
    training_record=dict(dataset='locuslab/TOFU',split='full',dataset_rows=len(sft_rows),
        dataset_sha256=full_dataset_sha256,examples_seen=examples_seen,
        completed_full_epochs=completed_full_epochs,optimizer_steps=step,
        max_length=MAX_LEN,seed=SEED,base_model=str(MODEL_PATH),
        checkpoint_type='merged_full_model_fp16',smoke_only=bool(options.get('smoke')))
    # Write completion metadata last, after weights and tokenizer are saved.
    (SFT_MODEL_DIR/'sft_training.json').write_text(json.dumps(training_record,indent=2),encoding='utf-8')
    del base_cpu,sft_merged; gc.collect()
    print('Saved complete fine-tuned model:', SFT_MODEL_DIR)
    print('Full epochs completed:', completed_full_epochs, '| Examples seen:', examples_seen)
''')
s = source(12).replace('sft_model=PeftModel.from_pretrained(load_fp16(),str(SFT_ADAPTER),is_trainable=False)', 'sft_model=load_fp16(SFT_MODEL_DIR)')
set_cell(12, s)
set_cell(13, source(13).replace('Increase SFT_STEPS', 'Increase SFT_EPOCHS'))
s = source(15).replace('reference=PeftModel.from_pretrained(load_fp16(),str(SFT_ADAPTER),is_trainable=False)', 'reference=load_fp16(SFT_MODEL_DIR)')
s = s.replace('# Policy starts at same SFT adapter checkpoint but stays trainable.', '# Reload the same complete SFT model, then attach a fresh trainable NPO adapter.')
s = s.replace('policy=PeftModel.from_pretrained(load_fp16(),str(SFT_ADAPTER),is_trainable=True)', 'policy=get_peft_model(load_fp16(SFT_MODEL_DIR),lora_config)')
set_cell(15, s)
set_cell(17, source(17).replace('str(MODEL_PATH)', 'str(SFT_MODEL_DIR)').replace('# Merge on CPU, not into bitsandbytes NF4 (avoids approximate 4-bit merge).', '# Merge the NPO adapter into the saved SFT base, preserving SFT in its own folder.'))
set_cell(0, source(0) + '''\nSaved model flow: `original model` → `sft_full_fp16/` → `merged_fp16/` (unlearned) → `quantized_nf4/`.
The fine-tuned checkpoint is loaded from disk for both baseline evaluation and NPO.
''')
for cell in cells:
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None
target = ROOT / 'Mother_Local_v2.ipynb'
backup = ROOT / 'notebooks' / 'Mother_Local_v2.before_local_pipeline.ipynb'
if not backup.exists():
    backup.write_bytes(target.read_bytes())
target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
print(target)
