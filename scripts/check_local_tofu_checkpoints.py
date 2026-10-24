"""Exercise the notebook's real SFT -> saved model -> NPO flow on a tiny CPU model."""
import ast
from collections import Counter
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import tempfile

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

root = Path(__file__).resolve().parents[1]
nb = json.loads((root/'Mother_Local_v2.ipynb').read_text(encoding='utf-8'))
source = lambda i: ''.join(nb['cells'][i]['source'])
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        ast.parse(source(i))
torch.manual_seed(42)
torch.set_num_threads(2)

with tempfile.TemporaryDirectory(prefix='tofu-checkpoints-') as directory:
    run = Path(directory)
    base = run/'original'
    model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2, max_position_embeddings=32))
    model.half().save_pretrained(base)
    del model
    backend = Tokenizer(WordLevel({str(i): i for i in range(32)}, unk_token='0'))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='0', pad_token='0', eos_token='2')
    tokenizer.save_pretrained(base)
    rows = [{'input_ids':[1,i,10,11], 'attention_mask':[1]*4,
             'labels':[-100,-100,10,11]} for i in range(3,8)]
    visits = []
    loads = []

    def collate(examples):
        visits.extend(row['input_ids'][1] for row in examples)
        return {key: torch.tensor([row[key] for row in examples]) for key in examples[0]}

    def load_fp16(path=base):
        loads.append(Path(path))
        return AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, local_files_only=True)

    ns = dict(torch=torch,F=F,gc=gc,json=json,math=math,random=random,
        AutoModelForCausalLM=AutoModelForCausalLM,PeftModel=PeftModel,get_peft_model=get_peft_model,
        tokenizer=tokenizer,collate=collate,load_fp16=load_fp16,
        lora_config=LoraConfig(r=2,lora_alpha=4,lora_dropout=0.0,
            target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'),
        MODEL_PATH=base,EXPERIMENT_DIR=run,SFT_MODEL_DIR=run/'sft_full_fp16',
        SFT_ADAPTER=run/'sft_adapter',NPO_ADAPTER=run/'npo_adapter',MERGED_DIR=run/'merged_fp16',
        REUSE_SFT_MODEL=None,SFT_EPOCHS=2,SFT_MAX_STEPS=None,GRAD_ACCUM=2,
        LR_SFT=1e-3,SEED=42,MAX_LEN=16,options={},full_dataset_sha256='test-data',
        sft_rows=rows,forget_rows=rows[:2],retain_rows=rows[2:],
        NPO_STEPS=1,LR_NPO=1e-3,BETA=0.1,RETAIN_COEFF=1.0)
    tree = ast.parse(source(8))
    fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='answer_logps')
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<loss>','exec'),ns)
    exec(compile(source(10),'<sft-cell>','exec'),ns)
    assert Counter(visits) == Counter({i:2 for i in range(3,8)}), visits
    meta = json.loads((ns['SFT_MODEL_DIR']/'sft_training.json').read_text())
    assert meta['completed_full_epochs']==2 and meta['optimizer_steps']==6
    assert meta['examples_seen']==10  # includes the one-row final batch of each epoch
    assert not (ns['SFT_MODEL_DIR']/'adapter_config.json').exists()
    assert (ns['SFT_MODEL_DIR']/'tokenizer.json').is_file()
    weights = ns['SFT_MODEL_DIR']/'model.safetensors'
    before = hashlib.sha256(weights.read_bytes()).hexdigest()
    # The full checkpoint reloads independently, including its tokenizer.
    reloaded = AutoModelForCausalLM.from_pretrained(ns['SFT_MODEL_DIR'],local_files_only=True)
    PreTrainedTokenizerFast.from_pretrained(ns['SFT_MODEL_DIR'],local_files_only=True)
    del reloaded
    # Reuse takes the skip-training path and performs no model loads.
    ns['REUSE_SFT_MODEL']=ns['SFT_MODEL_DIR']
    count_before=len(loads)
    exec(compile(source(10),'<reuse-cell>','exec'),ns)
    assert len(loads)==count_before
    loads.clear()
    exec(compile(source(15).replace("device='cuda'", "device='cpu'"),'<npo-cell>','exec'),ns)
    assert loads == [ns['SFT_MODEL_DIR'],ns['SFT_MODEL_DIR']], loads
    adapter_config=json.loads((ns['NPO_ADAPTER']/'adapter_config.json').read_text())
    assert Path(adapter_config['base_model_name_or_path'])==ns['SFT_MODEL_DIR']
    exec(compile(source(17),'<merge-cell>','exec'),ns)
    assert hashlib.sha256(weights.read_bytes()).hexdigest()==before, 'NPO modified the saved SFT weights'
    assert (ns['MERGED_DIR']/'config.json').exists()
    unlearned=AutoModelForCausalLM.from_pretrained(ns['MERGED_DIR'],local_files_only=True)
    sft=AutoModelForCausalLM.from_pretrained(ns['SFT_MODEL_DIR'],local_files_only=True)
    assert any(not torch.equal(a,b) for a,b in zip(unlearned.parameters(),sft.parameters()))
    del unlearned,sft
    # Reusing an incomplete checkpoint must fail before loading/training anything.
    meta['completed_full_epochs']=0
    (ns['SFT_MODEL_DIR']/'sft_training.json').write_text(json.dumps(meta))
    try:
        exec(compile(source(10),'<incomplete-reuse>','exec'),ns)
    except AssertionError as error:
        assert 'incomplete/smoke' in str(error)
    else:
        raise AssertionError('Incomplete SFT checkpoint was accepted')
    gc.collect()
print('PASS: full dataset coverage, partial accumulation, standalone SFT save/reload, reuse, NPO checkpoint lineage, preserved SFT weights, merged unlearned reload.')
