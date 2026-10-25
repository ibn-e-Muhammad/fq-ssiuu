"""Compare interrupted/resumed notebook SFT with uninterrupted training on CPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import ast
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM,
                          PreTrainedTokenizerFast, get_cosine_schedule_with_warmup)

root = Path(__file__).resolve().parents[1]
notebook = json.loads((root/'Mother_Local_v2.ipynb').read_text(encoding='utf-8'))
source = lambda i: ''.join(notebook['cells'][i]['source'])
tree = ast.parse(source(10))
helpers = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
body = [n for n in tree.body if n not in helpers]
helper_code = compile(ast.Module(body=helpers, type_ignores=[]), '<checkpoint-helpers>', 'exec')
train_code = compile(ast.Module(body=body, type_ignores=[]), '<notebook-sft>', 'exec')
loss_fn = next(n for n in ast.parse(source(8)).body if isinstance(n, ast.FunctionDef) and n.name == 'answer_logps')
loss_code = compile(ast.Module(body=[loss_fn], type_ignores=[]), '<answer-loss>', 'exec')
torch.set_num_threads(2)


class PowerCut(Exception):
    pass


def same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b), 'Tensor mismatch after recovery'
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            same(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            same(x, y)
    else:
        assert a == b, (a, b)


with tempfile.TemporaryDirectory(prefix='sft-recovery-') as folder:
    work = Path(folder)
    base = work/'base'
    torch.manual_seed(77)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32))
    model.half().save_pretrained(base)
    del model
    backend = Tokenizer(WordLevel({str(i): i for i in range(32)}, unk_token='0'))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='0', pad_token='0', eos_token='2')
    tokenizer.save_pretrained(base)
    rows = [{'input_ids':[1, i, 10, 11], 'attention_mask':[1]*4, 'labels':[-100,-100,10,11]} for i in range(3, 8)]

    def execute(name, resume=None, cut=None, mismatch=False):
        run = work/name
        run.mkdir()
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        visits = []
        def collate(examples):
            visits.extend(x['input_ids'][1] for x in examples)
            return {key: torch.tensor([r[key] for r in examples]) for key in examples[0]}
        def load_fp16(path):
            return AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16, local_files_only=True)
        ns = dict(torch=torch, F=F, gc=gc, hashlib=hashlib, json=json, math=math, random=random, np=np, Path=Path,
            AutoModelForCausalLM=AutoModelForCausalLM, PeftModel=PeftModel, get_peft_model=get_peft_model,
            get_cosine_schedule_with_warmup=get_cosine_schedule_with_warmup,
            tokenizer=tokenizer, collate=collate, load_fp16=load_fp16,
            lora_config=LoraConfig(r=2, lora_alpha=4, lora_dropout=0.15, target_modules=['q_proj','v_proj'], task_type='CAUSAL_LM'),
            MODEL_PATH=base, EXPERIMENT_DIR=run, SFT_MODEL_DIR=run/'sft_full_fp16', SFT_ADAPTER=run/'sft_adapter',
            REUSE_SFT_MODEL=None, SFT_RESUME_FROM=resume, SFT_SAVE_EVERY=1, SFT_KEEP_CHECKPOINTS=2, SFT_MICRO_BATCH_SIZE=2,
            SFT_EPOCHS=2, SFT_MAX_STEPS=None, GRAD_ACCUM=2, LR_SFT=0.002 if mismatch else 0.001,
            SEED=42, MAX_LEN=16, options={}, full_dataset_sha256='test-dataset', sft_rows=rows)
        exec(helper_code, ns); exec(loss_code, ns)
        real_save = ns['save_sft_checkpoint']
        if cut is not None:
            def save_and_interrupt(state, policy, tok):
                result = real_save(state, policy, tok)
                if state['step'] == cut:
                    raise PowerCut()
                return result
            ns['save_sft_checkpoint'] = save_and_interrupt
        try:
            exec(train_code, ns)
        except PowerCut:
            assert cut is not None
        except ValueError as error:
            assert mismatch and 'Resume settings/data differ' in str(error), str(error)
            assert not visits
            return run, visits, None
        else:
            assert cut is None and not mismatch
        latest = ns['find_sft_checkpoint'](run)
        state = torch.load(latest/'training_state.pt', weights_only=True, map_location='cpu')
        del ns
        gc.collect()
        return run, visits, state

    baseline, all_visits, expected = execute('baseline')
    assert len(all_visits) == 10
    assert expected['step'] == 6 and expected['completed_full_epochs'] == 2
    assert len(list((baseline/'sft_checkpoints').glob('step-*'))) == 2
    for cut in (2, 3, 6):  # mid-epoch, epoch boundary, all updates finished before merge
        interrupted, first_visits, _ = execute(f'interrupted-{cut}', cut=cut)
        resumed, next_visits, actual = execute(f'resumed-{cut}', resume=interrupted)
        assert first_visits + next_visits == all_visits, 'Skipped or repeated examples'
        same(expected, actual)
        model_a = AutoModelForCausalLM.from_pretrained(baseline/'sft_full_fp16', local_files_only=True)
        model_b = AutoModelForCausalLM.from_pretrained(resumed/'sft_full_fp16', local_files_only=True)
        same(model_a.state_dict(), model_b.state_dict())
        del model_a, model_b
    execute('settings-mismatch', resume=baseline, mismatch=True)
    # Latest interrupted checkpoint is damaged; earlier committed checkpoint must win.
    damaged, _, _ = execute('damaged', cut=3)
    (damaged/'sft_checkpoints/step-00000003/training_state.pt').write_bytes(b'power-cut damage')
    (damaged/'sft_checkpoints/.incomplete-test').mkdir()
    recovered, visits, actual = execute('fallback', resume=damaged)
    assert visits == all_visits[4:]
    same(expected, actual)
    # None starts fresh even with existing interrupted/completed runs present.
    fresh, visits, actual = execute('fresh')
    assert visits == all_visits
    same(expected, actual)
print('PASS: mid-epoch, epoch-boundary and final-step recovery; identical weights/optimizer/scheduler/RNG/data order; corrupt-save fallback; retention; config rejection; fresh training.')
