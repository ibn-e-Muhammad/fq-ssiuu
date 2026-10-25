"""Update evaluation/report cells while preserving the user's training code/settings."""
import ast
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
target=ROOT/'Mother_Local_v2.ipynb'
nb=json.loads(target.read_text(encoding='utf-8'))
backup=ROOT/'notebooks'/'Mother_Local_v2.before_interpretable_results.ipynb'
if not backup.exists():
    backup.write_bytes(target.read_bytes())

def source(i):
    return ''.join(nb['cells'][i]['source'])

def put(i,text):
    nb['cells'][i]['source']=text.splitlines(keepends=True)

put(0, '''# Local TOFU: learning, forgetting, and recovery after quantization

1. Load the local Llama model.
2. Fine-tune on the complete TOFU full split and save the complete model separately.
3. Reload and evaluate the fine-tuned checkpoint: establish what it can answer.
4. Apply NPO + retain regularization to a copy loaded from the saved SFT model.
5. Evaluate the unlearned FP16 model on the same questions.
6. Quantize that unlearned checkpoint to NF4, save/reload it, and ask the same questions again.
7. Read the paired report: exact-answer losses/returns, score changes, retained utility, and all generated answers.

Kernel: `C:/VENV/fyp/Scripts/python.exe`. Model: `C:/Model/Llama-3.2-1B-Instruct`.
This notebook includes its evaluation/report code and remains standalone.
''')
s=source(4)
if 'FORGET_SPLIT =' not in s:
    s=s.replace('EVAL_PER_SPLIT = 40', '''FORGET_SPLIT = 'forget05'  # choices: forget01, forget05, forget10
assert FORGET_SPLIT in ('forget01','forget05','forget10')
RETAIN_SPLIT = f'retain{100-int(FORGET_SPLIT[-2:]):02d}'
EVAL_FORGET_LIMIT = None   # None evaluates every forget question
EVAL_PER_SPLIT = 40       # fixed samples for retain/world utility''')
    s=s.replace('EVAL_PER_SPLIT, EVAL_NEW_TOKENS = 2, 16','EVAL_PER_SPLIT, EVAL_NEW_TOKENS = 2, 16\n    EVAL_FORGET_LIMIT = 2')
    s=s.replace('eval_new_tokens=EVAL_NEW_TOKENS, smoke_only=', 'eval_new_tokens=EVAL_NEW_TOKENS, forget_split=FORGET_SPLIT, retain_split=RETAIN_SPLIT,\n    eval_forget_limit=EVAL_FORGET_LIMIT, smoke_only=')
put(4,s)
put(6, '''from datasets import load_from_disk

DATA_DIR = Path(r'C:\\MY-FYP\\fq-ssiuu\\data\\tofu')
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = DATA_DIR/'hf_cache'

def get_split(name):
    # Evaluation-only resume uses this run's exact saved dataset snapshots.
    snapshot=EXPERIMENT_DIR/f'dataset_{name}.jsonl'
    if options.get('resume_eval'):
        assert snapshot.is_file(), f'Missing saved dataset: {snapshot}'
        return load_dataset('json',data_files=str(snapshot),split='train')
    local=DATA_DIR/name
    if local.exists():
        return load_from_disk(str(local))
    ds=load_dataset('locuslab/TOFU',name,split='train',cache_dir=str(CACHE_DIR))
    ds.save_to_disk(str(local))
    return ds

full,forget,retain,world=[get_split(name) for name in ('full',FORGET_SPLIT,RETAIN_SPLIT,'world_facts')]
for name,ds in [('full',full),(FORGET_SPLIT,forget),(RETAIN_SPLIT,retain),('world_facts',world)]:
    assert {'question','answer'}.issubset(ds.column_names)
    print(name,len(ds))
key=lambda row:(row['question'].strip(),row['answer'].strip())
assert {key(x) for x in forget}.isdisjoint({key(x) for x in retain}), 'forget and retain sets overlap'
assert {key(x) for x in full}=={key(x) for x in forget}|{key(x) for x in retain}
if not options.get('resume_eval'):
    for name,ds in [('full',full),(FORGET_SPLIT,forget),(RETAIN_SPLIT,retain),('world_facts',world)]:
        ds.to_json(str(EXPERIMENT_DIR/f'dataset_{name}.jsonl'))
full_dataset_sha256=hashlib.sha256(json.dumps(list(full),sort_keys=True,ensure_ascii=False).encode('utf-8')).hexdigest()
print('Verified disjoint partition:',FORGET_SPLIT,'+',RETAIN_SPLIT,'= full TOFU')
''')
put(11, '## 5 · Evaluate the saved fine-tuned model: baseline answers before unlearning')
s=source(12)
start=s.index('def select_eval_sets') if 'def select_eval_sets' in s else s.index('# Fixed samples')
end=s.index('def evaluate',start)
s=s[:start]+'''def select_eval_sets(forget,retain,world,forget_split,retain_split,seed,utility_limit,forget_limit=None):
    def choose(ds,limit):
        if limit is None: return [dict(row) for row in ds]
        assert limit>0
        return [dict(ds[i]) for i in random.Random(seed).sample(range(len(ds)),min(limit,len(ds)))]
    return {forget_split:choose(forget,forget_limit),retain_split:choose(retain,utility_limit),
            'world_facts':choose(world,utility_limit)}

# Construct once; every model is asked exactly these questions with identical decoding.
eval_sets=select_eval_sets(forget,retain,world,FORGET_SPLIT,RETAIN_SPLIT,SEED,EVAL_PER_SPLIT,EVAL_FORGET_LIMIT)
if options.get('resume_eval'):
    eval_sets=json.loads((EXPERIMENT_DIR/'eval_samples.json').read_text(encoding='utf-8'))
else:
    (EXPERIMENT_DIR/'eval_samples.json').write_text(json.dumps(eval_sets,indent=2),encoding='utf-8')
print('Evaluation counts:',{name:len(rows) for name,rows in eval_sets.items()})

'''+s[end:]
if 'for row_index,ex' not in s:
    s=s.replace('model.eval(); records=[]','model.eval(); records=[]\n    print("Evaluating",label,"on the fixed question set")')
    s=s.replace('for ex in examples:', 'for row_index,ex in enumerate(examples):')
    s=s.replace('pad_token_id=tokenizer.pad_token_id)', 'pad_token_id=tokenizer.pad_token_id,use_cache=True)')
    s=s.replace("reference=ex['answer']", "reference=ex['answer']\n            if (row_index+1)%10==0 or row_index+1==len(examples): print(label,subset,f'{row_index+1}/{len(examples)}')")
# The old trailing save is redundant; the manifest is now written before evaluation.
s=s.replace("(EXPERIMENT_DIR/'eval_samples.json').write_text(json.dumps(eval_sets, indent=2))",'')
put(12,s)
put(13, '''**Baseline interpretation:** Exact match is strict; token F1 is word overlap, not factual accuracy.
Inspect whether the fine-tuned model answers the target facts correctly. A question it
already gets wrong cannot establish successful forgetting. The final report conditions
exact-answer recovery on answers that were exact at SFT and lost after NPO.
''')
put(14,'## 6 · Apply NPO + retain regularization to the saved fine-tuned model')
put(18,'## 8 · Ask the same questions after NPO and measure forgetting')
put(21,'## 9 · Quantize the unlearned model, reload it, and ask the same questions again')
put(22, '''QUANTIZED_DIR=EXPERIMENT_DIR/'quantized_nf4'
if not (QUANTIZED_DIR/'config.json').exists():
    quantized=AutoModelForCausalLM.from_pretrained(str(MERGED_DIR),quantization_config=qcfg,
        device_map={'':0},torch_dtype=torch.float16,attn_implementation='sdpa',local_files_only=True)
    quantized.save_pretrained(str(QUANTIZED_DIR))
    tokenizer.save_pretrained(str(QUANTIZED_DIR))
    (QUANTIZED_DIR/'quantization_source.json').write_text(json.dumps({
        'source_checkpoint':str(MERGED_DIR.resolve()),'quantization':qcfg.to_dict()},indent=2),encoding='utf-8')
    del quantized; gc.collect(); torch.cuda.empty_cache()

# Test the saved quantized model, with exactly the same prompts and decoding.
quantized=AutoModelForCausalLM.from_pretrained(str(QUANTIZED_DIR),device_map={'':0},
    torch_dtype=torch.float16,attn_implementation='sdpa',local_files_only=True)
assert getattr(quantized,'is_loaded_in_4bit',False), 'Expected a saved 4-bit model'
eval_quant=evaluate(quantized,'npo_merged_nf4')
del quantized; gc.collect(); torch.cuda.empty_cache()
''')
put(23,'## 10 · Interpret forgetting, quantization recovery, and retained utility')
report_code=(ROOT/'scripts'/'report_local_tofu.py').read_text(encoding='utf-8').split('# Command-line entry point.')[0]
put(24, '''# Paired recovery report version 1. All functions are embedded for standalone use.
'''+report_code+'''
report=make_report(EXPERIMENT_DIR)
from IPython.display import Markdown,display
display(Markdown((EXPERIMENT_DIR/'interpretation'/'RESULTS.md').read_text(encoding='utf-8')))
print('Open the searchable side-by-side answer report:',EXPERIMENT_DIR/'interpretation'/'report.html')
''')
put(25, '''## How to read the result

- **Exact-answer recovery:** among answers that were exact before NPO and stopped being exact after NPO, how many become exact again after NF4? The report prints the numerator and denominator.
- **Likelihood/F1 reversal:** how much of the score change introduced by NPO did quantization reverse? This is a change in a score, not the percentage of facts recovered.
- **Factual inspection:** open the HTML report and compare each reference with all three generated answers. Word overlap can be high even when a name, date, or book title is wrong.
- **Retained utility:** check the retain/world-facts scores too. A model that loses knowledge everywhere has not selectively forgotten.
- All forget questions are evaluated by default. Exact matching can miss correct paraphrases; direct prompting does not prove deletion or resistance to stronger extraction attacks.

These are local experiment metrics, not official TOFU forget quality. See [TOFU's evaluation definitions](https://locuslab.github.io/tofu/).
''')
for cell in nb['cells']:
    if cell['cell_type']=='code':
        ast.parse(''.join(cell['source']))
        cell['execution_count']=None
        cell['outputs']=[]
target.write_text(json.dumps(nb,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
print('Updated',target)
print('Preserved previous notebook including its outputs:',backup)
