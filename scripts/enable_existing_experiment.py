"""Add existing-run continuation without changing the user's SFT training cell."""
import ast
import json
from pathlib import Path
import textwrap
from datetime import datetime, timezone

root = Path(__file__).resolve().parents[1]
path = root/'Mother_Local_v2.ipynb'
nb = json.loads(path.read_text(encoding='utf-8'))
source = lambda i: ''.join(nb['cells'][i]['source'])
if 'RUN_MODE =' in source(4):
    raise SystemExit('Existing-run support already installed; no changes made.')
backup = root/'notebooks'/('Mother_Local_v2.before_existing_run_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')+'.ipynb')
backup.write_bytes(path.read_bytes())

def put(i, content):
    nb['cells'][i]['source'] = content.splitlines(keepends=True)
    if nb['cells'][i]['cell_type'] == 'code':
        nb['cells'][i]['outputs'] = []
        nb['cells'][i]['execution_count'] = None

s = source(4).replace('# To repeat unlearning without fine-tuning again, paste a saved sft_full_fp16 path here.',
    "# 'continue' keeps the saved SFT run folder; 'new' starts a separate experiment.\nRUN_MODE = 'continue'\n# Select the complete SFT model to reuse below; leave None for fresh training.")
s = s.replace("options = globals().get('LOCAL_RUN_OPTIONS', {})", "options = globals().get('LOCAL_RUN_OPTIONS', {})\nRUN_MODE = options.get('run_mode') or RUN_MODE\nassert RUN_MODE in ('continue', 'new')")
s = s.replace("if options.get('resume_sft'):\n    SFT_RESUME_FROM", "if options.get('resume_sft'):\n    REUSE_SFT_MODEL = None\n    SFT_RESUME_FROM")
s = s.replace("if options.get('reuse_sft_model'):\n    REUSE_SFT_MODEL", "if options.get('reuse_sft_model'):\n    SFT_RESUME_FROM = None\n    REUSE_SFT_MODEL")
old = s[s.index('ROOT = Path.cwd()'):s.index("SFT_ADAPTER = EXPERIMENT_DIR/")]
new = '''ROOT = Path.cwd()

def select_experiment_dir(root, reuse_model, run_mode, options):
    if options.get('resume_eval'):
        folder = Path(options['resume_eval']).resolve()
        assert (folder/'merged_fp16'/'config.json').is_file()
        return folder, True
    if run_mode == 'continue' and reuse_model is not None:
        folder = Path(reuse_model).resolve().parent
        if Path(reuse_model).name != 'sft_full_fp16' or not (folder/'settings.json').is_file():
            raise ValueError("Continue requires this notebook's run/sft_full_fp16 folder; use RUN_MODE='new' for an external model.")
        return folder, True
    folder = Path(root)/'runs'/('local_tofu_npo_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
    folder.mkdir(parents=True, exist_ok=False)
    return folder, False

EXPERIMENT_DIR, CONTINUING_RUN = select_experiment_dir(ROOT, REUSE_SFT_MODEL, RUN_MODE, options)

def saved_stage_exists(folder, adapter=False):
    """Missing stage may be created; an incomplete existing stage is never overwritten."""
    folder = Path(folder)
    if not folder.exists():
        return False
    config = folder/('adapter_config.json' if adapter else 'config.json')
    index = folder/'model.safetensors.index.json'
    if not config.is_file():
        raise RuntimeError(f'Incomplete stage: {folder}. Preserve it and use a new run to rebuild.')
    json.loads(config.read_text(encoding='utf-8'))
    if adapter:
        names = ['adapter_model.safetensors']
    elif index.is_file():
        names = list(set(json.loads(index.read_text(encoding='utf-8'))['weight_map'].values()))
    else:
        names = ['model.safetensors']
    if not names or any(not (folder/name).is_file() or (folder/name).stat().st_size == 0 for name in names):
        raise RuntimeError(f'Missing or empty weights in {folder}. Use a new run to rebuild.')
    if not (folder/'tokenizer.json').is_file():
        raise RuntimeError(f'Missing tokenizer in {folder}. Use a new run to rebuild.')
    return True

def verify_continuation_settings(saved, current):
    keys = ('model_path', 'sft_model_dir', 'seed', 'max_length', 'npo_steps', 'accumulation',
            'lr_npo', 'beta', 'retain_coeff', 'eval_per_split', 'eval_new_tokens',
            'forget_split', 'retain_split', 'eval_forget_limit', 'smoke_only')
    changed = []
    for key in keys:
        a, b = saved.get(key), current.get(key)
        if key in ('model_path', 'sft_model_dir') and a and b:
            a, b = Path(a).resolve(), Path(b).resolve()
        if a != b:
            changed.append(key)
    if changed:
        raise ValueError(f"Settings differ from this saved experiment: {changed}. Restore them or set RUN_MODE='new'.")

'''
s = s.replace(old,new)
s = s.replace("if not options.get('resume_eval'):\n    (EXPERIMENT_DIR/'settings.json').write_text", "if CONTINUING_RUN and not options.get('resume_eval'):\n    saved_settings = json.loads((EXPERIMENT_DIR/'settings.json').read_text(encoding='utf-8'))\n    verify_continuation_settings(saved_settings, settings)\nif not CONTINUING_RUN:\n    (EXPERIMENT_DIR/'settings.json').write_text")
s = s.replace("(EXPERIMENT_DIR/'environment.json').write_text", "if not CONTINUING_RUN:\n    (EXPERIMENT_DIR/'environment.json').write_text")
s += "\nprint('Run mode:', 'continue existing experiment' if CONTINUING_RUN else 'new experiment')\n"
put(4,s)
# Saved datasets and question selection remain the experiment's source of truth.
put(6,source(6).replace("if options.get('resume_eval'):", 'if CONTINUING_RUN:').replace("if not options.get('resume_eval'):", 'if not CONTINUING_RUN:'))
put(12,source(12).replace("if options.get('resume_eval'):", "if CONTINUING_RUN and (EXPERIMENT_DIR/'eval_samples.json').is_file():"))

npo = source(15)
put(15,'''# Reopen completed stages without retraining or invalidating their quantized model.
if saved_stage_exists(NPO_ADAPTER, adapter=True):
    adapter_info = json.loads((NPO_ADAPTER/'adapter_config.json').read_text(encoding='utf-8'))
    assert Path(adapter_info['base_model_name_or_path']).resolve() == SFT_MODEL_DIR.resolve(), 'NPO adapter belongs to a different SFT model'
    npo_history = json.loads((EXPERIMENT_DIR/'npo_history.json').read_text(encoding='utf-8'))
    assert npo_history and npo_history[-1]['step'] == NPO_STEPS, 'Incomplete NPO stage; use a new run to rebuild'
    print('Reusing completed NPO adapter:', NPO_ADAPTER)
else:
    if MERGED_DIR.exists() or (EXPERIMENT_DIR/'quantized_nf4').exists():
        raise RuntimeError("Downstream models exist without their NPO adapter. Use RUN_MODE='new' to rebuild together.")
'''+textwrap.indent(npo,'    ')+'''
    if not npo_history or npo_history[-1]['step'] != NPO_STEPS:
        npo_history.append({'step': NPO_STEPS, 'completed': True})
        (EXPERIMENT_DIR/'npo_history.json').write_text(json.dumps(npo_history, indent=2))
''')
put(17,"if saved_stage_exists(MERGED_DIR):\n    print('Reusing merged FP16 model:', MERGED_DIR)\nelse:\n    if (EXPERIMENT_DIR/'quantized_nf4').exists():\n        raise RuntimeError(\"Quantized model exists without its merge source. Use RUN_MODE='new'.\")\n"+textwrap.indent(source(17),'    '))
quant = source(22).replace("if not (QUANTIZED_DIR/'config.json').exists():", "if not saved_stage_exists(QUANTIZED_DIR):")
quant = quant.replace('# Test the saved quantized model, with exactly the same prompts and decoding.', '''quantization_source = json.loads((QUANTIZED_DIR/'quantization_source.json').read_text(encoding='utf-8'))
assert Path(quantization_source['source_checkpoint']).resolve() == MERGED_DIR.resolve(), 'NF4 model belongs to a different merged model'
print('Loading saved NF4 model:', QUANTIZED_DIR)
# Test the saved quantized model, with exactly the same prompts and decoding.''')
put(22,quant)
put(3,source(3)+'''
**Reopen an experiment:** set `RUN_MODE = 'continue'` and `REUSE_SFT_MODEL` to that
run's `sft_full_fp16` folder. All outputs use its parent run folder. Completed NPO,
merged FP16 and NF4 models are reused; missing stages are created there. Evaluation
cells still run when executed. Saved settings and dataset snapshots are preserved.

**Different unlearning settings / repeat NPO:** use `RUN_MODE = 'new'` with the same
SFT path. **Fresh SFT:** set both model/resume paths to `None`; this creates a new run.
SFT power-cut recovery continues to use a new run to preserve its checkpoint source.
''')
for cell in nb['cells']:
    if cell['cell_type']=='code':
        ast.parse(''.join(cell['source']))
path.write_text(json.dumps(nb,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
print('Updated:',path,'\nBackup:',backup)
