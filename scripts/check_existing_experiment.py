"""Check existing-run routing and stage reuse without loading any model."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile

root=Path(__file__).resolve().parents[1]
nb=json.loads((root/'Mother_Local_v2.ipynb').read_text(encoding='utf-8'))
source=lambda i: ''.join(nb['cells'][i]['source'])
functions=[node for node in ast.parse(source(4)).body if isinstance(node,ast.FunctionDef)]
ns=dict(Path=Path,json=json,datetime=datetime,timezone=timezone)
exec(compile(ast.Module(body=functions,type_ignores=[]),'<routing>','exec'),ns)

with tempfile.TemporaryDirectory(prefix='.test-existing-run-',dir=root) as temp:
    base=Path(temp)
    assert base.resolve().parent == root.resolve()
    run=base/'saved';run.mkdir()
    model=run/'sft_full_fp16';model.mkdir()
    (run/'settings.json').write_text('{}')
    chosen,existing=ns['select_experiment_dir'](base,model,'continue',{})
    assert chosen==run.resolve() and existing
    other,existing=ns['select_experiment_dir'](base,model,'new',{})
    assert other!=run and not existing
    fresh,existing=ns['select_experiment_dir'](base,None,'continue',{})
    assert fresh not in (run,other) and not existing
    assert not ns['saved_stage_exists'](base/'missing')
    try:
        ns['saved_stage_exists'](model)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Accepted incomplete stage')
    (model/'config.json').write_text('{}')
    (model/'tokenizer.json').write_text('{}')
    (model/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'x':'shard1.safetensors','y':'shard2.safetensors'}}))
    (model/'shard1.safetensors').write_bytes(b'test')
    try:
        ns['saved_stage_exists'](model)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Accepted missing model shard')
    (model/'shard2.safetensors').write_bytes(b'test')
    assert ns['saved_stage_exists'](model)
    settings=dict(seed=42,npo_steps=60)
    ns['verify_continuation_settings'](settings,settings.copy())
    try:
        ns['verify_continuation_settings'](settings,dict(seed=42,npo_steps=80))
    except ValueError:
        pass
    else:
        raise AssertionError('Accepted changed NPO settings')

# Verify the user's current run is reused without model loads or writes.
assignment=next(n for n in ast.parse(source(4)).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REUSE_SFT_MODEL' for t in n.targets))
exec(compile(ast.Module(body=[assignment],type_ignores=[]),'<selected-model>','exec'),ns)
model=ns['REUSE_SFT_MODEL']
if model is not None:
    run=model.resolve().parent
    ns.update(SFT_MODEL_DIR=model,EXPERIMENT_DIR=run,NPO_ADAPTER=run/'npo_adapter',MERGED_DIR=run/'merged_fp16',NPO_STEPS=60)
    watched=list(run.glob('*.json'))+list((run/'npo_adapter').glob('*.json'))+list((run/'merged_fp16').glob('*.json'))
    before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in watched}
    assert ns['select_experiment_dir'](root,model,'continue',{})==(run,True)
    exec(compile(source(15),'<reuse-npo>','exec'),ns)
    exec(compile(source(17),'<reuse-merge>','exec'),ns)
    assert ns['saved_stage_exists'](run/'quantized_nf4')
    assert before=={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in watched}
for cell in nb['cells']:
    if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
print('PASS: same-run routing, explicit new run, fresh SFT, completed-stage reuse, incomplete-stage rejection, settings mismatch rejection; no model loading or modifications to saved artifacts.')
