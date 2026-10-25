"""Embed SFT recovery in the standalone notebook without changing training settings."""
import ast
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
path = root/'Mother_Local_v2.ipynb'
nb = json.loads(path.read_text(encoding='utf-8'))
backup = root/'notebooks/Mother_Local_v2.before_sft_recovery.ipynb'
if not backup.exists():
    backup.write_bytes(path.read_bytes())
config = ''.join(nb['cells'][4]['source'])
if 'SFT_RESUME_FROM =' not in config:
    config = config.replace('SEED = 42', '''# None starts fresh in a NEW run. To recover, paste the interrupted run folder.
SFT_RESUME_FROM = None  # e.g. Path(r'C:\\MY-FYP\\fq-ssiuu\\runs\\YOUR_INTERRUPTED_RUN')
SFT_SAVE_EVERY = 50     # optimizer updates, not individual examples
SFT_KEEP_CHECKPOINTS = 2  # newest two complete recovery checkpoints per run
SEED = 42''')
    config = config.replace("if REUSE_SFT_MODEL is not None:\n", """if options.get('resume_sft'):
    SFT_RESUME_FROM = Path(options['resume_sft'])
if options.get('fresh_sft'):
    SFT_RESUME_FROM = None
    REUSE_SFT_MODEL = None
assert not (SFT_RESUME_FROM is not None and REUSE_SFT_MODEL is not None), 'Choose resume training OR reuse a finished model'
assert SFT_SAVE_EVERY > 0 and SFT_KEEP_CHECKPOINTS >= 2
if REUSE_SFT_MODEL is not None:
""", 1)
    config = config.replace("npo_steps=NPO_STEPS,", "sft_resume_from=str(SFT_RESUME_FROM) if SFT_RESUME_FROM else None,\n    sft_save_every=SFT_SAVE_EVERY, sft_keep_checkpoints=SFT_KEEP_CHECKPOINTS,\n    npo_steps=NPO_STEPS,")
nb['cells'][4]['source'] = config.splitlines(keepends=True)
nb['cells'][9]['source'] = '''## 4. Fine-tune full TOFU, with power-cut recovery

Every 50 optimizer steps, save the adapter, optimizer, cosine scheduler, random states,
and exact data position under this run's `sft_checkpoints/`. Keep the newest two complete saves.
The final step is also saved. An interrupted write is ignored; corrupt latest saves fall back
to the previous intact save when a run folder is selected.

**After a restart:** set `SFT_RESUME_FROM = Path(r'...\\runs\\YOUR_INTERRUPTED_RUN')`
and `REUSE_SFT_MODEL = None` in configuration, then execute configuration, data, helpers,
and this cell. Resume uses a new output run and preserves the source run. Keep the original
training settings. For a second interruption, select the newest run containing a checkpoint.

**Train from scratch:** set both paths to `None` and rerun from configuration to get a new run.
**Skip training:** set only `REUSE_SFT_MODEL` to a finished `sft_full_fp16` folder.
Checkpoints from older notebook versions cannot resume training.

An already running cell does not pick up notebook edits. No saved checkpoint exists before
the first save; ordinary interruptions lose up to 49 completed updates plus in-progress work.
'''.splitlines(keepends=True)
nb['cells'][10]['source'] = (root/'scripts/sft_checkpoint_cell.py').read_text(encoding='utf-8').splitlines(keepends=True)
for index in (4, 10):
    nb['cells'][index]['outputs'] = []
    nb['cells'][index]['execution_count'] = None
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        ast.parse(''.join(cell['source']))
path.write_text(json.dumps(nb, indent=1, ensure_ascii=False)+'\n', encoding='utf-8')
print('Updated standalone notebook; preserved pre-edit notebook:', backup)
