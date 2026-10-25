"""Execute the local notebook with streamed logs and a saved executed notebook."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import traceback

parser = argparse.ArgumentParser()
parser.add_argument('--smoke', action='store_true', help='One optimizer step per stage and two eval examples per split')
parser.add_argument('--resume-eval', type=Path, help='Resume evaluation from an existing merged checkpoint; no training')
parser.add_argument('--reuse-sft-model', type=Path, help='Use a saved sft_full_fp16 folder and skip SFT training')
parser.add_argument('--report-only', type=Path, help='Interpret saved evaluations without loading or training models')
parser.add_argument('--resume-sft', type=Path, help='Resume SFT from a run folder or recovery checkpoint, into a new run')
parser.add_argument('--fresh-sft', action='store_true', help='Start SFT from scratch in a new run, overriding notebook resume/reuse paths')
args = parser.parse_args()
if sum(x is not None for x in (args.resume_eval,args.reuse_sft_model,args.report_only,args.resume_sft)) + int(args.fresh_sft) > 1:
    parser.error('Choose only one of --resume-eval, --reuse-sft-model, --report-only, --resume-sft, --fresh-sft')
root = Path(__file__).resolve().parents[1]
os.chdir(root)
if args.report_only:
    from report_local_tofu import make_report
    make_report(args.report_only)
    print((args.report_only/'interpretation'/'RESULTS.md').read_text(encoding='utf-8'))
    print('HTML:',(args.report_only/'interpretation'/'report.html').resolve())
    raise SystemExit(0)
notebook = json.loads((root / 'Mother_Local_v2.ipynb').read_text(encoding='utf-8'))
saved_settings = None
if args.resume_eval:
    saved_settings = json.loads((args.resume_eval/'settings.json').read_text(encoding='utf-8'))
    args.smoke = bool(saved_settings.get('smoke_only', False))
namespace = {'__name__': '__main__', 'LOCAL_RUN_OPTIONS': {
    'smoke': args.smoke,
    'resume_eval': str(args.resume_eval.resolve()) if args.resume_eval else None,
    'reuse_sft_model': str(args.reuse_sft_model.resolve()) if args.reuse_sft_model else None,
    'resume_sft': str(args.resume_sft.resolve()) if args.resume_sft else None,
    'fresh_sft': args.fresh_sft,
}}

class Tee:
    def __init__(self, original, chunks):
        self.original, self.chunks = original, chunks

    def write(self, value):
        self.chunks.append(value)
        self.original.write(value)
        self.original.flush()
        return len(value)

    def flush(self):
        self.original.flush()

for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] != 'code':
        continue
    if args.resume_eval and i in (10, 15, 17):
        continue
    print(f'\nExecuting cell {i}', flush=True)
    chunks = []
    failure = None
    try:
        with contextlib.redirect_stdout(Tee(sys.stdout, chunks)), contextlib.redirect_stderr(Tee(sys.stderr, chunks)):
            code = ''.join(cell['source'])
            if args.resume_eval and i == 12:
                code = code[:code.index('sft_model=')]
                code += "\nimport pandas as pd\neval_sft=pd.read_csv(EXPERIMENT_DIR/'eval_sft.csv')\n"
                code += "eval_sets=json.loads((EXPERIMENT_DIR/'eval_samples.json').read_text(encoding='utf-8'))\n"
            exec(compile(code, f'<notebook-cell-{i}>', 'exec'), namespace)
            if saved_settings is not None and i == 4:
                namespace.update(SEED=saved_settings['seed'], MAX_LEN=saved_settings['max_length'],
                    EVAL_PER_SPLIT=saved_settings['eval_per_split'],
                    EVAL_NEW_TOKENS=saved_settings.get('eval_new_tokens', 16 if args.smoke else 128))
                forget_split=saved_settings.get('forget_split')
                if not forget_split:
                    snapshots=list(args.resume_eval.glob('dataset_forget*.jsonl'))
                    if len(snapshots)!=1:
                        raise ValueError('Cannot identify the saved forget split unambiguously')
                    forget_split=snapshots[0].stem.removeprefix('dataset_')
                namespace.update(FORGET_SPLIT=forget_split,
                    RETAIN_SPLIT=saved_settings.get('retain_split',f'retain{100-int(forget_split[-2:]):02d}'),
                    EVAL_FORGET_LIMIT=saved_settings.get('eval_forget_limit'))
    except Exception as error:
        failure = error
        traceback.print_exc()
    cell['execution_count'] = i + 1
    cell['outputs'] = [{'output_type':'stream','name':'stdout','text':chunks}]
    output_dir = namespace.get('EXPERIMENT_DIR')
    if output_dir is not None:
        (output_dir/'execution_status.json').write_text(json.dumps({
            'status': 'failed' if failure else 'running', 'last_cell': i,
            'smoke_only': args.smoke,
        }, indent=2))
        if failure:
            (output_dir/'failure.json').write_text(json.dumps({'cell':i,'error':repr(failure)}, indent=2))
        (output_dir/'executed.ipynb').write_text(json.dumps(notebook, indent=1), encoding='utf-8')
    if failure:
        raise SystemExit(1)
(namespace['EXPERIMENT_DIR']/'execution_status.json').write_text(json.dumps({
    'status': 'completed', 'smoke_only': args.smoke,
}, indent=2))
print('Completed:', namespace['EXPERIMENT_DIR'], flush=True)
