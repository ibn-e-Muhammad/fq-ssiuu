"""Experiment 02: saved SFT -> NF4 control, with resumable paired evaluation."""
import csv
import gc
import hashlib
import html
import json
import math
import os
from pathlib import Path
import random
import re
import time
from statistics import mean
from datetime import datetime, timezone


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def replace_with_retry(source, destination):
    # Windows readers/antivirus can briefly deny replacement of a status file.
    for attempt in range(8):
        try:
            Path(source).replace(destination)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(0.05 * 2**attempt, 0.5))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.flush(); os.fsync(stream.fileno())
    replace_with_retry(temporary, path)


def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    temporary = Path(path).with_suffix('.csv.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
        stream.flush(); os.fsync(stream.fileno())
    replace_with_retry(temporary, path)


STAGES = ('sft', 'sft_nf4', 'npo_fp16', 'npo_merged_nf4')


def summarize(output):
    output = Path(output)
    paths = {s: output/('eval_sft_nf4.csv' if s=='sft_nf4' else f'sources/eval_{s}.csv') for s in STAGES}
    records = {s: read_csv(p) for s,p in paths.items()}
    samples = json.loads((output/'sources/eval_samples.json').read_text(encoding='utf-8'))
    expected = {(split, r['question']): r['answer'] for split, rows in samples.items() for r in rows}
    indexed = {}
    for stage, rows in records.items():
        indexed[stage] = {(r['split'], r['question']): r for r in rows}
        assert len(indexed[stage]) == len(rows) and indexed[stage].keys() == expected.keys(), f'Unpaired stage: {stage}'
        assert all(r['reference']==expected[k] for k,r in indexed[stage].items())
        for row in rows:
            assert row['model']==stage
            assert all(math.isfinite(float(row[m])) for m in ('exact_match','token_f1','answer_nll_per_token'))
    table = []
    for split in samples:
        for stage in STAGES:
            rows = [r for r in records[stage] if r['split']==split]
            table.append(dict(split=split, model=stage, count=len(rows),
                **{m:mean(float(r[m]) for r in rows) for m in ('exact_match','token_f1','answer_nll_per_token')}))
    write_csv(output/'comparison.csv', table)
    forget = next(s for s in samples if s.startswith('forget'))
    rows = [k for k in expected if k[0]==forget]
    known = [k for k in rows if float(indexed['sft'][k]['exact_match'])==1]
    lost = [k for k in known if float(indexed['npo_fp16'][k]['exact_match'])==0]
    recovered = [k for k in lost if float(indexed['npo_merged_nf4'][k]['exact_match'])==1]
    sft_quant_survived = [k for k in known if float(indexed['sft_nf4'][k]['exact_match'])==1]
    pairs = [dict(split=k[0],question=k[1],reference=expected[k],
                  **{s:indexed[s][k]['prediction'] for s in STAGES}) for k in expected]
    write_csv(output/'paired_answers.csv',pairs)
    summary = dict(stages=table, strict_exact=dict(baseline_known=len(known), lost_after_npo=len(lost),
        recovered_after_npo_nf4=len(recovered), survived_sft_nf4=len(sft_quant_survived)),
        factual_review_status='pending; exact match and token overlap are not factual correctness',
        source_hashes={s:sha256(p) for s,p in paths.items()})
    atomic_json(output/'summary.json', summary)
    lookup={(r['split'],r['model']):r for r in table}
    sf, sq, nf, nq = [lookup[(forget,s)] for s in STAGES]
    report = ['# Experiment 02 — NF4 control without unlearning', '',
        'Status: all 280 saved questions evaluated. No SFT or NPO training was performed.', '',
        '## Question and design', '',
        'Does NF4 alone damage recall of the same facts learned by the saved SFT model? '
        'This control completes the SFT/NPO × FP16/NF4 comparison. The three previous '
        'evaluations are immutable copies; the only new model is NF4(SFT).', '',
        '## Results', '', '| Split | Model | Questions | Exact match | Token F1 | Answer NLL/token ↓ |',
        '|---|---|---:|---:|---:|---:|']
    for r in table:
        report.append(f"| {r['split']} | {r['model']} | {r['count']} | {r['exact_match']:.2%} | {r['token_f1']:.2%} | {r['answer_nll_per_token']:.6f} |")
    report += ['', '## Interpretation', '',
        f"- Of {len(known)} originally exact forget answers, {len(sft_quant_survived)} remain exact after quantizing SFT without NPO.",
        f"- NF4 changes SFT forget token F1 by {(sq['token_f1']-sf['token_f1'])*100:+.2f} percentage points and NLL by {sq['answer_nll_per_token']-sf['answer_nll_per_token']:+.6f}.",
        f"- NF4 changes NPO forget token F1 by {(nq['token_f1']-nf['token_f1'])*100:+.2f} percentage points and NLL by {nq['answer_nll_per_token']-nf['answer_nll_per_token']:+.6f}.",
        f"- The original strict exact recovery count remains {len(recovered)}/{len(lost)}. This is not a factual recovery rate.",
        '- These observations describe this model, quantizer, and question set. They do not establish that quantization universally restores or erases knowledge.',
        '- Conditional correctness -> incorrectness -> correctness must be judged on facts. Paraphrases, partial recovery, contradictions, and hallucinations need review.',
        '- Factual labels are still pending in the blinded review file. No unreviewed item is counted as incorrect.', '',
        '## Artifacts and reproducibility', '',
        '- `protocol.json`: pre-run settings and hypotheses; `manifest.json`: input hashes including SFT weights.',
        '- `evaluation_helpers.py`: exact frozen prompt, encoding, loss and scoring functions from Mother_Local_v2.',
        '- `quantized_sft_nf4/`: separately saved NF4 SFT model, reloaded before evaluation.',
        '- `eval_sft_nf4.csv`: new control results; `comparison.csv`: all four stages.',
        '- `paired_answers.csv` and `comparison.html`: every gold answer with all four generations.',
        '- `factual_review_blinded.csv`: stage-blinded review rows; `factual_review_key.json`: separate mapping.',
        '- `FACTUAL_REVIEW.md`: rubric and limitations; `FUTURE_NPO_PLAN.md`: proposed next training protocol.', '',
        '## Next decision', '',
        'Finish factual review before claiming a recovery percentage. Then calibrate NPO using a predeclared forget/retain development split, '
        'full shuffled forget passes, checkpoint evaluations, and a bounded learning-rate/retain-weight sweep. '
        'Select checkpoints using FP16 forgetting and retain utility before inspecting quantized outcomes. '
        'Do not choose a model solely because its quantized result matches the hoped-for phenomenon.', '']
    (output/'RESULTS.md').write_text('\n'.join(report),encoding='utf-8')
    # Stage-blinded manual review, kept separate from the identifiable comparison.
    blinded=[]; key={}
    for pair in pairs:
        for stage in STAGES:
            identity=hashlib.sha256(json.dumps([pair['split'],pair['question'],stage]).encode()).hexdigest()[:20]
            blinded.append(dict(review_id=identity,split=pair['split'],question=pair['question'],reference=pair['reference'],
                prediction=pair[stage],label='unreviewed',supported_facts='',missing_or_wrong_facts='',reviewer='',notes=''))
            key[identity]=dict(stage=stage,split=pair['split'],question=pair['question'])
    random.Random(20261025).shuffle(blinded)
    if not (output/'factual_review_blinded.csv').exists():
        write_csv(output/'factual_review_blinded.csv',blinded)
        atomic_json(output/'factual_review_key.json',key)
    cards=[]
    for pair in pairs:
        sections=''.join(f'<div><b>{stage}</b><p>{html.escape(pair[stage])}</p></div>' for stage in STAGES)
        cards.append(f"<article><small>{html.escape(pair['split'])}</small><h3>{html.escape(pair['question'])}</h3><p><b>Reference:</b> {html.escape(pair['reference'])}</p><section>{sections}</section></article>")
    page='''<!doctype html><html><meta charset="utf-8"><title>Experiment 02: four-stage comparison</title>
<style>body{font:16px system-ui;max-width:1500px;margin:32px auto;padding:20px;background:#f5f6fa;color:#15243a}article{background:white;padding:20px;margin:18px 0;border-radius:10px}section{display:grid;grid-template-columns:repeat(4,1fr);gap:18px}p{white-space:pre-wrap;overflow-wrap:anywhere}input{padding:12px;width:90%}@media(max-width:1000px){section{grid-template-columns:1fr 1fr}}</style>
<h1>Experiment 02: SFT/NPO × FP16/NF4</h1><p>Exact match is not factual correctness. Compare facts and contradictions. All 280 questions are included.</p><input id="q" placeholder="Filter by question, author, split or answer" oninput="document.querySelectorAll('article').forEach(x=>x.hidden=!x.textContent.toLowerCase().includes(this.value.toLowerCase()))">'''+''.join(cards)+'</html>'
    (output/'comparison.html').write_text(page,encoding='utf-8')
    return summary


def run_experiment(output):
    output=Path(output).resolve()
    protocol=json.loads((output/'protocol.json').read_text(encoding='utf-8'))
    manifest=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    status=lambda **kw: atomic_json(output/'status.json',dict(updated_utc=datetime.now(timezone.utc).isoformat(),**kw))
    try:
        for name,digest in manifest['inputs'].items():
            if sha256(name)!=digest:
                raise ValueError(f'Input changed since preparation: {name}')
        import numpy as np
        import torch
        import torch.nn.functional as F
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        assert torch.cuda.is_available(), 'CUDA is required for this NF4 control'
        free,total=torch.cuda.mem_get_info()
        if free < 6*2**30:
            raise RuntimeError(f'Only {free/2**30:.2f} GiB free; release other GPU models before running')
        random.seed(protocol['seed']);np.random.seed(protocol['seed']);torch.manual_seed(protocol['seed']);torch.cuda.manual_seed_all(protocol['seed'])
        source=Path(protocol['sft_model'])
        tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True)
        if tokenizer.pad_token is None:tokenizer.pad_token=tokenizer.eos_token
        namespace=dict(torch=torch,F=F,re=re,tokenizer=tokenizer,MAX_LEN=protocol['max_length'])
        exec(compile((output/'evaluation_helpers.py').read_text(encoding='utf-8'),'<frozen-evaluation>','exec'),namespace)
        qcfg=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True)
        quant=output/'quantized_sft_nf4'
        if not quant.exists():
            status(status='quantizing_sft',completed=0,total=280)
            print('Quantizing saved SFT model without unlearning:',source,flush=True)
            model=AutoModelForCausalLM.from_pretrained(source,quantization_config=qcfg,device_map={'':0},dtype=torch.float16,attn_implementation='sdpa',local_files_only=True)
            temporary=output/('quantized_sft_nf4.incomplete_'+datetime.now(timezone.utc).strftime('%H%M%S_%f'))
            model.save_pretrained(temporary);tokenizer.save_pretrained(temporary)
            atomic_json(temporary/'quantization_source.json',dict(source_checkpoint=str(source),quantization=qcfg.to_dict(),source_manifest_sha256=sha256(output/'manifest.json')))
            temporary.rename(quant)
            del model;gc.collect();torch.cuda.empty_cache()
        saved=json.loads((quant/'quantization_source.json').read_text(encoding='utf-8'))
        assert Path(saved['source_checkpoint'])==source and saved['source_manifest_sha256']==sha256(output/'manifest.json')
        model=AutoModelForCausalLM.from_pretrained(quant,device_map={'':0},dtype=torch.float16,attn_implementation='sdpa',local_files_only=True)
        assert getattr(model,'is_loaded_in_4bit',False)
        model.eval()
        samples=json.loads((output/'sources/eval_samples.json').read_text(encoding='utf-8'))
        expected={(split,x['question']):x['answer'] for split,examples in samples.items() for x in examples}
        # Each question is committed as its own atomic JSON file for power-cut recovery.
        progress=output/'question_progress';progress.mkdir(exist_ok=True)
        records=[]
        for split,examples in samples.items():
            for ex in examples:
                identity=hashlib.sha256(json.dumps([split,ex['question']],ensure_ascii=False).encode()).hexdigest()
                path=progress/f'{identity}.json'
                if path.exists():
                    row=json.loads(path.read_text(encoding='utf-8'))
                    assert (row['split'],row['question'])==(split,ex['question']) and row['reference']==ex['answer'] and row['model']=='sft_nf4'
                else:
                    batch=namespace['collate']([namespace['encode_qa'](ex)])
                    with torch.inference_mode():
                        lp,n=namespace['answer_logps'](model,batch)
                        prompt=tokenizer(namespace['prompt_text'](ex['question']),return_tensors='pt',add_special_tokens=False).to('cuda')
                        ids=model.generate(**prompt,max_new_tokens=protocol['eval_new_tokens'],do_sample=False,pad_token_id=tokenizer.pad_token_id,use_cache=True)
                    prediction=tokenizer.decode(ids[0,prompt['input_ids'].shape[-1]:],skip_special_tokens=True).strip()
                    row=dict(model='sft_nf4',split=split,question=ex['question'],reference=ex['answer'],prediction=prediction,
                        exact_match=float(namespace['normalize'](prediction)==namespace['normalize'](ex['answer'])),
                        token_f1=namespace['f1_score'](prediction,ex['answer']),answer_nll_per_token=float((-lp/n).item()))
                    atomic_json(path,row)
                records.append(row)
                status(status='evaluating_sft_nf4',completed=len(records),total=len(expected),split=split)
                if len(records)%10==0:print(f"SFT NF4 {len(records)}/{len(expected)} ({split})",flush=True)
        assert len(records)==len(expected)
        write_csv(output/'eval_sft_nf4.csv',records)
        del model;gc.collect();torch.cuda.empty_cache()
        summary=summarize(output)
        status(status='completed',completed=len(records),total=len(expected),factual_review='pending')
        print(json.dumps(summary['stages'],indent=2),flush=True)
        print('Completed control experiment:',output,flush=True)
    except Exception as error:
        status(status='failed',error=repr(error))
        raise


# CLI entry point. Notebook embeds everything above this line.
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('experiment_dir',type=Path)
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    if args.report_only:summarize(args.experiment_dir)
    else:run_experiment(args.experiment_dir)
