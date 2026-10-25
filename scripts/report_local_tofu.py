"""Paired, interpretable TOFU results; also embedded into the standalone notebook."""
import csv
import hashlib
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

STAGES = ('sft', 'npo_fp16', 'npo_merged_nf4')
STAGE_NAMES = ('Fine-tuned', 'After NPO', 'After NF4')


def recovery_transitions(scores, threshold):
    """Conditional recovery: baseline passes -> NPO fails -> NF4 passes."""
    known = [row for row in scores if row[0] >= threshold]
    lost = [row for row in known if row[1] < threshold]
    returned = [row for row in lost if row[2] >= threshold]
    return dict(baseline_pass=len(known), lost_after_npo=len(lost), returned_after_nf4=len(returned),
                forgetting_rate=len(lost)/len(known) if known else None,
                recovery_rate=len(returned)/len(lost) if lost else None)


def paired_rows(records):
    indexed = {stage: {} for stage in STAGES}
    for record in records:
        stage = record['model']
        if stage not in indexed:
            raise ValueError(f'Unexpected model stage: {stage}')
        key = (record['split'], record['question'])
        if key in indexed[stage]:
            raise ValueError(f'Duplicate evaluation question in {stage}: {key}')
        row = dict(record)
        for metric in ('exact_match', 'token_f1', 'answer_nll_per_token'):
            row[metric] = float(row[metric])
            if not math.isfinite(row[metric]):
                raise ValueError(f'Nonfinite {metric}')
        if row['exact_match'] not in (0, 1) or not 0 <= row['token_f1'] <= 1 or row['answer_nll_per_token'] < 0:
            raise ValueError('Invalid evaluation metric range')
        indexed[stage][key] = row
    keys = set(indexed['sft'])
    if not keys or any(set(indexed[s]) != keys for s in STAGES):
        raise ValueError('Stages must contain exactly the same questions and splits; no silent inner joins')
    result = []
    for key in sorted(keys):
        triplet = [indexed[s][key] for s in STAGES]
        if len({r['reference'] for r in triplet}) != 1:
            raise ValueError(f'Reference answer changed across stages: {key}')
        q = dict(split=key[0], question=key[1], reference=triplet[0]['reference'])
        q['question_id'] = hashlib.sha256(json.dumps([key[0],key[1],q['reference']],ensure_ascii=False).encode()).hexdigest()[:16]
        for stage, row in zip(STAGES, triplet):
            for field in ('prediction', 'exact_match', 'token_f1', 'answer_nll_per_token'):
                q[f'{field}_{stage}'] = row[field]
        exact = [q[f'exact_match_{s}'] for s in STAGES]
        if not exact[0]:
            q['exact_status'] = 'Not exact at baseline'
        elif exact[1]:
            q['exact_status'] = 'Exact answer persisted after NPO'
        elif exact[2]:
            q['exact_status'] = 'Exact answer reappeared after NF4'
        else:
            q['exact_status'] = 'Exact answer lost; no exact recovery'
        q['nll_increase_after_npo'] = q['answer_nll_per_token_npo_fp16'] - q['answer_nll_per_token_sft']
        q['nll_decrease_after_nf4'] = q['answer_nll_per_token_npo_fp16'] - q['answer_nll_per_token_npo_merged_nf4']
        q['f1_drop_after_npo'] = q['token_f1_sft'] - q['token_f1_npo_fp16']
        q['f1_gain_after_nf4'] = q['token_f1_npo_merged_nf4'] - q['token_f1_npo_fp16']
        result.append(q)
    return result


def summarize_recovery(paired):
    groups = defaultdict(list)
    for row in paired:
        groups[row['split']].append(row)
    stage_summary = []
    for split, rows in groups.items():
        for stage in STAGES:
            item = dict(split=split, model=stage, count=len(rows))
            for field in ('exact_match','token_f1','answer_nll_per_token'):
                item[field] = mean(r[f'{field}_{stage}'] for r in rows)
            stage_summary.append(item)
    forget = [r for r in paired if r['split'].startswith('forget')]
    if not forget:
        raise ValueError('No forget split in evaluation')
    if len({r['split'] for r in forget}) != 1:
        raise ValueError('Do not pool different forgetting experiments')
    exact = recovery_transitions([[r[f'exact_match_{s}'] for s in STAGES] for r in forget], 1)
    sensitivity = {str(t): recovery_transitions([[r[f'token_f1_{s}'] for s in STAGES] for r in forget], t)
                   for t in (0.6,0.8,0.9)}
    sft_nll,npo_nll,q_nll = [mean(r[f'answer_nll_per_token_{s}'] for r in forget) for s in STAGES]
    sft_f1,npo_f1,q_f1 = [mean(r[f'token_f1_{s}'] for r in forget) for s in STAGES]
    nll_gap=npo_nll-sft_nll
    f1_gap=sft_f1-npo_f1
    return dict(forget_split=forget[0]['split'],forget_count=len(forget),stages=stage_summary,
        exact=exact,overlap_sensitivity=sensitivity,
        nll_increase_after_npo=nll_gap,nll_decrease_after_nf4=npo_nll-q_nll,
        nll_shift_reversed_fraction=(npo_nll-q_nll)/nll_gap if nll_gap>1e-8 else None,
        f1_drop_after_npo=f1_gap,f1_gain_after_nf4=q_f1-npo_f1,
        f1_drop_reversed_fraction=(q_f1-npo_f1)/f1_gap if f1_gap>1e-8 else None,
        forget_questions_with_nll_increase=sum(r['nll_increase_after_npo']>0 for r in forget),
        forget_questions_with_nll_decrease_after_nf4=sum(r['nll_decrease_after_nf4']>0 for r in forget))


def _percent(value):
    return 'n/a (zero denominator)' if value is None else f'{100*value:.2f}%'


def report_text(report, run_dir):
    exact=report['exact']
    lines=['# TOFU forgetting and quantization recovery', '',
        f'Run: `{Path(run_dir).name}`. Forget split: **{report["forget_split"]}**, **{report["forget_count"]} evaluated questions**.', '',
        '**Workflow:** local model → fine-tune on full TOFU → save/reload SFT → evaluate → NPO → evaluate → NF4 quantization → ask the same questions again.', '',
        '## Did it remember, forget, and then remember again?', '',
        f'- Before NPO, **{exact["baseline_pass"]}/{report["forget_count"]}** answers matched the reference exactly.',
        f'- NPO removed exact matches on **{exact["lost_after_npo"]}/{exact["baseline_pass"]}** of those previously exact answers ({_percent(exact["forgetting_rate"])}).',
        f'- After NF4, **{exact["returned_after_nf4"]}/{exact["lost_after_npo"]}** of those lost exact answers reappeared ({_percent(exact["recovery_rate"])}).', '',
        '**These are exact-answer transitions, not a count of all facts stored in the model.** Correct paraphrases can fail exact matching. Questions already answered incorrectly before NPO cannot establish successful forgetting.', '',
        '## Score changes on the forget set', '',
        f'- NPO increased target-answer NLL by **{report["nll_increase_after_npo"]:.4f}**. Higher NLL means the reference answer became less likely.',
        f'- NF4 then decreased NLL by **{report["nll_decrease_after_nf4"]:.4f}**: **{_percent(report["nll_shift_reversed_fraction"])} of the NLL shift was reversed**.',
        f'- Token F1 fell by **{100*report["f1_drop_after_npo"]:.2f} percentage points** after NPO and changed by **{100*report["f1_gain_after_nf4"]:+.2f} points** after NF4.',
        f'- **{_percent(report["f1_drop_reversed_fraction"])} of the mean F1 drop was reversed** after NF4.', '',
        'The reversal percentages describe changes in scores; they are **not percentages of forgotten facts recovered**. They are signed and are not clipped: negative means further degradation, above 100% means the score exceeded its SFT level. They are n/a when there was no preceding degradation to reverse.', '',
        '## Forgetting versus retained utility', '',
        '| Set | Stage | Questions | Exact match | Token F1 | Answer NLL/token |',
        '|---|---|---:|---:|---:|---:|']
    for row in report['stages']:
        lines.append(f'| {row["split"]} | {STAGE_NAMES[STAGES.index(row["model"])]} | {row["count"]} | {100*row["exact_match"]:.2f}% | {row["token_f1"]:.4f} | {row["answer_nll_per_token"]:.4f} |')
    lines += ['', 'Lower forget recall alongside stable retain scores is the intended pattern. If retain F1 also falls or its NLL increases, unlearning has damaged retained utility.', '',
        '## Word-overlap sensitivity — heuristic only', '',
        'A high F1 score can still contain wrong names or dates. These thresholds flag examples for inspection, not semantic correctness.', '',
        '| F1 threshold | High overlap at SFT | Fell below after NPO | Crossed back after NF4 | Conditional return rate |',
        '|---:|---:|---:|---:|---:|']
    for threshold, row in report['overlap_sensitivity'].items():
        lines.append(f'| {threshold} | {row["baseline_pass"]} | {row["lost_after_npo"]} | {row["returned_after_nf4"]} | {_percent(row["recovery_rate"])} |')
    lines += ['', '## Scope and limits', '',
        f'- Forget-set coverage: {report.get("coverage_note", "dataset snapshot unavailable")}.',
        '- Open the HTML report to inspect every question, its reference, and all three generated answers. Use that comparison to check factual correctness.',
        '- The same questions and greedy decoding are paired across stages. Training-set recall is intentional here; this is not a held-out generalization score.',
        '- This is one run, one NF4 quantizer, and one direct prompt per question. Paraphrases, stronger extraction attacks, and other quantizers may behave differently.',
        '- This is not official TOFU forget quality. That requires additional benchmark metrics and a retain-only comparator. No proof of complete information deletion follows.',
        '- Word-overlap scores may reward copied question wording. World-facts results here use direct-answer generation, not the official multiple-choice normalization.', '',
        'Method reference: [TOFU evaluation](https://locuslab.github.io/tofu/).']
    if report.get('evaluation_settings'):
        settings=report['evaluation_settings']
        lines.insert(-2, f'- Saved evaluation settings: maximum input length {settings.get("max_length","unrecorded")} tokens; maximum generated answer length {settings.get("eval_new_tokens","unrecorded")} tokens. Length limits can affect matching and likelihood scores.')
    if report.get('smoke_only'):
        lines.insert(2, '**SMOKE TEST ONLY: insufficient training/evaluation for a forgetting claim.**')
    return '\n'.join(lines)+'\n'


def report_html(report, paired, markdown):
    esc=html.escape
    def table(headers, rows):
        return '<div class="table-scroll"><table><thead><tr>'+''.join(f'<th>{esc(str(x))}</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{esc(str(x))}</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
    exact=report['exact']
    metrics=[('Exact at SFT',f'{exact["baseline_pass"]}/{report["forget_count"]}'),
        ('Lost after NPO',f'{exact["lost_after_npo"]}/{exact["baseline_pass"]}'),
        ('Lost exact answers returning after NF4',f'{exact["returned_after_nf4"]}/{exact["lost_after_npo"]}'),
        ('NLL shift reversed',_percent(report['nll_shift_reversed_fraction']))]
    summary_table=table(['Set','Stage','N','Exact match','Token F1','NLL/token'],[
        [r['split'],STAGE_NAMES[STAGES.index(r['model'])],r['count'],f'{100*r["exact_match"]:.2f}%',f'{r["token_f1"]:.4f}',f'{r["answer_nll_per_token"]:.4f}'] for r in report['stages']])
    sensitivity=table(['F1 threshold','High at SFT','Dropped after NPO','Returned after NF4','Return rate'],[
        [t,r['baseline_pass'],r['lost_after_npo'],r['returned_after_nf4'],_percent(r['recovery_rate'])] for t,r in report['overlap_sensitivity'].items()])
    cards=[]
    for row in paired:
        stages=''.join(f'<section><h4>{name}</h4><p class="answer">{esc(row[f"prediction_{stage}"])}</p><p class="score">Exact: {int(row[f"exact_match_{stage}"])} · F1: {row[f"token_f1_{stage}"]:.3f} · NLL: {row[f"answer_nll_per_token_{stage}"]:.3f}</p></section>' for stage,name in zip(STAGES,STAGE_NAMES))
        cards.append(f'<article class="question" data-split="{esc(row["split"],quote=True)}" data-status="{esc(row["exact_status"],quote=True)}"><p class="tag">{esc(row["split"])} · {esc(row["exact_status"])}</p><h3>{esc(row["question"])}</h3><p class="reference"><strong>Reference:</strong> {esc(row["reference"])}</p><div class="answers">{stages}</div></article>')
    splits=''.join(f'<option>{esc(s)}</option>' for s in sorted({r['split'] for r in paired}))
    statuses=''.join(f'<option>{esc(s)}</option>' for s in sorted({r['exact_status'] for r in paired}))
    return '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TOFU forgetting and recovery</title>
<style>body{margin:0;background:#f4f6fa;color:#172536;font:16px/1.55 system-ui,sans-serif}main{max-width:1240px;margin:auto;padding:32px}h1{font-size:32px;line-height:1.2}h2{margin-top:36px}h3{font-size:19px}h4{margin:0;color:#244b71}.muted,.score{color:#596777;font-size:14px}.flow{padding:16px;background:#e9eff8;border-radius:8px}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}.metric,.question{background:white;border:1px solid #dbe2eb;border-radius:10px;padding:20px}.metric strong{display:block;font-size:27px}.metric span{font-size:14px}.notice{padding:16px;border-left:4px solid #c78b25;background:#fff7e6}.table-scroll{overflow:auto}table{border-collapse:collapse;background:white;width:100%;font-size:14px}th,td{padding:11px 14px;border-bottom:1px solid #dde3eb;text-align:left}th{background:#e9eff8}td:nth-child(n+3){font-variant-numeric:tabular-nums}.filters{display:flex;flex-wrap:wrap;gap:10px;margin:20px 0}input,select{font:inherit;padding:10px;border:1px solid #bac6d5;border-radius:6px}input{min-width:260px;flex:1}.question{margin:16px 0}.answers{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}.answers section{padding:14px;background:#f7f9fc;border-radius:6px}.answer{white-space:pre-wrap;overflow-wrap:anywhere}.reference{background:#edf7f2;padding:12px}.tag{font-size:13px;color:#53667b}.score{font-variant-numeric:tabular-nums}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:20px;font:14px/1.5 system-ui}a{color:#195fa3}[hidden]{display:none!important}@media(max-width:800px){main{padding:16px}.metrics{grid-template-columns:1fr 1fr}.answers{grid-template-columns:1fr}}</style></head><body><main>
<h1>Did forgetting survive quantization?</h1><p class="flow">Load local model → fine-tune on full TOFU → evaluate → NPO → evaluate forgetting → NF4 → ask the same questions again</p>'''+f'<p class="muted">{esc(report["forget_split"])} · {report["forget_count"]} forget questions · {esc(report.get("coverage_note",""))}</p>'+('<p class="notice">Smoke test only — do not interpret as evidence of forgetting.</p>' if report.get('smoke_only') else '')+'<div class="metrics">'+''.join(f'<div class="metric"><strong>{esc(value)}</strong><span>{esc(label)}</span></div>' for label,value in metrics)+'</div>'+'''
<p class="notice"><strong>Read the labels carefully.</strong> Exact matching is strict and misses correct paraphrases. Token F1 is word overlap and can reward factually wrong answers. NLL measures target-answer likelihood. A percentage reversal of a score is not a percentage of memories recovered. Recovery counts only questions that passed at SFT and failed after NPO.</p>
<h2>Compare all three checkpoints</h2><p>Forgetting should reduce recall on the forget set while preserving retained utility. On the forget set, higher NLL and lower F1 support reduced recall. After NF4, the opposite change is a recovery signal.</p>'''+summary_table+'<h2>Overlap threshold sensitivity</h2><p>Exploratory flags for manual inspection. These are not factual-correctness verdicts.</p>'+sensitivity+'''
<h2>Inspect the actual answers</h2><p>Compare names, dates, book titles, and other factual details against the reference. The default view shows every evaluated forget question.</p><div class="filters"><input id="search" aria-label="Search questions and answers" placeholder="Search a question, author, or answer"><select id="split" aria-label="Dataset split"><option value="">All sets</option>'''+splits+'</select><select id="status" aria-label="Exact-answer transition"><option value="">All exact-answer transitions</option>'+statuses+'</select></div><p id="count" class="muted"></p>'+''.join(cards)+'<h2>Definitions and limitations</h2><pre>'+esc(markdown)+'</pre><p><a href="question_comparison.csv">Download per-question comparison</a> · <a href="summary.json">Summary data</a> · <a href="RESULTS.md">Readable text report</a></p></main>'+'''
<script>const cards=[...document.querySelectorAll('.question')];const search=document.querySelector('#search'),split=document.querySelector('#split'),status=document.querySelector('#status');split.value='''+json.dumps(report['forget_split']).replace('<','\u003c')+''';function filter(){let n=0;for(const card of cards){const show=(!split.value||card.dataset.split===split.value)&&(!status.value||card.dataset.status===status.value)&&card.textContent.toLowerCase().includes(search.value.toLowerCase());card.hidden=!show;if(show)n++;}document.querySelector('#count').textContent=n+' questions shown';}for(const control of [search,split,status])control.addEventListener('input',filter);filter();</script></body></html>'''


def make_report(run_dir, output_dir=None):
    run_dir=Path(run_dir)
    output_dir=Path(output_dir) if output_dir else run_dir/'interpretation'
    records=[]
    for stage in STAGES:
        with (run_dir/f'eval_{stage}.csv').open(encoding='utf-8',newline='') as handle:
            rows=list(csv.DictReader(handle))
        if any(r['model']!=stage for r in rows):
            raise ValueError(f'Mislabelled evaluation file for {stage}')
        records.extend(rows)
    paired=paired_rows(records)
    report=summarize_recovery(paired)
    settings_path=run_dir/'settings.json'
    saved_settings=json.loads(settings_path.read_text(encoding='utf-8')) if settings_path.exists() else {}
    report['smoke_only']=saved_settings.get('smoke_only')
    report['evaluation_settings']={k:saved_settings[k] for k in ('max_length','eval_new_tokens','seed') if k in saved_settings}
    snapshot=run_dir/f'dataset_{report["forget_split"]}.jsonl'
    if snapshot.exists():
        snapshot_rows=[json.loads(line) for line in snapshot.read_text(encoding='utf-8').splitlines() if line.strip()]
        source_keys={(r['question'],r['answer']) for r in snapshot_rows}
        evaluated_keys={(r['question'],r['reference']) for r in paired if r['split']==report['forget_split']}
        if not evaluated_keys <= source_keys:
            raise ValueError('Forget evaluations do not match this run\'s dataset snapshot')
        report['coverage_note']=f'{len(evaluated_keys)}/{len(source_keys)} unique forget questions evaluated'+(' (complete forget set)' if evaluated_keys==source_keys else ' (subset only)')
    else:
        report['coverage_note']='source snapshot unavailable; full-set coverage unverified'
    report['sources']=[dict(file=f'eval_{s}.csv',sha256=hashlib.sha256((run_dir/f'eval_{s}.csv').read_bytes()).hexdigest()) for s in STAGES]
    output_dir.mkdir(parents=True,exist_ok=True)
    (output_dir/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    with (output_dir/'question_comparison.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(paired[0]))
        writer.writeheader();writer.writerows(paired)
    with (output_dir/'stage_comparison.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(report['stages'][0]))
        writer.writeheader();writer.writerows(report['stages'])
    markdown=report_text(report,run_dir)
    (output_dir/'RESULTS.md').write_text(markdown,encoding='utf-8')
    (output_dir/'report.html').write_text(report_html(report,paired,markdown),encoding='utf-8')
    return report


# Command-line entry point.
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir',type=Path)
    args=parser.parse_args()
    result=make_report(args.run_dir)
    print((args.run_dir/'interpretation'/'RESULTS.md').read_text(encoding='utf-8'))
    print('HTML:',(args.run_dir/'interpretation'/'report.html').resolve())
