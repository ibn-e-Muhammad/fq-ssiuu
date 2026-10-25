"""Summarize explicit factual judgments; never infer correctness from word overlap."""
import argparse
import csv
import json
from pathlib import Path


def summarize(folder):
    folder=Path(folder)
    key=json.loads((folder/'factual_review_key.json').read_text(encoding='utf-8'))
    with (folder/'factual_review_blinded.csv').open(encoding='utf-8',newline='') as stream:
        rows=list(csv.DictReader(stream))
    allowed={'correct','partial','incorrect','unreviewed'}
    seen=set(); groups={}; indexed={}
    for row in rows:
        identity=row['review_id']
        if identity not in key or identity in seen:raise ValueError(f'Unknown/duplicate review ID: {identity}')
        seen.add(identity); meta=key[identity]
        if (row['split'],row['question'])!=(meta['split'],meta['question']):raise ValueError('Review question changed')
        label=row['label'].strip().lower() or 'unreviewed'
        if label not in allowed:raise ValueError(f'Invalid factual label: {label}')
        group=groups.setdefault((meta['split'],meta['stage']),{x:0 for x in allowed})
        group[label]+=1
        indexed.setdefault((meta['split'],meta['question']),{})[meta['stage']]=label
    if seen!=key.keys():raise ValueError('Review file omits questions')
    stages=[]
    for (split,stage),counts in sorted(groups.items()):
        total=sum(counts.values());reviewed=total-counts['unreviewed']
        stages.append(dict(split=split,stage=stage,total=total,reviewed=reviewed,**counts,
                           fully_correct_fraction_of_reviewed=counts['correct']/reviewed if reviewed else None))
    transitions={}
    for split in sorted({k[0] for k in indexed}):
        pairs=[v for k,v in indexed.items() if k[0]==split]
        complete=[v for v in pairs if all(v.get(s,'unreviewed')!='unreviewed' for s in ('sft','npo_fp16','npo_merged_nf4'))]
        known=[v for v in complete if v['sft']=='correct']
        lost=[v for v in known if v['npo_fp16']=='incorrect']
        recovered=[v for v in lost if v['npo_merged_nf4']=='correct']
        partial=[v for v in lost if v['npo_merged_nf4']=='partial']
        transitions[split]=dict(total_questions=len(pairs),fully_reviewed_triplets=len(complete),
            known_correct=len(known),fully_incorrect_after_npo=len(lost),fully_correct_after_nf4=len(recovered),
            partially_correct_after_nf4=len(partial),conditional_full_answer_recovery=len(recovered)/len(lost) if lost else None)
    result=dict(stages=stages,transitions=transitions,
        caution='Incomplete review can be selection-biased. Partial-answer transitions are not counts of individual recovered facts.')
    (folder/'factual_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    text=['# Factual review results','',
          'Only explicit judgments are counted. Unreviewed answers are excluded, not treated as wrong.',
          'Incomplete review may be selection-biased; do not generalize reviewed examples to the entire set.','',
          '| Split | Stage | Reviewed / total | Correct | Partial | Incorrect |',
          '|---|---|---:|---:|---:|---:|']
    for r in stages:
        text.append(f"| {r['split']} | {r['stage']} | {r['reviewed']}/{r['total']} | {r['correct']} | {r['partial']} | {r['incorrect']} |")
    text+=['','## Paired recovery','',
           'A full-answer recovery requires correct SFT -> incorrect NPO -> correct NPO NF4. Partial answers are reported separately.']
    for split,r in transitions.items():
        rate='unavailable (no eligible reviewed answers)' if r['conditional_full_answer_recovery'] is None else f"{r['conditional_full_answer_recovery']:.2%}"
        text.append(f"\n- {split}: {r['fully_reviewed_triplets']}/{r['total_questions']} triplets reviewed; eligible losses {r['fully_incorrect_after_npo']}; full returns {r['fully_correct_after_nf4']}; partial returns {r['partially_correct_after_nf4']}; full recovery rate {rate}.")
    text+=['','These are answer-level labels. Counting component-level factual recovery requires explicit component annotations and adjudication.','']
    (folder/'FACTUAL_RESULTS.md').write_text('\n'.join(text),encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('experiment_dir',type=Path)
    args=parser.parse_args();summarize(args.experiment_dir)
    print('Wrote factual_summary.json and FACTUAL_RESULTS.md')
