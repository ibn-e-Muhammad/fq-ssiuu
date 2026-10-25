"""Focused CPU-only checks for recovery metrics and paired evaluation selection."""
import ast
import copy
import json
import math
from pathlib import Path
import random
import unittest

from report_local_tofu import STAGES, paired_rows, recovery_transitions, summarize_recovery, report_html, report_text


def records(exact=(1,0,1),f1=(0.9,0.2,0.6),nll=(1,3,2),question='q'):
    return [dict(model=s,split='forget01',question=question,reference='reference',prediction='answer',
        exact_match=e,token_f1=f,answer_nll_per_token=n) for s,e,f,n in zip(STAGES,exact,f1,nll)]


class RecoveryTests(unittest.TestCase):
    def test_recovery_is_conditional_on_baseline_and_loss(self):
        result=recovery_transitions([(1,0,1),(1,0,0),(1,1,1),(0,0,1)],1)
        self.assertEqual(result['baseline_pass'],3)
        self.assertEqual(result['lost_after_npo'],2)
        self.assertEqual(result['returned_after_nf4'],1)
        self.assertEqual(result['recovery_rate'],0.5)

    def test_no_known_or_lost_answers_is_undefined_not_zero(self):
        self.assertIsNone(recovery_transitions([(0,0,1)],1)['forgetting_rate'])
        self.assertIsNone(recovery_transitions([(1,1,1)],1)['recovery_rate'])

    def test_pairing_by_question_not_csv_order(self):
        data=records(question='a')+records(exact=(0,1,0),question='b')
        random.Random(5).shuffle(data)
        pairs=paired_rows(data)
        self.assertEqual(pairs[0]['question'],'a')
        self.assertEqual(pairs[0]['exact_status'],'Exact answer reappeared after NF4')
        self.assertEqual(pairs[1]['exact_status'],'Not exact at baseline')

    def test_missing_duplicate_and_changed_references_fail(self):
        data=records()
        bad=copy.deepcopy(data);bad[1]['reference']='different'
        for values in (data[:-1],data+[data[0]],bad):
            with self.assertRaises(ValueError): paired_rows(values)

    def test_invalid_numbers_fail(self):
        for metric,value in [('token_f1',float('nan')),('token_f1',1.1),('answer_nll_per_token',-1),('exact_match',0.5)]:
            data=records();data[0][metric]=value
            with self.assertRaises(ValueError): paired_rows(data)

    def test_score_reversal_is_signed_and_not_clipped(self):
        self.assertAlmostEqual(summarize_recovery(paired_rows(records()))['nll_shift_reversed_fraction'],0.5)
        self.assertAlmostEqual(summarize_recovery(paired_rows(records(nll=(1,3,4))))['nll_shift_reversed_fraction'],-0.5)
        self.assertAlmostEqual(summarize_recovery(paired_rows(records(nll=(1,3,0))))['nll_shift_reversed_fraction'],1.5)
        self.assertIsNone(summarize_recovery(paired_rows(records(nll=(3,1,0))))['nll_shift_reversed_fraction'])

    def test_no_escaping_injected_html_from_answers(self):
        data=records(question='<script>evil()</script>')
        data[0]['prediction']='<img src=x onerror=evil()>'
        pairs=paired_rows(data);summary=summarize_recovery(pairs)
        rendered=report_html(summary,pairs,report_text(summary,'run'))
        self.assertNotIn('<script>evil()',rendered)
        self.assertNotIn('<img src=x',rendered)
        self.assertIn('&lt;img',rendered)

    def test_notebook_evaluates_all_forget_questions_and_keeps_fixed_samples(self):
        path=Path(__file__).resolve().parents[1]/'Mother_Local_v2.ipynb'
        notebook=json.loads(path.read_text(encoding='utf-8'))
        text=''.join(notebook['cells'][12]['source'])
        node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='select_eval_sets')
        ns={'random':random}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<selection>','exec'),ns)
        rows=[dict(question=f'q{i}',answer=f'a{i}') for i in range(200)]
        select=ns['select_eval_sets']
        first=select(rows,rows,rows,'forget05','retain95',42,40)
        self.assertEqual(len(first['forget05']),200)
        self.assertEqual(len(first['retain95']),40)
        self.assertEqual(first,select(rows,rows,rows,'forget05','retain95',42,40))
        self.assertEqual(len(select(rows,rows,rows,'forget05','retain95',42,2,2)['forget05']),2)


if __name__=='__main__':
    unittest.main(verbosity=2)
