"""Synthetic audit test; does not use or claim to replicate Kaggle's 70-task raw data."""
import unittest
from pathlib import Path
import importlib.util
p=Path(__file__).resolve().parents[1]/'scripts/audit_day6.py'
spec=importlib.util.spec_from_file_location('audit_day6',p)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class TestDay6Audit(unittest.TestCase):
    def test_counting_and_pairing(self):
        man={'final_holdout_ids':[1,2,3,4]}
        ini=[{'task_id':i,'visible_status':('passed' if i==4 else 'wrong_output')}
             for i in (1,2,3,4)]
        fixes=[{'task_id':i,'condition':c,'attempts':[{'correction_index':1}]}
               for i in (1,2,3) for c in m.MODELS]
        scores=[]
        for i in range(1,5):
            scores.append({'score_key':f'initial:{i}','mbppplus_pass':i in (3,4)})
        for i in (1,2,3):
            for c in m.MODELS:
                scores.append({'score_key':f'fixed:{c}:{i}:1',
                               'mbppplus_pass':(i==1 and c=='c2') or (i==2 and c=='c1')})
        result=m.analyze(man,ini,fixes,scores,expected_tasks=4)
        self.assertEqual(result['official_scoring_requests'],13)
        self.assertEqual(result['true_failed_starts'],2)
        self.assertEqual(result['officially_correct_visible_fail_ids'],[3])
        self.assertEqual(result['by_model']['c1']['passed'],1)
        self.assertEqual(result['by_model']['c2']['passed'],1)
        self.assertEqual(result['paired_C2_vs_C1']['discordant_c2_only'],1)
        self.assertEqual(result['paired_C2_vs_C1']['discordant_c1_only'],1)

if __name__=='__main__': unittest.main()
