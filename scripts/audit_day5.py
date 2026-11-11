#!/usr/bin/env python3
"""Recompute audited Day 4/5 development metrics from the saved Day 5 ZIP.

No hidden test results are fed into generation; this script is post-hoc analysis only.
Uses only Python stdlib, and does not need a GPU.
Usage: python audit_day5.py /path/to/gemma_day5_backup.zip --out /path/to/audit_results.json
"""
import argparse
import csv
import io
import json
import math
import random
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

OLD = 'gemma_day5_backup/day5_prior_day4/'
NEW = 'gemma_day5_backup/day5_outputs/'
CONDS = ('c0', 'c1', 'c2')


def read_lines(z, path):
    return [json.loads(x) for x in z.read(path).decode('utf-8').splitlines() if x.strip()]


def indexed(rows, key_fn, name):
    result = {}
    for row in rows:
        k = key_fn(row)
        if k in result:
            raise ValueError(f'Duplicate {name}: {k}')
        result[k] = row
    return result


def paired_delta_ci(a, b, seed=20261110, repeats=20000):
    """Paired percentile bootstrap of mean(a_i - b_i), inclusive 95% interval."""
    assert len(a) == len(b) and a
    d = [int(x) - int(y) for x,y in zip(a,b)]
    rng = random.Random(seed)
    n = len(d)
    draws = []
    for _ in range(repeats):
        draws.append(sum(d[rng.randrange(n)] for _ in range(n)) / n)
    draws.sort()
    return {'n':n,'difference':sum(d)/n,
            'ci_low':draws[int(.025*(repeats-1))],
            'ci_high':draws[int(.975*(repeats-1))],
            'discordant_a_only':sum(x==1 for x in d),
            'discordant_b_only':sum(x==-1 for x in d)}


def audit(path):
    with ZipFile(path) as z:
        assert z.testzip() is None, 'Corrupt ZIP'
        m = json.loads(z.read(NEW+'day5_partition_manifest.json'))
        day3_counts = {'c1':55,'c2':55}  # cross-checked by separate Day 3 backup
        assert not set(m['prior_day4_dev_ids']) & set(m['new_dev_ids'])
        assert not set(m['final_holdout_ids']) & (set(m['new_dev_ids']) | set(m['prior_day4_dev_ids']))
        assert len(m['final_holdout_ids']) == 70
        assert not m['training_overlap']

        initial = {}
        fixed = {}
        official = {}
        for pref in (OLD, NEW):
            for row in read_lines(z, pref+'base_initials.jsonl'):
                tid = int(row['task_id'])
                assert tid not in initial, f'Initial duplicate {tid}'
                initial[tid] = row
            for row in read_lines(z, pref+'fixed_corrections.jsonl'):
                key = (int(row['task_id']), row['condition'])
                assert key not in fixed, f'Fixed duplicate {key}'
                fixed[key] = row
            for row in read_lines(z, pref+'official_scores.jsonl'):
                score_key=row['score_key']
                assert score_key not in official, f'Score duplicate {score_key}'
                official[score_key] = bool(row['mbppplus_pass'])

        assert len(initial)==110, len(initial)
        assert len(fixed)==31*3, len(fixed)
        assert set(initial) == set(m['prior_day4_dev_ids']) | set(m['new_dev_ids'])
        visible_fail_ids = sorted(t for t,r in initial.items() if r['visible_status']!='passed')
        visible_ok_ids = sorted(set(initial)-set(visible_fail_ids))
        assert len(visible_fail_ids)==31

        # Official initial/visible-test disagreements, which were measured only after generation.
        initial_score = {t:official[f'initial:{t}'] for t in initial}
        false_alarm_ids = sorted(t for t in visible_fail_ids if initial_score[t])
        missed_fault_ids = sorted(t for t in visible_ok_ids if not initial_score[t])
        genuinely_wrong_ids = sorted(t for t in visible_fail_ids if not initial_score[t])
        assert len(false_alarm_ids)==2, false_alarm_ids
        assert len(missed_fault_ids)==17, missed_fault_ids
        assert len(genuinely_wrong_ids)==29
        visible_confusion = {
            'visible_pass_official_pass':sum(initial_score[t] for t in visible_ok_ids),
            'visible_pass_official_fail':len(missed_fault_ids),
            'visible_fail_official_pass':len(false_alarm_ids),
            'visible_fail_official_fail':len(genuinely_wrong_ids),
            'false_alarm_task_ids':false_alarm_ids,
            'hidden_only_failure_task_ids':missed_fault_ids,
        }

        fixed_all={}
        fixed_audited={}
        for name,task_ids,dest in [('all_visible_failure_starts', visible_fail_ids,fixed_all),
                                   ('officially_incorrect_starts',genuinely_wrong_ids,fixed_audited)]:
            dest['n']=len(task_ids)
            dest['by_model']={}
            for c in CONDS:
                successes=[official[f'fixed:{c}:{t}:1'] for t in task_ids]
                rows=[fixed[(t,c)] for t in task_ids]
                dest['by_model'][c]={
                    'passed':sum(successes), 'total':len(successes),
                    'rate':sum(successes)/len(successes),
                    'mean_completion_tokens':sum(r['total_completion_tokens'] for r in rows)/len(rows),
                    'visible_pass':sum(r['recovered_visible'] for r in rows),
                }
            dest['c2_minus_c1'] = paired_delta_ci(
                [official[f'fixed:c2:{t}:1'] for t in task_ids],
                [official[f'fixed:c1:{t}:1'] for t in task_ids]
            )

        # Sequential-feedback tasks. Score 'ever' and code actually returned by the agent.
        e2e = indexed(read_lines(z, NEW+'end_to_end.jsonl'),
                      lambda r:(int(r['task_id']),r['condition']), 'end_to_end')
        em = json.loads(z.read(NEW+'end_to_end_manifest.json'))
        taskids=em['task_ids']
        assert len(taskids)==30 and len(e2e)==90
        budget={}
        e2e_seq_by_model={}
        for c in CONDS:
            runs=[e2e[(int(t),c)] for t in taskids]
            ever_by_k={k:[] for k in [1,2,3]}
            final=[]
            changed_from_correct_to_incorrect=[]
            total_completion=sum(r['total_completion_tokens'] for r in runs)
            n_exec=sum(r['num_executions'] for r in runs)
            for r in runs:
                t=int(r['task_id'])
                tries=r['attempts']
                correctness=[official[f'e2e:{c}:{t}:{a["attempt"]}'] for a in tries]
                for k in [1,2,3]:
                    ever_by_k[k].append(any(x for a,x in zip(tries,correctness) if a['attempt']<=k))
                final.append(correctness[-1])
                if any(correctness) and not correctness[-1]:
                    changed_from_correct_to_incorrect.append(t)
            e2e_seq_by_model[c]=final
            budget[c]={
                'n':len(runs),'ever_solved_by_attempt':{str(k):sum(vals) for k,vals in ever_by_k.items()},
                'final_solved':sum(final),
                'final_success_rate':sum(final)/len(final),
                'total_generated_tokens':total_completion,
                'generated_tokens_per_final_solved':total_completion/sum(final) if sum(final) else None,
                'mean_executions_per_task':n_exec/len(runs),
                'regressed_from_officially_correct_earlier_attempt_task_ids':changed_from_correct_to_incorrect,
            }
        e2e_delta=paired_delta_ci(e2e_seq_by_model['c2'],e2e_seq_by_model['c1'])
        ablation = indexed(read_lines(z, NEW+'no_feedback_c2.jsonl'),lambda r:int(r['task_id']),'no_feedback')
        assert len(ablation)==24
        with_feedback=[official[f'fixed:c2:{t}:1'] for t in sorted(ablation)]
        no_feedback=[official[f'no_feedback:c2:{t}'] for t in sorted(ablation)]
        feedback={'n':len(ablation),'detailed_pass':sum(with_feedback),'vague_pass':sum(no_feedback),
                  'paired':paired_delta_ci(with_feedback,no_feedback)}

        human_scores=read_lines(z, NEW+'humaneval_official_scores.jsonl')
        human_agg={}
        for c in CONDS:
            ss=[x for x in human_scores if x['condition']==c]
            assert len(ss)==20, (c,len(ss))
            human_agg[c]={'passed':sum(x['mbppplus_pass'] for x in ss),
                         'n':len(ss)}
        # Check exact known key shape for HumanEval data to catch silent measurement mismatches.
        if not any('mbppplus_pass' in r for r in human_scores):
            raise RuntimeError('Unexpected HumanEval score schema: '+repr(human_scores[0].keys()))
        out={
            'source_zip':Path(path).name,
            'source_sha256':__import__('hashlib').sha256(Path(path).read_bytes()).hexdigest(),
            'scope':'development-only; frozen 70-task holdout was not used',
            'split':{'dev':110,'prior_dev':30,'additional_dev':80,'untouched_holdout':70},
            'visible_test_vs_official_score':visible_confusion,
            'fixed_state_on_all_visible_failures':fixed_all,
            'fixed_state_after_excluding_officially_correct_initial_states':fixed_audited,
            'end_to_end':budget,
            'end_to_end_final_c2_minus_c1':e2e_delta,
            'feedback_ablation_c2':feedback,
            'humaneval':human_agg,
            'audit_notes':[
                'Fixed-state corrections were generated only for initially visible-failing states; the post-hoc official-correct filter is for analysis, not an online decision.',
                'Ever-passed by k is an oracle diagnostic, not deployment-time final returned-code success; the agent does not observe hidden test status.',
                'Official test failures among visible-pass solutions were not retried.',
                'All CIs are percentile paired bootstrap on task IDs and are exploratory, with one training seed.',
            ]}
        return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('zip_path',type=Path)
    ap.add_argument('--out',type=Path,default=Path('day5_audit.json'))
    args=ap.parse_args()
    data=audit(args.zip_path)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(data,indent=2)+'\n')
    f=data['fixed_state_after_excluding_officially_correct_initial_states']
    print('AUDIT completed:', args.out)
    print('Officially incorrect initial states:', f['n'])
    for c,v in f['by_model'].items():
        print(f'  {c}: {v["passed"]}/{v["total"]} ({v["rate"]*100:.1f}%)')
    print('C2-C1 paired difference:', f['c2_minus_c1'])
    for c,v in data['end_to_end'].items():
        print(f'End-to-end {c}: final {v["final_solved"]}/{v["n"]}, ever by 3 {v["ever_solved_by_attempt"]["3"]}/{v["n"]}')
    print('Visible feedback disagreements:',data['visible_test_vs_official_score'])

if __name__=='__main__': main()
