"""Day 5: extended *development-only* repair analysis.

Generation receives original MBPP visible tests only. EvalPlus hidden tests are
NEVER imported by this module; score in a separate CPU-only process afterward.
Run candidate code only in an isolated credential-free environment.
"""
from __future__ import annotations

import ast
import difflib
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

from tqdm.auto import tqdm

from day4_eval import (
    BASE_ID, MODES, append_jsonl, read_jsonl, write_json, generate_code,
    record_base_initials, record_fixed_corrections, prepare_offline_scoring,
    paired_bootstrap, _feedback_message,
)
from agent_loop import build_initial_messages, extract_python
from test_runner import run_visible_tests


def _safe_zip_extract(zip_path, dest, prefix, allow_names):
    dest = Path(dest)
    with ZipFile(zip_path) as z:
        if z.testzip() is not None:
            raise ValueError(f'ZIP failed integrity check: {zip_path}')
        for member in z.infolist():
            parts = Path(member.filename).parts
            if member.is_dir() or '..' in parts or '__MACOSX' in parts:
                continue
            if len(parts) == 2 and parts[0] == prefix and parts[-1] in allow_names:
                dest.mkdir(parents=True, exist_ok=True)
                (dest / parts[-1]).write_bytes(z.read(member))


def restore_day4_results(kaggle_root='/kaggle/input', out='/kaggle/working/day5_prior_day4'):
    """Restore Day 4 development results from saved notebook outputs or ZIP."""
    root, out = Path(kaggle_root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    names = {
        'eval_manifest.json', 'base_initials.jsonl', 'fixed_corrections.jsonl',
        'official_scores.jsonl', 'scoring_inputs.jsonl', 'day4_summary.json',
        'evaluation_protocol.json',
    }
    if not (out / 'day4_summary.json').exists():
        for candidate in sorted(root.rglob('gemma_day4_backup*.zip')):
            _safe_zip_extract(candidate, out, 'day4_outputs', names)
        # Loose output files from an attached saved notebook.
        for filename in names:
            if (out / filename).exists():
                continue
            for p in sorted(root.rglob(filename)):
                if 'day4_outputs' in p.parts and p.is_file():
                    shutil.copy2(p, out / filename)
                    break
    missing = sorted(n for n in names if not (out / n).is_file())
    if missing:
        raise FileNotFoundError(
            'Attach Day 4 saved outputs or gemma_day4_backup.zip via Add Input. Missing: ' + str(missing))
    manifest = json.loads((out/'eval_manifest.json').read_text())
    old_initials = read_jsonl(out/'base_initials.jsonl')
    old_fixed = read_jsonl(out/'fixed_corrections.jsonl')
    old_scores = read_jsonl(out/'official_scores.jsonl')
    old_requests = read_jsonl(out/'scoring_inputs.jsonl')
    ids = {int(x) for x in manifest['task_ids']}
    assert len(ids) == len(old_initials) == len(manifest['task_ids'])
    assert {int(r['task_id']) for r in old_initials} == ids
    failed = {int(r['task_id']) for r in old_initials if r['visible_status'] != 'passed'}
    assert {(int(r['task_id']),r['condition']) for r in old_fixed} == {(i,c) for i in failed for c in MODES}
    assert len({r['score_key'] for r in old_scores}) == len(old_scores)
    assert {r['score_key'] for r in old_scores} == {r['score_key'] for r in old_requests}
    print(f'Restored Day 4: {len(ids)} tasks; {len(failed)} initial visible failures; {len(old_scores)} scores')
    return out, manifest



def restore_day5_checkpoint(kaggle_root='/kaggle/input', output='/kaggle/working/day5_outputs'):
    """Optionally restore saved Day5 partial/full backup from attached Kaggle Input.

    Never overwrite an existing local result with different bytes. The caller
    still checks frozen configuration and all task IDs before running anything.
    """
    root,out=Path(kaggle_root),Path(output)
    out.mkdir(parents=True,exist_ok=True)
    restored=0
    for pattern in ('day5_generation_checkpoint*.zip','gemma_day5_backup*.zip'):
        for archive in sorted(root.rglob(pattern)):
            with ZipFile(archive) as z:
                if z.testzip() is not None: raise ValueError(f'Corrupt resume ZIP: {archive}')
                for item in z.infolist():
                    parts=Path(item.filename).parts
                    if item.is_dir() or '..' in parts or len(parts)<2 or parts[0]!='day5_outputs':
                        continue
                    name=Path(*parts[1:])
                    if name.is_absolute():raise ValueError(f'Unsafe ZIP member: {name}')
                    target=out/name
                    content=z.read(item)
                    if target.exists():
                        if target.read_bytes()!=content:
                            raise RuntimeError(f'Checkpoint conflict: {target}. Reconcile, do not overwrite.')
                        continue
                    target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes(content)
                    restored+=1
    if restored:print(f'Restored {restored} Day 5 checkpoint files from Kaggle input.')
    return restored

def freeze_partitions(inputs, old_dir, output, n_new=80, n_final=70, seed=20261109):
    """Choose dev extension BEFORE inference; reserve untouched final IDs now.

    Use only original MBPP prompts/test_list and cached MBPP+ task IDs from Day 2.
    A reference-runner compatibility check is done on *dev* candidates only.
    Final task IDs are merely reserved, never loaded into generation/scoring.
    """
    from datasets import load_dataset
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    frozen_path = output/'day5_partition_manifest.json'
    m = json.loads((Path(inputs)/'mbpp_split_manifest.json').read_text())
    old_m = json.loads((Path(old_dir)/'eval_manifest.json').read_text())
    old_ids = {int(tid) for tid in old_m['task_ids']}
    train_ids = set(map(int, m['mbppplus_decontaminated']['train']['task_ids']))
    mbpp_test = load_dataset('google-research-datasets/mbpp', 'full')['test']
    rows = {int(r['task_id']):dict(r) for r in mbpp_test}
    eligible = sorted((set(rows) & set(map(int,m['evalplus_mbppplus_ids']))) - train_ids - old_ids)
    assert len(eligible) >= n_new+n_final, f'Only {len(eligible)} unseen eligible tasks for {n_new}+{n_final}'
    if frozen_path.is_file():
        plan = json.loads(frozen_path.read_text())
        assert plan['seed']==seed and plan['n_new_requested']==n_new and plan['n_final_reserved']==n_final
        dev, final = set(plan['new_dev_ids']), set(plan['final_holdout_ids'])
        assert len(dev)==n_new and len(final)==n_final and not(dev & final) and not(dev & old_ids)
        assert (dev|final) <= set(eligible)
        return [rows[i] for i in plan['new_dev_ids']],plan
    rng = random.Random(seed)
    rng.shuffle(eligible)
    chosen, excluded = [], []
    # Deterministic oracle-compatibility filter for dev only; we never inspect candidate model outputs here.
    for tid in tqdm(eligible, desc='Freeze compatible dev task list', unit='task'):
        oracle = run_visible_tests(rows[tid]['code'],rows[tid]['test_list'],rows[tid].get('test_setup_code') or '')
        if oracle['status']=='passed':
            chosen.append(tid)
        else:
            excluded.append({'task_id':tid,'reference_status':oracle['status']})
        if len(chosen)==n_new:
            break
    if len(chosen)<n_new: raise RuntimeError('Not enough runner-compatible development tasks.')
    final = [tid for tid in eligible if tid not in set(chosen)][:n_final]
    if len(final)!=n_final: raise RuntimeError('Not enough held-out tasks to reserve.')
    plan = {
        'name':'Day 5 extended-development partition, FINAL set never used in this notebook',
        'seed':seed, 'n_new_requested':n_new, 'n_final_reserved':n_final,
        'prior_day4_dev_ids':sorted(old_ids), 'new_dev_ids':chosen,
        'final_holdout_ids':final,'n_remaining_eligible':len(eligible)-len(chosen)-len(final),
        'dev_runner_exclusions':excluded,
        'training_overlap':sorted(train_ids & (old_ids | set(chosen) | set(final))),
        'protocol':'C0/C1/C2 on new dev IDs, reuse Day 4 dev; no final-holdout evaluation or training.',
    }
    assert not plan['training_overlap']
    write_json(frozen_path,plan)
    return [rows[i] for i in chosen],plan


def freeze_e2e_ids(new_problems, output, n=30, seed=20261110):
    output=Path(output); p=output/'end_to_end_manifest.json'
    ids=[int(x['task_id']) for x in new_problems]
    if p.exists():
        obj=json.loads(p.read_text())
        assert obj['seed']==seed and obj['requested_n']==n and set(obj['task_ids'])<=set(ids)
        return obj['task_ids']
    if n>len(ids): raise ValueError('E2E_N exceeds new development problems')
    rng=random.Random(seed); rng.shuffle(ids)
    chosen=ids[:n]
    write_json(p,{'seed':seed,'requested_n':n,'task_ids':chosen,
                'sampling':'Unconditional random dev subset; chosen before generation, not on observed failure'})
    return chosen


def record_e2e(problems, model, processor, output, max_new_tokens=384, max_attempts=3):
    """Each condition independently starts from problem only, uses feedback after failures."""
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    log=output/'end_to_end.jsonl'
    previous=read_jsonl(log)
    existing={(int(r['task_id']),r['condition']):r for r in previous}
    assert len(existing)==len(previous), 'Duplicate E2E checkpoint entries'
    total=len(problems)*len(MODES)
    with tqdm(total=total,initial=len(existing),desc='End-to-end 1–3 attempts',unit='condition',dynamic_ncols=True) as bar:
        for p in problems:
            tid=int(p['task_id'])
            for condition in MODES:
                key=(tid,condition)
                if key in existing: continue
                msgs=build_initial_messages(p)
                attempts=[]
                for i in range(1,max_attempts+1):
                    bar.set_postfix(task=tid,condition=condition,attempt=i)
                    g=generate_code(model,processor,msgs,condition,max_new_tokens)
                    code=extract_python(g['raw'])
                    chk=run_visible_tests(code,p['test_list'],p.get('test_setup_code') or '')
                    if chk['status']=='runner_error':
                        raise RuntimeError(f'Visible runner infrastructure error on {tid}/{condition}/{i}')
                    attempts.append({'attempt':i,'code':code,'visible_status':chk['status'],
                                     'visible_feedback':chk['feedback'], **g})
                    if chk['status']=='passed':break
                    if i<max_attempts:
                        msgs.extend([{'role':'assistant','content':code},
                                     {'role':'user','content':_feedback_message(chk['feedback'])}])
                rec={'task_id':tid,'condition':condition,'attempts':attempts,
                     'total_completion_tokens':sum(int(a['completion_tokens']) for a in attempts),
                     'total_prompt_tokens':sum(int(a['prompt_tokens']) for a in attempts),
                     'num_executions':len(attempts),'final_visible_pass':attempts[-1]['visible_status']=='passed'}
                append_jsonl(log,rec);existing[key]=rec;bar.update()
    return existing


def record_no_feedback(problems, initials, model, processor, output, max_new_tokens=384):
    """C2 from same failed code but vague 'failed' message; no concrete test outputs."""
    output=Path(output);log=output/'no_feedback_c2.jsonl'
    rows=read_jsonl(log);existing={int(r['task_id']):r for r in rows}
    assert len(rows)==len(existing)
    failures=[p for p in problems if initials[int(p['task_id'])]['visible_status']!='passed']
    with tqdm(total=len(failures),initial=len(existing),desc='C2 no-feedback ablation',unit='failure') as bar:
        for p in failures:
            tid=int(p['task_id'])
            if tid in existing: continue
            start=initials[tid]
            msgs=build_initial_messages(p)+[
                {'role':'assistant','content':start['code']},
                {'role':'user','content':'The previous code failed a visible test. Try a corrected full Python solution. Output code only.'},
            ]
            g=generate_code(model,processor,msgs,'c2',max_new_tokens)
            code=extract_python(g['raw'])
            chk=run_visible_tests(code,p['test_list'],p.get('test_setup_code') or '')
            if chk['status']=='runner_error':raise RuntimeError('Runner error on no-feedback ablation')
            r={'task_id':tid,'condition':'c2_no_feedback','code':code,'visible_status':chk['status'],
               'visible_feedback':chk['feedback'],**g}
            append_jsonl(log,r);existing[tid]=r;bar.update()
    return existing


def freeze_humaneval(output, n=20, seed=20261111):
    """Load ORIGINAL public HumanEval prompts; do not load HumanEval+ hidden tests."""
    from evalplus.data.humaneval import get_human_eval
    out=Path(output);p=out/'humaneval_manifest.json'
    tasks=get_human_eval()
    ids=sorted(tasks)
    if p.exists():
        m=json.loads(p.read_text())
        assert m['n_tasks']==n and m['seed']==seed and set(m['task_ids'])<=set(ids)
        return {tid:tasks[tid] for tid in m['task_ids']}
    if n>len(ids):raise ValueError('n exceeds HumanEval task count')
    rng=random.Random(seed);rng.shuffle(ids)
    selected=ids[:n]
    write_json(p,{'n_tasks':n,'seed':seed,'task_ids':selected,
                  'source':'Original public HumanEval prompt (NOT HumanEval+ tests)',
                  'evaluation':'Single-shot, separate offline official HumanEval+ scoring. Exploratory transfer.'})
    return {tid:tasks[tid] for tid in selected}


def record_humaneval(problems, model, processor, output, max_new_tokens=384):
    from day4_eval import generate_code
    output=Path(output);log=output/'humaneval_first_attempts.jsonl'
    rows=read_jsonl(log);existing={(r['task_id'],r['condition']):r for r in rows}
    assert len(existing)==len(rows)
    with tqdm(total=len(problems)*len(MODES),initial=len(existing),
              desc='HumanEval transfer (first attempt only)',unit='condition') as bar:
        for tid,p in problems.items():
            for condition in MODES:
                key=(tid,condition)
                if key in existing:continue
                prompt='Implement the following Python function. Return a complete, runnable Python implementation including its function definition, not just the body. Output code only.\n\n'+p['prompt']
                msgs=[{'role':'user','content':prompt}]
                g=generate_code(model,processor,msgs,condition,max_new_tokens)
                code=extract_python(g['raw'])
                rec={'task_id':tid,'condition':condition,'code':code,**g}
                append_jsonl(log,rec);existing[key]=rec;bar.update();bar.set_postfix(task=tid,condition=condition)
    return existing


def create_scoring_requests(new_problems, output, include_e2e=True, include_ablation=True,
                            include_humaneval=True):
    """Freeze ALL requests once before invoking any hidden test scoring."""
    output=Path(output)
    # day4_eval.prepare_offline_scoring checks fixed pair completeness first.
    new_path=prepare_offline_scoring(new_problems,output)
    mbpp=read_jsonl(new_path)
    task_ids={int(p['task_id']) for p in new_problems}
    if include_e2e:
        e2e=read_jsonl(output/'end_to_end.jsonl')
        expected={(tid,c) for tid in json.loads((output/'end_to_end_manifest.json').read_text())['task_ids'] for c in MODES}
        assert {(int(r['task_id']),r['condition']) for r in e2e} == expected, 'E2E incomplete'
        for r in e2e:
            for a in r['attempts']:
                mbpp.append({'score_key':f'e2e:{r["condition"]}:{r["task_id"]}:{a["attempt"]}',
                             'task_id':int(r['task_id']),'condition':r['condition'],
                             'stage':'end_to_end','attempt':a['attempt'],'code':a['code']})
    if include_ablation:
        ab=read_jsonl(output/'no_feedback_c2.jsonl')
        initials=read_jsonl(output/'base_initials.jsonl')
        failed={int(x['task_id']) for x in initials if x['visible_status']!='passed'}
        assert {int(x['task_id']) for x in ab}==failed, 'Ablation incomplete'
        for r in ab:
            mbpp.append({'score_key':f'no_feedback:c2:{r["task_id"]}',
                         'task_id':int(r['task_id']),'condition':'c2_no_feedback',
                         'stage':'no_feedback','attempt':1,'code':r['code']})
    assert len(mbpp)==len({r['score_key'] for r in mbpp}), 'Duplicate MBPP scorer keys'
    path=output/'scoring_inputs.jsonl'
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in mbpp))
    human=[]
    if include_humaneval:
        h=read_jsonl(output/'humaneval_first_attempts.jsonl')
        m=json.loads((output/'humaneval_manifest.json').read_text())
        assert {(r['task_id'],r['condition']) for r in h} == {(t,c) for t in m['task_ids'] for c in MODES}
        for r in h:
            human.append({'score_key':f'humaneval:{r["condition"]}:{r["task_id"]}',
                          'task_id':r['task_id'],'condition':r['condition'],
                          'stage':'humaneval','code':r['code']})
    human_path=output/'humaneval_scoring_inputs.jsonl'
    human_path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in human))
    write_json(output/'scoring_protocol.json', {
        'mbpp_requests':len(mbpp),'humaneval_requests':len(human),
        'generation_frozen_before_official_scoring':True,
        'source':'Original MBPP visible tests; original HumanEval public prompts',
        'hidden_use':'EvalPlus base+plus in separate CPU process only',
        'do_not_train_on_this_split':True,
    })
    return path,human_path


def _pass(scores,key):
    if key not in scores:
        raise KeyError(f'Missing official score: {key}')
    return bool(scores[key]['mbppplus_pass'])


def _normalize(code):
    try:
        return ast.unparse(ast.parse(code)).strip()
    except (SyntaxError,ValueError):
        return '\n'.join(line.strip() for line in str(code).splitlines() if line.strip())


def _repeat_stats(attempt_codes, statuses, threshold=0.95):
    exact=near=0
    for i in range(1,len(attempt_codes)):
        if statuses[i]=='passed':continue
        a,b=_normalize(attempt_codes[i-1]),_normalize(attempt_codes[i])
        if a==b:
            exact+=1
        elif difflib.SequenceMatcher(None,a,b).ratio()>=threshold:
            near+=1
    return exact,near


def _boot(c1,c2,seed=42):
    if not c1:return {'n_pairs':0,'difference_c2_minus_c1':None,'ci_low':None,'ci_high':None}
    return paired_bootstrap(c1,c2,seed=seed)


def analyze(old_dir,new_dir, include_e2e=True,include_ablation=True,include_humaneval=True):
    """Compute fixed-state, failure, budget, and compute statistics from OFFICIAL scores."""
    from statistics import mean
    import csv
    old_dir,new_dir=Path(old_dir),Path(new_dir)
    old_init=read_jsonl(old_dir/'base_initials.jsonl')
    new_init=read_jsonl(new_dir/'base_initials.jsonl')
    old_fix=read_jsonl(old_dir/'fixed_corrections.jsonl')
    new_fix=read_jsonl(new_dir/'fixed_corrections.jsonl')
    old_scores=read_jsonl(old_dir/'official_scores.jsonl')
    new_scores=read_jsonl(new_dir/'official_scores.jsonl')
    score_rows=old_scores+new_scores
    scoremap={r['score_key']:r for r in score_rows}
    assert len(scoremap)==len(score_rows),'Score key collision or duplicates'
    all_init=old_init+new_init
    assert len({int(r['task_id']) for r in all_init})==len(all_init),'Overlapping development IDs'
    all_fix=old_fix+new_fix
    assert len({(int(r['task_id']),r['condition']) for r in all_fix})==len(all_fix)
    failed={int(r['task_id']):r for r in all_init if r['visible_status']!='passed'}
    lookup={(int(r['task_id']),r['condition']):r for r in all_fix}
    assert set(lookup)=={(tid,c) for tid in failed for c in MODES}
    detail=[]
    for tid,start in sorted(failed.items()):
        entry={'task_id':tid,'first_failure_type':start['visible_status'],'starting_completion_tokens':start['completion_tokens']}
        for c in MODES:
            r=lookup[(tid,c)]
            recovered=any(_pass(scoremap,f'fixed:{c}:{tid}:{a["correction_index"]}') for a in r['attempts'])
            exact,near=_repeat_stats([start['code']]+[a['code'] for a in r['attempts']],
                                     [start['visible_status']]+[a['visible_status'] for a in r['attempts']])
            entry[c+'_pass']=int(recovered)
            entry[c+'_visible_pass']=int(r['recovered_visible'])
            entry[c+'_generated_tokens']=r['total_completion_tokens']
            entry[c+'_prompt_tokens']=r['total_prompt_tokens']
            entry[c+'_executions']=r['num_corrections']
            entry[c+'_repeated_exact']=exact
            entry[c+'_repeated_near']=near
        detail.append(entry)
    with (new_dir/'fixed_state_by_task.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(detail[0]) if detail else ['task_id'])
        w.writeheader();w.writerows(detail)
    fixed={}
    for c in MODES:
        fixed[c]={
            'n_failed_states':len(detail),
            'official_success':sum(r[c+'_pass'] for r in detail),
            'official_rate':sum(r[c+'_pass'] for r in detail)/len(detail) if detail else None,
            'visible_success':sum(r[c+'_visible_pass'] for r in detail),
            'mean_generated_tokens':mean(r[c+'_generated_tokens'] for r in detail) if detail else None,
            'mean_prompt_tokens':mean(r[c+'_prompt_tokens'] for r in detail) if detail else None,
            'exact_repeated_unsuccessful':sum(r[c+'_repeated_exact'] for r in detail),
            'near_repeated_unsuccessful':sum(r[c+'_repeated_near'] for r in detail),
        }
    types={}
    for status in sorted({r['first_failure_type'] for r in detail}):
        rows=[r for r in detail if r['first_failure_type']==status]
        types[status]={'n':len(rows),**{c+'_success':sum(r[c+'_pass'] for r in rows) for c in MODES}}
    summary={
        'scope':'DEV ONLY (30 old + new dev), not untouched final evaluation',
        'n_dev_tasks':len(all_init),'n_fixed_failed_states':len(detail),
        'n_visible_first_attempt_pass':sum(r['visible_status']=='passed' for r in all_init),
        'failure_types':types,'fixed_state_results':fixed,
        'paired_C2_minus_C1':_boot([r['c1_pass'] for r in detail],[r['c2_pass'] for r in detail]),
        'paired_C2_minus_C0':_boot([r['c0_pass'] for r in detail],[r['c2_pass'] for r in detail]),
        'fixed_case_records':'fixed_state_by_task.csv',
        'definitions':{
            'recovery':'official base AND plus tests pass after a correction from identical failed C0 state',
            'repeated_exact':'AST-normalized code unchanged from previous attempt AND current visible tests fail',
            'repeated_near':'normalized source similarity >=0.95, nonidentical, AND current visible tests fail',
            'tokens_per_solved':'all generated completion tokens across tasks divided by number of officially solved tasks',
            'three_attempt_success':'ever passed official tests among produced attempts up to k (adaptive visible-test stopping)',
            'confidence_intervals':'paired bootstrap resampling development task IDs; one training seed',
        },
    }
    if include_ablation:
        ab=read_jsonl(new_dir/'no_feedback_c2.jsonl')
        baseline_new=[r for r in detail if int(r['task_id']) in {int(x['task_id']) for x in new_init}]
        by_id={int(r['task_id']):r for r in ab}
        assert set(by_id)=={int(r['task_id']) for r in new_init if r['visible_status']!='passed'}
        yes=[r['c2_pass'] for r in baseline_new]
        no=[int(_pass(scoremap,f'no_feedback:c2:{r["task_id"]}')) for r in baseline_new]
        summary['no_feedback_ablation_new_dev_only']={
            'n_states':len(yes),'with_feedback_successes':sum(yes),
            'without_detailed_feedback_successes':sum(no),
            'paired_difference':_boot(no,yes),
            'interpretation':'Vague retry prompt vs exact original visible-test error; only on newly added development tasks',
        }
    if include_e2e:
        records=read_jsonl(new_dir/'end_to_end.jsonl')
        ids=json.loads((new_dir/'end_to_end_manifest.json').read_text())['task_ids']
        assert {(int(r['task_id']),r['condition']) for r in records}=={(i,c) for i in ids for c in MODES}
        budget={}
        by_task=defaultdict(dict)
        e2e_detail=[]
        for r in records:
            tid,c=int(r['task_id']),r['condition']
            ever=[int(any(_pass(scoremap,f'e2e:{c}:{tid}:{a["attempt"]}') for a in r['attempts'] if a['attempt']<=k)) for k in (1,2,3)]
            exact,near=_repeat_stats([a['code'] for a in r['attempts']], [a['visible_status'] for a in r['attempts']])
            item={'task_id':tid,'condition':c, 'success_1':ever[0],'success_2':ever[1],'success_3':ever[2],
                  'generated_tokens':r['total_completion_tokens'],'prompt_tokens':r['total_prompt_tokens'],
                  'test_executions':len(r['attempts']),'repeated_exact':exact,'repeated_near':near}
            by_task[tid][c]=item;e2e_detail.append(item)
        with (new_dir/'end_to_end_by_task.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(e2e_detail[0]) if e2e_detail else ['task_id'])
            w.writeheader();w.writerows(e2e_detail)
        assert all(set(conditions)==set(MODES) for conditions in by_task.values())
        for c in MODES:
            rs=[x for x in e2e_detail if x['condition']==c]
            solved=sum(x['success_3'] for x in rs)
            budget[c]={
                'n_tasks':len(rs), 'success_1':sum(x['success_1'] for x in rs),
                'success_2':sum(x['success_2'] for x in rs),'success_3':solved,
                'rate_1':mean(x['success_1'] for x in rs), 'rate_2':mean(x['success_2'] for x in rs),
                'rate_3':mean(x['success_3'] for x in rs),
                'total_generated_tokens':sum(x['generated_tokens'] for x in rs),
                'tokens_per_solved_problem':sum(x['generated_tokens'] for x in rs)/solved if solved else None,
                'mean_test_executions_per_task':mean(x['test_executions'] for x in rs),
                'exact_repeated_unsuccessful':sum(x['repeated_exact'] for x in rs),
                'near_repeated_unsuccessful':sum(x['repeated_near'] for x in rs),
            }
        summary['end_to_end_budget']=budget
        summary['end_to_end_paired_C2_minus_C1']={
            f'k={k}':_boot([by_task[i]['c1'][f'success_{k}'] for i in ids],
                            [by_task[i]['c2'][f'success_{k}'] for i in ids]) for k in (1,2,3)}
    if include_humaneval:
        h=read_jsonl(new_dir/'humaneval_official_scores.jsonl')
        counts={c:[r['mbppplus_pass'] for r in h if r['condition']==c] for c in MODES}
        m=json.loads((new_dir/'humaneval_manifest.json').read_text())
        assert all(len(counts[c])==len(m['task_ids']) for c in MODES)
        summary['humaneval_plus_single_shot_transfer']={c:{'n':len(counts[c]),'passed':sum(counts[c]),
            'rate':mean(counts[c])} for c in MODES}
        sm={(r['task_id'],r['condition']):r for r in h}
        summary['humaneval_paired_C2_minus_C1']=_boot(
            [int(sm[(tid,'c1')]['mbppplus_pass']) for tid in m['task_ids']],
            [int(sm[(tid,'c2')]['mbppplus_pass']) for tid in m['task_ids']])
    write_json(new_dir/'day5_summary.json',summary)
    return summary


def make_figures(summary,output):
    import matplotlib.pyplot as plt
    import math
    p=Path(output)/'figures';p.mkdir(exist_ok=True)
    fixed=summary['fixed_state_results']
    fig,ax=plt.subplots(figsize=(6.6,4))
    ax.bar(['C0','C1','C2'],[100*fixed[c]['official_rate'] for c in MODES]);ax.set_ylabel('Official MBPP+ recovery (%)')
    ax.set_title(f'Fixed-state recovery (n={summary["n_fixed_failed_states"]} failures)')
    fig.tight_layout();fig.savefig(p/'fixed_recovery.png',dpi=160);plt.close(fig)
    types=summary['failure_types'];cats=list(types)
    if cats:
        fig,ax=plt.subplots(figsize=(7.2,4.3))
        x=list(range(len(cats)));w=.25
        for i,c in enumerate(MODES):
            ax.bar([t+(i-1)*w for t in x], [types[k][c+'_success']/types[k]['n']*100 for k in cats], width=w,label=c.upper())
        ax.set_xticks(x,[f'{k}\n(n={types[k]["n"]})' for k in cats]);ax.set_ylabel('Recovery (%)')
        ax.legend();ax.set_title('Recovery by first-failure type (small groups are exploratory)')
        fig.tight_layout();fig.savefig(p/'by_failure_type.png',dpi=160);plt.close(fig)
    if 'end_to_end_budget' in summary:
        fig,ax=plt.subplots(figsize=(6.6,4))
        for c in MODES:
            b=summary['end_to_end_budget'][c]
            ax.plot([1,2,3],[100*b[f'rate_{k}'] for k in (1,2,3)],marker='o',label=c.upper())
        ax.set_xticks([1,2,3]);ax.set_xlabel('Maximum attempts');ax.set_ylabel('Cumulative official success (%)')
        ax.set_title('End-to-end code generation');ax.legend()
        fig.tight_layout();fig.savefig(p/'attempt_budget.png',dpi=160);plt.close(fig)
        fig,ax=plt.subplots(figsize=(6.6,4))
        vals=[summary['end_to_end_budget'][c]['tokens_per_solved_problem'] for c in MODES]
        ax.bar([c.upper() for c in MODES],[v if v is not None else 0 for v in vals])
        ax.set_ylabel('Total generated tokens / officially solved tasks')
        ax.set_title('Compute efficiency (report alongside solve rate)')
        fig.tight_layout();fig.savefig(p/'tokens_per_solved.png',dpi=160);plt.close(fig)
    return list(sorted(str(x) for x in p.glob('*.png')))
