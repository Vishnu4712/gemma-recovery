"""Day 4: paired, held-out MBPP+ recovery evaluation for Gemma 4.

Generate on visible ORIGINAL MBPP tests. Score with official EvalPlus ONLY
in a separate post-generation subprocess. Never include hidden feedback
in model prompts. Use only on disposable, credential-free runtimes.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from agent_loop import build_initial_messages, extract_python
from test_runner import run_visible_tests

BASE_ID = 'google/gemma-4-E4B-it'
MODES = ('c0', 'c1', 'c2')


def write_json(path, obj):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(p)


def read_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    for i, ln in enumerate(p.read_text(encoding='utf-8').splitlines(), 1):
        if ln.strip():
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError as exc:
                raise ValueError(f'Corrupt checkpoint {p} at line {i}: {exc}') from exc
    return rows


def append_jsonl(path, row):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _copy_selected_from_zip(zip_path, base, condition_dirs=False):
    """Safely extract a small allowlist; never do unrestricted extractall."""
    wanted_files = {'mbpp_split_manifest.json', 'training_plan.json'}
    out = Path(base)
    with ZipFile(zip_path, 'r') as z:
        if z.testzip() is not None:
            raise ValueError(f'Corrupt ZIP: {zip_path}')
        for item in z.infolist():
            if item.is_dir():
                continue
            parts = Path(item.filename).parts
            if '__MACOSX' in parts or '..' in parts:
                continue
            name = parts[-1]
            if name in wanted_files:
                (out / name).write_bytes(z.read(item))
            for cond in ('c1', 'c2'):
                if (f'adapter_{cond}' in parts and
                    name in {'adapter_model.safetensors', 'adapter_config.json'}):
                    dst = out / f'adapter_{cond}'
                    dst.mkdir(exist_ok=True)
                    (dst / name).write_bytes(z.read(item))


def restore_inputs(kaggle_root='/kaggle/input', out='/kaggle/working/day4_inputs'):
    """Locate saved Day 2 and Day 3 outputs or attached backup ZIPs.

    Add Day 2 AND Day 3 saved notebook outputs via Kaggle Add Input, or
    attach private Kaggle Datasets containing gemma_day2_backup.zip and
    gemma_day3_backup.zip.
    """
    root = Path(kaggle_root)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    required = lambda: all((out / f'adapter_{c}' / 'adapter_model.safetensors').is_file() and
                           (out / f'adapter_{c}' / 'adapter_config.json').is_file()
                           for c in ('c1', 'c2')) and (out/'mbpp_split_manifest.json').is_file()
    if required():
        return out
    if not root.exists():
        raise FileNotFoundError(f'No Kaggle input directory at {root}')
    # Loose files from notebook output: copy adapters and manifest.
    for p in sorted(root.rglob('mbpp_split_manifest.json')):
        if p.is_file():
            shutil.copy2(p, out/'mbpp_split_manifest.json')
            break
    for p in sorted(root.rglob('training_plan.json')):
        if p.is_file():
            shutil.copy2(p, out/'training_plan.json')
            break
    for cond in ('c1', 'c2'):
        configs = sorted(p for p in root.rglob('adapter_config.json') if p.parent.name == f'adapter_{cond}')
        for p in configs:
            w = p.parent/'adapter_model.safetensors'
            if w.is_file():
                dst = out/f'adapter_{cond}'; dst.mkdir(exist_ok=True)
                shutil.copy2(p, dst/p.name)
                shutil.copy2(w, dst/w.name)
                break
    # ZIP backups: only read trusted allowlisted files.
    for pattern in ('gemma_day2_backup*.zip', 'gemma_day3_backup*.zip'):
        for zpath in sorted(root.rglob(pattern)):
            _copy_selected_from_zip(zpath, out)
    if not required():
        missing = [str(p) for p in [out/'mbpp_split_manifest.json',
                  out/'adapter_c1/adapter_config.json',out/'adapter_c1/adapter_model.safetensors',
                  out/'adapter_c2/adapter_config.json',out/'adapter_c2/adapter_model.safetensors'] if not p.exists()]
        raise FileNotFoundError('Attach Day 2 and Day 3 output data. Missing: ' + str(missing))
    a1 = json.loads((out/'adapter_c1/adapter_config.json').read_text())
    a2 = json.loads((out/'adapter_c2/adapter_config.json').read_text())
    assert a1.get('base_model_name_or_path') == BASE_ID and a2.get('base_model_name_or_path') == BASE_ID, 'Adapter base checkpoint mismatch'
    return out


def select_dev_problems(inputs, output, n=30, seed=20261108):
    """Freeze a development set using the original MBPP test split.

    MBPP+ task IDs identify eligible held-out tasks, but NO hidden input/output
    is loaded during generation. Reference code is only used for a local runner
    compatibility check; exclude tasks if that runner cannot execute their
    reference code correctly. Report these exclusions.
    """
    from datasets import load_dataset
    from tqdm.auto import tqdm
    inputs = Path(inputs); output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((inputs/'mbpp_split_manifest.json').read_text())
    dev_path = output/'eval_manifest.json'
    ds = load_dataset('google-research-datasets/mbpp', 'full')
    rows = {int(row['task_id']): dict(row) for row in ds['test']}
    eval_ids = {int(x) for x in manifest['evalplus_mbppplus_ids']}
    trained_ids = set(manifest['mbppplus_decontaminated']['train']['task_ids'])
    candidate_ids = sorted(set(rows).intersection(eval_ids) - trained_ids)
    assert len(candidate_ids) >= n, f'Only {len(candidate_ids)} compatible ID candidates, requested {n}'
    if dev_path.exists():
        frozen = json.loads(dev_path.read_text())
        assert frozen['requested_n'] == n and frozen['seed'] == seed, 'Dev manifest already frozen for different settings; use new output directory'
        chosen = frozen['task_ids']
        assert set(chosen).issubset(candidate_ids), 'Frozen task set no longer matches dataset'
        return [rows[tid] for tid in chosen], frozen
    order = list(candidate_ids)
    random.Random(seed).shuffle(order)
    chosen, excluded = [], []
    for tid in tqdm(order, desc='Check visible-test runner compatibility', unit='task'):
        row = rows[tid]
        oracle = run_visible_tests(row['code'], row['test_list'], row.get('test_setup_code') or '')
        if oracle['status'] == 'passed':
            chosen.append(tid)
            if len(chosen) == n:
                break
        else:
            excluded.append({'task_id': tid, 'reason': oracle['status']})
    if len(chosen) < n:
        raise ValueError(f'Only {len(chosen)} local-runner-compatible tasks. Lower requested n or inspect runner.')
    frozen = {
        'name':'MBPP+ overlap with original MBPP test split, held-out dev set',
        'task_ids': chosen, 'seed': seed, 'requested_n': n,
        'candidate_count': len(candidate_ids),
        'runner_exclusions': excluded,
        'training_overlap': sorted(set(chosen).intersection(trained_ids)),
        'visible_tests': 'original MBPP test_list',
        'hidden_tests': 'official EvalPlus base_input/plus_input, scored only after generation',
        'do_not_use_for_final_claims': 'Use separate untouched test tasks in final evaluation after development is frozen.',
    }
    assert not frozen['training_overlap']
    write_json(dev_path, frozen)
    return [rows[tid] for tid in chosen], frozen


def load_all_adapters(inputs, model_id=BASE_ID):
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig
    from peft import PeftModel
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    print('Loading Gemma 4 in 4-bit on GPU 0 ...', flush=True)
    base = AutoModelForMultimodalLM.from_pretrained(
        model_id, quantization_config=quant, device_map={'':0}, dtype=torch.float16)
    base.config.use_cache = True
    processor = AutoProcessor.from_pretrained(model_id)
    model = PeftModel.from_pretrained(base, str(Path(inputs)/'adapter_c1'),
        adapter_name='c1', is_trainable=False)
    model.load_adapter(str(Path(inputs)/'adapter_c2'), adapter_name='c2', is_trainable=False)
    model.eval()
    model.set_adapter('c1')
    assert {'c1','c2'} <= set(model.peft_config.keys()), f'Adapters not loaded: {model.peft_config.keys()}'
    print('Loaded C1 + C2. GPU0 allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 2), 'GiB', flush=True)
    return model, processor


def generate_code(model, processor, messages, condition, max_new_tokens=384):
    """Deterministic inference; C0 disables all adapters temporarily."""
    import torch
    if condition not in MODES:
        raise ValueError(condition)
    if condition in ('c1','c2'):
        model.set_adapter(condition)
        ctx = contextlib.nullcontext()
    else:
        ctx = model.disable_adapter()
    inputs = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors='pt',
        add_generation_prompt=True, enable_thinking=False).to('cuda:0')
    n = int(inputs['input_ids'].shape[1])
    started = time.perf_counter()
    with ctx, torch.inference_mode():
        out = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
            pad_token_id=processor.tokenizer.pad_token_id)
    new = out[0][n:]
    return {
        'raw': processor.decode(new, skip_special_tokens=True),
        'prompt_tokens':n, 'completion_tokens':int(len(new)),
        'generation_seconds':round(time.perf_counter()-started, 3),
    }


def record_base_initials(problems, model, processor, output, max_new_tokens=384):
    from tqdm.auto import tqdm
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    log = output/'base_initials.jsonl'
    existing = {int(r['task_id']):r for r in read_jsonl(log)}
    if len(existing)!=len(read_jsonl(log)):
        raise ValueError('Duplicate base initial task IDs found in checkpoint')
    pending = [p for p in problems if int(p['task_id']) not in existing]
    with tqdm(total=len(problems), initial=len(problems)-len(pending),
              desc='C0 first attempts', unit='problem', dynamic_ncols=True) as bar:
        for p in pending:
            tid = int(p['task_id'])
            bar.set_postfix(task=tid)
            print(f'Generating initial code for MBPP {tid}', flush=True)
            g = generate_code(model, processor, build_initial_messages(p), 'c0', max_new_tokens)
            code = extract_python(g['raw'])
            check = run_visible_tests(code, p['test_list'], p.get('test_setup_code') or '')
            rec = {'task_id':tid,'condition':'c0','attempt':1,'code':code,
                   'visible_status':check['status'], 'visible_feedback':check['feedback'],
                   'visible_passed':check.get('passed',0), 'visible_total':check.get('total',0),
                   **g}
            append_jsonl(log, rec)
            existing[tid]=rec
            bar.update()
            bar.set_postfix(task=tid, visible=check['status'], sec=g['generation_seconds'])
    return existing


def _feedback_message(feedback):
    return 'The previous code FAILED the visible tests.\n' + feedback + '\nProvide a corrected full Python solution. Do not repeat the same code. Output code only.'


def record_fixed_corrections(problems, initials, model, processor, output, max_new_tokens=384, max_corrections=1):
    """Same failed C0 code and same initial feedback for C0/C1/C2.

    No corrections are issued after the visible tests pass, regardless of
    eventual hidden-test success/failure. Hidden results are never in prompts.
    """
    from tqdm.auto import tqdm
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    log=output/'fixed_corrections.jsonl'
    existing_rows=read_jsonl(log)
    existing={(int(r['task_id']),r['condition']):r for r in existing_rows}
    if len(existing)!=len(existing_rows):
        raise ValueError('Duplicate fixed-state records in checkpoint')
    tasks=[p for p in problems if initials[int(p['task_id'])]['visible_status']!='passed']
    count=len(tasks)*len(MODES)
    assert tasks, 'No visible failures in chosen tasks. Increase number of evaluation tasks; do not change their IDs after looking at hidden results.'
    pending=sum((int(p['task_id']),c) not in existing for p in tasks for c in MODES)
    with tqdm(total=count,initial=count-pending,desc='Paired recovery conditions',unit='condition',dynamic_ncols=True) as bar:
        for p in tasks:
            tid=int(p['task_id'])
            init=initials[tid]
            if init['visible_status']=='runner_error':
                raise ValueError(f'Runner error on task {tid}; inspect before evaluation')
            for condition in MODES:
                key=(tid,condition)
                if key in existing:
                    continue
                messages=build_initial_messages(p)
                messages.append({'role':'assistant','content':init['code']})
                messages.append({'role':'user','content':_feedback_message(init['visible_feedback'])})
                attempts=[]
                for retry in range(1,max_corrections+1):
                    print(f'Fixing MBPP {tid}: {condition}, correction {retry}/{max_corrections}',flush=True)
                    g=generate_code(model,processor,messages,condition,max_new_tokens)
                    code=extract_python(g['raw'])
                    chk=run_visible_tests(code,p['test_list'],p.get('test_setup_code') or '')
                    attempts.append({'correction_index':retry,'code':code,
                                     'visible_status':chk['status'], 'visible_feedback':chk['feedback'],
                                     'visible_passed':chk.get('passed',0),
                                     'visible_total':chk.get('total',0), **g})
                    if chk['status']=='passed':
                        break
                    if retry < max_corrections:
                        messages.extend([
                            {'role':'assistant','content':code},
                            {'role':'user','content':_feedback_message(chk['feedback'])}])
                rec={'task_id':tid,'condition':condition,
                     'starting_code':init['code'],
                     'starting_feedback':init['visible_feedback'],
                     'starting_visible_status':init['visible_status'],
                     'attempts':attempts,
                     'recovered_visible':any(a['visible_status']=='passed' for a in attempts),
                     'num_corrections':len(attempts),
                     'total_completion_tokens':sum(a['completion_tokens'] for a in attempts),
                     'total_prompt_tokens':sum(a['prompt_tokens'] for a in attempts)}
                append_jsonl(log,rec);existing[key]=rec
                bar.update()
                bar.set_postfix(task=tid,condition=condition,
                                visible=rec['recovered_visible'])
    return existing


def prepare_offline_scoring(problems, output):
    """Create evaluation inputs after all completions were generated."""
    output=Path(output)
    ids={int(p['task_id']) for p in problems}
    initials=read_jsonl(output/'base_initials.jsonl')
    fixed=read_jsonl(output/'fixed_corrections.jsonl')
    assert {int(r['task_id']) for r in initials} == ids, 'Initial generation incomplete'
    failed={int(r['task_id']) for r in initials if r['visible_status']!='passed'}
    for cond in MODES:
        assert {int(r['task_id']) for r in fixed if r['condition']==cond} == failed, f'Missing fixed corrections for {cond}'
    rows=[]
    for r in initials:
        rows.append({'score_key':f'initial:{int(r["task_id"])}',
                     'task_id':int(r['task_id']), 'condition':'c0',
                     'stage':'initial', 'attempt':1,'code':r['code']})
    for r in fixed:
        for a in r['attempts']:
            rows.append({'score_key':f'fixed:{r["condition"]}:{int(r["task_id"])}:{a["correction_index"]}',
                         'task_id':int(r['task_id']), 'condition':r['condition'],
                         'stage':'fixed_recovery','attempt':a['correction_index'],
                         'code':a['code']})
    assert len(rows)==len({r['score_key'] for r in rows}), 'Duplicate score keys'
    p=output/'scoring_inputs.jsonl'
    tmp=p.with_suffix('.jsonl.tmp')
    tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    tmp.replace(p)
    return p


def _rate(arr):
    return sum(arr)/len(arr) if arr else float('nan')


def paired_bootstrap(c1,c2,n_boot=3000,seed=42):
    assert len(c1)==len(c2)>0
    diffs=[int(b)-int(a) for a,b in zip(c1,c2)]
    rng=random.Random(seed)
    draws=[]
    for _ in range(n_boot):
        draws.append(sum(diffs[rng.randrange(len(diffs))] for _ in diffs)/len(diffs))
    draws.sort()
    return {'difference_c2_minus_c1':sum(diffs)/len(diffs),
            'ci_low':draws[int(0.025*n_boot)],
            'ci_high':draws[min(n_boot-1,int(0.975*n_boot))],
            'n_pairs':len(diffs),
            'discordant_c2_only':sum(d==1 for d in diffs),
            'discordant_c1_only':sum(d==-1 for d in diffs)}


def analyze_official(problems, output):
    output=Path(output)
    initials=read_jsonl(output/'base_initials.jsonl')
    corrections=read_jsonl(output/'fixed_corrections.jsonl')
    official=read_jsonl(output/'official_scores.jsonl')
    im={r['score_key']:r for r in official}
    expected_keys={r['score_key'] for r in read_jsonl(output/'scoring_inputs.jsonl')}
    if set(im)!=expected_keys or len(im)!=len(official):
        raise ValueError('Incomplete/duplicate official score log. Rerun scorer.')
    failed_ids=sorted(int(r['task_id']) for r in initials if r['visible_status']!='passed')
    f={(int(r['task_id']),r['condition']):r for r in corrections}
    successes={}
    for cond in MODES:
        hits=[];visible=[];tokens=[]
        for tid in failed_ids:
            r=f[(tid,cond)]
            hits.append(any(im[f'fixed:{cond}:{tid}:{a["correction_index"]}']['mbppplus_pass'] for a in r['attempts']))
            visible.append(r['recovered_visible'])
            tokens.append(r['total_completion_tokens'])
        successes[cond]={'official_recovery':sum(hits), 'n_failed_states':len(failed_ids),
             'recovery_rate':_rate(hits), 'visible_recovery_rate':_rate(visible),
             'mean_completion_tokens':_rate(tokens)}
    c1=[];c2=[]
    for tid in failed_ids:
        for arr,cond in ((c1,'c1'),(c2,'c2')):
            r=f[(tid,cond)]
            arr.append(any(im[f'fixed:{cond}:{tid}:{a["correction_index"]}']['mbppplus_pass'] for a in r['attempts']))
    base_pass=sum(im[f'initial:{int(r["task_id"])}']['mbppplus_pass'] for r in initials)
    summary={
        'benchmark': 'EvalPlus MBPP+ official base + plus oracle',
        'dev_only':True, 'n_dev_tasks':len(problems),
        'n_fixed_failed_states':len(failed_ids),
        'base_first_attempt_official_pass':base_pass,
        'base_first_attempt_official_rate':base_pass/len(initials),
        'fixed_state_results':successes,
        'paired_C2_vs_C1':paired_bootstrap(c1,c2),
        'methodology_note':'Fixed starting states are first-attempt visible failures from C0; hidden tests are only scored after all generations and never returned as feedback.',
    }
    write_json(output/'day4_summary.json',summary)
    return summary
