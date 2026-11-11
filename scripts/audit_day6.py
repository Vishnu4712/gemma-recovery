#!/usr/bin/env python3
"""Recompute frozen Day 6 MBPP+ recovery statistics from saved Kaggle outputs.

Usage:
  python scripts/audit_day6.py /path/to/gemma_final_holdout_backup.zip \
      --out results/final_holdout/day6_recomputed.json

Do NOT treat the summary extracted from the executed notebook as independently
reproduced row-level statistics. This script requires the ZIP (or output folder)
containing base_initials.jsonl, fixed_corrections.jsonl, official_scores.jsonl,
and frozen_manifest.json.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
from pathlib import Path
from zipfile import ZipFile

REQUIRED = (
    'frozen_manifest.json',
    'base_initials.jsonl',
    'fixed_corrections.jsonl',
    'official_scores.jsonl',
)
MODELS=('c0','c1','c2')


def read_source(source: Path):
    """Read only a narrow allowlist of JSON/JSONL artifacts; never extract code."""
    d={}
    if source.is_file():
        with ZipFile(source) as z:
            names=z.namelist()
            for name in REQUIRED:
                paths=[p for p in names if p == name or p.endswith('/'+name)]
                if len(paths)!=1: raise ValueError(f'Expected one {name}, got {paths}')
                d[name]=z.read(paths[0]).decode('utf-8')
    elif source.is_dir():
        for name in REQUIRED:
            paths=list(source.rglob(name))
            if len(paths)!=1: raise ValueError(f'Expected one {name}, got {paths}')
            d[name]=paths[0].read_text(encoding='utf-8')
    else: raise FileNotFoundError(source)
    return d


def json_lines(text: str, label: str):
    rows=[json.loads(line) for line in text.splitlines() if line.strip()]
    if not all(isinstance(x,dict) for x in rows): raise ValueError(f'Non-object record in {label}')
    return rows


def paired_bootstrap(a,b,draws=20_000,seed=20261110):
    if not a or len(a)!=len(b): raise ValueError('Pairs are missing or unequal')
    delta=[int(y)-int(x) for x,y in zip(a,b)]
    n=len(delta)
    r=random.Random(seed)
    bs=sorted(sum(delta[r.randrange(n)] for _ in range(n))/n for _ in range(draws))
    return {
        'difference_c2_minus_c1':sum(delta)/n,
        'ci_low':bs[int(.025*draws)],
        'ci_high':bs[min(draws-1,int(.975*draws))],
        'n_pairs':n,
        'discordant_c2_only':sum(x==1 for x in delta),
        'discordant_c1_only':sum(x==-1 for x in delta),
    }


def analyze(manifest, initial_rows, fixed_rows, score_rows, expected_tasks=70):
    ids=[int(x) for x in manifest['final_holdout_ids']]
    if len(ids)!=expected_tasks or len(set(ids))!=len(ids):
        raise ValueError('Frozen manifest count / uniqueness mismatch')
    idset=set(ids)
    init={int(r['task_id']):r for r in initial_rows}
    if len(init)!=len(initial_rows) or set(init)!=idset:
        raise ValueError('Initial records must contain exactly frozen task IDs once')
    failed=sorted(t for t,r in init.items() if r['visible_status']!='passed')
    fixed={(int(r['task_id']),str(r['condition'])):r for r in fixed_rows}
    if len(fixed)!=len(fixed_rows): raise ValueError('Duplicate corrections')
    if set(fixed)!={(t,c) for t in failed for c in MODELS}:
        raise ValueError('Corrections are missing, extra, or from unpaired tasks')
    # The frozen protocol allows exactly one correction after any local failure.
    if any(len(row.get('attempts',[]))!=1 or row['attempts'][0]['correction_index']!=1
           for row in fixed.values()):
        raise ValueError('Expected one correction attempt per fixed state')
    sc={r['score_key']:r for r in score_rows}
    if len(sc)!=len(score_rows): raise ValueError('Duplicate official score keys')
    expected={f'initial:{t}' for t in ids} | {f'fixed:{c}:{t}:1' for t in failed for c in MODELS}
    if set(sc)!=expected:
        raise ValueError(f'Official score keys mismatch: missing={sorted(expected-set(sc))[:5]} '
                         f'extra={sorted(set(sc)-expected)[:5]}')
    # Require explicit booleans; otherwise bool('False') would silently score true.
    if any(type(v.get('mbppplus_pass')) is not bool for v in sc.values()):
        raise ValueError('Scores must have boolean mbppplus_pass')
    passed=lambda key:sc[key]['mbppplus_pass']
    false_alarms=[t for t in failed if passed(f'initial:{t}')]
    incorrect=[t for t in failed if not passed(f'initial:{t}')]
    by_model={}
    for c in MODELS:
        hits=[passed(f'fixed:{c}:{t}:1') for t in incorrect]
        by_model[c]={'passed':sum(hits),'n':len(hits),
                     'rate':sum(hits)/len(hits) if hits else None}
    paired=paired_bootstrap(
        [passed(f'fixed:c1:{t}:1') for t in incorrect],
        [passed(f'fixed:c2:{t}:1') for t in incorrect]) if incorrect else None
    initial_pass=sum(passed(f'initial:{t}') for t in ids)
    visible_pass=[t for t in ids if t not in failed]
    return {
        'scope':'FROZEN FINAL HOLDOUT — source raw records independently audited',
        'n_tasks':len(ids),
        'first_attempt_official_pass':initial_pass,
        'visible_failed_starts':len(failed),
        'officially_correct_visible_fail_ids':false_alarms,
        'true_failed_starts':len(incorrect),
        'by_model':by_model,
        'paired_C2_vs_C1':paired,
        'official_scoring_requests':len(score_rows),
        'visible_official_confusion':{
          'visible_pass_official_pass':sum(passed(f'initial:{t}') for t in visible_pass),
          'visible_pass_official_fail':sum(not passed(f'initial:{t}') for t in visible_pass),
          'visible_fail_official_pass':len(false_alarms),
          'visible_fail_official_fail':len(incorrect),
        },
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path,help='Kaggle gemma_final_holdout_backup.zip or extracted output directory')
    parser.add_argument('--out',type=Path,default=Path('day6_recomputed.json'))
    parser.add_argument('--expected-tasks',type=int,default=70)
    a=parser.parse_args()
    d=read_source(a.input)
    result=analyze(json.loads(d['frozen_manifest.json']),
       json_lines(d['base_initials.jsonl'],'base_initials'),
       json_lines(d['fixed_corrections.jsonl'],'fixed_corrections'),
       json_lines(d['official_scores.jsonl'],'official_scores'),
       expected_tasks=a.expected_tasks)
    if a.input.is_file(): result['source_sha256']=hashlib.sha256(a.input.read_bytes()).hexdigest()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
