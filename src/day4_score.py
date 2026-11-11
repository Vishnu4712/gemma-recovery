"""Offline official EvalPlus MBPP+ scoring for Day 4.

Run ONLY after generation, in a separate CPU subprocess. No hidden feedback
is supplied to the coding agent. IMPORTANT: EvalPlus executes generated code;
run only in an isolated, credential-free container/runtime.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
from tqdm.auto import tqdm

from day4_eval import append_jsonl, read_jsonl, sha256_bytes


def score(input_jsonl, output_jsonl):
    # Import lazily: model-generation process does not load hidden oracle.
    from evalplus.data import get_mbpp_plus, get_mbpp_plus_hash
    from evalplus.evaluate import check_correctness, get_groundtruth
    from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
    source=Path(input_jsonl)
    dest=Path(output_jsonl)
    requests=read_jsonl(source)
    if not requests:
        raise RuntimeError(f'No scoring requests: {source}')
    score_keys=[r['score_key'] for r in requests]
    if len(set(score_keys))!=len(score_keys):
        raise ValueError('Duplicate score keys')
    cache_inputs=dest.with_suffix('.source_sha256')
    hash_input=sha256_bytes(source.read_bytes())
    if cache_inputs.exists() and cache_inputs.read_text().strip()!=hash_input:
        raise RuntimeError('Score requests changed since scoring began; do not mix checkpoints')
    cache_inputs.parent.mkdir(parents=True,exist_ok=True)
    cache_inputs.write_text(hash_input+'\n')
    old=read_jsonl(dest)
    if len(old)!=len({r['score_key'] for r in old}):
        raise ValueError('Duplicate score checkpoints')
    done={r['score_key']:r for r in old}
    if not set(done).issubset(score_keys):
        raise ValueError('Checkpoint contains scores for different task set')
    problems_all=get_mbpp_plus()
    key_by_id = {int(str(key).split('/')[-1]): key for key in problems_all}
    requested_ids = {int(r['task_id']) for r in requests}
    assert requested_ids.issubset(key_by_id), 'Some selected IDs are absent from official EvalPlus MBPP+'
    ids = {key_by_id[tid] for tid in requested_ids}
    problems={tid:problems_all[tid] for tid in ids}
    print('Loaded',len(problems),'official EvalPlus MBPP+ problems',flush=True)
    # Hash includes dataset version AND selected subset: NEVER corrupt full-benchmark cache.
    subhash=hashlib.sha256((get_mbpp_plus_hash() + '|' + ','.join(sorted(ids))).encode()).hexdigest()[:24]
    expected=get_groundtruth(problems,'day4_dev_subset_'+subhash,MBPP_OUTPUT_NOT_NONE_TASKS)
    remaining=[r for r in requests if r['score_key'] not in done]
    with tqdm(total=len(requests),initial=len(done),desc='Official EvalPlus scoring',unit='solution',dynamic_ncols=True) as bar:
        for r in remaining:
            tid=key_by_id[int(r['task_id'])]
            try:
                result=check_correctness(
                    'mbpp', 0, problems[tid], r['code'], expected[tid],
                    base_only=False, fast_check=True,
                )
                base_status=result['base'][0]
                plus_status=result['plus'][0]
            except Exception as exc:
                # Treat scorer exceptions as infrastructure issues, not incorrect code.
                raise RuntimeError(f'Official evaluator failed on {r["score_key"]}: {exc}') from exc
            rec={'score_key':r['score_key'],'task_id':int(r['task_id']),
                 'stage':r['stage'],'condition':r['condition'],
                 'evalplus_base_status':base_status,
                 'evalplus_plus_status':plus_status,
                 'mbppplus_pass':base_status=='pass' and plus_status=='pass'}
            append_jsonl(dest,rec)
            bar.update()
            bar.set_postfix(task=tid,pass_plus=rec['mbppplus_pass'])
    print('Finished:',len(read_jsonl(dest)),'official scores saved at',dest,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--requests',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    score(args.requests,args.output)
