"""Offline CPU-only EvalPlus scoring for Day 5. Run AFTER all model generation.

The scoring process imports hidden tests. NEVER import this in a generation cell.
Executing generated code is dangerous without strong container isolation.
"""
from __future__ import annotations
import argparse,hashlib
from pathlib import Path
from tqdm.auto import tqdm

from day4_eval import append_jsonl, read_jsonl, sha256_bytes
from day4_score import score as score_mbpp


def score_humaneval(input_jsonl, output_jsonl):
    from evalplus.data import get_human_eval_plus,get_human_eval_plus_hash
    from evalplus.evaluate import check_correctness,get_groundtruth
    inp=Path(input_jsonl);out=Path(output_jsonl)
    req=read_jsonl(inp)
    if not req:
        print('No optional HumanEval+ requests: skipping')
        return
    keys=[r['score_key'] for r in req]
    assert len(keys)==len(set(keys)), 'Duplicate scoring keys'
    digest=sha256_bytes(inp.read_bytes());proof=out.with_suffix('.source_sha256')
    if proof.exists() and proof.read_text().strip()!=digest:
        raise ValueError('HumanEval scoring inputs changed since checkpoint creation; do not combine scores.')
    proof.parent.mkdir(parents=True,exist_ok=True);proof.write_text(digest+'\n')
    existing=read_jsonl(out);done={r['score_key']:r for r in existing}
    if len(existing)!=len(done) or not set(done)<=set(keys):
        raise ValueError('HumanEval checkpoint contains conflicting or duplicate keys')
    all_problems=get_human_eval_plus()
    ids={r['task_id'] for r in req}
    assert ids <= set(all_problems), 'Unknown HumanEval+ IDs'
    problems={tid:all_problems[tid] for tid in ids}
    sig=hashlib.sha256((get_human_eval_plus_hash()+'|'+','.join(sorted(ids))).encode()).hexdigest()[:24]
    expected=get_groundtruth(problems,'day5_humaneval_subset_'+sig,[])
    pending=[r for r in req if r['score_key'] not in done]
    with tqdm(total=len(req),initial=len(done),desc='Official HumanEval+ scoring',unit='solution') as bar:
        for r in pending:
            try:
                check=check_correctness('humaneval',0,problems[r['task_id']],r['code'],
                                        expected[r['task_id']],base_only=False,fast_check=True)
            except Exception as exc:
                raise RuntimeError(f'HumanEval infrastructure error on {r["score_key"]}: {exc}') from exc
            base=check['base'][0];plus=check['plus'][0]
            result={'score_key':r['score_key'],'task_id':r['task_id'],'condition':r['condition'],
                    'stage':'humaneval','evalplus_base_status':base,'evalplus_plus_status':plus,
                    'mbppplus_pass':base=='pass' and plus=='pass'}
            append_jsonl(out,result);bar.update();bar.set_postfix(task=r['task_id'],passed=result['mbppplus_pass'])
    print(f'HumanEval+ results saved: {len(read_jsonl(out))}')

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--kind',choices=['mbpp','humaneval'],required=True)
    ap.add_argument('--requests',required=True)
    ap.add_argument('--output',required=True)
    args=ap.parse_args()
    if args.kind=='mbpp':score_mbpp(args.requests,args.output)
    else:score_humaneval(args.requests,args.output)
