"""Day 3 matched-budget QLoRA for Gemma 4 E4B; uses Day 2 prompt/completion JSONL."""
from collections import Counter
from pathlib import Path
import hashlib, json, random, shutil


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]


def write_json(path, value):
    p=Path(path);p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False),encoding='utf-8')


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()


def discover_day2_input(root='/kaggle/input', output='/kaggle/working/day3_input'):
    """Discover Day 2 files via Kaggle Add Input or its saved backup ZIP."""
    from zipfile import ZipFile
    root,output=Path(root),Path(output)
    req={'success_only.jsonl','recovery.jsonl','dataset_summary.json'}
    if output.is_dir() and req <= {p.name for p in output.iterdir() if p.is_file()}:
        return output
    if root.exists():
        for p in sorted(root.rglob('dataset_summary.json')):
            if p.is_file() and req <= {q.name for q in p.parent.iterdir() if q.is_file()}:
                output.mkdir(parents=True,exist_ok=True)
                for name in ['success_only.jsonl','recovery.jsonl','recovery_self.jsonl',
                             'recovery_reference.jsonl','c2_mixed.jsonl',
                             'dataset_summary.json','data_collection_config.json']:
                    src=p.parent/name
                    if src.is_file(): shutil.copy2(src,output/name)
                ms=list(root.rglob('mbpp_split_manifest.json'))
                if ms: shutil.copy2(ms[0],output/'mbpp_split_manifest.json')
                write_json(output/'input_location.json',{'source':str(p.parent),'mode':'loose_files'})
                return output
    zips=sorted(root.rglob('gemma_day2_backup*.zip')) if root.exists() else []
    if not zips: raise FileNotFoundError('Attach the Day 2 saved outputs OR the Day 2 backup ZIP via Kaggle Add Input. Searched '+str(root))
    zip_path=zips[0]; output.mkdir(parents=True,exist_ok=True)
    with ZipFile(zip_path) as z:
        assert z.testzip() is None,'Corrupt Day 2 archive'
        found=set()
        for member in z.infolist():
            parts=Path(member.filename).parts
            if '__MACOSX' in parts or member.is_dir(): continue
            basename=parts[-1]
            if 'day2_data' in parts and basename in {'success_only.jsonl','recovery.jsonl',
                'recovery_self.jsonl','recovery_reference.jsonl','c2_mixed.jsonl',
                'dataset_summary.json','data_collection_config.json'}:
                (output/basename).write_bytes(z.read(member));found.add(basename)
            if basename=='mbpp_split_manifest.json':
                (output/basename).write_bytes(z.read(member))
        if not req <= found: raise ValueError(f'Day 2 ZIP missing files: {req-found}')
    write_json(output/'input_location.json',{'source':str(zip_path),'mode':'zip','sha256':sha256(zip_path)})
    return output


def validate_day2(data_dir):
    d=Path(data_dir)
    success=read_jsonl(d/'success_only.jsonl');recovery=read_jsonl(d/'recovery.jsonl')
    summary=json.loads((d/'dataset_summary.json').read_text())
    assert success and recovery,'Empty training pool'
    assert len(success)==summary['c1_success_examples']
    assert len(recovery)==summary['c2_recovery_combined_examples']
    for row in success+recovery:
        assert row.get('prompt') and row.get('completion') and row.get('target_code')
        assert row['completion'][-1]['role']=='assistant'
        assert row['completion'][-1]['content']==row['target_code']
        assert row.get('verified_on')=='original MBPP visible tests only'
    assert not summary.get('evaluation_overlap_ids'), 'Day 2 reported benchmark overlap'
    mpath=d/'mbpp_split_manifest.json'
    if mpath.exists():
        manifest=json.loads(mpath.read_text())
        approved=set(manifest['mbppplus_decontaminated']['train']['task_ids'])
        held_out=set(manifest['evalplus_mbppplus_ids'])
        actual={int(r['task_id']) for r in success+recovery}
        assert actual <= approved, f'Not in eligible training split: {sorted(actual-approved)}'
        assert actual.isdisjoint(held_out), f'Benchmark contamination: {sorted(actual & held_out)}'
    return {'success':success,'recovery':recovery,'summary':summary}


def _ids(processor,messages,add_generation_prompt):
    ans=processor.apply_chat_template(messages,tokenize=True,return_dict=True,
        return_tensors='pt',add_generation_prompt=add_generation_prompt,enable_thinking=False)
    x=ans['input_ids']
    return x[0].tolist() if getattr(x,'ndim',None)==2 else list(x)


def tokenize_example(row,processor,max_seq_len):
    """Mask all prompt tokens (including earlier assistant mistakes)."""
    prefix=_ids(processor,row['prompt'],True)
    full=_ids(processor,row['prompt']+row['completion'],False)
    if full[:len(prefix)]!=prefix:
        raise ValueError('Chat template prefix mismatch. STOP: inspect Gemma tokenizer formatting; do not train on wrong labels.')
    target=full[len(prefix):]
    assert target,'No supervised tokens'
    if len(full)>max_seq_len:return None
    labels=[-100]*len(prefix)+target
    assert len(labels)==len(full)
    return {'task_id':int(row['task_id']),'sample_index':row.get('sample_index'),
        'example_type':row['example_type'],'source':row['source'],
        'input_ids':full,'attention_mask':[1]*len(full),'labels':labels,
        'target_tokens':len(target),'prompt_tokens':len(prefix),'sequence_tokens':len(full)}


def tokenize_pools(data,processor,max_seq_len):
    pools={'success':[],'recovery':[]};dropped={'success':[],'recovery':[]}
    from tqdm.auto import tqdm
    for k in pools:
        for r in tqdm(data[k],desc=f'Tokenize {k}',unit='example'):
            t=tokenize_example(r,processor,max_seq_len)
            if t is None:dropped[k].append({'task_id':r['task_id'],'source':r['source']})
            else:pools[k].append(t)
        print(k,len(pools[k]),'fit;',len(dropped[k]),'too long')
    if not pools['success'] or not pools['recovery']:
        raise ValueError('Insufficient fitting examples; inspect MAX_SEQ_LEN')
    return {'pools':pools,'dropped':dropped}


def matched_plans(tokenized,seed=42,tries=6000,tolerance=0.04):
    """Same sample count / optimizer steps, matched supervised *target* tokens.

    C1 = all available success rows. C2 = every recovery row plus a
    subset of success rows selected to match C1's target-token count.
    """
    success=tokenized['pools']['success'];recovery=tokenized['pools']['recovery']
    if len(recovery)>len(success):raise ValueError('Recovery pool exceeds success pool')
    n=len(success);k=n-len(recovery)
    target=sum(r['target_tokens'] for r in success)
    r_tokens=sum(r['target_tokens'] for r in recovery)
    rng=random.Random(seed);ids=list(range(n));best=None
    for _ in range(max(1,tries)):
        chosen=rng.sample(ids,k)
        observed=r_tokens+sum(success[i]['target_tokens'] for i in chosen)
        delta=abs(observed-target)
        if best is None or delta<best[0]:best=(delta,tuple(sorted(chosen)),observed)
        if delta==0:break
    c1=list(success);c2=list(recovery)+[success[i] for i in best[1]]
    rng1=random.Random(seed+101);rng2=random.Random(seed+202)
    rng1.shuffle(c1);rng2.shuffle(c2)
    delta=abs(best[2]-target)/target
    print('C1:',len(c1),'rows,',target,'target tokens')
    print('C2:',len(c2),'rows,',best[2],'target tokens')
    print('Token difference:',f'{delta:.2%}','| C2 sources:',dict(Counter(r['source'] for r in c2)))
    if delta>tolerance:
        raise ValueError(f'C1 vs C2 target tokens differ by {delta:.2%}, exceeding {tolerance:.0%}. ' 
                         'Change MAX_SEQ_LEN or explicit matching strategy before training.')
    manifest={
        'seed':seed,'c1_count':len(c1),'c2_count':len(c2),
        'c1_target_tokens':target,'c2_target_tokens':best[2],
        'relative_token_difference':delta,
        'c1_sources':dict(Counter(r['source'] for r in c1)),
        'c2_sources':dict(Counter(r['source'] for r in c2)),
        'c1_task_ids':sorted(set(r['task_id'] for r in c1)),
        'c2_task_ids':sorted(set(r['task_id'] for r in c2)),
        'c1_rows':[{'task_id':r['task_id'],'sample_index':r['sample_index'],
                    'source':r['source'],'target_tokens':r['target_tokens']} for r in c1],
        'c2_rows':[{'task_id':r['task_id'],'sample_index':r['sample_index'],
                    'source':r['source'],'target_tokens':r['target_tokens']} for r in c2],
        'sequence_exclusions':tokenized['dropped'],
        'notes':['C1 contains all fitting success-only rows.',
                 'C2 contains all fitting recovery rows and selected success rows.',
                 'Same training examples count and gradient updates; target tokens closely matched.',
                 'Different prompt lengths are not controlled; report prompt/computation costs separately.',
                 'Recovery corrections include separately labeled self and reference sources.',
                 'All training examples were verified on original MBPP visible tests only.']}
    return {'c1':c1,'c2':c2,'manifest':manifest}


class CompletionCollator:
    def __init__(self,pad_token_id):self.pad=pad_token_id
    def __call__(self,features):
        import torch
        n=max(len(f['input_ids']) for f in features)
        ids=[r['input_ids']+[self.pad]*(n-len(r['input_ids'])) for r in features]
        am=[r['attention_mask']+[0]*(n-len(r['attention_mask'])) for r in features]
        labels=[r['labels']+[-100]*(n-len(r['labels'])) for r in features]
        return {'input_ids':torch.tensor(ids,dtype=torch.long),
                'attention_mask':torch.tensor(am,dtype=torch.long),
                'labels':torch.tensor(labels,dtype=torch.long)}


def build_dataset(plan):
    from datasets import Dataset
    cols=['input_ids','attention_mask','labels']
    return Dataset.from_dict({k:[r[k] for r in plan] for k in cols})


def load_fresh_model(model_id,lora_rank=8):
    import torch
    from transformers import AutoModelForMultimodalLM,BitsAndBytesConfig
    from peft import LoraConfig,get_peft_model
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
    base=AutoModelForMultimodalLM.from_pretrained(
        model_id,quantization_config=quant,device_map={'':0},dtype=torch.float16)
    base.config.use_cache=False
    # T4 VRAM fix: avoid PEFT casting huge frozen Gemma 4 embedding tables to FP32.
    # Retain 4-bit / fp16 weights; freeze manually and enable checkpointing.
    for param in base.parameters():
        param.requires_grad_(False)
    base.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={'use_reentrant': False}
    )
    if hasattr(base, 'enable_input_require_grads'):
        base.enable_input_require_grads()
    print('Base frozen; gradient checkpointing enabled (no FP32 embedding upcast).')
    print('GPU allocated:',
          round(torch.cuda.memory_allocated(0) / 1024**3, 2), 'GiB')
    # Explicit regex limits LoRA to *language* attention, not vision/audio layers.
    lora=LoraConfig(task_type='CAUSAL_LM',r=lora_rank,lora_alpha=2*lora_rank,
        lora_dropout=0.05,bias='none',
        target_modules=r'.*language_model.*\.(q_proj|v_proj)')
    model=get_peft_model(base,lora)
    model.config.use_cache=False
    # Some Gemma4 Transformers releases mishandle loss normalization with gradient accumulation.
    model.accepts_loss_kwargs=False
    model.print_trainable_parameters()
    n=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n<1000:raise RuntimeError('Unexpectedly no trainable language-model LoRA weights')
    return model


def train_condition(condition,plan,model_id,output_root,seed=42,max_seq_len=1024,
                    lr=1e-4,epochs=1,grad_accum=4,lora_rank=8,
                    save_steps=5,resume=False):
    """Train C1/C2 independently; use built-in Trainer tqdm and checkpoints."""
    import gc,torch
    from transformers import Trainer,TrainingArguments,AutoProcessor,set_seed
    from transformers.trainer_utils import get_last_checkpoint
    root=Path(output_root);root.mkdir(parents=True,exist_ok=True)
    out=root/f'train_{condition}';adapter=root/f'adapter_{condition}'
    if (adapter/'adapter_config.json').exists():
        raise RuntimeError(f'{adapter} already exists. Do not overwrite a completed adapter.')
    set_seed(seed);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(0)
    model=load_fresh_model(model_id,lora_rank=lora_rank)
    model.train()
    processor=AutoProcessor.from_pretrained(model_id)
    pad=processor.tokenizer.pad_token_id
    if pad is None:pad=processor.tokenizer.eos_token_id
    assert pad is not None
    ds=build_dataset(plan)
    args=TrainingArguments(
        output_dir=str(out),
        per_device_train_batch_size=1,gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,learning_rate=lr,lr_scheduler_type='constant',
        optim='adamw_torch',weight_decay=0.0,
        fp16=True,bf16=False, # T4 GPUs do not support bf16 natively
        logging_strategy='steps',logging_steps=1,
        save_strategy='steps',save_steps=save_steps,save_total_limit=2,
        remove_unused_columns=False,dataloader_num_workers=0,
        disable_tqdm=False,report_to='none',seed=seed,data_seed=seed,
        max_grad_norm=0.3,gradient_checkpointing=False, # handled above
    )
    trainer=Trainer(model=model,args=args,train_dataset=ds,
                    data_collator=CompletionCollator(pad))
    checkpoint=get_last_checkpoint(str(out)) if out.exists() and resume else None
    if checkpoint:print('Resuming from',checkpoint,flush=True)
    result=trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(str(adapter))
    metrics=dict(result.metrics)
    metrics.update({'condition':condition,'base_model':model_id,
        'num_examples':len(plan),
        'supervised_target_tokens_per_epoch':sum(r['target_tokens'] for r in plan),
        'max_seq_len':max_seq_len,'learning_rate':lr,'epochs':epochs,
        'gradient_accumulation_steps':grad_accum,'lora_rank':lora_rank,
        'seed':seed,'adapter_path':str(adapter),
        'gpu0_peak_allocated_gib':round(torch.cuda.max_memory_allocated(0)/1024**3,3)})
    write_json(root/f'{condition}_metrics.json',metrics)
    write_json(root/f'{condition}_log_history.json',trainer.state.log_history)
    print('FINISHED',condition,'training_loss:',metrics.get('train_loss'),flush=True)
    del trainer,model,processor,ds
    gc.collect();torch.cuda.empty_cache()
    return metrics
