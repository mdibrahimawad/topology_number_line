"""Authorized matched-task experiments, bounded compute and resumable artifacts."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import subprocess
import sys
import time
import uuid
import modal

HERE=Path(__file__).resolve().parent
REMOTE='/root/taskrules'
RUN='taskrules_20260920_v1'
BASE=Path('/artifacts')/RUN
PROFILE='mdibrahimawad2'
GPU_SECONDS=900
CPU_SECONDS=1800
ALLOWANCE=14.5  # leaves room inside the user's $17; estimate, not live billing.

common={'PYTHONPATH':REMOTE,'PYTHONUNBUFFERED':'1','HF_HOME':'/cache/huggingface',
        'TOKENIZERS_PARALLELISM':'false','MPLBACKEND':'Agg','MPLCONFIGDIR':'/tmp/matplotlib',
        'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
gpu_image=(modal.Image.debian_slim(python_version='3.11')
    .pip_install_from_requirements(str(HERE/'requirements-crystal-fullrange.txt'))
    .env(common).add_local_dir(HERE/'numzig',REMOTE+'/numzig',ignore=['__pycache__','*.pyc']))
cpu_image=(modal.Image.debian_slim(python_version='3.11')
    .pip_install_from_requirements(str(HERE/'requirements-ph.txt'))
    .pip_install('plotly==5.24.1').env(common)
    .add_local_dir(HERE/'numzig',REMOTE+'/numzig',ignore=['__pycache__','*.pyc']))
app=modal.App('numberline-taskrules-v1')
volume=modal.Volume.from_name('numberline-zigzag-results',create_if_missing=False)
cache=modal.Volume.from_name('numberline-zigzag-hf-cache',create_if_missing=False)

@app.function(image=gpu_image,gpu='L40S',cpu=(2,2),memory=(32768,32768),
    timeout=GPU_SECONDS,startup_timeout=120,retries=0,max_containers=10,scaledown_window=2,
    volumes={'/artifacts':volume,'/cache':cache.with_mount_options(read_only=True)})
def gpu_job(model,stage,keys,contract):
    from numzig.fullrange.storage import read_json
    from numzig.taskrules.runtime import extract_shard,run_pilot
    volume.reload();cache.reload()
    rows=read_json(BASE/'dataset.json')
    if stage=='pilot':
        from numzig.taskrules.data import pilot_records
        return run_pilot(BASE/'pilot'/model,model,pilot_records(rows),contract,volume.commit)
    return extract_shard(BASE/model,model,rows,keys,contract,volume.commit)

@app.function(image=cpu_image,cpu=(2,2),memory=(8192,8192),timeout=CPU_SECONDS,
    startup_timeout=120,retries=0,max_containers=24,scaledown_window=2,volumes={'/artifacts':volume})
def analysis_job(model,task,ctx,contract):
    from numzig.fullrange.storage import read_json
    from numzig.taskrules.runtime import analyze_saved_group
    volume.reload();started=time.monotonic()
    result=analyze_saved_group(BASE/model,model,read_json(BASE/'dataset.json'),task,ctx,contract,volume.commit)
    volume.commit()
    return dict(model=model,task=task,context_id=ctx,seconds=time.monotonic()-started,summary=result)

@app.function(image=cpu_image,cpu=(.25,.25),memory=(2048,2048),timeout=14400,
    startup_timeout=120,retries=0,max_containers=1,scaledown_window=2,volumes={'/artifacts':volume})
def pipeline(stage='pilot'):
    from numzig.fullrange.storage import write_json,read_json,digest,fingerprint
    from numzig.taskrules.data import make_dataset
    from numzig.taskrules.runtime import MODELS,chunk_rows,chunk_valid
    from numzig.ph_overnight import reserve_budget,settle_budget
    from numzig.taskrules import data,extract,runtime
    started=time.monotonic();volume.reload();BASE.mkdir(exist_ok=True,parents=True)
    rows=make_dataset()
    contract=fingerprint(dict(dataset=fingerprint(rows),
        sources={Path(m.__file__).name:digest(m.__file__) for m in (data,extract,runtime)},schema=1))
    if (BASE/'contract.json').exists():
        if read_json(BASE/'contract.json')['fingerprint']!=contract:
            raise ValueError('Changed extraction contract: choose a new run namespace, do not overwrite')
    else:
        write_json(BASE/'contract.json',dict(fingerprint=contract,dataset_hash=fingerprint(rows),
            points_per_model=len(rows),models=list(MODELS),budget_allowance=ALLOWANCE,
            note='Matched contexts analyzed separately. Native model final normalization preserved.'))
        write_json(BASE/'dataset.json',rows)
        volume.commit()
    ledger=read_json(BASE/'budget.json') if (BASE/'budget.json').exists() else []
    outcomes=[]
    def checkpoint(state):
        write_json(BASE/'budget.json',ledger)
        write_json(BASE/'progress.json',dict(stage=stage,state=state,updated=time.time(),
            conservative_compute_usd=sum(r['charged_usd'] for r in ledger),allowance_usd=ALLOWANCE,
            outcomes=outcomes,excludes_exact_account_billing=True))
        volume.commit()
    def reserve(label,seconds,cores,gib,gpus=0):
        index=reserve_budget(ledger,label,seconds,cores,gib,gpus,limit=ALLOWANCE)
        checkpoint('running');return index
    coordinator=reserve('coordinator/'+stage,14400,.25,2)
    def batch(function,jobs,workers,seconds,cores,gib,gpus=0):
        pending=list(jobs)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            running={}
            while pending or running:
                while pending and len(running)<workers:
                    args=pending[0]
                    try:index=reserve(str(args[:3]),seconds,cores,gib,gpus)
                    except RuntimeError:
                        if not running:raise
                        break
                    pending.pop(0)
                    future=pool.submit(function.remote,*args);running[future]=index
                if not running:break
                future=next(as_completed(running))
                index=running.pop(future)
                try:
                    result=future.result();volume.reload()
                    settle_budget(ledger,index,min(float(result['seconds']),seconds))
                    outcomes.append({k:v for k,v in result.items() if k not in ('summary','files','meta')})
                except Exception as exc:
                    # Uncertain failed allocations keep their entire reserved cost.
                    outcomes.append(dict(error=repr(exc),job=ledger[index]['job']))
                    checkpoint('failed');raise
                checkpoint('running')
    try:
        if stage in ('pilot','all'):
            batch(gpu_job,[(m,'pilot',[],contract) for m in MODELS],3,GPU_SECONDS,2,32,1)
        if stage in ('all','extract'):
            for m in MODELS:
                if not (BASE/'pilot'/m/'receipt.json').exists():
                    raise ValueError('A completed technical/behavioral pilot is required before bulk extraction')
            jobs=[]
            for m in MODELS:
                missing=[key for key,part in chunk_rows(rows) if not chunk_valid(BASE/m,key,part,contract)]
                # Stable chunk identities; worker count never enters the checkpoint contract.
                for i in range(min(10,len(missing))):jobs.append((m,'extract',missing[i::10],contract))
            # Interleave model workers so the pool uses at most ten GPUs total.
            jobs.sort(key=lambda j:j[2][0] if j[2] else '')
            batch(gpu_job,jobs,10,GPU_SECONDS,2,32,1)
        if stage in ('all','analyze'):
            groups=sorted(set((r['task'],r['context_id']) for r in rows))
            batch(analysis_job,[(m,t,c,contract) for m in MODELS for t,c in groups],24,CPU_SECONDS,2,8)
        settle_budget(ledger,coordinator,min(time.monotonic()-started,14400))
        checkpoint('complete')
    except Exception:
        checkpoint('failed_or_budget_paused');raise
    return read_json(BASE/'progress.json')

@app.local_entrypoint()
def main(stage:str='pilot'):
    if stage not in ('pilot','all','extract','analyze'):raise ValueError('Invalid stage')
    if os.environ.get('MODAL_PROFILE')!=PROFILE:raise ValueError('Explicit mdibrahimawad2 profile required')
    apps=json.loads(subprocess.check_output([sys.executable,'-m','modal','app','list','--env','main','--json']))
    for row in apps:
        if row['description']==app.name and row['app_id']!=app.app_id and (row['state'].lower()!='stopped' or int(row['tasks'])):
            raise RuntimeError('Previous taskrules app is still active; wait before resume')
    print(pipeline.remote(stage))
