"""Detached, budgeted all-layer PH. No model inference or model downloads."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import json,os,subprocess,sys,time
import modal

LOCAL=Path(__file__).resolve().parent
REV='30243c0c752de26d7fdf6e41f08bf7b840ca4744'
base=(modal.Image.debian_slim(python_version='3.12').pip_install_from_requirements(str(LOCAL/'requirements-ph.txt'))
      .add_local_dir(LOCAL/'numzig','/root/numzig',ignore=['__pycache__','*.pyc']))
gpu_image=(modal.Image.from_registry('nvidia/cuda:11.8.0-devel-ubuntu22.04',add_python='3.12')
 .apt_install('git','cmake','build-essential')
 .pip_install_from_requirements(str(LOCAL/'requirements-ph.txt'))
 .run_commands(f'git clone https://github.com/simonzhang00/ripser-plusplus.git /opt/ripser-plusplus && cd /opt/ripser-plusplus && git checkout {REV}',
  'cmake -S /opt/ripser-plusplus -B /opt/rpp-build -DCUDA_ARCH_LIST=8.9 -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=/opt/ripser-plusplus/ripserplusplus',
  'cmake --build /opt/rpp-build -j 4')
 .env({'PYTHONPATH':'/root:/opt/ripser-plusplus','OMP_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','MPLCONFIGDIR':'/tmp/matplotlib','PYTHONUNBUFFERED':'1'})
 .add_local_dir(LOCAL/'numzig','/root/numzig',ignore=['__pycache__','*.pyc']))
volume=modal.Volume.from_name('numberline-zigzag-results',create_if_missing=False)

app=modal.App('numberline-ph-overnight')
ROOT=Path('/artifacts/ph_overnight_v1')

def child(stage,model='',layer=0,seconds=1740):
    from numzig.fullrange.storage import read_json
    start=time.monotonic();log=ROOT/'logs'/f'{stage}_{model}_{layer}_{time.time_ns()}.txt'
    log.parent.mkdir(parents=True,exist_ok=True)
    args=[sys.executable,'-m','numzig.ph_overnight',stage,'/artifacts',str(ROOT)]
    if stage in ('prepare','gpu'):args += [model,str(layer)]
    try:
        with log.open('w') as f:
            subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=seconds,cwd='/root',
                env={**os.environ,'PYTHONPATH':os.environ.get('PYTHONPATH','/root'),
                     'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
                     'MPLCONFIGDIR':f'/tmp/mpl-{os.getpid()}-{model}-{layer}','PYTHONUNBUFFERED':'1'})
        return dict(stage=stage,model=model,layer=layer,status='complete',seconds=time.monotonic()-start)
    except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as exc:
        tail=log.read_text()[-5000:]
        return dict(stage=stage,model=model,layer=layer,status='timeout' if isinstance(exc,subprocess.TimeoutExpired) else 'failed',
                    seconds=time.monotonic()-start,error=str(exc),log=str(log),tail=tail)

@app.function(image=base,cpu=(4,4),memory=(8192,8192),timeout=1800,startup_timeout=120,retries=0,max_containers=8,scaledown_window=2,volumes={'/artifacts':volume})
def prepare_worker(model,layer):
    volume.reload();r=child('prepare',model,layer);volume.commit();return r

@app.function(image=base,cpu=(4,4),memory=(12288,12288),timeout=2400,startup_timeout=120,retries=0,max_containers=1,scaledown_window=2,volumes={'/artifacts':volume})
def cache_worker():
    volume.reload();r=child('cache',seconds=2340);volume.commit();return r

@app.function(image=gpu_image,gpu='L40S',cpu=(6,6),memory=(40960,40960),timeout=1080,startup_timeout=120,retries=0,max_containers=10,scaledown_window=2,volumes={'/artifacts':volume})
def gpu_group(jobs,seconds=900):
    if not 1<=len(jobs)<=4:raise ValueError('One to four independent layers per GPU')
    volume.reload();started=time.monotonic()
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures=[pool.submit(child,'gpu',m,l,seconds) for m,l in jobs]
        outcomes=[f.result() for f in futures]
    volume.reload();volume.commit()
    return dict(status='complete' if all(r['status']=='complete' for r in outcomes) else 'partial',
                outcomes=outcomes,seconds=time.monotonic()-started)

@app.function(image=base,cpu=(2,2),memory=(8192,8192),timeout=3600,startup_timeout=120,retries=0,max_containers=1,scaledown_window=2,volumes={'/artifacts':volume})
def collect_worker():
    volume.reload();r=child('collect',seconds=3540);volume.commit();return r

@app.function(image=base,cpu=(.25,.25),memory=(2048,2048),timeout=57600,startup_timeout=120,retries=0,max_containers=1,scaledown_window=2,volumes={'/artifacts':volume})
def pipeline(resume=False):
    from numzig import ph
    from numzig.ph_overnight import contract,reserve_budget,settle_budget,legacy_root
    from numzig.fullrange.storage import Store,read_json,write_json,digest
    started=time.monotonic();volume.reload();ROOT.mkdir(parents=True,exist_ok=True)
    lock=ROOT/'coordinator.lock'
    if lock.exists() and not resume:raise RuntimeError('Existing lock: verify prior app stopped and explicitly resume')
    lock.unlink(missing_ok=True)
    write_json(lock,dict(started=time.time(),app_id=app.app_id));volume.commit()
    ledger=read_json(ROOT/'budget.json') if (ROOT/'budget.json').exists() else []
    outcomes=[];finished=set();running={};retries={};blocked=[];collection=None
    def checkpoint(state):
        write_json(ROOT/'budget.json',ledger)
        write_json(ROOT/'progress.json',dict(state=state,completed_layers=len(finished),expected_layers=97,
            completed=sorted([list(v) for v in finished]),outcomes=outcomes,blocked=blocked,
            committed_compute_allowance=sum(r['charged_usd'] for r in ledger),limit_usd=20,
            note='Conservative compute accounting, not account billing; reserved jobs include timeout and startup margin.',updated=time.time()))
        volume.commit()
    def reserve(label,seconds,cores,gib,gpus=0):
        i=reserve_budget(ledger,label,seconds,cores,gib,gpus);checkpoint('running');return i
    def done(model,level):
        p=ROOT/'results'/model/f'layer_{level:02d}'
        if not (p/'manifest.json').exists():return False
        s=Store(p);_,_,_,identity=ph.source_info(Path('/artifacts')/ph.MODELS[model][0],level)
        if s.manifest['config']!=contract(identity,'ph'):raise RuntimeError('Changed scientific code/config; separate output root required')
        return all(s.valid(k) for k in ['h0','h1','plots'])
    def charge(index,result):
        if isinstance(result,dict) and 'seconds' in result:settle_budget(ledger,index,min(result['seconds'],ledger[index]['seconds_limit']))
    coordinator_index=None
    try:
        coordinator_index=reserve('coordinator',57600,.25,2)
        # Reserve graph generation up front so exhausted workers cannot consume it.
        collect_index=reserve('gallery_and_archive',3600,2,8)
        identities=[]
        for model,(experiment,levels,dim) in ph.MODELS.items():
            _,_,meta,identity=ph.source_info(Path('/artifacts')/experiment,0)
            if meta['expected_levels']!=levels or meta['expected_hidden_dim']!=dim:raise ValueError('Unexpected source shape')
            identities.append(identity['raw_prompts'])
        if len(set(identities))!=1:raise ValueError('Models do not share prompts')
        all_jobs=[(m,l) for l in range(37) for m,v in ph.MODELS.items() if l<v[1]]
        finished.update(j for j in all_jobs if done(*j))
        # Establish real multi-process/GPU equivalence before admitting bulk GPU jobs.
        gate_jobs=[('crystal',1),('crystal',7),('starcoderbase-3b',1),('openllama-3b',1)]
        gate_jobs=[j for j in gate_jobs if j not in finished]
        with ThreadPoolExecutor(max_workers=20) as pool:
            cache_i=reserve('crystal_cache',2400,4,12)
            cache_future=pool.submit(cache_worker.remote)
            if gate_jobs:
                gate_i=reserve('shared_gpu_equivalence_gate',1080,6,40,1)
                gate=gpu_group.remote(gate_jobs);volume.reload();charge(gate_i,gate)
                outcomes.extend(gate['outcomes']);finished.update((r['model'],r['layer']) for r in gate['outcomes'] if r['status']=='complete')
                if gate['status']!='complete':
                    cache=cache_future.result();volume.reload();charge(cache_i,cache)
                    raise RuntimeError('Shared-GPU/reference gate failed; inspect per-layer logs before continuing')
            cache=cache_future.result();volume.reload();charge(cache_i,cache);outcomes.append(cache)
            if cache['status']!='complete':raise RuntimeError('Crystal source-cache preparation failed')
            checkpoint('shared_gpu_gate_passed')
            todo=deque(j for j in all_jobs if j not in finished);ready=deque();halt=False
            while todo or ready or running:
                prep_active=sum(t[0]=='prepare' for t in running.values())
                gpu_active=sum(t[0]=='gpu' for t in running.values())
                # Fill GPU work first so preparation cannot consume its allowance.
                while not halt and ready and gpu_active<10 and (len(ready)>=4 or not todo or prep_active==0):
                    batch=[ready.popleft() for _ in range(min(4,len(ready)))]
                    retry=any(retries.get(j,0) for j in batch)
                    if retry:
                        ready.extendleft(reversed(batch[1:]));batch=batch[:1]
                    timeout=1980 if retry else 1080
                    try:i=reserve('gpu/'+str(batch),timeout,6,40,1)
                    except RuntimeError:
                        ready.extendleft(reversed(batch));break
                    worker=gpu_group.with_options(timeout=timeout)
                    f=pool.submit(worker.remote,batch,1800 if retry else 900);running[f]=('gpu',batch,i);gpu_active+=1
                while not halt and todo and prep_active<8:
                    job=todo.popleft()
                    if legacy_root('/artifacts',*job):ready.append(job);continue
                    try:i=reserve('prepare/'+str(job),1800,4,8)
                    except RuntimeError:todo.appendleft(job);break
                    f=pool.submit(prepare_worker.remote,*job);running[f]=('prepare',[job],i);prep_active+=1
                if not running:
                    if ready or todo:
                        # If some ready work was newly produced, admit it before
                        # treating lack of in-flight work as budget exhaustion.
                        next_timeout=1980 if ready and retries.get(ready[0],0) else 1080
                        if ready and sum(r['charged_usd'] for r in ledger)+1.25*(next_timeout+120)*(.000542+6*.0000131+40*.00000222)<=20:
                            continue
                        blocked.extend(dict(model=m,layer=l,reason='compute_allowance') for m,l in list(ready)+list(todo))
                    break
                completed,_=wait(running,timeout=30,return_when=FIRST_COMPLETED)
                for f in completed:
                    kind,jobs,i=running.pop(f)
                    try:result=f.result()
                    except Exception as exc:
                        result=dict(status='failed',error=str(exc))
                    volume.reload();charge(i,result)
                    rows=result.get('outcomes',[result])
                    for j in jobs:
                        r=next((r for r in rows if (r.get('model'),r.get('layer'))==j),dict(model=j[0],layer=j[1],status='failed',error=result.get('error','remote worker failed')))
                        outcomes.append(r)
                        if r['status']=='complete':
                            if kind=='prepare':ready.append(j)
                            else:finished.add(j)
                        elif 'ValueError' in r.get('tail',''):
                            blocked.append(dict(model=j[0],layer=j[1],reason='validation_failure',detail=r));halt=True
                        elif retries.get(j,0)<1:
                            retries[j]=1
                            (todo if kind=='prepare' else ready).append(j)
                        else:blocked.append(dict(model=j[0],layer=j[1],reason='retry_exhausted',detail=r))
                    checkpoint('running' if not halt else 'validation_failure_draining')
                if halt and not running:
                    blocked.extend(dict(model=m,layer=l,reason='halted_after_validation_failure') for m,l in list(todo)+list(ready));break
        collection=collect_worker.remote();volume.reload();charge(collect_index,collection)
        if collection['status']!='complete':blocked.append(dict(reason='gallery_failed',detail=collection))
        state='complete' if len(finished)==97 and collection['status']=='complete' else 'needs_attention'
        settle_budget(ledger,coordinator_index,min(time.monotonic()-started,57600));coordinator_index=None
        checkpoint(state)
        report=read_json(ROOT/'progress.json');write_json(ROOT/'report.json',report);volume.commit();return report
    except Exception as exc:
        blocked.append(dict(reason='coordinator_error',error=str(exc)))
        checkpoint('needs_attention');write_json(ROOT/'report.json',read_json(ROOT/'progress.json'));volume.commit()
        raise
    finally:
        lock.unlink(missing_ok=True);volume.commit()

@app.local_entrypoint()
def main(resume: bool=False):
    if os.environ.get('MODAL_PROFILE')!='mdibrahimawad2':raise ValueError('Wrong Modal profile')
    rows=json.loads(subprocess.check_output([sys.executable,'-m','modal','app','list','--env','main','--json']))
    names=['numberline-layer-ph','numberline-ph-acceleration','numberline-ph-overnight']
    for r in rows:
        if r['description'] in names and r['app_id']!=app.app_id and (r['state']!='stopped' or int(r['tasks'])):
            raise RuntimeError('Another PH app remains active; do not overlap runs')
    print(json.dumps(pipeline.remote(resume),indent=2))
