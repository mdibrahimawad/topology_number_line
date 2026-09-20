"""Analysis-only Modal app. Import does not launch jobs or access model weights."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import modal

ROOT = Path(__file__).resolve().parent
REMOTE = '/root/numberline_ph'
APP_NAME = 'numberline-layer-ph'
image = (modal.Image.debian_slim(python_version='3.12')
    .pip_install_from_requirements(str(ROOT/'requirements-ph.txt'))
    .env({'PYTHONPATH':REMOTE,'MPLBACKEND':'Agg','MPLCONFIGDIR':'/tmp/matplotlib',
          'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4','PYTHONUNBUFFERED':'1'})
    .add_local_dir(ROOT/'numzig',REMOTE+'/numzig',ignore=['__pycache__','*.pyc']))
app=modal.App(APP_NAME,image=image)
volume=modal.Volume.from_name('numberline-zigzag-results',create_if_missing=False)
OUT=Path('/artifacts/ph_fullrange_v1')
RESULTS=OUT/'direct_rips'


@app.function(cpu=(4,4),memory=(16384,16384),timeout=3900,startup_timeout=120,
              retries=0,max_containers=24,scaledown_window=2,volumes={'/artifacts':volume})
def layer_job(model: str, layer: int, seconds: int):
    from numzig.ph import MODELS
    from numzig.fullrange.storage import write_json
    volume.reload()
    dest=RESULTS/model/f'layer_{layer:02d}'
    # Child isolates native allocations and makes native-code timeouts enforceable.
    script='''from pathlib import Path
import modal, sys
from numzig.ph import run_layer
from numzig.fullrange.storage import read_json
v=modal.Volume.from_name('numberline-zigzag-results',create_if_missing=False)
old=Path(sys.argv[4]);reuse=None
if (old/'manifest.json').exists():
    m=read_json(old/'manifest.json')
    if all(m['artifacts'].get(f'distance/{i:05d}',{}).get('status')=='complete' for i in range(0,10000,256)):
        reuse=old
run_layer(Path(sys.argv[1]),Path(sys.argv[2]),int(sys.argv[3]),commit=v.commit,reuse_preprocessing=reuse)
'''
    started=time.monotonic()
    try:
        subprocess.run([sys.executable,'-c',script,str(Path('/artifacts')/MODELS[model][0]),str(dest),str(layer),str(OUT/model/f'layer_{layer:02d}')],
                       check=True,timeout=seconds,cwd=REMOTE)
        return dict(model=model,layer=layer,status='complete',seconds=time.monotonic()-started)
    except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as exc:
        # Reload child commits before publishing a separate failure receipt.
        volume.reload()
        failure=dict(model=model,layer=layer,status='timeout' if isinstance(exc,subprocess.TimeoutExpired) else 'failed',
                     seconds=time.monotonic()-started,error=str(exc))
        write_json(OUT/'failures'/f'{model}_{layer:02d}_{time.time_ns()}.json',failure)
        volume.commit()
        return failure


@app.function(cpu=(.25,.25),memory=(1024,1024),timeout=14400,startup_timeout=120,
              retries=0,max_containers=1,scaledown_window=2,volumes={'/artifacts':volume})
def pipeline(stage: str, workers: int, seconds: int):
    from numzig.ph import MODELS,PILOT,reserve,packages,source_info
    from numzig.fullrange.storage import Store,read_json,write_json,digest
    volume.reload()
    OUT.mkdir(parents=True,exist_ok=True)
    # A durable lock fails closed on coordinator preemption. The operator first
    # verifies no active app, then explicitly clears it for a deliberate resume.
    lock=OUT/'coordinator.lock'
    if lock.exists():raise RuntimeError('Existing coordinator lock: verify old app stopped before explicit resume')
    write_json(lock,dict(stage=stage,started=time.time()))
    volume.commit()
    ledger_path=OUT/'budget_ledger.json'
    ledger=read_json(ledger_path) if ledger_path.exists() else []
    outcomes=[]
    try:
        reserve(ledger,stage,'coordinator',.25,1,14400)
        write_json(ledger_path,ledger);volume.commit()
        prompt_hashes=set()
        for model,(experiment,levels,dimension) in MODELS.items():
            _,_,metadata,identity=source_info(Path('/artifacts')/experiment,0)
            if metadata['expected_levels']!=levels or metadata['expected_hidden_dim']!=dimension:
                raise ValueError(f'Unexpected model shape: {model}')
            prompt_hashes.add(identity['raw_prompts'])
        if len(prompt_hashes)!=1:raise ValueError('Saved models do not share identical raw prompts')
        if stage=='full':
            review=read_json(OUT/'pilot_report.json')
            if not review['passed'] or review['code_sha256']!=digest(REMOTE+'/numzig/ph.py') or review['packages']!=packages():
                raise RuntimeError('A complete passing pilot under this code/packages is required')
            # Conservative estimate from slowest observed pilot layer per model.
            estimate=sum(max(r['seconds'] for r in review['outcomes'] if r['model']==m)*v[1]
                         for m,v in MODELS.items())*(4*.0000131+16*.00000222)*1.5
            if estimate>19-sum(r['reserved_usd'] for r in ledger if r['stage']=='full'):
                raise RuntimeError(f'Pilot predicts {estimate:.2f} USD compute; full allowance insufficient')
        todo=[]
        for model,(_,levels,_) in MODELS.items():
            for layer in PILOT[model] if stage=='pilot' else range(levels):
                dest=RESULTS/model/f'layer_{layer:02d}'
                if (dest/'manifest.json').exists():
                    store=Store(dest)
                    if store.manifest['config']['code_sha256']!=digest(REMOTE+'/numzig/ph.py'):
                        raise RuntimeError('Code changed: use a separate PH output root; never force checkpoint hashes')
                    if store.valid('h1') and store.valid('plots'):
                        metrics=read_json(dest/'metrics.json')
                        outcomes.append(dict(model=model,layer=layer,status='complete',seconds=metrics['invocation_seconds'],reused=True))
                        continue
                todo.append((model,layer))
        if stage=='pilot':
            todo.sort(key=lambda job:(PILOT[job[0]].index(job[1]),list(MODELS).index(job[0])))
        # Reserve the entire next wave before submission; no failed jobs are retried.
        for start in range(0,len(todo),workers):
            batch=todo[start:start+workers]
            trial=list(ledger)
            for model,layer in batch:reserve(trial,stage,f'{model}/{layer}',4,16,seconds)
            ledger=trial;write_json(ledger_path,ledger);volume.commit()
            with ThreadPoolExecutor(max_workers=workers) as pool:
                bounded=layer_job.with_options(timeout=seconds+120)
                futures=[pool.submit(bounded.remote,m,l,seconds) for m,l in batch]
                results=[f.result() for f in as_completed(futures)]
            outcomes.extend(results);volume.reload()
            write_json(OUT/f'{stage}_progress.json',dict(outcomes=outcomes,total=len(todo),reserved_usd=sum(r['reserved_usd'] for r in ledger)))
            volume.commit()
            if any(r['status']!='complete' for r in results):break
        expected=12 if stage=='pilot' else 97
        report=dict(stage=stage,passed=len(outcomes)==expected and all(r['status']=='complete' for r in outcomes),
                    outcomes=outcomes,packages=packages(),code_sha256=digest(REMOTE+'/numzig/ph.py'),
                    reserved_usd=sum(r['reserved_usd'] for r in ledger),
                    note='Conservative compute reservations, not Modal billing or an account hard spend cap. No refunds. Plot/collect locally.')
        write_json(OUT/f'{stage}_report.json',report);volume.commit()
        return report
    finally:
        # A killed coordinator leaves its lock. Normal failure releases it only
        # after synchronous worker waves have drained.
        lock.unlink(missing_ok=True);volume.commit()


@app.function(cpu=(.25,.25),memory=(512,512),timeout=60,retries=0,volumes={'/artifacts':volume})
def clear_lock():
    volume.reload();(OUT/'coordinator.lock').unlink(missing_ok=True);volume.commit()


@app.local_entrypoint()
def main(stage: str='pilot',workers: int=2,seconds: int=600,resume: bool=False):
    if stage not in ('pilot','full') or not 1<=workers<=24 or not 60<=seconds<=3600:
        raise ValueError('Invalid stage/workers/seconds')
    if os.environ.get('MODAL_PROFILE')!='mdibrahimawad2':
        raise ValueError('Use the intended MODAL_PROFILE=mdibrahimawad2')
    # Refuse concurrent PH app launches; no changes to account/plan/secrets.
    active=json.loads(subprocess.check_output([sys.executable,'-m','modal','app','list','--env','main','--json']))
    for row in active:
        if not all(k in row for k in ('description','state','app_id','tasks')):
            raise ValueError('Unknown Modal app-list schema; inspect before launching')
        name=row['description'];state=str(row['state']).lower();ident=row['app_id']
        if name==APP_NAME and ident!=app.app_id and (state!='stopped' or int(row['tasks'])!=0):
            raise RuntimeError('Another PH app may be active; wait for it to stop before resuming')
    if resume:clear_lock.remote()
    print(json.dumps(pipeline.remote(stage,workers,seconds),indent=2))
