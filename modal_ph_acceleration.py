"""One bounded GPU benchmark; no inference and no full-run launch."""
from pathlib import Path
import json,os,subprocess,sys,time,threading
import modal

ROOT=Path(__file__).resolve().parent
REV='30243c0c752de26d7fdf6e41f08bf7b840ca4744'
base=(modal.Image.debian_slim(python_version='3.12').pip_install_from_requirements(str(ROOT/'requirements-ph.txt'))
      .add_local_dir(ROOT/'numzig','/root/numzig',ignore=['__pycache__','*.pyc']))
gpu_image=(modal.Image.from_registry('nvidia/cuda:11.8.0-devel-ubuntu22.04',add_python='3.12')
 .apt_install('git','cmake','build-essential')
 .pip_install_from_requirements(str(ROOT/'requirements-ph.txt'))
 .run_commands(f'git clone https://github.com/simonzhang00/ripser-plusplus.git /opt/ripser-plusplus && cd /opt/ripser-plusplus && git checkout {REV}',
  'cmake -S /opt/ripser-plusplus -B /opt/rpp-build -DCUDA_ARCH_LIST=8.9 -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=/opt/ripser-plusplus/ripserplusplus',
  'cmake --build /opt/rpp-build -j 4')
 .env({'PYTHONPATH':'/root:/opt/ripser-plusplus','OMP_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','MPLCONFIGDIR':'/tmp/matplotlib','PYTHONUNBUFFERED':'1'})
 .add_local_dir(ROOT/'numzig','/root/numzig',ignore=['__pycache__','*.pyc']))
app=modal.App('numberline-ph-acceleration')
volume=modal.Volume.from_name('numberline-zigzag-results',create_if_missing=False)
OUT=Path('/artifacts/ph_fullrange_v1')

@app.function(image=gpu_image,gpu='L40S',cpu=(8,8),memory=(32768,32768),timeout=1020,startup_timeout=120,retries=0,max_containers=1,scaledown_window=2,volumes={'/artifacts':volume})
def gpu_benchmark(attempt, efficient=False):
    from numzig.fullrange.storage import write_json,read_json
    volume.reload();dest=OUT/'acceleration_v1'/attempt
    seconds=180 if efficient else 900
    cores,gib=(4,12) if efficient else (8,32)
    layers=[7] if efficient else [1,7]
    deadline=time.monotonic()+seconds;outcomes=[];samples=[];stop=threading.Event()
    def monitor():
        while not stop.is_set():
            r=subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
            samples.append(dict(time=time.time(),gpu=r.stdout.strip()))
            stop.wait(3)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try:
        for layer in layers:
            remaining=deadline-time.monotonic()
            if remaining<=1:break
            started=time.monotonic();path=dest/f'layer_{layer:02d}'
            try:
                subprocess.run([sys.executable,'-m','numzig.ph_acceleration','/artifacts',str(path),str(layer)],check=True,timeout=remaining,
                               env={**os.environ,'OMP_NUM_THREADS':str(cores)})
                outcomes.append(dict(layer=layer,status='complete',seconds=time.monotonic()-started))
            except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as exc:
                outcomes.append(dict(layer=layer,status='timeout' if isinstance(exc,subprocess.TimeoutExpired) else 'failed',seconds=time.monotonic()-started,error=str(exc)))
                break
    finally:
        stop.set();thread.join(timeout=12);volume.reload()
        report=dict(outcomes=outcomes,passed=len(outcomes)==len(layers) and all(r['status']=='complete' for r in outcomes),gpu='L40S',cpus=cores,memory_gib=gib,revision=REV)
        write_json(dest/'report.json',report);write_json(dest/'gpu_samples.json',samples);volume.commit()
    return report

@app.function(image=base,cpu=(.25,.25),memory=(1024,1024),timeout=1300,retries=0,max_containers=1,scaledown_window=2,volumes={'/artifacts':volume})
def coordinate(efficient=False):
    from numzig.fullrange.storage import read_json,write_json
    volume.reload();lock=OUT/'coordinator.lock'
    if lock.exists():raise RuntimeError('Existing coordinator lock; inspect prior app before resuming')
    attempt=str(time.time_ns());ledger=read_json(OUT/'budget_ledger.json')
    seconds,cores,gib,coord_seconds=(180,4,12,480) if efficient else (900,8,32,1300)
    gpu_cost=1.25*(seconds+240)*(.000542+cores*.0000131+gib*.00000222)
    cpu_cost=1.25*(coord_seconds+240)*(.25*.0000131+.00000222)
    if sum(r['reserved_usd'] for r in ledger if r['stage']=='pilot')+gpu_cost+cpu_cost>3:
        raise RuntimeError('Remaining $3 pilot allowance insufficient')
    write_json(lock,dict(stage='gpu_benchmark',attempt=attempt));volume.commit()
    try:
        for job,cost in [('gpu_benchmark',gpu_cost),('gpu_coordinator',cpu_cost)]:
            ledger.append(dict(stage='pilot',job=job,reserved_usd=cost,status='reserved',started=time.time()))
        write_json(OUT/'budget_ledger.json',ledger);volume.commit()
        worker=gpu_benchmark.with_options(cpu=(cores,cores),memory=(gib*1024,gib*1024),timeout=seconds+120)
        result=worker.remote(attempt,efficient);volume.reload()
        result.update(attempt=attempt,reserved_usd=sum(r['reserved_usd'] for r in ledger),note='Reservations are not actual billing; builds/storage excluded.')
        write_json(OUT/'acceleration_v1/latest.json',result);volume.commit()
        return result
    finally:lock.unlink(missing_ok=True);volume.commit()

@app.local_entrypoint()
def main(efficient: bool=False):
    if os.environ.get('MODAL_PROFILE')!='mdibrahimawad2':raise ValueError('Wrong Modal profile')
    rows=json.loads(subprocess.check_output([sys.executable,'-m','modal','app','list','--env','main','--json']))
    for r in rows:
        if r['description'] in ('numberline-layer-ph','numberline-ph-acceleration') and r['app_id']!=app.app_id and (r['state']!='stopped' or int(r['tasks'])):
            raise RuntimeError('Another PH app is active')
    coordinator=coordinate.with_options(timeout=480 if efficient else 1300)
    print(json.dumps(coordinator.remote(efficient),indent=2))
