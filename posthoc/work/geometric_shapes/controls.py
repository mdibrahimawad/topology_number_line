"""Exploratory negative controls for shortlisted shapes; not significance tests."""
import os
for key in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import argparse,json,hashlib,time
import numpy as np
import fit
from fit import fit_all
from benchmark_cached import register_cached
fit.register=register_cached
from templates import make_templates
from inventory import load_cloud,ROOT
from run import write,VERSION
OUT=ROOT/'outputs/geometric_shapes'

def select():
    rows=[json.loads(p.read_text()) for p in (OUT/'fits').glob('*.json')]
    assert len(rows)==1649 and all(r['version']==VERSION for r in rows)
    primary=[r for r in rows if r['entry']['group']!='overlay' and r['entry']['layer']>0 and 'fits' in r]
    selected={}
    def add(row,reason):
        key=row['entry']['id']
        selected.setdefault(key,dict(entry=row['entry'],reasons=[],observed_fits=row['fits']))['reasons'].append(reason)
    # All provisional passes (if any), and closest original/new example of every template.
    for r in primary:
        if r['screen_matches']:add(r,'passes descriptive threshold')
    names=[t['name'] for t in make_templates()]
    for cohort in ('original','new'):
        pool=[r for r in primary if r['entry']['group']==cohort]
        for name in names:
            row=min(pool,key=lambda r:next(f['symmetric_rms'] for f in r['fits'] if f['name']==name))
            add(row,f'closest {cohort} {name}')
    write(OUT/'control_selection.json',list(selected.values()))
    return list(selected.values())

@np.errstate(over='ignore',divide='ignore',invalid='ignore')
def control_job(job):
    row,kind,rep=job;e=row['entry'];path=OUT/'controls'/f"{e['id']}__{kind}_{rep}.json"
    if path.exists():return path.name
    y,_,_,_=load_cloud(e)
    seed=int(hashlib.sha256(f"{e['id']}_{kind}_{rep}".encode()).hexdigest()[:8],16)
    rng=np.random.default_rng(seed)
    if kind=='permuted':
        x=np.column_stack([rng.permutation(y[:,j]) for j in range(3)])
    else:
        # Gaussian with matching covariance in expectation, same point count.
        eigen,v=np.linalg.eigh(np.cov(y.T));a=v*np.sqrt(np.maximum(eigen,0))
        x=rng.normal(size=y.shape)@a.T+y.mean(0)
    tic=time.monotonic();result=fit_all(x,make_templates())
    fits=[{k:f[k] for k in ('name','family','dimension','symmetric_rms','data_rms','coverage_rms','screen_pass')} for f in result['fits']]
    write(path,dict(id=e['id'],kind=kind,replicate=rep,seed=seed,version=VERSION,fits=fits,seconds=time.monotonic()-tic))
    return path.name

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=6);ap.add_argument('--replicates',type=int,default=3);args=ap.parse_args()
    (OUT/'controls').mkdir(exist_ok=True)
    selected=select();jobs=[(row,kind,rep) for row in selected for kind in ('gaussian','permuted') for rep in range(args.replicates)]
    print('Selected',len(selected),'clouds;',len(jobs),'complete-library control scans',flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        for n,f in enumerate(as_completed([pool.submit(control_job,j) for j in jobs]),1):
            f.result()
            if n%10==0:print(n,'/',len(jobs),flush=True)
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
