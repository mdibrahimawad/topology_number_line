"""Resume-safe local scan of every existing numerical 3D coordinate view."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['VECLIB_MAXIMUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['MPLCONFIGDIR']='/tmp/geometric_shapes_mpl'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,hashlib,json,time
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist
from fit import fit_all
from templates import make_templates
from inventory import inventory,load_cloud,projection_diagnostics,ROOT

OUT=ROOT/'outputs/geometric_shapes'
VERSION='regular32-v2'

def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,separators=(',',':'),allow_nan=False))
    os.replace(tmp,path)

def work(entry):
    p=OUT/'fits'/f"{entry['id']}.json"
    sha=hashlib.sha256((ROOT/entry['source']).read_bytes()).hexdigest()
    if p.exists():
        prior=json.loads(p.read_text())
        if prior.get('source_sha256')==sha and prior.get('version')==VERSION:return entry['id'],'cached'
    tic=time.monotonic()
    y,targets,labels,evr=load_cloud(entry)
    result=dict(entry=entry,version=VERSION,source_sha256=sha)
    if entry['status']=='valid':
        result.update(fit_all(y,make_templates()))
        result['projection']=projection_diagnostics(entry,y,targets)
        result['unique_points']=int(len(np.unique(y,axis=0)))
        # Compare regular vertices with unrestricted four-/nine-centroid descriptions.
        norm=result['normalization']
        with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
            v=((y-np.array(norm['center']))/norm['radius'])@np.array(norm['basis']).T
        train=v[norm['train_ids']];test=v[norm['test_ids']]
        for k in (4,9):
            if len(np.unique(train,axis=0))<k:continue
            options=[kmeans2(train,k,iter=30,minit='++',seed=42+i)[0] for i in range(3)]
            centers=min(options,key=lambda c:np.min(cdist(train,c),axis=1).sum())
            distances=cdist(test,centers)
            residual=distances.min(1)
            d=np.linalg.norm(centers[:,None]-centers[None,:],axis=2)[np.triu_indices(k,1)]
            result['baselines'][f'free_{k}_centroids']=dict(data_rms=float(np.sqrt(np.mean(residual**2))),pair_distance_cv=float(np.std(d)/np.mean(d)),occupancy=np.bincount(distances.argmin(1),minlength=k).tolist())
        result['fits_count']=len(result['fits'])
        result['screen_matches']=[f['name'] for f in result['fits'] if f['screen_pass']]
    result['seconds']=time.monotonic()-tic
    write(p,result)
    return entry['id'],result.get('screen_matches',[])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=10);ap.add_argument('--limit',type=int);ap.add_argument('--only');args=ap.parse_args()
    (OUT/'fits').mkdir(parents=True,exist_ok=True)
    entries=inventory()
    if args.only:entries=[e for e in entries if args.only in e['id']]
    if args.limit:entries=entries[:args.limit]
    write(OUT/'templates.json',[{k:v for k,v in t.items() if k!='points'} for t in make_templates()])
    tic=time.monotonic();done=0
    with ProcessPoolExecutor(args.workers) as pool:
        futures={pool.submit(work,e):e for e in entries}
        for f in as_completed(futures):
            name,value=f.result();done+=1
            if done%20==0 or value not in ([], 'cached') or done==len(entries):
                print(done,'/',len(entries),name,value,'elapsed',round(time.monotonic()-tic),flush=True)
                write(OUT/'progress.json',dict(done=done,total=len(entries),seconds=time.monotonic()-tic,last=name))
    print('COMPLETE',done,flush=True)

if __name__=='__main__':main()
