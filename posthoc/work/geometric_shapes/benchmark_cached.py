"""Benchmark exact reuse of ICP nearest-neighbor queries; leaves live code unchanged."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import json
import time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import fit
from inventory import load_cloud
from templates import make_templates


def register_cached(y, template, starts=4, iterations=24):
    tree_y=cKDTree(y)
    transforms=[]
    for r in fit.initial_rotations(template):
        z=template@r
        transforms.append((fit.score(y,z,tree_y),1.,r,np.zeros(3)))
    transforms.sort(key=lambda t:t[0])
    best=None
    for _,scale,r,offset in transforms[:starts]:
        previous=np.inf
        z=scale*template@r+offset
        a=cKDTree(z).query(y)[1]
        b=tree_y.query(z)[1]
        w=np.r_[np.full(len(y),.5/len(y)),np.full(len(template),.5/len(template))]
        for k in range(iterations):
            source=np.concatenate([template[a],template])
            dest=np.concatenate([y,y[b]])
            scale,r,offset=fit.similarity(source,dest,w)
            z=scale*template@r+offset
            d,a=cKDTree(z).query(y)
            e,b=tree_y.query(z)
            value=float((np.mean(d*d)+np.mean(e*e))/2)
            if abs(previous-value)<1e-7:break
            previous=value
        item=(value,scale,r,offset,k+1)
        if best is None or item[0]<best[0]:best=item
    return best


if __name__=='__main__':
    root=Path(__file__).resolve().parents[2]
    entries=json.loads((root/'outputs/geometric_shapes/inventory.json').read_text())
    names=('original_crystal_L07','new_crystal_word_copy_ctx0_L32')
    entries=[next(e for e in entries if e['id']==name) for name in names]
    templates=make_templates()
    original=fit.register
    records=[]
    for entry in entries:
        y,*_=load_cloud(entry)
        tic=time.perf_counter(); expected=fit.fit_all(y,templates); t1=time.perf_counter()-tic
        fit.register=register_cached
        tic=time.perf_counter(); actual=fit.fit_all(y,templates); t2=time.perf_counter()-tic
        fit.register=original
        assert expected==actual, entry['id']
        record={'id':entry['id'],'baseline_seconds':t1,'cached_seconds':t2,'speedup':t1/t2,'all_outputs_exactly_equal':True}
        records.append(record)
        print(json.dumps(record),flush=True)
    fit.register=register_cached
    import test_fit
    test_fit.main()
    (Path(__file__).with_name('cached_benchmark.json')).write_text(json.dumps({'records':records,'synthetics_passed':True},indent=2)+'\n')
