"""Bounded GPU benchmark on existing, verified full-point distances."""
import ctypes
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
from scipy.spatial.distance import cdist
from numzig.ph import MODELS, inherited_distances, source_info, h1, normalization, packages, verified_file
from numzig.fullrange.storage import Store, digest, read_json, write_json, write_npz

REVISION='30243c0c752de26d7fdf6e41f08bf7b840ca4744'

def sorted_diagram(a):
    a=np.asarray(a,dtype=np.float64).reshape(-1,2)
    if not np.isfinite(a).all() or np.any(a[:,0]<0) or np.any(a[:,1]<a[:,0]):
        raise ValueError('Invalid full-range H1 diagram')
    return a[np.lexsort((a[:,1],a[:,0]))]

def compare(a,b,scale):
    a,b=sorted_diagram(a),sorted_diagram(b)
    tolerance=float(4*np.spacing(np.float32(max(scale,np.finfo(np.float32).tiny))))
    if a.shape!=b.shape:raise ValueError(f'Backend bar counts differ: {a.shape} vs {b.shape}')
    error=float(np.max(np.abs(a-b),initial=0))
    if error>tolerance:raise ValueError(f'Backend endpoints differ: {error} > {tolerance}')
    return dict(bars=len(a),max_endpoint_difference=error,tolerance=tolerance,passed=True)

def lower_triangle(d):
    if d.ndim!=2 or d.shape[0]!=d.shape[1] or not np.isfinite(d).all() or np.any(d<0):
        raise ValueError('Invalid distance matrix')
    if np.any(np.diag(d)!=0) or not np.array_equal(d,d.T):raise ValueError('Not a symmetric zero-diagonal metric')
    n=len(d);a=np.empty(n*(n-1)//2,dtype=np.float32)
    for i in range(1,n):a[i*(i-1)//2:i*(i+1)//2]=d[i,:i]
    if len(a)>np.iinfo(np.int32).max:raise ValueError('Native binding entry limit exceeded')
    return a

def gpu_h1(d):
    # Same native entry point as upstream, passing an existing NumPy buffer.
    # Upstream Python expands 50 million values into Python/ctypes objects.
    # This avoids that allocation without changing a single input value.
    import ripserplusplus
    from ripserplusplus.Ripser_plusplus_Converter import Ripser_plusplus_result
    a=lower_triangle(d);n=len(d)
    path=Path(ripserplusplus.__file__).parent
    ctypes.CDLL(str(path/'libphmap.so'))
    lib=ctypes.CDLL(str(path/'libpyripser++.so'))
    fn=lib.run_main
    fn.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_char_p),ctypes.POINTER(ctypes.c_float),ctypes.c_int,ctypes.c_int,ctypes.c_int]
    fn.restype=Ripser_plusplus_result
    args=(ctypes.c_char_p*4)(b'--format',b'lower-distance',b'--dim',b'1')
    result=fn(4,args,a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),len(a),n,n)
    if result.num_dimensions!=2:raise ValueError('Native backend returned unexpected dimensions')
    bars=result.set_of_barcodes[1]
    diagram=np.array([(bars.barcodes[i].birth,bars.barcodes[i].death) for i in range(bars.num_barcodes)],dtype=np.float64).reshape(-1,2)
    # Native result owns malloc allocations. Each benchmark is also isolated
    # in its own short-lived process to release remaining library allocations.
    libc=ctypes.CDLL(None);libc.free.argtypes=[ctypes.c_void_p]
    for dim in range(result.num_dimensions):libc.free(result.set_of_barcodes[dim].barcodes)
    libc.free(result.set_of_barcodes)
    return sorted_diagram(diagram)

def controls():
    import ripserplusplus
    theta=np.linspace(0,2*np.pi,48,endpoint=False)
    d=cdist(np.c_[np.cos(theta),np.sin(theta)],np.c_[np.cos(theta),np.sin(theta)])
    official=ripserplusplus.run('--format distance --dim 1',d)[1]
    if official.dtype.names:official=np.column_stack([official['birth'],official['death']])
    gpu=gpu_h1(d)
    wrapper=compare(gpu,official,float(d.max()))
    independent=compare(gpu,h1(d,'ripser'),float(d.max()))
    if len(gpu)!=1:raise ValueError('Circle control failed')
    line=cdist(np.arange(32)[:,None],np.arange(32)[:,None])
    if len(gpu_h1(line)):raise ValueError('Line control failed')
    return dict(circle_wrapper=wrapper,circle_reference=independent,line_passed=True)

def benchmark(artifacts,output,layer):
    import modal
    volume=modal.Volume.from_name('numberline-zigzag-results',create_if_missing=False)
    artifacts,output=Path(artifacts),Path(output)
    source=artifacts/MODELS['crystal'][0]
    _,_,_,identity=source_info(source,layer)
    old=artifacts/'ph_fullrange_v1/crystal'/f'layer_{layer:02d}'
    config=dict(source=identity,backend='ripser++',revision=REVISION,code_sha256=digest(__file__),
        preprocessing_manifest=digest(old/'manifest.json'),packages=packages(),metric='euclidean',coeff=2,threshold=None)
    store=Store(output,config,volume.commit);dep=store.manifest['config_fingerprint']
    if store.valid('h1',dep):return read_json(output/'metrics.json')
    start=time.perf_counter();control=controls()
    x,d=inherited_distances(old,identity);del x
    ids=np.sort(np.random.default_rng(42).choice(len(d),128,replace=False));sub=d[np.ix_(ids,ids)]
    gate=compare(gpu_h1(sub),h1(sub,'ripser'),float(sub.max()))
    write_json(output/'gates.json',dict(synthetic=control,real_subset=gate,point_ids=ids.tolist()))
    store.finish('gates',[output/'gates.json'],dep)
    print(f'GPU PH starting: Crystal L{layer}, {len(d)} points',flush=True)
    phstart=time.perf_counter();diagram=gpu_h1(d);phseconds=time.perf_counter()-phstart
    norm=normalization(d)
    equality=None
    if layer==1:
        reference=artifacts/'ph_fullrange_v1/direct_rips/crystal/layer_01'
        m=read_json(reference/'manifest.json')
        if m['config']['source']!=identity:raise ValueError('CPU reference source differs')
        with np.load(verified_file(reference,m,'h1','h1.npz')) as f:equality=compare(diagram,f['raw'],float(d.max()))
    write_npz(output/'h1.npz',raw=diagram,normalized=diagram/norm['scale'],right_censored=np.zeros(len(diagram),dtype=bool))
    metrics=dict(model='crystal',layer=layer,point_count=len(d),status='complete',backend='ripser++',
        revision=REVISION,ph_seconds=phseconds,total_seconds=time.perf_counter()-start,
        normalization=norm,h1_finite_count=len(diagram),h1_right_censored_count=0,
        h1_max_finite_lifetime=float(np.max(np.diff(diagram,axis=1),initial=0)/norm['scale']),
        full_cpu_reference=equality,subset_gate=gate,finite_precision='float32 filtration; no point subsampling',
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    write_json(output/'metrics.json',metrics)
    store.finish('h1',[output/'h1.npz',output/'metrics.json'],dep)
    print(json.dumps(metrics),flush=True)
    return metrics

if __name__=='__main__':benchmark(sys.argv[1],sys.argv[2],int(sys.argv[3]))
