import os,sys,subprocess,json,time
from pathlib import Path
base=Path('/Users/mohammed.awad/Documents/Codex/2026-09-19/c/work/ph_backend_bench');base.mkdir(exist_ok=True)
if len(sys.argv)>1:
 import numpy as np
 from scipy.spatial.distance import cdist
 from gph import ripser_parallel
 n=int(sys.argv[1]);collapse=bool(int(sys.argv[2]))
 path='/Users/mohammed.awad/Documents/numberline_zigzag_modal_FINAL/results/native_models_modal_20260919/full_complete/starcoderbase-3b_fullrange_1_10000_ctx1234_k4_seed42/analysis_cache/layer_01.npy'
 x=np.load(path,mmap_mode='r');ids=np.sort(np.random.default_rng(42).choice(len(x),n,replace=False));d=cdist(x[ids],x[ids])
 t=time.perf_counter();a=ripser_parallel(d,metric='precomputed',maxdim=1,n_threads=1,collapse_edges=collapse)['dgms'][1];seconds=time.perf_counter()-t
 np.save(base/f'{n}_{int(collapse)}.npy',a)
 print(json.dumps(dict(n=n,collapse_edges=collapse,seconds=seconds,h1_bars=len(a))))
else:
 rows=[]
 for n in [256,512,1024]:
  for flag in [0,1]:
   try:
    r=subprocess.run([sys.executable,__file__,str(n),str(flag)],capture_output=True,text=True,timeout=60,check=True)
    row=json.loads(r.stdout.strip());rows.append(row);print(row,flush=True)
   except subprocess.TimeoutExpired:rows.append(dict(n=n,collapse_edges=bool(flag),status='timeout_60s'))
 (base/'timings.json').write_text(json.dumps(rows,indent=2))
