import os
os.environ['VECLIB_MAXIMUM_THREADS']='2'
from pathlib import Path
import numpy as np,time,json
from scipy.sparse.linalg import LinearOperator,eigsh
p=Path('/Users/mohammed.awad/Documents/numberline_zigzag_modal_FINAL/results/crystal_fullrange_modal_20260919T123307Z/full_complete/crystal_fullrange_1_10000_ctx1234_k4_seed42')
t=time.monotonic();x=np.empty((10000,4096),dtype=np.float64)
for f in sorted((p/'hidden').glob('*_ids.npy')):
 ids=np.load(f);a=np.load(f.with_name(f.name.replace('_ids','')),mmap_mode='r');x[ids]=a[7]
print('load',time.monotonic()-t,flush=True)
with np.load(p/'analysis/layer_07.npz') as z:mean=z['mean'];basis=z['components'];old=z['scores'];ev=z['explained_variance'];ratio=z['explained_variance_ratio']
x-=mean;calls=0

def mv(v):
 global calls
 calls+=1
 v=v-basis.T@(basis@v)
 y=x.T@(x@v)
 return y-basis.T@(basis@y)
t=time.monotonic();vals,vec=eigsh(LinearOperator((4096,4096),matvec=mv,dtype=np.float64),k=1,which='LA',tol=1e-10,v0=np.random.default_rng(42).normal(size=4096))
v=vec[:,0];v-=basis.T@(basis@v);v/=np.linalg.norm(v);pc=x@v
res=np.linalg.norm(x.T@pc-vals[0]*v)/vals[0]
print(json.dumps(dict(solve_seconds=time.monotonic()-t,calls=calls,eigval=float(vals[0]),relative_eigen_residual=float(res),old_score_max_error=float(np.max(abs(x@basis.T-old))),pc3_variance=float(pc@pc/9999),pc2_variance=float(ev[1]),pc3_variance_ratio=float(pc@pc/np.sum(x*x)),cpu_count=os.cpu_count())),flush=True)
