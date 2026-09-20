import os
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','2')
import sys,time,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pca3d'))
from build import MODELS
OUT=Path(__file__).resolve().parents[2]/'outputs/pca3d_analysis';result=[]
for model,cfg in MODELS.items():
 tic=time.monotonic();root=cfg['path'];sample=np.load(root/'hidden/00000.npy',mmap_mode='r');L,_,D=sample.shape
 sums=np.zeros((L,D));sq=np.zeros(L);groups=[np.zeros((L,10,D)) for _ in range(4)];counts=[np.zeros(10,int) for _ in range(4)];seen=[]
 for f in sorted((root/'hidden').glob('*_ids.npy')):
  ids=np.load(f);take=(ids>=999)&(ids<9999)
  if not take.any():continue
  n=ids[take]+1;seen.extend(n.tolist());a=np.asarray(np.load(f.with_name(f.name.replace('_ids','')),mmap_mode='r')[:,take,:],dtype=np.float64)
  sums+=a.sum(1);sq+=np.einsum('lij,lij->l',a,a)
  for j,k in enumerate([1000,100,10,1]):
   label=n//k%10
   for d in np.unique(label):
    mask=label==d;groups[j][:,d,:]+=a[:,mask,:].sum(1);counts[j][d]+=int(mask.sum())
 assert sorted(seen)==list(range(1000,10000))
 mean=sums/9000;total=sq-9000*np.einsum('ld,ld->l',mean,mean)
 for l in range(1,L):
  values={}
  for j,name in enumerate(['thousands','hundreds','tens','units']):
   present=counts[j]>0;groupmean=groups[j][l,present,:]/counts[j][present,None];var=float(np.einsum('gd,gd,g->',groupmean-mean[l],groupmean-mean[l],counts[j][present])/total[l]);values[name]=var
  assert np.isfinite(total[l]) and total[l]>0 and min(values.values())>=0 and sum(values.values())<1+1e-6
  result.append(dict(model=model,layer=l,space='original hidden vectors',four_digit_count=9000,total_centered_ss=float(total[l]),digit_variance_fractions=values,additive_R2=sum(values.values())))
 print(model,'complete',round(time.monotonic()-tic,2),'seconds',flush=True)
 (OUT/'hidden_digit_contributions.json').write_text(json.dumps(result,indent=2,allow_nan=False))
assert len(result)==94
print('COMPLETE',flush=True)
