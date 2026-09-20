import os
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from pathlib import Path
import json, csv, hashlib
import numpy as np
from scipy.spatial import cKDTree
from scipy.linalg import orthogonal_procrustes
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[2]; SRC=ROOT/'outputs/pca3d'; OUT=ROOT/'outputs/pca3d_analysis';OUT.mkdir(exist_ok=True)
idx=json.loads((SRC/'index.json').read_text()); records=[]; clouds=[]; samples=[]
for m in idx['models']:
 for l in m['layers']:
  p=SRC/l['download']
  with np.load(p) as z:
   y=z['scores']; n=z['targets']; lead=z['leading_digits']; lens=z['digit_counts'];nn=z['neighbors']
   v=z['explained_variance_ratio']
  r=dict(model=m['id'],name=m['name'],layer=l['layer'],last=m['last'],status=l['status'],variance=v.tolist(),variance3=float(v.sum()),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
  if l['status']=='valid' and l['layer']>0:
   total=np.sum((y-y.mean(0))**2)
   for key,labels in [('leading_digit',lead),('digit_length',lens)]:
    pred=np.vstack([y[labels==g].mean(0) for g in np.unique(labels)])[np.searchsorted(np.unique(labels),labels)]
    r[key+'_between_fraction']=float(1-np.sum((y-pred)**2)/total)
   q=y[999:9999]; g=n[999:9999]//1000
   pred=np.vstack([q[g==d].mean(0) for d in range(1,10)])[g-1]
   r['four_digit_leading_between_fraction']=float(1-np.sum((q-pred)**2)/np.sum((q-q.mean(0))**2))
   r['pc3_fraction_of_visible_variance']=float(v[2]/v.sum())
   r['variance_participation_3d']=float(v.sum()**2/np.sum(v*v))
   # Fraction of the saved original-space four-neighbor IDs recovered in PCA3.
   _,kn=cKDTree(y).query(y,k=5,workers=2)
   nn3=np.stack([row[row!=i][:4] for i,row in enumerate(kn)])
   r['neighbor4_retention3d']=float(np.mean([len(set(a)&set(b))/4 for a,b in zip(nn,nn3)]))
   r['same_leading_neighbor4_3d']=float(np.mean(lead[nn3]==lead[:,None]))
   r['same_leading_neighbor4_hidden']=float(np.mean(lead[nn]==lead[:,None]))
   r['consecutive_to_random_step_ratio_4digit']=float(np.median(np.linalg.norm(np.diff(q,axis=0),axis=1))/np.median(np.linalg.norm(q-q[np.random.default_rng(42).permutation(len(q))],axis=1)))
   c=np.vstack([q[g==d].mean(0) for d in range(1,10)]); e=np.linalg.svd(c-c.mean(0),compute_uv=False)**2
   r['leading_centroid_planarity3d']=float(e[:2].sum()/e.sum())
   r['centroid_adjacent_to_all_distance_ratio']=float(np.mean(np.linalg.norm(np.diff(c,axis=0),axis=1))/np.mean([np.linalg.norm(c[i]-c[j]) for i in range(9) for j in range(i+1,9)]))
   r['centroid_end_to_step_ratio']=float(np.linalg.norm(c[-1]-c[0])/np.mean(np.linalg.norm(np.diff(c,axis=0),axis=1)))
  records.append(r);clouds.append(y);samples.append(y[999:9999].reshape(90,100,3).mean(1))
print('Loaded all',len(records),'views.',flush=True)
Y=np.stack(clouds); valid=np.array([r['status']=='valid' and r['layer']>0 for r in records]); inds=np.flatnonzero(valid)
# Four-digit targets: 9,000 equally weighted examples, 1,000 per leading digit.
n=np.arange(1000,10000); x=(n-n.mean())/np.std(n); constant=np.ones((len(n),1)); linear=np.column_stack([constant,x]); g=n//1000
cat=np.eye(9)[g-1]; within=((n%1000)-499.5)/1000
piecewise=np.column_stack([cat,cat*within[:,None]])
bases={'linear':linear,'cubic':np.column_stack([linear,x*x,x*x*x]),'leading_digit':cat,'piecewise_digit_lines':piecewise,'decimal_digits':np.column_stack([cat]+[np.eye(10)[n//k%10][:,1:] for k in [100,10,1]])}
periods=[2,5,10,100,1000,3000,10000]
for p in periods:
 harmonic=np.column_stack([np.cos(2*np.pi*n/p),np.sin(2*np.pi*n/p)]) if p!=2 else np.cos(np.pi*n)[:,None]
 bases[f'helix_T{p}']=np.column_stack([linear,harmonic])
 bases[f'within_digit_T{p}']=np.column_stack([piecewise,harmonic])
# Same fixed split for every layer and model. Block folds withhold entire 100-number ranges.
rng=np.random.default_rng(1729); randomfold=np.empty(len(n),int);randomfold[rng.permutation(len(n))]=np.arange(len(n))%5
blockfold=np.repeat(np.concatenate([rng.permutation(np.tile(np.arange(5),2)) for _ in range(9)]),100)
y=Y[valid,999:9999,:].transpose(1,0,2).reshape(len(n),-1)
SST=((y-y.mean(0))**2).sum(0).reshape(-1,3).sum(1)
fit={}
with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
 for key,b in bases.items():
  score={}
  for split,fold in [('random',randomfold),('block100',blockfold)]:
   err=np.zeros(y.shape[1])
   for f in range(5):
    train=fold!=f;test=~train;coef=np.linalg.pinv(b[train])@y[train];res=y[test]-b[test]@coef;err+=(res*res).sum(0)
   r2=1-err.reshape(-1,3).sum(1)/SST
   assert np.isfinite(r2).all()
   score[split]=r2
  fit[key]=score
  print('Fit',key,flush=True)
for j,i in enumerate(inds):
 records[i]['fits']={k:{s:float(a[j]) for s,a in value.items()} for k,value in fit.items()}
 records[i]['best_repeated_helix']=max([5,10,100,1000,3000],key=lambda t:records[i]['fits'][f'helix_T{t}']['block100'])
 records[i]['best_within_digit_period']=max([2,5,10,100,1000,3000],key=lambda t:records[i]['fits'][f'within_digit_T{t}']['block100']-records[i]['fits']['piecewise_digit_lines']['block100'])
# Rotation/reflection/translation/scale invariant labeled shape comparison of 90 number-bin means.
B=np.stack(samples)[valid];B-=B.mean(1,keepdims=True);B/=np.linalg.norm(B,axis=(1,2))[:,None,None]
sim=np.empty((len(B),len(B)))
with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
 for i in range(len(B)):
  for j in range(i,len(B)):
   sim[i,j]=sim[j,i]=np.linalg.svd(B[i].T@B[j],compute_uv=False).sum()
assert np.allclose(np.diag(sim),1) and np.min(sim)>=-1e-10 and np.max(sim)<=1+1e-10
np.savez_compressed(OUT/'comparisons.npz',similarity=sim,record_indices=inds,binned_means=np.stack(samples),bin_targets=np.arange(1049.5,10000,100))
(OUT/'metrics.json').write_text(json.dumps(records,indent=2,allow_nan=False))
fields=['model','layer','status','variance3','pc3_fraction_of_visible_variance','leading_digit_between_fraction','digit_length_between_fraction','four_digit_leading_between_fraction','neighbor4_retention3d','consecutive_to_random_step_ratio_4digit','leading_centroid_planarity3d','centroid_adjacent_to_all_distance_ratio']
with (OUT/'metrics.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows({k:r.get(k,'') for k in fields} for r in records)
# Check that the test recognizes an actual sampled helix and cannot label its circular basis rank one.
q=np.column_stack([x,np.cos(2*np.pi*n/1000),np.sin(2*np.pi*n/1000)])
b=bases['helix_T1000'];res=q-b@np.linalg.lstsq(b,q,rcond=None)[0]
assert np.max(abs(res))<1e-10
(OUT/'validation.json').write_text(json.dumps(dict(views=len(records),pca_valid_views=sum(r['status']=='valid' for r in records),four_digit_fit_views=int(valid.sum()),contextual_layers=sum(r['layer']>0 for r in records),targets_per_view=10000,helix_synthetic_exact_fit=True,all_metrics_finite=True,comparison_self_similarity_one=True,split_seed=1729,four_digit_targets=9000,periods=periods),indent=2))
print('COMPLETE',flush=True)
