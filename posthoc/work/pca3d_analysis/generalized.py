import os
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','2')
from pathlib import Path
import json,numpy as np
ROOT=Path(__file__).resolve().parents[2];O=ROOT/'outputs/pca3d_analysis';R=json.loads((O/'metrics.json').read_text());n=np.arange(1000,10000);x=(n-n.mean())/n.std();rng=np.random.default_rng(1729);randomfold=np.empty(len(n),int);randomfold[rng.permutation(len(n))]=np.arange(len(n))%5;blockfold=np.repeat(np.concatenate([rng.permutation(np.tile(np.arange(5),2)) for _ in range(9)]),100)
ids=[i for i,r in enumerate(R) if r['layer']];Y=np.stack([np.load(ROOT/'outputs/pca3d/data'/f"{R[i]['model']}_{R[i]['layer']:02}.npz")['scores'][999:9999] for i in ids]);y=Y.transpose(1,0,2).reshape(len(n),-1);sst=((y-y.mean(0))**2).sum(0).reshape(-1,3).sum(1)
with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
 for key,periods in [('generalized_decimal',[2,5,10,100,1000]),('generalized_plus_global_bend',[2,5,10,100,1000,10000])]:
  cols=[np.ones(len(n)),x]
  for p in periods:
   cols.append(np.cos(2*np.pi*n/p))
   if p!=2:cols.append(np.sin(2*np.pi*n/p))
  b=np.column_stack(cols)
  for split,fold in [('random',randomfold),('block100',blockfold)]:
   err=np.zeros(y.shape[1])
   for f in range(5):
    tr=fold!=f;te=~tr;coef=np.linalg.pinv(b[tr])@y[tr];res=y[te]-b[te]@coef;err+=(res*res).sum(0)
   r2=1-err.reshape(-1,3).sum(1)/sst
   for j,i in enumerate(ids):R[i]['fits'].setdefault(key,{})[split]=float(r2[j])
(O/'metrics.json').write_text(json.dumps(R,indent=2,allow_nan=False))
# One reproducible bin-mean similarity is paired with pointwise similarity and split-bin stability.
Z=np.load(O/'comparisons.npz');S=Z['similarity'];rid=Z['record_indices'];vr=[R[i] for i in rid];summ=[]
def procrustes(a,b):
 a=a-a.mean(0);b=b-b.mean(0);a/=np.linalg.norm(a);b/=np.linalg.norm(b)
 with np.errstate(over='ignore',divide='ignore',invalid='ignore'):return float(np.linalg.svd(a.T@b,compute_uv=False).sum())
for ma,mb in [('crystal','starcoder'),('crystal','openllama'),('starcoder','openllama')]:
 A=[i for i,r in enumerate(vr) if r['model']==ma];B=[i for i,r in enumerate(vr) if r['model']==mb];v=S[np.ix_(A,B)];best=np.unravel_index(np.argmax(v),v.shape)
 for kind,i,j in [('best_bin_means',A[best[0]],B[best[1]]),('final',A[-1],B[-1])]:
  a=Y[list(ids).index(int(rid[i]))];b=Y[list(ids).index(int(rid[j]))];pa=[];pb=[];qa=[];qb=[]
  random=np.random.default_rng(63)
  for k in range(90):
   order=random.permutation(100);pa.append(a[k*100+order[:50]].mean(0));pb.append(b[k*100+order[:50]].mean(0));qa.append(a[k*100+order[50:]].mean(0));qb.append(b[k*100+order[50:]].mean(0))
  summ.append(dict(kind=kind,model_a=ma,layer_a=vr[i]['layer'],model_b=mb,layer_b=vr[j]['layer'],bin_mean_similarity=float(S[i,j]),pointwise_similarity=procrustes(a,b),first_half_bin_similarity=procrustes(np.array(pa),np.array(pb)),second_half_bin_similarity=procrustes(np.array(qa),np.array(qb))))
(O/'shape_comparisons.json').write_text(json.dumps(summ,indent=2));print(json.dumps(summ,indent=2))
