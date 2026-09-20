import os
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','2')
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/pca3d_analysis';SRC=ROOT/'outputs/pca3d/data'
R=json.loads((OUT/'metrics.json').read_text());result=[]
for r in R:
 if not r['layer']:continue
 with np.load(SRC/f"{r['model']}_{r['layer']:02}.npz") as z:y=z['scores'][999:9999]
 n=np.arange(1000,10000);yc=y-y.mean(0);total=np.sum(yc*yc);pred=np.zeros_like(y);d={}
 with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
  for k,name in [(1000,'thousands'),(100,'hundreds'),(10,'tens'),(1,'units')]:
   lab=n//k%10;values=np.unique(lab);means=np.stack([yc[lab==v].mean(0) for v in values]);p=means[np.searchsorted(values,lab)];pred+=p
   center_var=float(np.sum(p*p)/total)
   basis=np.column_stack([np.ones(len(values)),np.cos(2*np.pi*values/10),np.sin(2*np.pi*values/10)])
   coefficients=np.linalg.lstsq(basis,means,rcond=None)[0];fit=basis@coefficients
   circleR2=1-np.sum((means-fit)**2)/np.sum((means-means.mean(0))**2)
   s=np.linalg.svd(coefficients[1:],compute_uv=False)
   d[name]=dict(variance_fraction=center_var,circular_first_harmonic_centroid_R2=float(circleR2),ellipse_axis_ratio=float(s[-1]/s[0]),means=means.tolist(),values=values.tolist())
  additiveR2=float(1-np.sum((yc-pred)**2)/total)
  assert abs(sum(v['variance_fraction'] for v in d.values())-additiveR2)<1e-10
  steps=np.linalg.norm(np.diff(y,axis=0),axis=1);previous=n[:-1]
  carries={str(k):float(np.median(steps[(previous+1)%k==0])/np.median(steps[(previous+1)%10!=0])) for k in [10,100,1000]}
  result.append(dict(model=r['model'],layer=r['layer'],digit_contributions=d,additive_R2=additiveR2,unexplained_fraction=1-additiveR2,carry_step_ratio=carries))
(OUT/'digit_contributions.json').write_text(json.dumps(result,indent=2,allow_nan=False))
for m in ['crystal','starcoder','openllama']:
 rows=[r for r in result if r['model']==m];print('\n',m)
 for l in [1,4,6,7,8,rows[-2]['layer'],rows[-1]['layer']]:
  r=next(r for r in rows if r['layer']==l);print(l,{k:round(d['variance_fraction'],3) for k,d in r['digit_contributions'].items()},'sum',round(r['additive_R2'],3))
  if l==1:print('circleR2',{k:round(d['circular_first_harmonic_centroid_R2'],3) for k,d in r['digit_contributions'].items()})
