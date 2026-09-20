from pathlib import Path
import json,csv
import numpy as np
from scipy.spatial.distance import pdist,squareform
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/Users/mohammed.awad/Documents/numberline_zigzag_modal_FINAL/results')
OUT=Path('/Users/mohammed.awad/Documents/Codex/2026-09-19/c/outputs/digit_order_check');OUT.mkdir(exist_ok=True)
paths={
 'Crystal': ROOT/'crystal_fullrange_modal_20260919T123307Z/full_complete/crystal_fullrange_1_10000_ctx1234_k4_seed42',
 'StarCoder':ROOT/'native_models_modal_20260919/full_complete/starcoderbase-3b_fullrange_1_10000_ctx1234_k4_seed42',
 'OpenLLaMA':ROOT/'native_models_modal_20260919/full_complete/openllama-3b_fullrange_1_10000_ctx1234_k4_seed42'}
records=[]; plotdata={}
def stats(c):
 dist=squareform(pdist(c)); low=pdist(c[:5]).mean();hi=pdist(c[6:]).mean()
 nearest=np.min(dist+np.eye(9)*np.max(dist)*10,axis=1)
 return dict(high7to9_vs_low1to5_centroid_spread=float(hi/low),digitgap_distance_rho=float(spearmanr(pdist(np.arange(1,10)[:,None]),pdist(c)).statistic),nearest_other_centroid_by_digit=nearest.tolist(),adjacent_centroid_distances=np.diag(dist,1).tolist(),pair_distances=dist.tolist())
for name,p in paths.items():
 ds=json.loads((p/'dataset.json').read_text())['records'];ds=sorted(ds,key=lambda r:r['point_id'])
 assert [r['point_id'] for r in ds]==list(range(10000))
 leads=np.array([r['leading_digit'] for r in ds]);digits=np.array([r['digit_count'] for r in ds]);targets=np.array([r['target'] for r in ds])
 assert np.array_equal(targets,np.arange(1,10001))
 n=max(int(f.stem.split('_')[1]) for f in (p/'analysis').glob('layer_*.npz'));late=list(range(n-4,n+1));
 for l in range(1,n+1):
  with np.load(p/'analysis'/f'layer_{l:02}.npz') as z:
   assert np.array_equal(z['point_ids'],np.arange(10000));s=z['scores'];ev=float(z['explained_variance_ratio'].sum())
  for subset in ['all','4digit']:
   mask=np.ones(10000,bool) if subset=='all' else digits==4
   c=np.stack([s[mask&(leads==d)].mean(0) for d in range(1,10)])
   vals=stats(c);vals['pc1_centroid_digit_rho_abs']=float(abs(spearmanr(np.arange(1,10),c[:,0]).statistic))
   vals['within_group_rms']= [float(np.sqrt(((s[mask&(leads==d)]-c[d-1])**2).sum(1).mean())) for d in range(1,10)]
   records.append(dict(model=name,layer=l,space='PCA2',subset=subset,variance_retained=ev,centroids=c.tolist(),**vals))
   if l==n and subset=='all':plotdata[name]=(s,leads,c,ev,n)
 # Full-space checks at the final five layers, including equal-length targets.
 first=np.load(p/'hidden/00000.npy',mmap_mode='r');width=first.shape[2];assert first.shape[0]==n+1
 sums={q:np.zeros((5,9,width),np.float64) for q in ['all','4digit']};counts={q:np.zeros(9,int) for q in sums};seen=[]
 for f in sorted((p/'hidden').glob('*.npy')):
  if f.stem.endswith('_ids'):continue
  ids=np.load(f.with_name(f.stem+'_ids.npy'));x=np.load(f,mmap_mode='r')[late];seen.extend(ids.tolist())
  for q in sums:
   eligible=np.ones(len(ids),bool) if q=='all' else digits[ids]==4
   for d in range(1,10):
    mask=eligible&(leads[ids]==d);sums[q][:,d-1]+=x[:,mask,:].sum(axis=1,dtype=np.float64);counts[q][d-1]+=mask.sum()
 assert sorted(seen)==list(range(10000)) and np.all(counts['4digit']==1000)
 for q in sums:
  cc=sums[q]/counts[q][None,:,None]
  for i,l in enumerate(late):
   # PCA of the full-space group means must match means of saved scores.
   with np.load(p/'analysis'/f'layer_{l:02}.npz') as z:projected=(cc[i]-z['mean'])@z['components'].T
   original=next(r for r in records if r['model']==name and r['layer']==l and r['space']=='PCA2' and r['subset']==q)
   assert np.allclose(projected,original['centroids'],atol=1e-5,rtol=1e-5)
   records.append(dict(model=name,layer=l,space='hidden',subset=q,**stats(cc[i])))
 print(name,'final',json.dumps([r for r in records if r['model']==name and r['layer']==n],default=float),flush=True)
(OUT/'metrics.json').write_text(json.dumps(records,indent=2))
scalar=[{k:v for k,v in r.items() if not isinstance(v,list)} for r in records]
keys=list(dict.fromkeys(k for r in scalar for k in r));
with (OUT/'metrics.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(scalar)
fig,axs=plt.subplots(1,3,figsize=(14,5.2));colors=plt.get_cmap('tab10')(np.array([0,1,2,3,4,5,6,7,9]))
for ax,(name,(s,leads,c,ev,n)) in zip(axs,plotdata.items()):
 scale=np.sqrt((s*s).sum(1).mean());s=s/scale;c=c/scale
 for d in range(1,10):
  ax.scatter(*s[leads==d].T,s=3,color=colors[d-1],alpha=.055,rasterized=True)
  ax.scatter(*c[d-1],s=145,color=colors[d-1],edgecolor='white',linewidth=1.5,zorder=5)
  ax.annotate(str(d),c[d-1],xytext=(7,7),textcoords='offset points',fontsize=12,fontweight='bold',zorder=6)
 ax.set_aspect('equal',adjustable='datalim');ax.set_title(f'{name} · final layer {n}\nTwo PCs retain {100*ev:.1f}% of variance',fontsize=12)
 ax.set_xlabel('PC1 / RMS projected radius');ax.set_ylabel('PC2 / RMS projected radius');ax.grid(alpha=.15)
fig.suptitle('Where are the leading-digit group centers?',fontsize=17)
fig.text(.5,.025,'Large labeled dots = group means; faint points = all 10,000 targets. Equal axis scales; no ordering imposed.',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.06,1,.91));fig.savefig(OUT/'final_centroids.png',dpi=180);fig.savefig(OUT/'final_centroids.svg');plt.close(fig)
summary={}
for name in paths:
 rows=[r for r in records if r['model']==name];n=max(r['layer'] for r in rows);summary[name]={}
 for space in ['PCA2','hidden']:
  for q in ['all','4digit']:
   rr=[r for r in rows if r['space']==space and r['subset']==q and r['layer']>=n-4]
   summary[name][space+'_'+q]={'final_spread_ratio':rr[-1]['high7to9_vs_low1to5_centroid_spread'],'last5_spread_ratios':[r['high7to9_vs_low1to5_centroid_spread'] for r in rr],'last5_digitgap_rhos':[r['digitgap_distance_rho'] for r in rr]}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
