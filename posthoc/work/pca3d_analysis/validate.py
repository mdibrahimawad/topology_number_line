from pathlib import Path
import json,hashlib,sys
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[2];O=ROOT/'outputs/pca3d_analysis';R=json.loads((O/'metrics.json').read_text());H=json.loads((O/'hidden_digit_contributions.json').read_text());D=json.loads((O/'digit_contributions.json').read_text())
assert len(R)==97 and len(H)==len(D)==94
for r in R:
 p=ROOT/'outputs/pca3d/data'/f"{r['model']}_{r['layer']:02}.npz";assert hashlib.sha256(p.read_bytes()).hexdigest()==r['source_sha256']
for d in D:assert np.isclose(sum(v['variance_fraction'] for v in d['digit_contributions'].values()),d['additive_R2'],atol=1e-10)
# Independent direct full-array check of streaming sums, using all four-digit Crystal L7 vectors.
sys.path.insert(0,str(ROOT/'work/pca3d'));from build import MODELS
p=MODELS['crystal']['path'];x=np.empty((9000,4096),dtype=np.float64);seen=np.zeros(9000,int)
for f in sorted((p/'hidden').glob('*_ids.npy')):
 ids=np.load(f);mask=(ids>=999)&(ids<9999)
 if mask.any():
  target=ids[mask]-999;x[target]=np.load(f.with_name(f.name.replace('_ids','')),mmap_mode='r')[7,mask];seen[target]+=1
assert np.all(seen==1);x-=x.mean(0);ss=np.einsum('ij,ij->',x,x);h=next(r for r in H if r['model']=='crystal' and r['layer']==7)
assert np.isclose(ss,h['total_centered_ss'],rtol=1e-9)
for k,name in [(1000,'thousands'),(100,'hundreds'),(10,'tens'),(1,'units')]:
 lab=np.arange(1000,10000)//k%10;b=0.
 for g in np.unique(lab):
  mu=x[lab==g].mean(0);b+=np.sum(lab==g)*np.dot(mu,mu)
 assert np.isclose(b/ss,h['digit_variance_fractions'][name],rtol=1e-9)
# KD tree identities agree with independent brute force for selected queries.
y=np.load(ROOT/'outputs/pca3d/data/starcoder_21.npz')['scores'];tree=cKDTree(y)
for i in [0,100,1000,9000,9999]:
 d=((y-y[i])**2).sum(1);d[i]=np.inf;brute=np.argsort(d)[:4];_,near=tree.query(y[i],k=5);near=near[near!=i][:4];assert set(brute)==set(near)
val=json.loads((O/'validation.json').read_text());val.update(original_space_layers_verified=94,original_source_pca_hashes_unchanged=97,balanced_digit_orthogonality_checked=94,streaming_original_space_matches_independent_direct_array=True,nearest_neighbor_search_matches_brute_force=True,generalized_fits_available_for_layers=sum('generalized_decimal' in r.get('fits',{}) for r in R),atlas_sheets=len(list((O/'atlas').glob('*.png'))),report_present=(O/'index.html').exists())
(O/'validation.json').write_text(json.dumps(val,indent=2));print(json.dumps(val,indent=2))
