from pathlib import Path
import json
import numpy as np
ROOT=Path('/Users/mohammed.awad/Documents/numberline_zigzag_modal_FINAL/results')
PH=Path('outputs/ph_fullrange_overnight/results/results')
OUT=Path('outputs/final_layer_change');OUT.mkdir(exist_ok=True)
models=[('crystal','Crystal',32,ROOT/'crystal_fullrange_modal_20260919T123307Z/full_complete/crystal_fullrange_1_10000_ctx1234_k4_seed42'),('starcoderbase-3b','StarCoder',36,ROOT/'native_models_modal_20260919/full_complete/starcoderbase-3b_fullrange_1_10000_ctx1234_k4_seed42'),('openllama-3b','OpenLLaMA',26,ROOT/'native_models_modal_20260919/full_complete/openllama-3b_fullrange_1_10000_ctx1234_k4_seed42')]
rows=[]
for key,name,last,path in models:
 ds=sorted(json.loads((path/'dataset.json').read_text())['records'],key=lambda r:r['point_id']);digits=np.array([r['leading_digit'] for r in ds]);counts=np.bincount(digits,minlength=10)[1:]
 for layer in [last-1,last]:
  with np.load(path/'analysis'/f'layer_{layer:02}.npz') as z:
   s=z['scores'].astype(np.float64);assert np.array_equal(z['point_ids'],np.arange(10000))
   ev=float(z['explained_variance_ratio'].sum())
  centers=np.array([s[digits==d].mean(0) for d in range(1,10)])
  total=np.mean(np.sum((s-s.mean(0))**2,axis=1));between=np.sum(counts*np.sum((centers-s.mean(0))**2,axis=1))/len(s)
  within=np.mean(np.sum((s-centers[digits-1])**2,axis=1));assert np.isclose(total,between+within,rtol=1e-12)
  ph=PH/key/f'layer_{layer:02}';meta=json.loads((ph/'metrics.json').read_text());h0=np.load(ph/'h0.npz')['normalized'];h1=np.load(ph/'h1.npz')['normalized'];end=meta['max_normalized_distance'];assert np.isfinite(h1).all()
  def count_at(diag,t):return int(np.sum((diag[:,0]<=t)&(diag[:,1]>t)))
  assert count_at(h0,end)==1 and count_at(h1,end)==0
  rows.append(dict(model=name,layer=layer,pca_total_variance=total,pca_between_fraction=between/total,pca_between_to_within_rms_ratio=float(np.sqrt(between/within)),pca_variance_retained=ev,typical_hidden_pair_distance=meta['normalization']['scale'],end_of_full_filtration_beta0=1,end_of_full_filtration_beta1=0))
(OUT/'metrics.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
print(json.dumps(rows,indent=2))
