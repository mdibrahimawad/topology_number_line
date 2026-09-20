"""Validate downloaded pilot science artifacts without cloud computation."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
repo=Path('/Users/mohammed.awad/Documents/numberline_zigzag_modal_FINAL')
sys.path.insert(0,str(repo))
from numzig.ph import MODELS,PILOT,source_info,packages
from numzig.fullrange.storage import Store
root=repo/'results/ph_execution_20260919/direct_lightweight'
sources={
 'crystal':repo/'results/crystal_fullrange_modal_20260919T123307Z/full_complete'/MODELS['crystal'][0],
 **{m:repo/'results/native_models_modal_20260919/full_complete'/MODELS[m][0] for m in MODELS if m!='crystal'}
}
rows=[]
for model,layers in PILOT.items():
 for layer in layers:
  p=root/model/f'layer_{layer:02d}'
  if not (p/'manifest.json').exists():continue
  s=Store(p)
  if not all(s.valid(k) for k in ('h0','h1','plots')):continue
  metrics=json.loads((p/'metrics.json').read_text())
  _,_,_,identity=source_info(sources[model],layer)
  assert s.manifest['config']['source']==identity
  assert s.manifest['config']['code_sha256']==hashlib.sha256((repo/'numzig/ph.py').read_bytes()).hexdigest()
  assert metrics['point_count']==10000 and metrics['dimensions']==MODELS[model][2]
  assert metrics['coefficient_field']==2 and metrics['normalized_threshold'] is None
  assert metrics['status']=='complete' and metrics['backend_gate']['passed']
  assert metrics['packages']==packages() and metrics['edge_collapse'] is False
  scale=metrics['normalization']['scale'];assert scale>0
  with np.load(p/'h0.npz') as h0,np.load(p/'h1.npz') as h1:
   assert h0['raw'].shape==(10000,2) and h0['mst_edges'].shape==(9999,2)
   assert np.array_equal(h0['point_ids'],np.arange(10000))
   assert np.all(h0['raw'][:,0]==0) and np.isinf(h0['raw'][-1,1])
   assert np.isfinite(h0['raw'][:-1]).all()
   assert np.array_equal(np.sort(h0['mst_distances']),h0['raw'][:-1,1])
   assert np.array_equal(h0['normalized'],h0['raw']/scale)
   a=h1['raw'];assert a.ndim==2 and a.shape[1]==2 and np.isfinite(a).all()
   assert np.all(a[:,1]>=a[:,0]) and np.all(a>=0)
   assert not h1['right_censored'].any() and metrics['h1_right_censored_count']==0
   assert len(a)==metrics['h1_finite_count']
   assert np.array_equal(h1['normalized'],a/scale)
   assert np.isclose(metrics['h1_max_finite_lifetime'],np.max(np.diff(a/scale,axis=1),initial=0))
   previous=repo/'results/ph_execution_20260919/lightweight'/model/f'layer_{layer:02d}'/'h0.npz'
   if previous.exists():
    with np.load(previous) as old:
     for k in old.files:assert np.array_equal(old[k],h0[k]),(model,layer,k)
  rows.append({'model':model,'layer':layer,'h1_bars':metrics['h1_finite_count'],
    'max_normalized_lifetime':metrics['h1_max_finite_lifetime'],
    'ph_seconds':metrics['ph_seconds'],'invocation_seconds':metrics['invocation_seconds'],
    'peak_rss_gib':metrics['peak_rss_bytes']/2**30,'backend_difference':metrics['backend_gate']['bottleneck'],
    'all_artifact_hashes_valid':True,'source_identity_matches':True})
result={'completed_validated':len(rows),'expected_pilot':12,'layers':rows,
 'scope':'Checks artifact hashes, full target/source identity, diagram integrity, normalization, saved sampled-backend gate, and exact legacy H0 agreement where available. Does not independently recompute full 10000-point H1.'}
(root/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
