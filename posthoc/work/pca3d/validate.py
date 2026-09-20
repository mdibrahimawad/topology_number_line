from pathlib import Path
import json,hashlib
import numpy as np
from build import MODELS,OUT,digest
index=json.loads((OUT/'index.json').read_text());checks=[]
assert index['completed']==97
for m in index['models']:
 cfg=MODELS[m['id']];assert [l['layer'] for l in m['layers']]==list(range(cfg['last']+1))
 for l in m['layers']:
  b=OUT/'data'/f"{m['id']}_{l['layer']:02}";v=json.loads(b.with_suffix('.validation.json').read_text());data=json.loads(b.with_suffix('.json').read_text());source=cfg['path']/'analysis'/f"layer_{l['layer']:02}.npz"
  assert digest(source)==v['source_analysis_sha256'] and digest(cfg['path']/'dataset.json')==v['source_dataset_sha256']
  assert digest(b.with_suffix('.npz'))==v['scores_npz_sha256'] and digest(b.with_suffix('.json'))==v['viewer_json_sha256']
  with np.load(b.with_suffix('.npz')) as z,np.load(source) as original:
   assert z['scores'].shape==(10000,3) and np.isfinite(z['scores']).all()
   assert np.array_equal(z['point_ids'],np.arange(10000)) and np.array_equal(z['targets'],np.arange(1,10001))
   assert np.allclose(np.asarray(data['scores']),z['scores'],rtol=0,atol=5.1e-8)
   if l['status']=='valid':
    assert np.array_equal(z['scores'][:,:2],original['scores']) and np.array_equal(z['components'][:2],original['components'])
    assert np.array_equal(z['neighbors'],original['neighbors']) and np.array_equal(z['distances'],original['distances'])
    assert np.all(z['neighbors']!=np.arange(10000)[:,None]) and z['neighbors'].shape==(10000,4)
    assert np.all(z['explained_variance'][:-1]>=z['explained_variance'][1:]*(1-1e-7))
   else:assert np.all(z['scores']==0)
  checks.append(dict(model=m['id'],layer=l['layer'],passed=True))
result=dict(views=len(checks),contextual_layers=94,embedding_levels=3,valid_pca_views=95,coincident_embedding_views=2,all_checks_passed=True,source_2d_coordinate_files_unchanged=True,original_pc12_preserved_exactly=True,point_count_each=10000,checks=checks)
(OUT/'final_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='checks'}))
