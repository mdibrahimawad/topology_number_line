"""Read-only real-output audit. Never calls a model or fits a new PCA."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from PIL import Image
from numzig.fullrange import configuration
from numzig.fullrange.dataset import load_dataset, SMOKE_TARGETS
from numzig.fullrange.extract import inventory, read_layer, chunks
from numzig.fullrange.storage import Store, digest, read_json, write_json

p=argparse.ArgumentParser()
p.add_argument('root', type=Path)
p.add_argument('--report', type=Path, required=True)
a=p.parse_args()
s=Store(a.root)
assert s.manifest['config'] == configuration(s.manifest['config']['smoke'])
assert all(s.manifest['stages'].get(k)=='complete' for k in ['dataset','extraction','analysis','plots'])
for key,item in s.manifest['artifacts'].items():
    assert item['status']=='complete', key
    for f,h in item['files'].items():
        assert (s.root/f).is_file() and digest(s.root/f)==h, (key,f)
rows=load_dataset(s)
n=len(rows)
ids=np.array([r['point_id'] for r in rows],dtype=np.int32)
target=np.array([r['target'] for r in rows])
assert target.tolist()==(SMOKE_TARGETS if s.manifest['config']['smoke'] else list(range(1,10001)))
assert np.array_equal(ids,target-1) and len(set(ids))==n
assert all(r['final_token_text']=='=' and r['prompt'].endswith('=') and r['extraction_position']==len(r['input_token_ids'])-1 for r in rows)
meta=read_json(s.root/'tokenizer_provenance.json')
assert not meta.get('synthetic',False)
for k in ['model_revision','tokenizer_revision','wrapper_revision']:
    assert meta[k]==s.manifest['config']['model_revision']
valid,missing,shape=inventory(s,rows,require_complete=True)
assert shape==(meta['expected_levels'],meta['expected_hidden_dim'])==(33,4096)
plans=[]
for path in (s.root/'plans').glob('*.json'):
    plan=read_json(path)
    keys=[k for assignment in plan['assignments'] for k in assignment]
    assert len(keys)==len(set(keys))
    plans.append(dict(kind=plan['kind'],workers=len(plan['assignments']),assigned=len(keys)))
for _,part,key in chunks(s,rows):
    item=s.manifest['artifacts'][key]
    receipt=read_json(s.root/'receipts'/(key+'.json'))
    assert receipt['status']=='complete' and receipt['files']==item['files']
    plan_id,slot=receipt['worker'].split('/')
    assert key in read_json(s.root/'plans'/(plan_id+'.json'))['assignments'][int(slot)]
runtimes=[read_json(f) for f in (s.root/'workers').rglob('extraction_runtime.json')]
assert runtimes and all(r['weights_loads_this_worker']==1 and r['expected_shape']==[33,4096] and 'L40S' in r['hardware'] for r in runtimes)
assert all(r['model_revision']==s.manifest['config']['model_revision'] and r['batch_size']==1 and r['use_cache'] is False for r in runtimes)
viewer=read_json(s.root/'viewer/index.json')
assert [r['point_id'] for r in viewer['rows']]==ids.tolist()
assert sorted(viewer['drawing_order'])==list(range(n))
colors=read_json(s.root/'display.json')['leading_digit_colors']
assert viewer['colors']==[colors[str(i)] for i in range(1,10)]
previous=None
metrics=[]
reference_layers=set(range(shape[0])) if n<100 else {1,8,16,24,27,28,32}
for layer in range(shape[0]):
    metric=read_json(s.root/f'analysis/layer_{layer:02d}.json')
    with np.load(s.root/f'analysis/layer_{layer:02d}.npz',allow_pickle=False) as z:
        data={k:z[k] for k in z.files}
    assert np.array_equal(data['point_ids'],ids)
    x=read_layer(s,layer,rows).astype(np.float64)
    assert x.shape==(n,4096) and np.isfinite(x).all()
    v=read_json(s.root/f'viewer/layer_{layer:02d}.json')
    assert v['status']==metric['status']
    if metric['status']=='degenerate':
        assert 'neighbors' not in data and 'scores' not in data
        assert metric['diagnostic']['rms_centered_norm']<=metric['diagnostic']['threshold']
        if layer==0:
            assert np.array_equal(x,np.broadcast_to(x[0],x.shape))
    else:
        nb,ds=data['neighbors'],data['distances']
        assert nb.shape==ds.shape==(n,4)
        assert metric['directed_relation_count']==n*4
        assert np.isin(nb,ids).all() and not np.any(nb==ids[:,None])
        assert np.all(np.diff(np.sort(nb,axis=1),axis=1)>0)
        assert np.isfinite(ds).all() and (ds>=0).all()
        directed={(int(i),int(j)) for i, ns in zip(ids,nb) for j in ns}
        union=sorted({tuple(sorted(e)) for e in directed})
        assert np.array_equal(data['union_edges'],union)
        assert data['low_to_high'].tolist()==[(u,w) in directed for u,w in union]
        assert data['high_to_low'].tolist()==[(w,u) in directed for u,w in union]
        assert np.array_equal(data['mutual'],data['low_to_high'] & data['high_to_low'])
        assert np.isclose(metric['mutual_neighbor_fraction'],2*data['mutual'].sum()/(n*4))
        ix=np.searchsorted(ids,nb)
        ld=np.array([r['leading_digit'] for r in rows])
        dl=np.array([r['digit_count'] for r in rows])
        same=ld[:,None]==ld[ix]
        cross=dl[:,None]!=dl[ix]
        assert np.isclose(metric['same_leading_digit_fraction'],same.mean())
        assert metric['conditional_denominator']==int(cross.sum())
        assert metric['same_leading_digit_cross_length_count']==int((same&cross).sum())
        if not cross.any():
            assert metric['same_leading_digit_given_cross_length'] is None
        else:
            assert np.isclose(metric['same_leading_digit_given_cross_length'],same[cross].mean())
        count=np.zeros((9,9),dtype=np.int64)
        np.add.at(count,(np.repeat(ld-1,4),ld[ix].ravel()-1),1)
        assert np.array_equal(count,data['digit_counts'])
        sums=count.sum(axis=1)
        np.testing.assert_allclose(data['digit_fractions'][sums>0],count[sums>0]/sums[sums>0,None])
        assert np.isnan(data['digit_fractions'][sums==0]).all()
        if previous is not None:
            overlaps=np.array([len(set(aa)&set(bb)) for aa,bb in zip(previous,nb)])
            assert np.array_equal(overlaps,data['previous_neighbor_overlap_per_point'])
        previous=nb
        sample=np.arange(n) if n<100 else np.unique(np.r_[0,1,8,9,98,99,998,999,9998,9999,np.random.default_rng(42).choice(n,8,replace=False)])
        np.testing.assert_allclose(data['mean'],x.mean(axis=0),rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose((x[sample]-data['mean'])@data['components'].T,data['scores'][sample],rtol=1e-8,atol=1e-8)
        total_variance=np.var(x,axis=0,ddof=1).sum()
        np.testing.assert_allclose(data['explained_variance']/total_variance,data['explained_variance_ratio'],rtol=1e-8,atol=1e-10)
        for component in [0,1]:
            rho=spearmanr(target,data['scores'][:,component]).statistic
            if np.isfinite(rho):
                assert rho>=-1e-12
                assert np.isclose(rho,metric['pca']['spearman_signed'][component])
            else:
                assert metric['pca']['spearman_signed'][component] is None
        np.testing.assert_array_equal(v['scores'],data['scores'])
        np.testing.assert_array_equal(v['neighbors'],nb)
        np.testing.assert_array_equal(v['distances'],ds)
        if layer in reference_layers:
            # Independent query audit over ALL candidates, direct differences and lexicographic ties.
            distance=cdist(x[sample],x,'euclidean')
            distance[np.arange(len(sample)),sample]=np.inf
            expected=np.stack([ids[np.lexsort((ids,row))[:4]] for row in distance])
            assert np.array_equal(nb[sample],expected),(layer,'reference neighbors')
            for q in sample:
                direct=np.linalg.norm(x[q]-x[np.searchsorted(ids,nb[q])],axis=1)
                np.testing.assert_allclose(ds[q],direct,rtol=1e-12,atol=1e-12)
    for mode in ['graph','digits','heatmap','magnitude']:
        with Image.open(s.root/f'figures/layer_{layer:02d}_{mode}.png') as im:
            im.verify()
    metrics.append(metric)
    print(f'AUDIT level={layer} status={metric["status"]} all_points={n}',flush=True)
for name in ['overview.png','depth.png']:
    with Image.open(s.root/'figures'/name) as im:
        im.verify()
assert len(list((s.root/'figures').glob('contact_*.png')))==6
baseline=read_json(s.root/'baselines.json')
counts=np.bincount([r['leading_digit'] for r in rows],minlength=10)[1:]
assert baseline['frequency']['counts']==counts.tolist()
expected_frequency=(counts-1)/(n-1)
np.testing.assert_allclose(baseline['frequency']['probability_same_digit_by_source'],expected_frequency)
assert np.isclose(baseline['frequency']['weighted_same_digit_probability'],np.dot(counts/n,expected_frequency))
with np.load(s.root/'baselines.npz',allow_pickle=False) as b:
    assert np.array_equal(b['point_ids'],ids)
    # In sorted scalar data the nearest four must lie within four positions.
    for q in range(n):
        candidates=np.array([j for j in range(max(0,q-4),min(n,q+5)) if j!=q])
        gaps=np.abs(target[candidates]-target[q])
        order=np.lexsort((ids[candidates],gaps))[:4]
        assert np.array_equal(b['neighbors'][q],ids[candidates[order]])
        np.testing.assert_array_equal(b['distances'][q],gaps[order])
assert baseline['numerical_distance']['directed_relation_count']==n*4
report=dict(passed=True,points=n,hidden_shape=[shape[0],n,shape[1]],chunks=len(valid),plans=plans,
    model_revision=meta['model_revision'],runtime_hardware=[r['hardware'] for r in runtimes],
    artifact_count=len(s.manifest['artifacts']),all_artifact_hashes_valid=True,
    degenerate_layers=[m['layer'] for m in metrics if m['status']=='degenerate'],
    valid_layers=sum(m['status']=='valid' for m in metrics),reference_neighbor_layers=sorted(reference_layers-{0}),
    all_layer_graph_structure_checked=True,all_layer_saved_projection_checks=True,
    metric_summary=[{k:m.get(k) for k in ['layer','status','same_leading_digit_fraction','same_digit_length_fraction','cross_digit_length_fraction','same_leading_digit_given_cross_length','conditional_denominator','knn_seconds','pca_seconds','process_peak_rss_bytes']} for m in metrics])
write_json(a.report,report)
print(json.dumps({k:v for k,v in report.items() if k!='metric_summary'},indent=2))
