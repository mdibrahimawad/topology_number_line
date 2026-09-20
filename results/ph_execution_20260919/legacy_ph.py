"""Offline, full-point Euclidean H0/H1. Never imports an inference backend.

Each layer owns a Store; concurrent layers never share a mutable manifest.
Sources are read-only. Checkpoints contain distances, not kNN/PCA geometry.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import time

import numpy as np
from scipy.spatial.distance import cdist

from numzig.fullrange.storage import Store, atomic, digest, fingerprint, read_json, write_json, write_npz

MODELS = {
    'crystal': ('crystal_fullrange_1_10000_ctx1234_k4_seed42', 33, 4096),
    'starcoderbase-3b': ('starcoderbase-3b_fullrange_1_10000_ctx1234_k4_seed42', 37, 2816),
    'openllama-3b': ('openllama-3b_fullrange_1_10000_ctx1234_k4_seed42', 27, 3200),
}
PILOT = {'crystal': [1, 7, 16, 32], 'starcoderbase-3b': [1, 18, 21, 36],
         'openllama-3b': [1, 8, 13, 26]}


def packages():
    return {p: importlib.metadata.version(p) for p in
            ('numpy', 'scipy', 'ripser', 'giotto-ph', 'matplotlib', 'scikit-learn', 'threadpoolctl')}


def verified_file(root, manifest, key, relative):
    item = manifest['artifacts'].get(key, {})
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Source artifact escapes its root')
    expected = item.get('files', {}).get(relative)
    if item.get('status') != 'complete' or expected is None or digest(path) != expected:
        raise ValueError(f'Missing/corrupt committed source: {key}: {relative}')
    return path


def source_info(root, layer, expected_points=10000):
    root = Path(root)
    m = read_json(root / 'manifest.json')
    if fingerprint(m['config']) != m['config_fingerprint']:
        raise ValueError('Invalid source configuration fingerprint')
    data = read_json(verified_file(root, m, 'dataset', 'dataset.json'))
    meta = read_json(verified_file(root, m, 'dataset', 'tokenizer_provenance.json'))
    rows = data['records']
    if (len(rows) != expected_points or [r['point_id'] for r in rows] != list(range(expected_points))
            or [r['target'] for r in rows] != list(range(1, expected_points + 1))):
        raise ValueError('Expected every canonical target exactly once, in order')
    if not 0 <= layer < meta['expected_levels']:
        raise ValueError('Layer outside saved levels')
    cache = f'cache/{layer:02d}'
    keys = [cache] if cache in m['artifacts'] else sorted(k for k in m['artifacts'] if k.startswith('chunk/'))
    if not keys or any(m['artifacts'][k]['status'] != 'complete' for k in keys):
        raise ValueError('No complete saved hidden source')
    identity = dict(experiment=m['config']['experiment'], config=m['config_fingerprint'],
        dataset=m['artifacts']['dataset']['files']['dataset.json'],
        raw_prompts=fingerprint([(r['target'],r['prompt']) for r in rows]),
        provenance=m['artifacts']['dataset']['files']['tokenizer_provenance.json'],
        input_artifacts={k: m['artifacts'][k]['files'] for k in keys}, layer=layer,
        shape=[expected_points, meta['expected_hidden_dim']])
    return m, rows, meta, identity


def read_vectors(root, layer, m, shape):
    root = Path(root)
    key = f'cache/{layer:02d}'
    if key in m['artifacts']:
        path = verified_file(root, m, key, f'analysis_cache/layer_{layer:02d}.npy')
        x = np.load(path, allow_pickle=False)
    else:
        x = np.empty(shape, dtype=np.float32)
        seen = np.zeros(shape[0], dtype=bool)
        for key in sorted(k for k in m['artifacts'] if k.startswith('chunk/')):
            stem = key.split('/')[1]
            ids = np.load(verified_file(root, m, key, f'hidden/{stem}_ids.npy'), allow_pickle=False)
            path = verified_file(root, m, key, f'hidden/{stem}.npy')
            a = np.load(path, mmap_mode='r', allow_pickle=False)
            try:
                if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or len(np.unique(ids)) != len(ids)
                        or np.any(ids < 0) or np.any(ids >= len(x)) or seen[ids].any()
                        or a.ndim != 3 or layer >= a.shape[0] or a.shape[1:] != (len(ids), shape[1])):
                    raise ValueError('Invalid hidden shard shape or overlapping IDs')
                x[ids] = a[layer]
                seen[ids] = True
            finally:
                a._mmap.close()
        if not seen.all():
            raise ValueError('Missing target vectors')
    if x.shape != tuple(shape) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError('Invalid saved vector shape/dtype/finiteness')
    return x


def dense_distances(store, x, block=256):
    """Resumable float64 direct-difference distances; no Gram cancellation."""
    n = len(x)
    dep = store.manifest['config_fingerprint']
    x64 = np.asarray(x, dtype=np.float64)
    paths = []
    for start in range(0, n, block):
        path = store.root / 'distances' / f'{start:05d}.npy'
        key = f'distance/{start:05d}'
        if not store.valid(key, dep):
            a = cdist(x64[start:start + block], x64, metric='euclidean')
            atomic(path, lambda f: np.save(f, a, allow_pickle=False))
            store.finish(key, [path], dep)
        paths.append(path)
    # Assemble in RAM. Saved blocks remain reusable after an interrupted PH call.
    d = np.empty((n, n), dtype=np.float64)
    for start, path in zip(range(0, n, block), paths):
        a = np.load(path, allow_pickle=False)
        if a.shape != (min(block, n-start), n) or not np.isfinite(a).all() or np.any(a < 0):
            raise ValueError('Invalid distance checkpoint')
        d[start:start + len(a)] = a
    if not np.array_equal(d, d.T) or np.any(np.diag(d) != 0):
        raise ValueError('Distance matrix must be symmetric with zero diagonal')
    return d


def mst_h0(d):
    """Dense Prim, O(N²) time/O(N) auxiliary space; preserves zero-length edges."""
    n = len(d)
    used = np.zeros(n, dtype=bool)
    best = np.full(n, np.inf)
    parent = np.full(n, -1, dtype=np.int32)
    best[0] = 0
    edges, weights = [], []
    for _ in range(n):
        v = int(np.argmin(np.where(used, np.inf, best)))
        if not np.isfinite(best[v]):
            raise ValueError('Nonfinite distance graph')
        used[v] = True
        if parent[v] >= 0:
            edges.append((parent[v], v)); weights.append(best[v])
        update = (~used) & (d[v] < best)
        best[update] = d[v, update]
        parent[update] = v
    h0 = np.column_stack([np.zeros(n), np.r_[np.sort(weights), np.inf]])
    return h0, np.asarray(edges, dtype=np.int32).reshape(-1, 2), np.asarray(weights)


def normalization(d, seed=42, pairs=100000):
    # Independent of model/layer: same sampled canonical index pairs for every layer.
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(d), size=pairs)
    j = rng.integers(0, len(d)-1, size=pairs)
    j += j >= i
    median = float(np.median(d[i, j]))
    scale = median
    method = 'median of 100000 seeded off-diagonal pair draws with replacement'
    if scale == 0:
        # StarCoder L0 contains many exact duplicates; never divide by zero or hide it.
        scale = float(d.max())
        method = 'diameter fallback: sampled median is zero'
    return dict(scale=scale, sampled_median=median, method=method, seed=seed, pairs=pairs,
                degenerate=scale == 0)


def h1(d, backend='gph', threads=4, threshold=None):
    if backend == 'gph':
        from gph import ripser_parallel
        result = ripser_parallel(d, metric='precomputed', maxdim=1, coeff=2,
                                thresh=np.inf if threshold is None else threshold,
                                n_threads=threads, collapse_edges=True)
    elif backend == 'ripser':
        from ripser import ripser
        result = ripser(d, distance_matrix=True, maxdim=1, coeff=2,
                        thresh=np.inf if threshold is None else threshold)
    else:
        raise ValueError('Unknown PH backend')
    a = np.asarray(result['dgms'][1], dtype=np.float64).reshape(-1, 2)
    if np.isnan(a).any() or np.any(a[:, 0] < 0) or np.any(a[:, 1] < a[:, 0]):
        raise ValueError('Invalid backend persistence diagram')
    if threshold is None and np.isinf(a).any():
        raise ValueError('Full finite metric Rips H1 cannot have essential bars')
    return a[np.lexsort((a[:, 1], a[:, 0]))]


def backend_gate(d):
    """Independent backends on a fixed subset: a gate, not proof for the full cloud."""
    from persim import bottleneck
    ids=np.sort(np.random.default_rng(42).choice(len(d),min(128,len(d)),replace=False))
    sub=d[np.ix_(ids,ids)]
    a=h1(sub,'gph',threads=1);b=h1(sub,'ripser')
    error=float(bottleneck(a,b))
    tolerance=float(4*np.spacing(np.float32(max(float(sub.max()),np.finfo(np.float32).tiny))))
    if error>tolerance:
        raise ValueError(f'Independent H1 backends disagree: {error} > {tolerance}')
    return dict(passed=True,point_ids=ids.tolist(),bottleneck=error,tolerance=tolerance,
                scope='seeded subset only; not full-candidate certification')


def compute(store, x, layer, backend='gph', threads=4, threshold=None, stage='all'):
    dep = store.manifest['config_fingerprint']
    started = time.perf_counter()
    if store.valid('h0', dep) and store.valid('h1', dep):
        if stage == 'all':
            plot_layer(store)
        return read_json(store.root / 'metrics.json')
    d = dense_distances(store, x)
    norm = normalization(d)
    if not store.valid('h0', dep):
        h0, edges, weights = mst_h0(d)
        path = store.root / 'h0.npz'
        write_npz(path, raw=h0, normalized=h0 / norm['scale'] if norm['scale'] else h0,
                  mst_edges=edges, mst_distances=weights, point_ids=np.arange(len(d)))
        write_json(store.root / 'normalization.json', norm)
        store.finish('h0', [path, store.root / 'normalization.json'], dep)
    if stage == 'h0':
        return dict(layer=layer, status='h0_complete', normalization=norm)
    ph_started = time.perf_counter()
    gate=backend_gate(d) if not norm['degenerate'] else dict(passed=True,scope='all vectors identical')
    # Both maintained backends use float32 filtration values internally. Measure
    # the maximum rounding error; do not describe this as float64-exact persistence.
    rounding = max(float(np.max(np.abs(d[s:s+256].astype(np.float32).astype(np.float64)-d[s:s+256])))
                   for s in range(0, len(d), 256))
    limit = None if threshold is None else threshold * norm['scale']
    diagram = np.empty((0, 2)) if norm['degenerate'] else h1(d, backend, threads, limit)
    normalized = diagram / norm['scale'] if norm['scale'] else diagram
    path = store.root / 'h1.npz'
    censored = np.isinf(diagram[:, 1])
    write_npz(path, raw=diagram, normalized=normalized, right_censored=censored)
    finite = normalized[~censored]
    lifetimes = finite[:, 1]-finite[:, 0]
    summary = dict(layer=layer, point_count=len(d), dimensions=x.shape[1], backend=backend,
        status='degenerate' if norm['degenerate'] else ('complete' if threshold is None else 'truncated'),
        max_normalized_distance=float(d.max()/norm['scale']) if norm['scale'] else 0.,
        normalized_threshold=threshold, normalization=norm, coefficient_field=2,
        h1_finite_count=len(finite), h1_right_censored_count=int(censored.sum()),
        h1_max_finite_lifetime=float(lifetimes.max()) if len(lifetimes) else 0.,
        distance_dtype='float64', filtration_dtype='float32', max_filtration_rounding_error=rounding,
        approximation='none; full points, finite precision; threshold truncation explicitly reported',
        ph_seconds=time.perf_counter()-ph_started, invocation_seconds=time.perf_counter()-started,
        backend_gate=gate,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if os.uname().sysname=='Darwin' else 1024),
        requested_ph_threads=threads, platform=os.uname().sysname, packages=packages())
    write_json(store.root / 'metrics.json', summary)
    store.finish('h1', [path, store.root / 'metrics.json'], dep)
    plot_layer(store)
    return summary


def run_layer(source, output, layer, backend='gph', threads=4, threshold=None,
              commit=lambda: None, stage='all', expected_points=10000):
    if threads < 1 or (threshold is not None and (not math.isfinite(threshold) or threshold <= 0)):
        raise ValueError('Invalid threads/threshold')
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('PH output and immutable source must be separate trees')
    m, rows, meta, identity = source_info(source, layer, expected_points)
    config = dict(schema=1, source=identity, backend=backend, threshold=threshold,
                  metric='euclidean', coeff=2, seed=42, block=256, packages=packages(),
                  code_sha256=digest(__file__), storage_sha256=digest(Path(__file__).parent/'fullrange/storage.py'))
    store = Store(output, config, commit)
    dep = store.manifest['config_fingerprint']
    if not store.valid('input', dep):
        x = read_vectors(source, layer, m, identity['shape'])
        p = output / 'vectors.npy'
        atomic(p, lambda f: np.save(f, x, allow_pickle=False))
        write_json(output / 'source.json', dict(identity=identity, metadata=meta,
            annotations=[{k: r.get(k) for k in ('point_id','target','leading_digit','digit_count','token_count','prompt','demonstrations')} for r in rows]))
        store.finish('input', [p, output / 'source.json'], dep)
    else:
        x = np.load(output / 'vectors.npy', allow_pickle=False)
    return compute(store, x, layer, backend, threads, threshold, stage)


def plot_layer(store):
    dep = store.artifact_fingerprint('h1')
    if store.valid('plots', dep):
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics = read_json(store.root / 'metrics.json')
    with np.load(store.root/'h0.npz') as f:
        h0 = f['normalized'].copy()
    with np.load(store.root/'h1.npz') as f:
        h1a = f['normalized'].copy()
    end = metrics['normalized_threshold'] or metrics['max_normalized_distance'] or 1.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for a, color, label in [(h0, '#3366aa','H0'),(h1a,'#cc6633','H1')]:
        finite = a[np.isfinite(a[:,1])]
        axes[0].scatter(finite[:,0], finite[:,1], s=7, alpha=.55, label=label, color=color)
        t = np.linspace(0, end, 256)
        counts = np.searchsorted(np.sort(a[:,0]),t,side='right')-np.searchsorted(np.sort(a[:,1]),t,side='right')
        axes[2].plot(t,counts,label=label,color=color)
    axes[0].plot([0,end],[0,end],color='gray',lw=.5)
    axes[0].set(xlabel='Birth / layer scale',ylabel='Death / layer scale',title='Finite persistence diagram')
    axes[0].legend()
    order = np.argsort(h1a[:,1]-h1a[:,0])[::-1][:30]
    for rank,i in enumerate(order):
        birth, death = h1a[i]
        axes[1].plot([birth,min(death,end)],[rank,rank],color='#cc6633')
        if np.isinf(death):
            axes[1].plot(end,rank,marker='>',color='red')
    axes[1].set(xlabel='Distance / layer scale',ylabel='Feature rank',title='Longest 30 H1 bars; arrows = censored')
    axes[2].set(xlabel='Distance / layer scale',ylabel='Features alive',yscale='symlog',title='Betti curves (symlog count)')
    axes[2].legend()
    experiment=store.manifest['config'].get('source',{}).get('experiment','synthetic')
    fig.suptitle(f"{experiment}\nL{metrics['layer']} · {metrics['point_count']} points · {metrics['status']} · {metrics['normalization']['method']}",fontsize=9)
    paths=[]
    for ext in ('png','svg'):
        p=store.root/f'persistence.{ext}'; fig.savefig(p,dpi=160); paths.append(p)
    plt.close(fig)
    store.finish('plots',paths,dep)


def aggregate(root):
    """Plot-only collection of completed layers; missing layers remain explicit."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from html import escape
    root=Path(root); rows=[]; stores=[]
    for p in sorted(root.glob('*/layer_*/manifest.json')):
        s=Store(p.parent)
        if not s.valid('h0') or not s.valid('h1'):
            continue
        plot_layer(s)
        metrics=read_json(s.root/'metrics.json')
        rows.append(dict(model=p.parent.parent.name,path=str(p.parent.relative_to(root)),**metrics)); stores.append(s)
    if not rows:
        raise ValueError('No completed PH layers to collect')
    contracts={(r['backend'],r['normalized_threshold'],json.dumps(r['packages'],sort_keys=True),
                s.manifest['config'].get('code_sha256')) for r,s in zip(rows,stores)}
    if len(contracts)!=1:
        raise ValueError('Refuse mixed backend/threshold/package comparisons')
    end=max(r['normalized_threshold'] or r['max_normalized_distance'] for r in rows) or 1.
    t=np.linspace(0,end,256)
    fig,axes=plt.subplots(3,2,figsize=(13,12),constrained_layout=True)
    for mi,(model,(_,levels,_)) in enumerate(MODELS.items()):
        selected=[(r,s) for r,s in zip(rows,stores) if r['model']==model]
        for dim in (0,1):
            a=np.full((levels,len(t)),np.nan)
            for r,s in selected:
                with np.load(s.root/f'h{dim}.npz') as f: d=f['normalized']
                counts=np.searchsorted(np.sort(d[:,0]),t,side='right')-np.searchsorted(np.sort(d[:,1]),t,side='right')
                a[r['layer']]=counts
                if r['normalized_threshold'] is not None: a[r['layer'],t>r['normalized_threshold']]=np.nan
            ceiling=max(r['point_count'] if dim==0 else max(1,r['h1_finite_count']+r['h1_right_censored_count']) for r in rows)
            im=axes[mi,dim].imshow(np.log1p(a),origin='lower',aspect='auto',extent=[0,end,-.5,levels-.5],interpolation='nearest',vmin=0,vmax=np.log1p(ceiling))
            axes[mi,dim].set(title=f'{model}: H{dim} (blank = unavailable)',xlabel='Distance / layer scale',ylabel='Saved layer')
            fig.colorbar(im,ax=axes[mi,dim],label='log(1 + features alive)')
    fig.savefig(root/'topology_heatmaps.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for model in MODELS:
        rr=sorted((r for r in rows if r['model']==model),key=lambda r:r['layer'])
        axes[0].plot([r['layer'] for r in rr],[r['h1_max_finite_lifetime'] for r in rr],'.-',label=model)
        axes[1].plot([r['layer'] for r in rr],[r['normalization']['scale'] for r in rr],'.-',label=model)
    axes[0].set(xlabel='Saved layer',ylabel='Normalized lifetime',title='Longest completed H1 feature')
    axes[1].set(xlabel='Saved layer',ylabel='Raw distance',title='Layer normalization scale')
    for ax in axes:ax.legend(fontsize=8)
    fig.savefig(root/'depth_trends.png',dpi=160);plt.close(fig)
    # Bound comparison cost, retaining the complete diagrams on disk. Removing
    # a bar costs half its lifetime in bottleneck distance; triangle inequality
    # supplies rigorous bounds relative to the saved complete H1 diagrams.
    from persim import bottleneck
    comparable=[(r,s) for r,s in zip(rows,stores) if r['status']=='complete' and r['layer']>0]
    diagrams=[]; errors=[]; labels=[]
    for r,s in comparable:
        with np.load(s.root/'h1.npz') as f: a=f['normalized'].copy()
        keep,error=diagram_sketch(a,32)
        diagrams.append(keep);errors.append(error);labels.append(f"{r['model']}/L{r['layer']}")
    distances=np.zeros((len(diagrams),len(diagrams)))
    for i in range(len(diagrams)):
        for j in range(i):
            distances[i,j]=distances[j,i]=float(bottleneck(diagrams[i],diagrams[j]))
    error=np.add.outer(errors,errors)
    lower=np.maximum(0,distances-error);upper=distances+error
    np.fill_diagonal(upper,0)
    write_json(root/'h1_comparison.json',dict(labels=labels,sketch_bottleneck=distances.tolist(),
        full_diagram_lower_bound=lower.tolist(),full_diagram_upper_bound=upper.tolist(),
        discarded_bar_error=errors,max_bars=32,
        note='Comparison uses longest 32 bars only; bounds apply to saved full H1 diagrams. All original bars remain saved. Excludes L0, degenerate and right-truncated levels. A zero distance is not proof of identical numerical organization.'))
    if len(diagrams):
        fig,ax=plt.subplots(figsize=(10,9),constrained_layout=True)
        im=ax.imshow(distances,interpolation='nearest');fig.colorbar(im,ax=ax,label='Bottleneck distance between diagram sketches')
        ticks=np.arange(0,len(labels),max(1,len(labels)//16))
        ax.set_xticks(ticks,[labels[i] for i in ticks],rotation=90,fontsize=7)
        ax.set_yticks(ticks,[labels[i] for i in ticks],fontsize=7)
        ax.set_title('H1 comparison: longest 32 bars; full-diagram bounds in JSON')
        fig.savefig(root/'h1_comparison.png',dpi=160);plt.close(fig)
    missing={model:sorted(set(range(v[1]))-{r['layer'] for r in rows if r['model']==model}) for model,v in MODELS.items()}
    write_json(root/'summary.json',dict(layers=rows,missing_layers=missing,
        note='Per-layer scale persistence, not tracking identical features across layers. No inference or causal conclusions.'))
    links=''.join(f'<li>{escape(r["model"])} L{r["layer"]}: <a href="{r["path"]}/persistence.png">diagram / barcode / Betti curves</a> ({r["status"]})</li>' for r in rows)
    comparison='<img src="h1_comparison.png"><p><a href="h1_comparison.json">Comparison bounds and settings</a></p>' if diagrams else ''
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Layer PH</title><style>body{font:17px system-ui;max-width:1200px;margin:40px auto}img{max-width:100%}</style><h1>Per-layer persistent homology</h1><p>Original hidden-vector Euclidean distances. Blank layers are not completed. L0 is an embedding control. Infinite H0 is essential; truncated H1 bars are censored, not essential.</p><img src="topology_heatmaps.png"><img src="depth_trends.png">'+comparison+'<ul>'+links+'</ul>')
    return dict(completed=len(rows),missing=missing)


def diagram_sketch(a, limit):
    a=np.asarray(a,dtype=float).reshape(-1,2)
    if limit<1 or not np.isfinite(a).all() or np.any(a[:,1]<a[:,0]):
        raise ValueError('Sketch requires finite valid bars and positive limit')
    life=a[:,1]-a[:,0]
    order=np.argsort(-life,kind='stable')
    omitted=order[limit:]
    return a[order[:limit]],float(life[omitted].max()/2) if len(omitted) else 0.


def reserve(ledger, stage, job, cores, gib, seconds):
    """Conservative compute allowance, not an account billing cap.

    No refund: timeout/failure/relaunch reservations remain spent in this ledger.
    Rates checked 2026-09-19, standard Modal Functions; no region premium.
    """
    if stage not in ('pilot','full') or min(cores,gib,seconds)<=0 or not all(map(math.isfinite,(cores,gib,seconds))):
        raise ValueError('Invalid budget reservation')
    limit={'pilot':3.,'full':19.}[stage]
    cost=1.25*(seconds+240)*(cores*.0000131+gib*.00000222)
    spent=sum(r['reserved_usd'] for r in ledger if r['stage']==stage)
    if spent+cost>limit:
        raise RuntimeError(f'{stage} reservation allowance exhausted; completed checkpoints remain saved')
    item=dict(stage=stage,job=job,reserved_usd=cost,started=time.time(),status='reserved')
    ledger.append(item)
    return item


def local_roots(repo):
    r=Path(repo)/'results'
    return {'crystal':r/'crystal_fullrange_modal_20260919T123307Z/full_complete'/MODELS['crystal'][0],
            **{k:r/'native_models_modal_20260919/full_complete'/v[0] for k,v in MODELS.items() if k!='crystal'}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['inventory','layer','h0','collect'])
    p.add_argument('--source',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--layer',type=int);p.add_argument('--backend',choices=['gph','ripser'],default='gph')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--threshold',type=float,help='Optional normalized distance cutoff; deaths beyond it are censored')
    a=p.parse_args()
    if a.stage=='inventory':
        report={}
        for k,root in local_roots(Path(__file__).resolve().parents[1]).items():
            _,rows,meta,identity=source_info(root,0)
            report[k]=dict(source=str(root),points=len(rows),levels=meta['expected_levels'],dimensions=meta['expected_hidden_dim'],dataset=identity['dataset'],raw_prompts=identity['raw_prompts'],pilot_layers=PILOT[k])
        if len({v['raw_prompts'] for v in report.values()})!=1:
            raise ValueError('Models do not contain identical target/prompt pairs')
        print(json.dumps(report,indent=2));return
    if a.output is None:p.error('--output required')
    if a.stage=='collect':
        print(json.dumps(aggregate(a.output)));return
    if a.source is None or a.layer is None:p.error('--source and --layer required')
    print(json.dumps(run_layer(a.source,a.output,a.layer,a.backend,a.threads,a.threshold,stage='h0' if a.stage=='h0' else 'all'),indent=2))


if __name__=='__main__':
    main()
