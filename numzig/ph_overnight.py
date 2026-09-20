"""Full-point PH stages; immutable sources and one manifest per layer."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json,os,resource,shutil,sys,time
import numpy as np
from scipy.spatial.distance import cdist
from numzig import ph
from numzig.ph_acceleration import gpu_h1,compare,controls,REVISION
from numzig.fullrange.storage import Store,atomic,digest,read_json,write_json,write_npz

BLOCK=256

def reserve_budget(ledger,label,seconds,cores,gib,gpus=0,limit=20.):
    if min(seconds,cores,gib)<=0 or gpus not in (0,1):raise ValueError('Invalid reservation')
    rate=cores*.0000131+gib*.00000222+gpus*.000542
    cost=1.25*(seconds+120)*rate
    if sum(r['charged_usd'] for r in ledger)+cost>limit:raise RuntimeError('Compute allowance exhausted')
    row=dict(job=label,reserved_usd=cost,charged_usd=cost,rate=rate,seconds_limit=seconds,status='reserved',started=time.time())
    ledger.append(row);return len(ledger)-1

def settle_budget(ledger,index,elapsed):
    row=ledger[index]
    if not np.isfinite(elapsed) or elapsed<0 or elapsed>row['seconds_limit']+1:raise ValueError('Invalid measured duration')
    if row['status']!='reserved':raise ValueError('Reservation already settled')
    row.update(status='settled',measured_seconds=elapsed,
               charged_usd=min(row['reserved_usd'],1.25*(elapsed+120)*row['rate']))

def contract(identity,role):
    return dict(schema=1,role=role,source=identity,code_sha256=digest(__file__),
                ph_sha256=digest(ph.__file__),gpu_adapter_sha256=digest(Path(__file__).with_name('ph_acceleration.py')),
                packages=ph.packages(),backend='ripser++',revision=REVISION,metric='euclidean',coeff=2,threshold=None)

def crystal_cache(artifacts,root,commit=lambda:None):
    """Read/hash each original shard once, then save all layer caches together."""
    source=Path(artifacts)/ph.MODELS['crystal'][0]
    m,_,meta,identity=ph.source_info(source,0)
    s=Store(Path(root)/'crystal_cache',contract(identity,'source_cache'),commit);dep=s.manifest['config_fingerprint']
    levels,dim=meta['expected_levels'],meta['expected_hidden_dim'];n=identity['shape'][0]
    if s.valid('all',dep):return str(s.root)
    paths=[s.root/f'layer_{l:02d}.npy' for l in range(levels)]
    arrays=[np.lib.format.open_memmap(str(p)+'.partial',mode='w+',dtype=np.float32,shape=(n,dim)) for p in paths]
    seen=np.zeros(n,dtype=bool)
    try:
        for key in sorted(k for k in m['artifacts'] if k.startswith('chunk/')):
            stem=key.split('/')[1]
            ids=np.load(ph.verified_file(source,m,key,f'hidden/{stem}_ids.npy'))
            a=np.load(ph.verified_file(source,m,key,f'hidden/{stem}.npy'),mmap_mode='r')
            if (ids.ndim!=1 or ids.dtype.kind not in 'iu' or len(np.unique(ids))!=len(ids)
                    or np.any(ids<0) or np.any(ids>=n) or seen[ids].any() or a.shape!=(levels,len(ids),dim)
                    or a.dtype!=np.float32 or not np.isfinite(a).all()):raise ValueError('Invalid Crystal shard')
            for l,x in enumerate(arrays):x[ids]=a[l]
            seen[ids]=True;a._mmap.close()
        if not seen.all():raise ValueError('Incomplete Crystal cache')
        for p,a in zip(paths,arrays):a.flush();a._mmap.close();os.replace(str(p)+'.partial',p)
    finally:
        for a in arrays:
            if not a._mmap.closed:a._mmap.close()
    s.finish('all',paths,dep)
    return str(s.root)

def parallel_distances(store,x,threads=4,block=BLOCK):
    dep=store.manifest['config_fingerprint'];n=len(x);x64=np.asarray(x,dtype=np.float64)
    todo=[i for i in range(0,n,block) if not store.valid(f'distance/{i:05d}',dep)]
    # Submit one bounded wave at a time: at most threads distance blocks in RAM.
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for pos in range(0,len(todo),threads):
            batch=todo[pos:pos+threads]
            futures=[pool.submit(cdist,x64[i:i+block],x64,metric='euclidean') for i in batch]
            for i,f in zip(batch,futures):
                a=f.result();p=store.root/f'distances/{i:05d}.npy'
                atomic(p,lambda file,a=a:np.save(file,a,allow_pickle=False))
                store.finish(f'distance/{i:05d}',[p],dep)
    return load_distances(store,n,block)

def load_distances(store,n,block=BLOCK):
    d=np.empty((n,n),dtype=np.float64)
    for i in range(0,n,block):
        p=ph.verified_file(store.root,store.manifest,f'distance/{i:05d}',f'distances/{i:05d}.npy')
        a=np.load(p,allow_pickle=False)
        if a.shape!=(min(block,n-i),n) or a.dtype!=np.float64 or not np.isfinite(a).all() or np.any(a<0):
            raise ValueError('Invalid distance block')
        d[i:i+len(a)]=a
    if not np.array_equal(d,d.T) or np.any(np.diag(d)!=0):raise ValueError('Invalid full distance matrix')
    return d

def legacy_root(artifacts,model,layer):
    p=Path(artifacts)/'ph_fullrange_v1'/model/f'layer_{layer:02d}'
    if not (p/'manifest.json').exists():return None
    m=read_json(p/'manifest.json')
    if all(m['artifacts'].get(f'distance/{i:05d}',{}).get('status')=='complete' for i in range(0,10000,BLOCK)):
        return p
    return None

def prepare(source,output,layer,commit=lambda:None,cache=None,points=10000):
    source,output=Path(source),Path(output)
    if output.is_relative_to(source) or source.is_relative_to(output):raise ValueError('Source and output overlap')
    m,_,_,identity=ph.source_info(source,layer,points)
    s=Store(output,contract(identity,'preparation'),commit);dep=s.manifest['config_fingerprint']
    if s.valid('h0',dep) and s.valid('geometry',dep):
        geometry=read_json(output/'geometry.json')
        if geometry['implicit_zero_distances'] or all(s.valid(f'distance/{i:05d}',dep) for i in range(0,points,BLOCK)):
            return geometry
    if cache is None:x=ph.read_vectors(source,layer,m,identity['shape'])
    else:
        cache=Path(cache);cm=read_json(cache/'manifest.json');ci=cm['config']['source']
        expected={**identity,'layer':0}
        if cm['config']!=contract(expected,'source_cache'):raise ValueError('Crystal cache provenance changed')
        x=np.load(ph.verified_file(cache,cm,'all',f'layer_{layer:02d}.npy'),allow_pickle=False)
        if x.shape!=tuple(identity['shape']) or x.dtype!=np.float32 or not np.isfinite(x).all():raise ValueError('Invalid cache vectors')
    identical=bool(np.all(x==x[0]))
    if identical:
        norm=dict(scale=0.,sampled_median=0.,method='all vectors exactly identical',seed=42,pairs=100000,degenerate=True)
        h0=np.column_stack([np.zeros(points),np.r_[np.zeros(points-1),np.inf]])
        edges=np.column_stack([np.zeros(points-1,dtype=np.int32),np.arange(1,points,dtype=np.int32)])
        weights=np.zeros(points-1);maximum=rounding=0.
    else:
        d=parallel_distances(s,x);norm=ph.normalization(d);h0,edges,weights=ph.mst_h0(d)
        maximum=float(d.max())
        rounding=max(float(np.max(np.abs(d[i:i+BLOCK]-d[i:i+BLOCK].astype(np.float32).astype(np.float64)))) for i in range(0,len(d),BLOCK))
    write_npz(output/'h0.npz',raw=h0,normalized=h0/norm['scale'] if norm['scale'] else h0,
              mst_edges=edges,mst_distances=weights,point_ids=np.arange(points))
    write_json(output/'normalization.json',norm)
    s.finish('h0',[output/'h0.npz',output/'normalization.json'],dep)
    geometry=dict(normalization=norm,max_distance=maximum,rounding_error=rounding,point_count=points,
                  dimensions=identity['shape'][1],implicit_zero_distances=identical)
    write_json(output/'geometry.json',geometry);s.finish('geometry',[output/'geometry.json'],dep)
    return geometry

def layer(artifacts,root,model,level,commit=lambda:None):
    artifacts,root=Path(artifacts),Path(root);source=artifacts/ph.MODELS[model][0]
    _,_,meta,identity=ph.source_info(source,level)
    output=root/'results'/model/f'layer_{level:02d}'
    s=Store(output,contract(identity,'ph'),commit);dep=s.manifest['config_fingerprint']
    if s.valid('h1',dep) and s.valid('h0',dep):ph.plot_layer(s);return read_json(output/'metrics.json')
    started=time.perf_counter();legacy=legacy_root(artifacts,model,level)
    if legacy:
        x,d=ph.inherited_distances(legacy,identity);del x
        prep=Store(legacy)
        norm=ph.normalization(d)
        geometry=dict(normalization=norm,max_distance=float(d.max()),point_count=10000,dimensions=identity['shape'][1],
          rounding_error=max(float(np.max(np.abs(d[i:i+BLOCK]-d[i:i+BLOCK].astype(np.float32).astype(np.float64)))) for i in range(0,len(d),BLOCK)))
    else:
        prep=Store(root/'prepared'/model/f'layer_{level:02d}')
        if prep.manifest['config']!=contract(identity,'preparation'):raise ValueError('Preparation provenance differs')
        if not prep.valid('geometry') or not prep.valid('h0'):raise ValueError('Missing validated preprocessing')
        geometry=read_json(prep.root/'geometry.json');norm=geometry['normalization']
        d=None if geometry['implicit_zero_distances'] else load_distances(prep,10000)
    for name in ['h0.npz','normalization.json']:
        p=ph.verified_file(prep.root,prep.manifest,'h0',name)
        atomic(output/name,lambda f,p=p:f.write(p.read_bytes()))
    s.finish('h0',[output/'h0.npz',output/'normalization.json'],dep)
    write_json(output/'source_metadata.json',dict(identity=identity,metadata=meta,preparation=str(prep.root),preparation_manifest=digest(prep.path)))
    if norm['degenerate']:
        diagram=np.empty((0,2));gate=dict(passed=True,scope='all vectors identical');ph_seconds=0.;reference=None
    else:
        control=controls()
        duplicate=np.repeat(np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]]),2,axis=0);duplicate=cdist(duplicate,duplicate)
        compare(gpu_h1(duplicate),ph.h1(duplicate,'ripser'),float(duplicate.max()))
        ids=np.sort(np.random.default_rng(42).choice(10000,128,replace=False));sub=d[np.ix_(ids,ids)]
        gate=dict(**compare(gpu_h1(sub),ph.h1(sub,'ripser'),float(sub.max())),scope='seeded subset',point_ids=ids.tolist())
        write_json(output/'gates.json',dict(synthetic=control,real=gate));s.finish('gates',[output/'gates.json'],dep)
        begin=time.perf_counter();diagram=gpu_h1(d);ph_seconds=time.perf_counter()-begin
        reference=None
        if level==1:
            old=artifacts/'ph_fullrange_v1/direct_rips'/model/'layer_01';m=read_json(old/'manifest.json')
            if m['config']['source']!=identity:raise ValueError('CPU reference source differs')
            with np.load(ph.verified_file(old,m,'h1','h1.npz')) as a:reference=compare(diagram,a['raw'],geometry['max_distance'])
        elif model=='crystal' and level==7:
            old=artifacts/'ph_fullrange_v1/acceleration_v1/1789846256274332641/layer_07';m=read_json(old/'manifest.json')
            if m['config']['source']!=identity or m['config']['revision']!=REVISION:raise ValueError('GPU reference source differs')
            with np.load(ph.verified_file(old,m,'h1','h1.npz')) as a:reference=compare(diagram,a['raw'],geometry['max_distance'])
    normalized=diagram/norm['scale'] if norm['scale'] else diagram
    write_npz(output/'h1.npz',raw=diagram,normalized=normalized,right_censored=np.zeros(len(diagram),dtype=bool))
    metrics=dict(model=model,layer=level,point_count=10000,dimensions=identity['shape'][1],backend='ripser++',revision=REVISION,
        status='degenerate' if norm['degenerate'] else 'complete',normalization=norm,coefficient_field=2,
        normalized_threshold=None,max_normalized_distance=geometry['max_distance']/norm['scale'] if norm['scale'] else 0.,
        h1_finite_count=len(diagram),h1_right_censored_count=0,
        h1_max_finite_lifetime=float(np.max(np.diff(normalized,axis=1),initial=0)),
        distance_dtype='float64',filtration_dtype='float32',max_filtration_rounding_error=geometry['rounding_error'],
        approximation='none: full 10000 points; finite-precision full Rips H1',
        ph_seconds=ph_seconds,invocation_seconds=time.perf_counter()-started,
        backend_gate=gate,full_reference_gate=reference,packages=ph.packages(),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    write_json(output/'metrics.json',metrics)
    s.finish('h1',[output/'h1.npz',output/'metrics.json',output/'source_metadata.json'],dep)
    ph.plot_layer(s)
    return metrics

def collect(root):
    """Make a lightweight archive and graph gallery, without interpretation."""
    import tarfile
    from html import escape
    root=Path(root);results=root/'results';rows=[]
    for p in sorted(results.glob('*/layer_*/manifest.json')):
        s=Store(p.parent)
        if all(s.valid(k) for k in ['h0','h1','plots']):
            rows.append(dict(model=p.parent.parent.name,path=str(p.parent.relative_to(results)),**read_json(p.parent/'metrics.json')))
    missing={m:sorted(set(range(v[1]))-{r['layer'] for r in rows if r['model']==m}) for m,v in ph.MODELS.items()}
    write_json(results/'summary.json',dict(completed=len(rows),expected=97,missing=missing,layers=rows))
    links=''.join(f'<li>{escape(r["model"])} L{r["layer"]}: <a href="{r["path"]}/persistence.png">PH graphs</a> · <a href="{r["path"]}/persistence.svg">SVG</a></li>' for r in rows)
    (results/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>All-layer PH</title><style>body{font:18px system-ui;max-width:1100px;margin:40px auto;line-height:1.5}</style><h1>All-layer persistent homology</h1><p>'+str(len(rows))+'/97 levels complete. Each uses all 10,000 targets. H0: components; H1: loops. Original-vector Euclidean distances; full-range Rips H1 over F2, float32 filtration. Layer 0 is the saved embedding level.</p><p><a href="summary.json">Saved metrics and missing levels</a></p><ul>'+links+'</ul>')
    with tarfile.open(root/'lightweight.tar.gz.partial','w:gz') as tar:tar.add(results,arcname='results')
    os.replace(root/'lightweight.tar.gz.partial',root/'lightweight.tar.gz')
    write_json(root/'package.json',dict(sha256=digest(root/'lightweight.tar.gz'),completed=len(rows),missing=missing))
    return dict(completed=len(rows),missing=missing)

if __name__=='__main__':
    import modal
    v=modal.Volume.from_name('numberline-zigzag-results',create_if_missing=False)
    stage,artifacts,root=sys.argv[1:4]
    if stage=='cache':crystal_cache(artifacts,root,v.commit)
    elif stage=='prepare':
        model,level=sys.argv[4],int(sys.argv[5]);source=Path(artifacts)/ph.MODELS[model][0]
        prepare(source,Path(root)/'prepared'/model/f'layer_{level:02d}',level,v.commit,
                cache=Path(root)/'crystal_cache' if model=='crystal' else None)
    elif stage=='gpu':layer(artifacts,root,sys.argv[4],int(sys.argv[5]),v.commit)
    elif stage=='collect':collect(root);v.commit()
    else:raise ValueError('Unknown stage')
