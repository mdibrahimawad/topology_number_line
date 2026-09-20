"""Existing-diagram comparisons only; no PH reruns, inference, or cloud calls."""
from pathlib import Path
import csv, json, time, itertools, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from persim import bottleneck

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/'outputs/ph_fullrange_overnight/results/results'
OUT = ROOT/'outputs/ph_analysis/comparisons'
OUT.mkdir(parents=True, exist_ok=True)
MODELS = {'crystal':32, 'starcoderbase-3b':36, 'openllama-3b':26}
NAMES = {'crystal':'Crystal', 'starcoderbase-3b':'StarCoder', 'openllama-3b':'OpenLLaMA'}
COLORS = {'crystal':'#3974b8','starcoderbase-3b':'#cf6d33','openllama-3b':'#36906a'}


def events(d):
    x = np.r_[d[:,0],d[:,1]]
    w = np.r_[np.ones(len(d),dtype=np.int32),-np.ones(len(d),dtype=np.int32)]
    order = np.argsort(x,kind='stable')
    return x[order],w[order]


def curve_l1(a,b,amplitude=10000):
    x = np.r_[a[0],b[0]]
    w = np.r_[a[1],-b[1]]
    ix = np.argsort(x,kind='stable')
    x,w = x[ix],w[ix]
    return float(np.dot(np.diff(x),np.abs(np.cumsum(w)[:-1]))/amplitude)


def betti_at(d,t):
    return (np.searchsorted(np.sort(d[:,0]),t,side='right')-np.searchsorted(np.sort(d[:,1]),t,side='right'))/10000


def landscape(d,t,k=5):
    # All bars compete for the top five landscape levels at every grid point.
    result = np.zeros((k,len(t)))
    if not len(d):return result
    for start in range(0,len(t),32):
        z = np.maximum(0,np.minimum(t[None,start:start+32]-d[:,0,None],d[:,1,None]-t[None,start:start+32]))
        v = np.sort(np.partition(z,-min(k,len(d)),axis=0)[-k:],axis=0)[::-1]
        result[:len(v),start:start+32]=v
    return result


def witness_lower(retained,full):
    # Any complete matching must give every retained bar a partner, possibly diagonal.
    lower=0.
    for start in range(0,len(retained),32):
        a=retained[start:start+32]
        cost=np.maximum(np.abs(a[:,None,0]-full[None,:,0]),np.abs(a[:,None,1]-full[None,:,1]))
        best=np.minimum((a[:,1]-a[:,0])/2,cost.min(axis=1))
        lower=max(lower,float(best.max()))
    return lower


def write_csv(name,rows):
    with (OUT/name).open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def selfcheck():
    a=np.array([[0.,2.],[1.,3.]])
    b=np.array([[0.,1.]])
    assert curve_l1(events(a),events(b),1)==3
    assert curve_l1(events(a),events(a),1)==0
    assert np.allclose(landscape(a,np.array([0.,1.,2.,3.]),2),[[0,1,1,0],[0,0,0,0]])
    assert np.allclose(betti_at(a,np.array([0.,1.,2.,3.]))*10000,[1,2,1,0])
    # Equal-event boundaries contribute zero width; half-open bars are used.
    assert curve_l1(events(np.array([[0.,1.],[1.,2.]])),events(np.array([[0.,2.]])),1)==0
    assert witness_lower(np.array([[0.,2.]]),np.array([[3.,4.]]))==1.
    assert witness_lower(np.array([[0.,2.]]),np.array([[.1,2.1]]))<.100000001


def main():
    selfcheck();started=time.monotonic();rows=[];inputs=[]
    for model,last in MODELS.items():
        for layer in range(1,last+1):
            p=SRC/model/f'layer_{layer:02d}'
            h0=np.load(p/'h0.npz')['normalized'];h1=np.load(p/'h1.npz')['normalized']
            assert h0.shape==(10000,2) and np.isinf(h0[:,1]).sum()==1
            h0=h0[np.isfinite(h0[:,1])]
            assert np.isfinite(h1).all() and (h1[:,1]>=h1[:,0]).all()
            rows.append(dict(model=model,layer=layer,depth=layer/last,label=f'{model}:L{layer:02d}',h0=h0,h1=h1,events=events(h1)))
            for f in ['h0.npz','h1.npz']:
                path=p/f;inputs.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    labels=[r['label'] for r in rows];n=len(rows);assert n==94
    h0mat=np.zeros((n,n));h1mat=np.zeros((n,n))
    for i,a in enumerate(rows):
        for j in range(i):
            b=rows[j]
            h0mat[i,j]=h0mat[j,i]=np.abs(np.sort(a['h0'][:,1])-np.sort(b['h0'][:,1])).sum()/10000
            h1mat[i,j]=h1mat[j,i]=curve_l1(a['events'],b['events'])
        if i%20==0:print('Exact Betti comparisons',i,n,flush=True)
    tmax=max(float(r['h1'][:,1].max()) if len(r['h1']) else 0 for r in rows)
    grid=np.linspace(0,tmax,1025)
    landscapes=[]
    for i,r in enumerate(rows):
        landscapes.append(landscape(r['h1'],grid))
        if i%20==0:print('Landscapes',i,n,flush=True)
    landscapes=np.asarray(landscapes)
    landmat=np.zeros((n,n))
    for i in range(n):
        for j in range(i):
            landmat[i,j]=landmat[j,i]=float(np.trapezoid(np.abs(landscapes[i]-landscapes[j]),grid,axis=1).mean())
    np.savez_compressed(OUT/'matrices.npz',labels=labels,h0_betti_l1=h0mat,h1_betti_l1=h1mat,landscape_top5_grid_l1=landmat,landscape_grid=grid,landscapes=landscapes)
    matrices={'h0':h0mat,'h1':h1mat,'landscape5':landmat}
    pairs=[]
    for i,a in enumerate(rows):
        for j in range(i):
            b=rows[j];pairs.append(dict(a=a['label'],b=b['label'],same_model=a['model']==b['model'],depth_difference=abs(a['depth']-b['depth']),**{k:float(v[i,j]) for k,v in matrices.items()}))
    write_csv('pairs.csv',pairs)
    idx={(r['model'],r['layer']):i for i,r in enumerate(rows)}
    selections={'best_rho':dict(zip(MODELS,[7,21,8])),'best_ev2':dict.fromkeys(MODELS,1),'final':dict(MODELS)}
    for depth in [.10,.25,.50,.75,.90]:
        selections[f'depth_{int(depth*100)}']={m:min(range(1,last+1),key=lambda l:(abs(l/last-depth),l)) for m,last in MODELS.items()}
    selected=[]
    for name,selection in selections.items():
        for a,b in itertools.combinations(MODELS,2):
            i,j=idx[a,selection[a]],idx[b,selection[b]]
            row={'selection':name,'a':rows[i]['label'],'b':rows[j]['label']}
            for k,v in matrices.items():
                cross=[v[x,y] for x in range(n) for y in range(n) if rows[x]['model']==a and rows[y]['model']==b]
                row[k]=float(v[i,j]);row[k+'_cross_pair_percentile']=float(100*np.mean(np.array(cross)<=v[i,j]))
            selected.append(row)
    write_csv('selected_comparisons.csv',selected)
    adjacent=[]
    for m,last in MODELS.items():
        for l in range(2,last+1):
            i,j=idx[m,l-1],idx[m,l]
            adjacent.append(dict(model=m,from_layer=l-1,to_layer=l,depth=l/last,**{k:float(v[i,j]) for k,v in matrices.items()}))
    write_csv('adjacent_changes.csv',adjacent)
    # Whole-window comparisons use all unique within-model or all cross-model pairs.
    windows=[]
    for window in ['early','middle','late','final3','final5']:
        for a,b in itertools.combinations_with_replacement(MODELS,2):
            def keep(r,m):
                if r['model']!=m:return False
                if window=='early':return r['depth']<=1/3
                if window=='middle':return 1/3<r['depth']<=2/3
                if window=='late':return r['depth']>2/3
                return r['layer']>MODELS[m]-int(window[-1])
            ia=[i for i,r in enumerate(rows) if keep(r,a)];ib=[i for i,r in enumerate(rows) if keep(r,b)]
            ij=[(i,j) for i in ia for j in ib if a!=b or i<j]
            windows.append(dict(window=window,a=a,b=b,pairs=len(ij),**{k+'_median':float(np.median([v[i,j] for i,j in ij])) for k,v in matrices.items()}))
    write_csv('window_comparisons.csv',windows)
    nearest=[]
    for i,a in enumerate(rows):
        for metric,mat in matrices.items():
            for group in ['other_model','same_model_nonadjacent']:
                candidates=[j for j,b in enumerate(rows) if (b['model']!=a['model'] if group=='other_model' else b['model']==a['model'] and abs(b['layer']-a['layer'])>1)]
                j=min(candidates,key=lambda j:mat[i,j]);nearest.append(dict(layer=a['label'],metric=metric,group=group,nearest=rows[j]['label'],distance=float(mat[i,j]),depth_difference=abs(a['depth']-rows[j]['depth'])))
    write_csv('nearest_layers.csv',nearest)
    summary={
        'scope':{'layers':94,'exclude_embedding':True,'all_targets_per_layer':10000,'h0_essential_bar':'One infinite bar in each connected full-Rips diagram cancels; finite 9999 bars included.','units':'Distances normalized per layer by saved median sampled pairwise distance. H0/H1 Betti amplitudes divided by 10000.','h0_h1_distance':'Exact integral absolute difference of Betti step curves on full event support; a pseudometric on diagrams, not a complete topology identifier.','landscape':'Top five landscape levels, all bars participate; mean absolute difference integrated by trapezoids on a common 1025-point grid. Levels above five omitted.','landscape_interval':[0,tmax],'landscape_grid_step':float(grid[1]-grid[0]),'landscape_trapezoid_conservative_absolute_error_bound':float(tmax*(grid[1]-grid[0])),'layer_matching':'Relative depth = layer/final transformer layer. Nearest matching; exact ties choose lower layer.','percentiles':'Descriptive within each cross-model pair distribution; smaller = more similar; not p-values.'},
        'selections':selections,'selected_comparisons':selected,'windows':windows,
        'largest_adjacent_changes':{m:{k:sorted([r for r in adjacent if r['model']==m],key=lambda r:r[k],reverse=True)[:3] for k in matrices} for m in MODELS},
        'all_pairs':{},'inputs':inputs}
    for metric,mat in matrices.items():
        same=[r[metric] for r in pairs if r['same_model']]
        cross=[r[metric] for r in pairs if not r['same_model']]
        near=[r[metric] for r in pairs if not r['same_model'] and r['depth_difference']<=.05]
        summary['all_pairs'][metric]={'within_model_median':float(np.median(same)),'cross_model_median':float(np.median(cross)),'matched_depth_within_5_percent_cross_model_median':float(np.median(near)),'within_pairs':len(same),'cross_pairs':len(cross),'depth_matched_pairs':len(near)}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    # Plot every layer in the same model/depth ordering, and selected curves on shared scales.
    fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    bounds=np.cumsum([0,*MODELS.values()])
    for ax,(name,mat) in zip(axes,matrices.items()):
        im=ax.imshow(mat,cmap='magma',origin='upper');fig.colorbar(im,ax=ax,shrink=.7)
        for cut in bounds[1:-1]:ax.axhline(cut-.5,color='white',lw=.8);ax.axvline(cut-.5,color='white',lw=.8)
        centers=(bounds[:-1]+bounds[1:]-1)/2
        ax.set_xticks(centers,[NAMES[m] for m in MODELS],rotation=35,ha='right');ax.set_yticks(centers,[NAMES[m] for m in MODELS]);ax.set_title({'h0':'H0 Betti: exact integrated difference','h1':'H1 Betti: exact integrated difference','landscape5':'Top-5 landscapes: grid approximation'}[name])
    fig.suptitle('Similarity across all 94 transformer levels · darker = more similar',fontsize=14)
    fig.savefig(OUT/'all_layer_similarity.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(16,11),layout='constrained')
    curve_tmax=max(float(rows[idx[m,selections[sel][m]]]['h1'][:,1].max()) for sel in ['best_rho','best_ev2','final'] for m in MODELS)
    plotgrid=np.linspace(0,curve_tmax,1500)
    h0max=max(float(rows[idx[m,selections[sel][m]]]['h0'][:,1].max()) for sel in ['best_rho','best_ev2','final'] for m in MODELS)
    h0grid=np.linspace(0,h0max,1500)
    for rr,sel in enumerate(['best_rho','best_ev2','final']):
        for m in MODELS:
            i=idx[m,selections[sel][m]];r=rows[i];label=f'{NAMES[m]} L{r["layer"]}'
            axes[rr,0].plot(h0grid,betti_at(r['h0'],h0grid)+1/10000,label=label,color=COLORS[m])
            axes[rr,1].plot(plotgrid,betti_at(r['h1'],plotgrid),label=label,color=COLORS[m])
            axes[rr,2].plot(grid,landscapes[i,0],label=label,color=COLORS[m])
        for cc,title in enumerate(['H0 components / 10,000','H1 loops / 10,000','First persistence landscape']):
            axes[rr,cc].set_title(sel.replace('_',' ')+' · '+title);axes[rr,cc].set_xlabel('Distance / layer median pair distance');axes[rr,cc].legend(fontsize=8);axes[rr,cc].grid(alpha=.2)
    fig.savefig(OUT/'selected_curves.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(11,10),layout='constrained')
    for ax,k in zip(axes,matrices):
        for m in MODELS:
            rr=[r for r in adjacent if r['model']==m]
            ax.plot([r['depth'] for r in rr],[r[k] for r in rr],'.-',label=NAMES[m],color=COLORS[m])
        ax.set_xlabel('Relative depth of later layer');ax.set_ylabel(k+' adjacent distance');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Where consecutive layers change most')
    fig.savefig(OUT/'adjacent_changes.png',dpi=150);plt.close(fig)
    print('Matrices and plots finished',time.monotonic()-started,flush=True)
    # Bounded bottleneck comparisons retain the longest bars; matching is exact ONLY for retained bars.
    # Diagram distance to the retained subdiagram is at most half the largest omitted lifetime.
    br=[]
    for sel in ['best_rho','best_ev2','final']:
        for a,b in itertools.combinations(MODELS,2):
            da=rows[idx[a,selections[sel][a]]]['h1'];db=rows[idx[b,selections[sel][b]]]['h1']
            ordera=np.argsort(da[:,1]-da[:,0])[::-1];orderb=np.argsort(db[:,1]-db[:,0])[::-1]
            for top in [128,512]:
                if top==512 and (sel=='best_ev2'):continue
                aa=da[ordera[:top]];bb=db[orderb[:top]]
                ea=float((da[ordera[top],1]-da[ordera[top],0])/2) if len(da)>top else 0.
                eb=float((db[orderb[top],1]-db[orderb[top],0])/2) if len(db)>top else 0.
                tick=time.monotonic();d=float(bottleneck(aa,bb));sec=time.monotonic()-tick
                witness=max(witness_lower(aa,db),witness_lower(bb,da))
                lower=max(0.,d-ea-eb,witness);upper=max(d,ea,eb)
                assert lower<=upper+1e-12
                br.append(dict(selection=sel,a=f'{a}:L{selections[sel][a]:02d}',b=f'{b}:L{selections[sel][b]:02d}',retained_max=top,a_original_bars=len(da),b_original_bars=len(db),retained_diagrams_exact_bottleneck=d,discard_error_a=ea,discard_error_b=eb,witness_lower=witness,full_bottleneck_lower=lower,full_bottleneck_upper=upper,certified_exact_full=abs(upper-lower)<1e-12,seconds=sec))
                write_csv('selected_bottleneck.csv',br)
                print('Bottleneck',sel,a,b,top,d,'full interval',(lower,upper),'seconds',sec,flush=True)
    summary['bottleneck']={'method':'Exact persim matching only on retained longest bars. For full diagrams: lower=max(0,d_filtered-eA-eB,witness), upper=max(d_filtered,eA,eB), eX=half maximum discarded lifetime. Upper extends filtered matching by sending omitted bars to diagonal. Witness lower=max over retained bars of min(half lifetime, nearest L-infinity distance to ANY bar of the FULL opposite diagram), in both directions. Triangle inequality supplies the other lower bound. Equal lower/upper certifies full bottleneck without full matching (numerical tolerance1e-12). Otherwise only an interval is certified.','comparisons':br}
    summary['runtime_seconds']=time.monotonic()-started
    summary['code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print('DONE',summary['runtime_seconds'],flush=True)

def area_sensitivity():
    z=np.load(OUT/'matrices.npz');labels=z['labels'].tolist()
    ev=[]
    for label in labels:
        m,l=label.split(':L');d=np.load(SRC/m/f'layer_{int(l):02d}'/'h1.npz')['normalized']
        x,w=events(d);area=float(np.sum(d[:,1]-d[:,0]));assert area>0
        ev.append((x,w/area))
    mat=np.zeros((len(ev),len(ev)))
    for i in range(len(ev)):
        for j in range(i):mat[i,j]=mat[j,i]=curve_l1(ev[i],ev[j],1)
    np.savez_compressed(OUT/'unit_area_h1.npz',labels=labels,unit_area_h1_betti_l1=mat)
    selected=list(csv.DictReader((OUT/'selected_comparisons.csv').open()));result=[]
    for r in selected:
        i,j=labels.index(r['a']),labels.index(r['b']);a,b=r['a'].split(':')[0],r['b'].split(':')[0]
        cross=[mat[x,y] for x in range(len(ev)) for y in range(len(ev)) if labels[x].split(':')[0]==a and labels[y].split(':')[0]==b]
        result.append(dict(selection=r['selection'],a=r['a'],b=r['b'],unit_area_h1_betti_l1=float(mat[i,j]),cross_pair_percentile=float(100*np.mean(np.array(cross)<=mat[i,j]))))
    write_csv('unit_area_selected.csv',result)
    note={'method':'Exact integral absolute difference of unit-area H1 Betti curves. Each curve divided by its own total persistence (amplitude normalized). Values between0 and2. All bars included; pseudometric on diagrams. Checks shape along filtration scale independently of total H1 lifetime.','selected':result}
    (OUT/'unit_area_summary.json').write_text(json.dumps(note,indent=2))
    print('UNIT AREA DONE',flush=True)

def validate_comparisons():
    rng=np.random.default_rng(17)
    for _ in range(20):
        a=rng.random((8,2));a[:,1]+=a[:,0]
        b=rng.random((10,2));b[:,1]+=b[:,0]
        a=a[np.argsort(a[:,1]-a[:,0])[::-1]];b=b[np.argsort(b[:,1]-b[:,0])[::-1]]
        ea=(a[4,1]-a[4,0])/2;eb=(b[4,1]-b[4,0])/2
        d=bottleneck(a[:4],b[:4]);exact=bottleneck(a,b)
        lo=max(0,d-ea-eb,witness_lower(a[:4],b),witness_lower(b[:4],a));hi=max(d,ea,eb)
        assert lo-1e-12<=exact<=hi+1e-12
    z=np.load(OUT/'matrices.npz');summary=json.loads((OUT/'summary.json').read_text())
    rows=[];cache={};grid=np.linspace(0,z['landscape_grid'][-1],4097)
    for r in summary['selected_comparisons']:
        if r['selection'] not in ['best_rho','best_ev2','final']:continue
        for label in [r['a'],r['b']]:
            if label not in cache:
                m,l=label.split(':L');d=np.load(SRC/m/f'layer_{int(l):02d}'/'h1.npz')['normalized'];cache[label]=landscape(d,grid)
        fine=float(np.trapezoid(np.abs(cache[r['a']]-cache[r['b']]),grid,axis=1).mean())
        rows.append(dict(selection=r['selection'],a=r['a'],b=r['b'],grid1025=r['landscape5'],grid4097=fine,absolute_difference=abs(fine-r['landscape5'])))
    write_csv('landscape_grid_check.csv',rows)
    for name in ['h0_betti_l1','h1_betti_l1','landscape_top5_grid_l1']:
        m=z[name];assert m.shape==(94,94) and np.isfinite(m).all() and (m>=0).all() and np.allclose(m,m.T) and np.allclose(np.diag(m),0)
    check={'betti_fixture_checks':'passed','bottleneck_bound_vs_full_matching_random_fixtures':20,'matrices':'94x94, finite, nonnegative, symmetric, diagonal zero','landscape_selected_pairs_grid_refinement':rows,'max_selected_landscape_grid_change':max(r['absolute_difference'] for r in rows),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'validation.json').write_text(json.dumps(check,indent=2))
    print('VALIDATION DONE',flush=True)

def depth_sensitivity():
    z=np.load(OUT/'matrices.npz');labels=z['labels'].tolist()
    matrices={'h0':z['h0_betti_l1'],'h1':z['h1_betti_l1'],'landscape5':z['landscape_top5_grid_l1']};result=[]
    for a,b in itertools.combinations(MODELS,2):
        ij=[(i,j) for i,x in enumerate(labels) for j,y in enumerate(labels) if x.split(':')[0]==a and y.split(':')[0]==b]
        for method in ['all_pairs','same_absolute_layer','relative_depth_within_0.05']:
            keep=[]
            for i,j in ij:
                la=int(labels[i].split(':L')[1]);lb=int(labels[j].split(':L')[1])
                if method=='all_pairs' or (method=='same_absolute_layer' and la==lb) or (method=='relative_depth_within_0.05' and abs(la/MODELS[a]-lb/MODELS[b])<=.05):keep.append((i,j))
            result.append(dict(a=a,b=b,comparison=method,pairs=len(keep),**{k+'_median':float(np.median([v[i,j] for i,j in keep])) for k,v in matrices.items()}))
    write_csv('absolute_vs_relative_depth.csv',result)

def empty_baseline():
    rows=list(csv.DictReader((OUT/'selected_bottleneck.csv').open()))
    for r in rows:
        vals=[]
        for label in [r['a'],r['b']]:
            m,l=label.split(':L');d=np.load(SRC/m/f'layer_{int(l):02d}'/'h1.npz')['normalized'];vals.append(float(np.max(d[:,1]-d[:,0])/2))
        r['a_distance_to_empty']=vals[0];r['b_distance_to_empty']=vals[1];r['delete_all_bars_upper']=max(vals)
        r['full_lower_over_delete_all']=float(r['full_bottleneck_lower'])/max(vals);r['full_upper_over_delete_all']=float(r['full_bottleneck_upper'])/max(vals)
    write_csv('bottleneck_vs_empty.csv',rows)

if __name__=='__main__':
    main()
    area_sensitivity()
    validate_comparisons()
    depth_sensitivity()
    empty_baseline()
