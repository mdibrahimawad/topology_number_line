"""Offline descriptive analysis of saved full-range PH; no inference or PH rerun."""
from pathlib import Path
import csv, hashlib, json, sys
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/Users/mohammed.awad/Documents/Codex/2026-09-19/c')
REPO=Path('/Users/mohammed.awad/Documents/numberline_zigzag_modal_FINAL')
sys.path.insert(0,str(REPO))
from numzig.fullrange.storage import Store,digest
DATA=ROOT/'outputs/ph_fullrange_overnight/results/results'
OUT=ROOT/'outputs/ph_analysis';(OUT/'figures').mkdir(parents=True,exist_ok=True)
MODELS={'crystal':('Crystal',32,'#2166ac'), 'starcoderbase-3b':('StarCoderBase-3B',36,'#d66a18'), 'openllama-3b':('OpenLLaMA-3B',26,'#278260')}
paths=json.loads((ROOT/'outputs/ph_fullrange_overnight/layer_selection_verified.json').read_text())
SCORES={p['model'].split('_fullrange')[0]:{r['layer']:r for r in json.loads(Path(p['source']).read_text())} for p in paths}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130,'savefig.dpi':170})

def dump(name,value): (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False))
def csvout(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def corr(x,y):
    if np.ptp(x)<=1e-12*max(1,float(np.max(np.abs(x)))) or np.ptp(y)<=1e-12*max(1,float(np.max(np.abs(y)))):return None
    return float(spearmanr(x,y).statistic)
def betti(a):
    t=np.r_[a[:,0],a[:,1]];delta=np.r_[np.ones(len(a),dtype=int),-np.ones(len(a),dtype=int)]
    u,inv=np.unique(t,return_inverse=True);v=np.cumsum(np.bincount(inv,weights=delta)).astype(int)
    return u,v
def curve(a,grid):return np.searchsorted(np.sort(a[:,0]),grid,side='right')-np.searchsorted(np.sort(a[:,1]),grid,side='right')

# Mathematical checks independent of real data: event ties and area identity.
toy=np.array([[0.,2.],[1.,3.],[2.,2.]])
t,b=betti(toy);assert np.array_equal(b,[1,2,1,0])
assert np.dot(np.diff(t),b[:-1])==np.diff(toy,axis=1).sum()

rows=[];diagrams={};provenance=[]
for model,(label,L,color) in MODELS.items():
    for layer in range(1,L+1):
        p=DATA/model/f'layer_{layer:02d}';s=Store(p)
        assert all(s.valid(k) for k in ('h0','h1','plots'))
        metric=json.loads((p/'metrics.json').read_text());old=SCORES[model][layer]
        with np.load(p/'h0.npz') as a,np.load(p/'h1.npz') as b:
            h0=a['normalized'].copy();h1=b['normalized'].copy();raw=b['raw'].copy()
            assert np.array_equal(a['point_ids'],np.arange(10000))
        assert old['status']=='valid' and metric['point_count']==10000 and np.isfinite(h1).all()
        diagrams[(model,layer)]=(h0,h1)
        life=h1[:,1]-h1[:,0];positive=life[life>0];ordered=np.sort(life)[::-1]
        h0d=np.sort(h0[np.isfinite(h0[:,1]),1]);t,beta=betti(h1)
        area=float(np.dot(np.diff(t),beta[:-1]))
        assert np.isclose(area,life.sum(),rtol=1e-11,atol=1e-11)
        prob=positive/positive.sum();entropy=float(-np.sum(prob*np.log(prob)))
        q=np.quantile(life,[.25,.5,.75,.9,.99]);mx=int(np.argmax(life))
        r=dict(model=model,layer=layer,depth=layer/L,phase=('early' if layer/L<=1/3 else 'middle' if layer/L<=2/3 else 'late'),
            rho_pc1=old['pca']['spearman_absolute'][0],rho_pc2=old['pca']['spearman_absolute'][1],
            ev_pc1=old['pca']['explained_variance_ratio'][0],ev_pc12=sum(old['pca']['explained_variance_ratio']),
            same_leading_digit=old['same_leading_digit_fraction'],same_digit_length=old['same_digit_length_fraction'],
            neighbor_numeric_gap=old['absolute_numerical_gaps']['median'],neighbor_log_gap=old['absolute_log10_gaps']['median'],
            distance_scale=metric['normalization']['scale'],h0_median=float(np.median(h0d)),h0_q90=float(np.quantile(h0d,.9)),h0_connect=float(h0d[-1]),
            h1_count=len(life),h1_max=float(ordered[0]),h1_second=float(ordered[1]),h1_top_ratio=float(ordered[0]/ordered[1]),
            h1_q25=float(q[0]),h1_median=float(q[1]),h1_q75=float(q[2]),h1_q90=float(q[3]),h1_q99=float(q[4]),
            h1_top_birth=float(h1[mx,0]),h1_top_death=float(h1[mx,1]),
            h1_total=float(life.sum()),h1_top_share=float(ordered[0]/life.sum()),h1_top30_share=float(ordered[:30].sum()/life.sum()),
            h1_entropy=entropy,h1_entropy_normalized=entropy/np.log(len(positive)),
            h1_peak_betti=int(beta.max()),h1_peak_scale=float(t[np.argmax(beta)]),
            h1_first_birth=float(h1[:,0].min()),h1_last_death=float(h1[:,1].max()),
            h1_raw_max=float(np.max(raw[:,1]-raw[:,0])),
            h1_rounding_bound=2*metric['max_filtration_rounding_error']/metric['normalization']['scale'])
        for threshold in [.01,.02,.05,.1,.2]:r['h1_count_gt_'+str(threshold)]=int(np.sum(life>threshold))
        for radius in [.1,.25,.5,1.,2.]:
            r['h0_components_at_'+str(radius)]=int(1+np.sum(h0d>radius))
            r['h1_betti_at_'+str(radius)]=int(curve(h1,np.array([radius]))[0])
        r['h1_below_rounding_bound']=int(np.sum(life<=r['h1_rounding_bound']))
        rows.append(r);provenance.append(dict(model=model,layer=layer,manifest_sha256=digest(p/'manifest.json'),h0_sha256=digest(p/'h0.npz'),h1_sha256=digest(p/'h1.npz')))
assert len(rows)==94
csvout('layer_metrics.csv',rows);dump('layer_metrics.json',rows);dump('input_hashes.json',provenance)

selectors={'best_rho_pc1':'rho_pc1','best_rho_pc2':'rho_pc2','best_ev_pc1':'ev_pc1','best_ev_pc12':'ev_pc12'}
anchors={};groups=[]
for model,(_,L,_) in MODELS.items():
    rr=[r for r in rows if r['model']==model]
    anchors[model]={k:max(rr,key=lambda r:r[v])['layer'] for k,v in selectors.items()}
    anchors[model]['final']=L
    for d in [.1,.25,.5,.75,.9,1.]:anchors[model]['depth_'+str(d)]=min(rr,key=lambda r:(abs(r['depth']-d),r['layer']))['layer']
    for name,subset in [(p,[r for r in rr if r['phase']==p]) for p in ['early','middle','late']]+[(f'last_{n}',rr[-n:]) for n in [3,5]]:
        groups.append(dict(model=model,group=name,layers=[r['layer'] for r in subset],**{k:float(np.median([r[k] for r in subset])) for k in ['rho_pc1','ev_pc12','h1_max','h1_median','h1_count','h0_connect','same_leading_digit','h1_top_ratio']}))
dump('selections.json',anchors);dump('phase_summaries.json',groups)

associations=[]
features=['h1_max','h1_median','h1_count','h1_total','h1_entropy_normalized','h0_connect','h1_peak_betti']
targets=['rho_pc1','rho_pc2','ev_pc12','ev_pc1','same_leading_digit','same_digit_length','neighbor_numeric_gap','depth']
for model in MODELS:
    rr=[r for r in rows if r['model']==model]
    for feature in features:
        for target in targets:
            x=np.array([r[feature] for r in rr]);y=np.array([r[target] for r in rr])
            associations.append(dict(model=model,topology=feature,comparison=target,spearman=corr(x,y),adjacent_change_spearman=corr(np.diff(x),np.diff(y))))
csvout('associations.csv',associations);dump('associations.json',associations)

fig,axes=plt.subplots(3,2,figsize=(13,12),layout='constrained')
for ax,key,title in zip(axes.flat,['rho_pc1','ev_pc12','h1_max','h1_median','h1_count','h0_connect'],['Numerical ordering (PC1 rho)','Variance retained in 2D','Longest H1 lifetime','Median H1 lifetime','Number of H1 intervals','Distance needed for full connectivity']):
    for model,(label,L,color) in MODELS.items():
        rr=[r for r in rows if r['model']==model];ax.plot([r['depth'] for r in rr],[r[key] for r in rr],label=label,color=color,lw=2)
        if key=='h1_max':
            r=next(r for r in rr if r['layer']==anchors[model]['best_rho_pc1']);ax.scatter(r['depth'],r[key],s=65,color=color,edgecolor='black',zorder=3)
    ax.set_title(title,loc='left');ax.set_xlabel('Relative layer depth');ax.grid(alpha=.15)
    if key.startswith('h1_') and key!='h1_count' or key=='h0_connect':ax.set_ylabel('Distance / layer median distance')
axes[0,0].legend(fontsize=9);fig.suptitle('All 94 non-embedding layers · saved full-point PH\nBlack-outlined dots mark best-rho layers',fontsize=15)
fig.savefig(OUT/'figures/depth_profiles.png');plt.close(fig)

fig,axes=plt.subplots(3,3,figsize=(15,12),layout='constrained')
for rid,(name,title) in enumerate([('best_rho_pc1','Best numerical ordering'),('best_ev_pc12','Best 2D explained variance'),('final','Final layer')]):
    for model,(label,L,color) in MODELS.items():
        layer=anchors[model][name];h0,h1=diagrams[(model,layer)];life=np.sort(h1[:,1]-h1[:,0]);grid=np.linspace(0,2,2001)
        axes[rid,0].plot(life,np.arange(1,len(life)+1)/len(life),color=color,label=f'{label} L{layer}')
        axes[rid,1].plot(grid,curve(h1,grid),color=color)
        deaths=np.sort(h0[np.isfinite(h0[:,1]),1]);axes[rid,2].plot(grid,1+len(deaths)-np.searchsorted(deaths,grid,side='right'),color=color)
    axes[rid,0].set_title(title+' · H1 lifetime distribution',loc='left');axes[rid,0].set_xlabel('Normalized lifetime');axes[rid,0].set_ylabel('Fraction of H1 intervals');axes[rid,0].legend(fontsize=8)
    axes[rid,1].set_title('Loops alive at each distance',loc='left');axes[rid,1].set_yscale('symlog',linthresh=1);axes[rid,1].set_xlabel('Normalized distance (display 0–2)')
    axes[rid,2].set_title('Components alive at each distance',loc='left');axes[rid,2].set_yscale('log');axes[rid,2].set_xlabel('Normalized distance (display 0–2)')
fig.suptitle('Anchor comparisons · all saved intervals used; curve display limited to 0–2',fontsize=15);fig.savefig(OUT/'figures/anchor_comparisons.png');plt.close(fig)

fig,axes=plt.subplots(1,3,figsize=(16,6),layout='constrained')
for ax,(model,(label,_,_)) in zip(axes,MODELS.items()):
    mat=np.array([[next(r['spearman'] for r in associations if r['model']==model and r['topology']==f and r['comparison']==t) for t in ['rho_pc1','ev_pc12','same_leading_digit','depth']] for f in features],dtype=float)
    im=ax.imshow(mat,vmin=-1,vmax=1,cmap='RdBu_r',aspect='auto');ax.set_title(label);ax.set_xticks(range(4),['PC1 rho','2D variance','Leading digit','Depth'],rotation=40,ha='right');ax.set_yticks(range(len(features)),['Longest H1','Median H1','H1 count','Total H1','H1 entropy','H0 connect','Peak loops'])
    for (i,j),v in np.ndenumerate(mat):ax.text(j,i,f'{v:.2f}',ha='center',va='center',color='white' if abs(v)>.65 else 'black',fontsize=9)
fig.colorbar(im,ax=axes,label='Descriptive Spearman correlation');fig.suptitle('Associations across layers · related layers, no significance claims',fontsize=15);fig.savefig(OUT/'figures/associations.png');plt.close(fig)

# Complete diagram atlas: every interval appears; shared axes within each model.
for model,(label,L,color) in MODELS.items():
    fig,axes=plt.subplots(int(np.ceil(L/6)),6,figsize=(21,3.25*int(np.ceil(L/6))),layout='constrained')
    vmax=max(float(diagrams[(model,l)][1].max()) for l in range(1,L+1))*1.02
    for i,ax in enumerate(axes.flat):
        layer=i+1
        if layer>L:ax.axis('off');continue
        h0,h1=diagrams[(model,layer)];life=h1[:,1]-h1[:,0];idx=int(np.argmax(life))
        ax.scatter(h1[:,0],h1[:,1],s=1,alpha=.16,color=color,rasterized=True)
        ax.scatter(*h1[idx],s=20,facecolors='none',edgecolors='black',zorder=4)
        ax.plot([0,vmax],[0,vmax],color='.7',lw=.7);ax.set_xlim(0,vmax);ax.set_ylim(0,vmax)
        ax.set_title(f'L{layer} · {len(h1):,} intervals\nlongest lifetime {life[idx]:.3f}',fontsize=9);ax.set_xlabel('Birth / scale',fontsize=8);ax.set_ylabel('Death / scale',fontsize=8);ax.tick_params(labelsize=8)
    fig.suptitle(f'{label}: every non-embedding H1 diagram · circle marks longest interval\nShared axes within model; all intervals shown; point darkness is not statistical significance',fontsize=15)
    fig.savefig(OUT/f'figures/{model}_all_diagrams.png');plt.close(fig)

summary={}
for m in MODELS:
    rr=[r for r in rows if r['model']==m];best=next(r for r in rr if r['layer']==anchors[m]['best_rho_pc1']);last=rr[-1]
    summary[m]=dict(anchors=anchors[m],best_rho=best,final=last,minimum_h1_max=min(rr,key=lambda r:r['h1_max']),maximum_h1_max=max(rr,key=lambda r:r['h1_max']),
        minimum_h1_median=min(rr,key=lambda r:r['h1_median']),maximum_h1_median=max(rr,key=lambda r:r['h1_median']),
        scale_range=[min(r['distance_scale'] for r in rr),max(r['distance_scale'] for r in rr)])
dump('summary.json',summary)
dump('validation.json',dict(layers=94,hash_checks_passed=True,interval_area_checks_passed=True,event_tie_unit_check_passed=True,
    all_intervals_used=True,finite_precision_warning='H1 uses saved float32 filtration. Two times measured distance rounding error / scale is recorded as a numerical sensitivity bound, not a scientific noise threshold.',
    statistics='Descriptive. No independence of layers assumed; no p-values. H0 essential interval excluded from finite-lifetime summaries.',
    selection='Non-embedding levels only. Best means maximum stored value, tie broken by earliest layer. Relative depths choose nearest layer, earlier on exact ties. Thirds: (0,1/3], (1/3,2/3], (2/3,1].'))
print(json.dumps({m:{'best_rho_layer':s['best_rho']['layer'],'best_rho_h1_max':s['best_rho']['h1_max'],'final_h1_max':s['final']['h1_max'],'final_h1_median':s['final']['h1_median'],'final_count':s['final']['h1_count'],'scale_range':s['scale_range']} for m,s in summary.items()},indent=2))
