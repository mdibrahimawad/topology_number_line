"""Human-readable findings and scientific figures from the saved shape scan."""
import os
for key in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
os.environ['MPLCONFIGDIR']='/tmp/geometric_shapes_mpl'
from collections import Counter
from pathlib import Path
import csv,html,json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from templates import make_templates
from inventory import load_cloud,ROOT
from run import VERSION,write

OUT=ROOT/'outputs/geometric_shapes'
PALETTE=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22']

def pretty(name):return name.replace('_',' ').replace('pitch','pitch ')
def ref(row,shape=None):
    e=row['entry'];label=f"{e['model']} · {e['task']} · "+(f"context {e['context']+1} · " if e['context'] is not None else '')+f"L{e['layer']}"
    return f"[{label}](index.html#id={e['id']}"+(f"&shape={shape}" if shape else '')+")"
def fitof(row,name):return next(f for f in row['fits'] if f['name']==name)

def draw_cloud(ax,row,fit):
    y,_,labels,_=load_cloud(row['entry']);norm=row['normalization']
    with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
        y=(y-np.array(norm['center']))/norm['radius']@np.array(norm['basis']).T
    ids=np.arange(len(y))
    if len(ids)>1600:ids=np.sort(np.random.default_rng(42).choice(ids,1600,False))
    ax.scatter(*y[ids].T,s=2,c=[PALETTE[(int(k)-1)%9] for k in labels[ids]],alpha=.28,rasterized=True)
    template=next(t for t in make_templates() if t['name']==fit['name'])
    z=fit['scale']*template['points']@np.array(fit['rotation'])+np.array(fit['offset'])
    if template['family'] in ('line','circle','helix'):
        if template['family']=='circle':z=np.vstack([z,z[0]])
        ax.plot(*z.T,color='#142233',lw=1.7)
    else:ax.scatter(*z.T,c='#101924',s=9 if template['dimension']==0 else 2,alpha=.75)
    center=(np.max(y,axis=0)+np.min(y,axis=0))/2
    half=max(np.ptp(y,axis=0).max()/2,np.max(abs(z-center)))*1.08
    ax.set(xlim=(center[0]-half,center[0]+half),ylim=(center[1]-half,center[1]+half),zlim=(center[2]-half,center[2]+half))
    ax.set_box_aspect((1,1,1));ax.view_init(22,40);ax.set_xticks([]);ax.set_yticks([]);ax.set_zticks([])
    e=row['entry'];title=f"{e['model']} L{e['layer']} | {e['task']}"+(f" c{e['context']+1}" if e['context'] is not None else '')
    ax.set_title(title+'\n'+pretty(fit['name'])+f" · error {fit['symmetric_rms']:.3f}",fontsize=9)

def main():
    rows=[json.loads(p.read_text()) for p in sorted((OUT/'fits').glob('*.json'))]
    assert len(rows)==1649 and all(r['version']==VERSION for r in rows)
    valid=[r for r in rows if 'fits' in r]
    primary=[r for r in valid if r['entry']['group']!='overlay' and r['entry']['layer']>0]
    matches=[r for r in primary if r['screen_matches']]
    controls=[json.loads(p.read_text()) for p in sorted((OUT/'controls').glob('*.json'))]
    grouped={}
    for c in controls:grouped.setdefault(c['id'],[]).append(c)
    templates=make_templates();template_names=[t['name'] for t in templates]
    byname={r['entry']['id']:r for r in rows}
    result=dict(total_views=len(rows),nonconstant_views=len(valid),contextual_individual_clouds=len(primary),
        shape_candidates=len(templates),candidate_fits=len(valid)*len(templates),
        degenerate_views=len(rows)-len(valid),positional_views=sum(r['entry']['positional'] for r in valid),
        primary_threshold_passes=len(matches),control_clouds=len(grouped),control_scans=len(controls),
        scope_counts=dict(Counter(r['entry']['group'] for r in rows)),
        matches=[dict(id=r['entry']['id'],shapes=r['screen_matches']) for r in matches])
    best_by_shape=[]
    for t in templates:
        group=[]
        for cohort in ('original','new'):
            r=min((r for r in primary if r['entry']['group']==cohort),key=lambda r:fitof(r,t['name'])['symmetric_rms'])
            f=fitof(r,t['name']); group.append(dict(cohort=cohort,id=r['entry']['id'],**{k:f[k] for k in ('name','symmetric_rms','data_rms','coverage_rms','screen_pass')}))
        best_by_shape.append(dict(name=t['name'],dimension=t['dimension'],best=group))
    result['best_by_shape']=best_by_shape
    evr=np.array([sum(r['entry']['evr']) for r in primary])
    result['pca_variance']=dict(min=float(evr.min()),median=float(np.median(evr)),max=float(evr.max()))
    write(OUT/'summary.json',result)
    fields=['id','group','model','task','context','layer','candidate','dimension','data_rms','coverage_rms','symmetric_rms','data_p95','coverage_p95','screen_pass','retained_variance','positional']
    with (OUT/'all_fits.csv').open('w') as file:
        writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader()
        for r in valid:
            for f in r['fits']:
                writer.writerow({**{k:r['entry'][k] for k in ('id','group','model','task','context','layer','positional')},'candidate':f['name'],'retained_variance':sum(r['entry']['evr']),**{k:f[k] for k in ('dimension','data_rms','coverage_rms','symmetric_rms','data_p95','coverage_p95','screen_pass')}})
    # Candidate library figure; different intrinsic dimensions are visible rather than conflated.
    fig=plt.figure(figsize=(16,13))
    for i,t in enumerate(templates):
        ax=fig.add_subplot(4,8,i+1,projection='3d');v=t['points'];ax.scatter(*v.T,s=10 if t['dimension']==0 else 1,c='#254a73');ax.set_box_aspect((1,1,1));ax.set_axis_off();ax.set_title(pretty(t['name']),fontsize=8)
    fig.suptitle('The 32 tested templates — fixed proportions, freely rotated and uniformly scaled',fontsize=15);fig.tight_layout(rect=(0,0,1,.97));fig.savefig(OUT/'candidate_library.png',dpi=150);plt.close(fig)
    # Closest tetrahedral vertices, nearest helix, nearest ring, and overall examples per cohort.
    exemplars=[]
    for cohort in ('original','new'):
        pool=[r for r in primary if r['entry']['group']==cohort]
        for names in (['tetrahedron_vertices'],['tetrahedron_edges'],['tetrahedron_surface'],['circle'],[t['name'] for t in templates if t['family']=='helix']):
            r,f=min(((r,fitof(r,n)) for r in pool for n in names),key=lambda pair:pair[1]['symmetric_rms'])
            exemplars.append((r,f))
    fig=plt.figure(figsize=(19,8))
    for i,(r,f) in enumerate(exemplars):draw_cloud(fig.add_subplot(2,5,i+1,projection='3d'),r,f)
    fig.suptitle('Closest examples in each cohort — a closest fit can still fail the match criteria',fontsize=14);fig.tight_layout(rect=(0,0,1,.95));fig.savefig(OUT/'closest_examples.png',dpi=170);plt.close(fig)
    # Errors across layer depth; isolate original and each new task, plus contexts.
    fig,axes=plt.subplots(3,7,figsize=(19,8),sharey=True)
    tasks=['original_copy','copy4','reverse4','swap_first4','swap_last4','numeric_copy','word_copy']
    for i,model in enumerate(('crystal','starcoder','openllama')):
        for j,task in enumerate(tasks):
            ax=axes[i,j]
            for ctx in ([None] if task=='original_copy' else [0,1]):
                rs=sorted([r for r in primary if r['entry']['model']==model and r['entry']['task']==task and r['entry']['context']==ctx],key=lambda r:r['entry']['layer'])
                ax.plot([r['entry']['layer'] for r in rs],[r['fits'][0]['symmetric_rms'] for r in rs],lw=1.2,label='original' if ctx is None else f'context {ctx+1}')
            ax.axhline(.15,color='#9a6666',ls='--',lw=.8);ax.set_ylim(0,.75);ax.grid(alpha=.2)
            if i==0:ax.set_title(pretty(task),fontsize=9)
            if j==0:ax.set_ylabel(model+'\ncombined residual')
            if i==2:ax.set_xlabel('saved layer')
    axes[0,1].legend(fontsize=7);fig.suptitle('Minimum residual among all 32 templates (descriptive search; lower is closer)',fontsize=13);fig.tight_layout(rect=(0,0,1,.95));fig.savefig(OUT/'all_layer_errors.png',dpi=160);plt.close(fig)
    lines=['# Do familiar geometric shapes emerge?', '',
        f'We tested **32 fixed-proportion shape templates on all {len(valid):,} nonconstant saved 3D views**. Another {len(rows)-len(valid)} views were constant and have no shape to fit. This includes every old and new individual layer plus the shared-PCA comparison views.', '',
        f'**{len(matches)} of the {len(primary):,} contextual individual clouds passed our stated descriptive match screen.** '+('Passing examples are listed below; these are projected resemblances, not unique shape identifications.' if matches else 'The closest candidate should therefore not be read as an identified tetrahedron, sphere, torus, or helix.'), '',
        '[Open every fitted view](index.html) · [Every candidate score](all_fits.csv) · [Machine-readable summary](summary.json) · [Validation](validation.json)', '',
        '## What did we look at?', '',
        '| Question | Test | What it tells us |','|---|---|---|',
        '| Does the cloud resemble a regular polyhedron? | Tetrahedron, cube, octahedron, icosahedron and dodecahedron: separate vertices, edges and surface models. | A regular geometric resemblance, rather than an arbitrary collection of clusters. |',
        '| Is there a simple curve or surface? | Line, circle, disk, sphere, cylinder and cone. | Whether a familiar simple description covers the entire cloud. |',
        '| Is it a donut or spiral? | Two fixed torus proportions; one-, two- and three-turn circular helices at three pitches. | These particular whole-cloud geometries, not every possible bent or twisted manifold. |',
        '| Is the apparent fit an empty outline? | Distances in both directions, using held-out observations. | Rejects a whole circle fitted to only a short populated arc. |',
        '| Does an ordinary cloud fit just as well? | Gaussian and shuffled-coordinate controls for shortlisted cases. | A descriptive check against chance resemblance, not a corrected significance test. |',
        '| Does projection explain the result? | Retained PCA variance and saved original-space distance/neighborhood checks. | How limited a claim about the full hidden geometry would be. |', '',
        '![Candidate library](candidate_library.png)', '',
        '## Closest examples, including tetrahedra', '',
        '| Cohort | View | Closest requested candidate | Data residual | Empty-shape residual | Pass? |',
        '|---|---|---|---:|---:|---|']
    for r,f in exemplars:lines.append(f"| {r['entry']['group']} | {ref(r,f['name'])} | {pretty(f['name'])} | {f['data_rms']:.3f} | {f['coverage_rms']:.3f} | {'Yes' if f['screen_pass'] else 'No'} |")
    lines += ['', 'Errors are in units of the cloud’s RMS radius. A value of 0.30 is roughly 30% of that overall spread. These examples were selected after searching the layers; their rank is exploratory.', '', '![Data with fitted templates](closest_examples.png)', '', '## Every layer, rather than selected pictures', '', '![All layer errors](all_layer_errors.png)', '',
        'The dashed line is 0.15. Falling below it is only a necessary condition: both directional RMS errors and both 95th-percentile errors must pass separately. Curves, surfaces and vertex models have different flexibility, so the lowest raw error alone cannot choose a unique manifold class. Original clouds have about ten times as many held-out observations as new individual clouds; this mechanically improves template coverage. Do not interpret a raw cross-cohort coverage advantage as a stronger scientific geometry. Controls match each cloud’s own point count.', '',
        '## What did the controls show?', '',
        f'We refitted the complete 32-candidate library to {len(controls)} control clouds for {len(grouped)} shortlisted real views. Each selected view gets three Gaussian clouds with matching covariance in expectation and three independently shuffled-coordinate clouds preserving its individual PCA-coordinate marginals. Point counts and fitting procedure are unchanged.', '',
        '| Shortlisted example | Candidate | Real combined error | Gaussian median | Shuffled-coordinate median |','|---|---|---:|---:|---:|']
    for r,f in exemplars:
        cs=grouped.get(r['entry']['id'],[])
        if not cs:continue
        vals={kind:[next(x['symmetric_rms'] for x in c['fits'] if x['name']==f['name']) for c in cs if c['kind']==kind] for kind in ('gaussian','permuted')}
        lines.append(f"| {ref(r,f['name'])} | {pretty(f['name'])} | {f['symmetric_rms']:.3f} | {np.median(vals['gaussian']):.3f} | {np.median(vals['permuted']):.3f} |")
    lines += ['', 'Lower real error than the controls indicates some joint geometric organization under this comparison. It does **not** establish that the named shape is a good fit. Three controls of each kind are insufficient for small p-values, and we have not corrected a significance test across all layers and candidates. Selection was based on the closest old/new example for every candidate plus any passing individual clouds. The controls are explicitly exploratory.', '',
        '## What about the two demonstration contexts?', '',
        'The second context is a useful repeat, but it does not turn correlated layers into independent experiments. For any passing candidate, require it to pass at the same layer in the other context before calling the resemblance context-stable. The tables and viewer expose both contexts.', '']
    if matches:
        lines+=['| Passing view | Candidate | Same candidate passes other context? |','|---|---|---|']
        for r in matches:
            e=r['entry'];other=None
            if e['context'] is not None:other=byname.get(e['id'].replace(f"ctx{e['context']}",f"ctx{1-e['context']}"))
            for n in r['screen_matches']:lines.append(f"| {ref(r,n)} | {pretty(n)} | "+('No second context' if other is None else ('Yes' if fitof(other,n)['screen_pass'] else 'No'))+' |')
    else:lines+=['There is no passing contextual individual-cloud candidate to replicate under the current screen.']
    lines += ['', '## Useful conclusion', '',
        'The useful outcome is a checked comparison, including negative results. Irregular digit groups, arcs and bands may still be real and scientifically useful even when they do not form a regular named solid. A whole-cloud negative fit does not rule out a shape inside one subgroup or a different coordinate subspace. We did not search arbitrary nonregular polyhedra, all possible aspect ratios, arbitrary deformations, or every subcluster.', '',
        'A freely fitted four-centroid model is included as a reference. Any four noncoplanar centroids define some tetrahedron; that alone is not a discovery. A regular tetrahedron instead requires equal edge lengths and appropriately occupied corners. We keep the regular constraint during fitting.', '',
        '## What this does not establish', '',
        f'- The three displayed PCs retain **{100*evr.min():.1f}%–{100*evr.max():.1f}%** of full hidden variance across these contextual individual clouds (median **{100*np.median(evr):.1f}%**). A good fit here would describe that projection.',
        '- Existing 2D scatterplots and kNN plots reuse these observations; they are not extra independent datasets. PH diagrams and Betti curves are summaries, not coordinates of the original point cloud, and we did not fit solids to those chart axes.',
        '- Shared-PCA overlays mix tasks, so separation can be caused by task offsets. We fitted all of them, but kept them outside the main count of individual-cloud findings.',
        '- Embedding-level positional clouds and constant clouds are flagged separately. They are not evidence of a number-manipulation mechanism.',
        '- Poor behavioral accuracy in some reversal/swap conditions limits interpretation: a prompt-conditioned shape is not proof that the model successfully carried out the transformation.',
        '- Shape, topology and mechanism are different claims. A tetrahedral surface and sphere share ideal topology; an open helix is a curve with no intrinsic closed loop. Existing H0/H1 alone cannot name a unique geometric object. No H2 computation or new model inference was performed.', '',
        '## Reproducible method', '',
        'Saved full-precision PCA coordinates are the input. A deterministic target-row split (seed 1729) assigns half the points to training and half to evaluation. The entire training half determines centering, RMS scale and an orthogonal PCA reorientation; at most 384 training points fit the shape pose. The pre-existing PCA coordinates themselves were computed using all targets, so this is held-out **shape fitting in a fixed descriptive projection**, not a fully held-out end-to-end benchmark. Original clouds have 10,000 targets; new individual clouds have 1,000. Overlays reuse 2,000 or 4,000 observations.', '',
        'The bounded similarity fit allows translation, rotation/reflection and one scale (0.25–3 in normalized units), with no independent axis stretching. It scores 56 initial orientations and refines the four best through up to 24 iterations of symmetric nearest-neighbor registration. No global optimum is guaranteed. Continuous templates use 384 deterministic samples; vertex templates use their actual vertices. Sampling causes a nonzero numerical floor, and raw error depends on template dimension and sampling density.', '',
        'The screening rule, fixed before examining the real-data outcomes, requires both data-to-template and template-to-held-out-data RMS errors ≤0.15, both 95th-percentile errors ≤0.30, and a fitted scale away from the bounds. This is a transparent descriptive threshold, not a learned posterior or statistical significance level. All per-candidate results, poses, train/test row IDs and source hashes are saved.', '',
        'Synthetic checks recover a noisy tetrahedron, sphere and helix; preserve a helix fit under rotation, reflection, translation and scale; reject Gaussian clouds as a tetrahedron/ring; and reject a short arc as a full circle. A detected missing-reflection initialization bug was corrected before this final version; all final records use the corrected fitter.', '',
        'Nearest-neighbor registration needs initialization and can find local rather than global optima; see the [Point Cloud Library registration documentation](https://pointclouds.org/documentation/group__registration.html). Ordinary Chamfer-style distances also have density limitations; see [Wu et al., NeurIPS 2021](https://papers.nips.cc/paper/2021/file/f3bd5ad57c8389a8a1a541a76be463bf-Paper.pdf). Those limitations motivate separate directional errors, held-out evaluation and explicit controls here.', '',
        'All computation was local using saved data. Previous experiment files and results were left unchanged. **No Modal credits or new inference were used.**', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    # Simple HTML companion keeps the report readable in the existing browser.
    from markdown_it import MarkdownIt
    body=MarkdownIt('commonmark').enable('table').render('\n'.join(lines))
    (OUT/'REPORT.html').write_text('<!doctype html><meta charset="utf-8"><title>Geometric shape findings</title><style>body{font:16px/1.6 system-ui;max-width:1100px;margin:36px auto;padding:0 24px;color:#223047}h1,h2{line-height:1.2}h2{margin-top:44px}a{color:#245ba4}img{max-width:100%;border:1px solid #ddd}table{border-collapse:collapse;font-size:14px;width:100%}th,td{border-bottom:1px solid #ddd;padding:10px;text-align:left}th{background:#f1f4f8}li{margin-bottom:9px}</style>'+body)
    shutil.copyfile(ROOT/'work/geometric_shapes/synthetic_validation.json',OUT/'synthetic_validation.json')
    print(json.dumps({k:v for k,v in result.items() if k not in ('best_by_shape','matches')},indent=2))

if __name__=='__main__':main()
