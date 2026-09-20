"""Read saved MSTs only: exact full-cloud H0 connectivity, no inference or PH reruns."""
from pathlib import Path
import json, csv, hashlib
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

ROOT = Path('/Users/mohammed.awad/Documents/Codex/2026-09-19/c')
SOURCE = ROOT/'outputs/ph_fullrange_overnight/results/results'
OUT = ROOT/'outputs/ph_analysis/h0_groups'
OUT.mkdir(parents=True, exist_ok=True)
TARGETS = np.arange(1, 10001)
LEAD = np.array([int(str(n)[0]) for n in TARGETS])
DIGITS = np.array([len(str(n)) for n in TARGETS])
MODELS = {'crystal': (32, 7), 'starcoderbase-3b': (36, 21), 'openllama-3b': (26, 8)}

def partition(edges, weights, radius, n=10000):
    use = weights <= radius
    graph = coo_matrix((np.ones(use.sum()), (edges[use,0], edges[use,1])), shape=(n,n))
    return connected_components(graph, directed=False)

def component_summary(ids):
    vals = TARGETS[ids]
    return {'size': len(ids), 'minimum': int(vals.min()), 'maximum': int(vals.max()),
            'leading_digit_counts': np.bincount(LEAD[ids], minlength=10)[1:].tolist(),
            'digit_length_counts': np.bincount(DIGITS[ids], minlength=6)[1:].tolist(),
            'contiguous_numeric_runs': int(1+np.count_nonzero(np.diff(vals)>1)),
            'targets_if_at_most_100': vals.tolist() if len(ids)<=100 else None}

def describe(edges, weights, radius, kind, requested=None, ami=False):
    count, labels = partition(edges, weights, radius)
    sizes = np.sort(np.bincount(labels))[::-1]
    record = {'kind': kind, 'requested': requested, 'radius': float(radius), 'components': count,
              'largest_component': int(sizes[0]), 'second_largest_component': int(sizes[1]) if count>1 else 0,
              'singleton_components': int(np.count_nonzero(sizes==1)),
              'components_at_least_100': int(np.count_nonzero(sizes>=100)),
              'mass_in_components_at_least_100': int(sizes[sizes>=100].sum()),
              'lead_ari': adjusted_rand_score(LEAD, labels), 'digits_ari': adjusted_rand_score(DIGITS, labels)}
    if ami:
        record.update(lead_ami=adjusted_mutual_info_score(LEAD, labels), digits_ami=adjusted_mutual_info_score(DIGITS,labels))
    return record, labels

def analyze(model, layer):
    folder = SOURCE/model/f'layer_{layer:02d}'
    data = np.load(folder/'h0.npz')
    assert np.array_equal(data['point_ids'], TARGETS-1)
    edges, raw = data['mst_edges'], data['mst_distances']
    scale = json.loads((folder/'normalization.json').read_text())['scale']
    weights = raw/scale
    assert edges.shape == (9999,2) and np.all(np.isfinite(weights)) and np.all(weights >= 0)
    order = np.argsort(weights, kind='stable'); sorted_weights=weights[order]
    assert np.allclose(np.sort(data['normalized'][:-1,1]),sorted_weights,rtol=1e-12,atol=1e-12)
    assert partition(edges,weights,float(weights.max()))[0]==1
    records=[]
    for radius in [.25,.5,.75,1.,1.25,1.5,2.]:
        item,_ = describe(edges,weights,radius,'fixed_radius'); records.append(item)
    for k in [2,3,5,9,10,20,50,100]:
        radius=sorted_weights[10000-k-1]
        item,labels=describe(edges,weights,radius,'fixed_component_count',k,ami=(k==9))
        # Ties can skip component counts: record actual count rather than breaking ties artificially.
        records.append(item)
        if k==9: k9=item; k9groups=[component_summary(np.flatnonzero(labels==i)) for i in np.argsort(np.bincount(labels))[::-1]]
    gaps=np.diff(sorted_weights)
    late=np.arange(10000-100-1,9998)
    gapindices={'widest_anywhere':int(np.argmax(gaps)), 'widest_with_2_to_100_components':int(late[np.argmax(gaps[late])])}
    plateaus={}
    for name,j in gapindices.items():
        lo,hi=sorted_weights[j:j+2]
        item,labels=describe(edges,weights,float(lo),'plateau',ami=True)
        item.update(start=float(lo),end=float(hi),width=float(hi-lo),
                    component_details=[component_summary(np.flatnonzero(labels==i)) for i in np.argsort(np.bincount(labels))[::-1]])
        plateaus[name]=item
    late_merges=[]
    parent=np.arange(10000);members={i:[i] for i in range(10000)}
    lead_counts=np.eye(9,dtype=np.int64)[LEAD-1]
    digit_counts=np.eye(5,dtype=np.int64)[DIGITS-1]
    label_pairs=np.array([np.sum(np.bincount(LEAD)*(np.bincount(LEAD)-1)/2),np.sum(np.bincount(DIGITS)*(np.bincount(DIGITS)-1)/2)])
    cluster_pairs=0.;agree_pairs=np.zeros(2);total_pairs=10000*9999/2
    curve=[]
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]; x=parent[x]
        return x
    for position,ix in enumerate(order):
        a,b=map(find,edges[ix]); assert a!=b
        if position>=9989:
            groups=sorted([component_summary(np.sort(members[a])),component_summary(np.sort(members[b]))],key=lambda x:x['size'],reverse=True)
            late_merges.append({'radius':float(weights[ix]),'edge_targets':(edges[ix]+1).tolist(),
                                'components_before':10000-position,'joining_components':groups})
        if len(members[a])<len(members[b]): a,b=b,a
        cluster_pairs+=len(members[a])*len(members[b])
        agree_pairs+=np.array([np.dot(lead_counts[a],lead_counts[b]),np.dot(digit_counts[a],digit_counts[b])])
        lead_counts[a]+=lead_counts[b];digit_counts[a]+=digit_counts[b]
        parent[b]=a;members[a].extend(members.pop(b))
        if position==9998 or sorted_weights[position+1]>weights[ix]:
            expected=cluster_pairs*label_pairs/total_pairs
            ari=(agree_pairs-expected)/(.5*(cluster_pairs+label_pairs)-expected)
            curve.append([weights[ix],9999-position,*ari])
    curve=np.array(curve)
    np.savez_compressed(OUT/f'{model}_layer_{layer:02d}_threshold_curve.npz',radius=curve[:,0],components=curve[:,1].astype(int),lead_ari=curve[:,2],digits_ari=curve[:,3])
    maxima={}
    for name,column in [('leading_digit',2),('digit_length',3)]:
        i=int(np.argmax(curve[:,column]));maximum,labels=describe(edges,weights,float(curve[i,0]),'maximum_ARI_over_all_distinct_radii',ami=True)
        assert np.isclose(maximum['lead_ari'],curve[i,2],atol=1e-10) and np.isclose(maximum['digits_ari'],curve[i,3],atol=1e-10)
        maximum['component_details']=[component_summary(np.flatnonzero(labels==j)) for j in np.argsort(np.bincount(labels))[::-1]][:20]
        maximum['threshold_bands']={}
        for threshold in [.5,.8,.95,.99]:
            eligible=curve[:-1,column]>=threshold
            start=np.flatnonzero(eligible & np.r_[True,~eligible[:-1]])
            end=np.flatnonzero(eligible & np.r_[~eligible[1:],True])+1
            bands=[{'start':float(curve[a,0]),'end':float(curve[b,0]),'width':float(curve[b,0]-curve[a,0])}for a,b in zip(start,end)]
            maximum['threshold_bands'][str(threshold)]={'number_of_intervals':len(bands),'total_width':sum(x['width']for x in bands),'widest':max(bands,key=lambda x:x['width'])if bands else None}
        maxima[name]=maximum
    numeric_jump=np.abs(TARGETS[edges[:,0]]-TARGETS[edges[:,1]])
    result={'model':model,'layer':layer,'relative_depth':layer/MODELS[model][0],
            'source_sha256':hashlib.sha256((folder/'h0.npz').read_bytes()).hexdigest(),
            'scale':scale,'h0_complete_connection_radius':float(weights.max()),
            'h0_merge_q50':float(np.quantile(weights,.5)), 'h0_merge_q95':float(np.quantile(weights,.95)),
            'h0_merge_q99':float(np.quantile(weights,.99)),
            'mst_same_leading_digit_fraction':float(np.mean(LEAD[edges[:,0]]==LEAD[edges[:,1]])),
            'mst_same_digit_length_fraction':float(np.mean(DIGITS[edges[:,0]]==DIGITS[edges[:,1]])),
            'mst_numeric_jump_median':float(np.median(numeric_jump)), 'mst_numeric_jump_mean':float(np.mean(numeric_jump)),
            'mst_consecutive_number_fraction':float(np.mean(numeric_jump==1)),
            'k9_lead_ari':k9['lead_ari'],'k9_digits_ari':k9['digits_ari'],
            'k9_components':k9['components'],'k9_lead_ami':k9['lead_ami'],'k9_digits_ami':k9['digits_ami'],
            'maximum_lead_ari':maxima['leading_digit']['lead_ari'],'maximum_digits_ari':maxima['digit_length']['digits_ari'],
            'all_threshold_maxima':maxima,'cuts':records,'plateaus':plateaus,'k9_groups':k9groups,'last_10_merges':late_merges}
    return result

def selfcheck():
    edges=np.array([[0,1],[1,2],[2,3]])
    weights=np.array([.1,.5,.8])
    assert partition(edges,weights,.5,n=4)[0]==2
    assert partition(edges,weights,.8,n=4)[0]==1
    assert partition(edges,weights,.09,n=4)[0]==4

def render_outputs(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names={'crystal':'Crystal','starcoderbase-3b':'StarCoderBase-3B','openllama-3b':'OpenLLaMA-3B'}
    colors={'crystal':'#20717d','starcoderbase-3b':'#b5572e','openllama-3b':'#7562ac'}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    metrics=[('maximum_lead_ari','Best leading-digit agreement across all radii','ARI (chance-adjusted)'),('maximum_digits_ari','Best digit-length agreement across all radii','ARI (chance-adjusted)'),('mst_same_leading_digit_fraction','MST edges joining the same leading digit','Fraction of 9,999 edges'),('mst_numeric_jump_median','Numeric gap across a typical MST edge','Median absolute number difference')]
    for ax,(key,title,ylabel) in zip(axs.ravel(),metrics):
        for m in MODELS:
            r=[x for x in rows if x['model']==m]
            ax.plot([x['relative_depth']for x in r],[x[key]for x in r],label=names[m],color=colors[m],lw=2)
        ax.set(title=title,xlabel='Relative layer depth',ylabel=ylabel);ax.grid(alpha=.2)
        if key.endswith('_median'):ax.set_yscale('log')
        else:ax.set_ylim(-.025,1.025)
    axs[0,0].legend(frameon=False);axs[1,0].axhline(.1110222222,ls=':',c='gray',label='Uniform-pair baseline')
    fig.suptitle('H0 connectivity changes from digit-length to leading-digit organization\nMaxima over radius are exploratory; all 10,000 targets remain included.',fontsize=14)
    fig.savefig(OUT/'h0_structure_by_depth.png');fig.savefig(OUT/'h0_structure_by_depth.svg');plt.close(fig)
    fig,axs=plt.subplots(3,3,figsize=(13,9),sharey=True,layout='constrained')
    for i,(m,(last,best)) in enumerate(MODELS.items()):
        for j,(l,role) in enumerate([(1,'Best 2D variance'),(best,'Best PC1 rho'),(last,'Final layer')]):
            c=np.load(OUT/f'{m}_layer_{l:02d}_threshold_curve.npz');ax=axs[i,j]
            ax.step(c['radius'],c['lead_ari'],where='post',label='Leading digit',c='#237685')
            ax.step(c['radius'],c['digits_ari'],where='post',label='Digit length',c='#c07338')
            ax.set(title=f'{names[m]} L{l} | {role}',xlabel='Radius / median pair distance',ylim=(-.05,1.05));ax.grid(alpha=.18)
            if j==0:ax.set_ylabel('Component-label ARI')
    axs[0,0].legend(frameon=False)
    fig.suptitle('What do connected components represent at different distance thresholds?',fontsize=15)
    fig.savefig(OUT/'h0_selected_threshold_profiles.png');fig.savefig(OUT/'h0_selected_threshold_profiles.svg');plt.close(fig)
    def fmt(value):return f'{value:.3f}'
    lines=['# H0 connectivity: what the components actually contain','',
           '**Scope:** all 94 non-embedding layers, all 10,000 targets in each layer. This analysis only reads saved exact Euclidean minimum-spanning trees (MSTs); it does not rerun PH or model inference.','',
           '## What we looked at','',
           'At each distance threshold, points linked by a chain of edges belong to the same connected component. Thresholding the exact MST gives exactly the same components as thresholding the complete Euclidean distance graph. We checked every distinct MST edge distance, not just selected screenshots. We compared component membership to leading digit and digit length using adjusted Rand index (ARI): 1 means identical partitions, about 0 means chance-level agreement under the fixed-size random-partition baseline. Here, the widest plateau means the widest interval between successive merge distances, excluding the initial all-isolated regime and the final permanently connected regime.','',
           '## Main findings','',
           '1. **All three models develop leading-digit components late in the network.** At the final layers, the largest nine components at the best-agreement threshold contain nearly pure leading-digit groups. The leftover points are retained as smaller components, not discarded. Crystal and StarCoder produce cleaner and more threshold-stable separation than OpenLLaMA in this dataset.','',
           '| Final layer | Best leading-digit ARI | Targets in largest 9 components | Leading-digit purity within those 9 cores | Total radius width with ARI ≥ 0.8 |','|---|---:|---:|---:|---:|']
    for m,(last,best) in MODELS.items():
        r=next(x for x in rows if x['model']==m and x['layer']==last);q=r['all_threshold_maxima']['leading_digit'];cores=q['component_details'][:9];mass=sum(c['size']for c in cores);purity=sum(max(c['leading_digit_counts'])for c in cores)/mass
        lines.append(f'| {names[m]} L{last} | {q["lead_ari"]:.4f} | {mass:,}/10,000 | {100*purity:.3f}% | {q["threshold_bands"]["0.8"]["total_width"]:.5f} |')
    lines+=['','2. **Best numerical ordering is not the same as strongest leading-digit component separation.** Crystal and OpenLLaMA have very little leading-digit component agreement at their best-ρ layers, even when the threshold is optimized. StarCoder is already partly separated at its best-ρ layer. This is a distinction between two measured properties, not evidence that numerical understanding is lost later.','',
            '| Best-ρ selection | Maximum leading-digit ARI | Maximum digit-length ARI | Median numeric jump across MST edges |','|---|---:|---:|---:|']
    for m,(last,best) in MODELS.items():
        r=next(x for x in rows if x['model']==m and x['layer']==best)
        lines.append(f'| {names[m]} L{best} | {r["maximum_lead_ari"]:.4f} | {r["maximum_digits_ari"]:.4f} | {r["mst_numeric_jump_median"]:g} |')
    lines+=['','3. **StarCoder’s first layer has an especially clear digit-length split.** Its widest H0 plateau runs from normalized radius 1.050464 to 1.544184. It has exactly five components: 1–9, 10–99, 100–999, 1000–9999, and 10000. Digit-length ARI is exactly 1. This is strong descriptive surface-form alignment, but it does not establish that tokenization caused it. Neither other model has this same five-way partition at layer 1.','',
            '4. **The last surviving H0 component is always a tiny group.** In all 94 layers, the final merge joins a group containing at most nine numbers to the rest. In 90/94 layers, the widest H0 plateau already has at least 9,900 numbers in one giant component. Therefore, the longest H0 bar or final connection distance is often about an outlier, not two substantial populations. Crystal L7 and OpenLLaMA L8 both leave the number 1 until the final merge. At the final layers, the last detached points are Crystal: 213; StarCoder: 1288; OpenLLaMA: 1223.','',
            '5. **Forcing exactly nine total components is a misleading diagnostic here.** Its leading-digit ARI is essentially zero in all 94 layers. That does not mean there are no leading-digit groups: by the time only nine total components remain, the large digit groups have already merged while a few outliers remain. At earlier thresholds, nine large digit groups coexist with many small residual components.','',
            '6. **The change happens at different depths.** The first layer whose best-across-radius leading-digit ARI exceeds 0.8 is Crystal L26/32 (81.25% depth), StarCoder L22/36 (61.11%), and OpenLLaMA L19/26 (73.08%). These are descriptive thresholds picked for comparison, not estimated change points or significance results. Crystal exceeds 0.95 from L27; StarCoder first does so at L23; OpenLLaMA never reaches 0.95.','',
            '7. **Tree edges become more numerically local, but this does not prove a straight number line.** At final layers, median absolute numeric gaps across MST edges are 40, 61, and 51 for Crystal, StarCoder and OpenLLaMA; at layer 1 they are 2000, 2022 and 2623. Nearly all final-layer tree edges share a leading digit (99.88%, 99.92%, 99.74%). The uniform random-pair reference for same leading digit is 11.10%, but this is only a descriptive baseline—not a null-model significance test. A tree construction itself cannot prove the original cloud is tree-shaped.','',
            '## Was this useful?','',
            '- **Useful:** H0 adds actual component membership to the visual PH diagrams, confirms late leading-digit structure, identifies StarCoder’s early digit-length split, and exposes outlier-driven long bars.','- **Useful but limited:** maximum ARI gives a systematic comparison after inspecting all distance scales; the accompanying threshold widths show whether a high maximum occupies a broad or narrow interval. These maxima are exploratory because the threshold was chosen using labels.','- **Unhelpful by itself:** largest H0 lifetime, exactly-nine-component cuts, and purity without coverage can give the wrong impression. We provide coverage and chance-adjusted agreement to prevent those mistakes.','- **Not answered:** which points create H1 loops, whether any cloud is a donut, or whether tokenization caused the structures. H0 component membership cannot answer those questions.','',
            '## All-layer outputs','',
            '- [All 94 layers, scalar summary](layer_summary.csv)','- [All layers with component memberships, late merges and thresholds](all_layers_h0_groups.json)','- [Depth profiles](h0_structure_by_depth.png)','- [Selected-layer threshold profiles](h0_selected_threshold_profiles.png)','- Each `<model>_layer_NN_threshold_curve.npz` contains exact radii, component counts, leading-digit ARI and digit-length ARI across all distinct merge distances.','',
            '## Reliability checks and interpretation limits','',
            'MSTs have 9,999 edges and all 10,000 canonical target IDs. Sorted MST distances were checked against saved H0 deaths. The maximum threshold yields one component. An independent sparse-graph component calculation agrees with the incremental all-threshold ARI calculation at every selected maximum. A four-point self-check covers threshold inclusion and connectivity. Tied distances are included together; no artificial tie breaking is used to create a requested component count.','',
            'All thresholds use each layer’s saved median-pair-distance normalization. H0 comes from float64 Euclidean distances. Label associations are observational; leading digit, digit length and numerical magnitude are related. Adjacent layers are not independent repetitions. The 94 layers are one model set and one saved prompt seed. These results neither establish generalization nor justify causal tokenizer claims.']
    (OUT/'H0_FINDINGS.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    selfcheck()
    allrows=[]
    for model,(last,best) in MODELS.items():
        for layer in range(1,last+1):
            row=analyze(model,layer);allrows.append(row)
            if layer in [1,best,last]:
                p=row['plateaus']['widest_anywhere']
                print(model,layer,'k9 ARI',round(row['k9_lead_ari'],4),'widest plateau',round(p['width'],3),'components',p['components'],'top sizes',[x['size'] for x in p['component_details'][:5]],'final smaller',row['last_10_merges'][-1]['joining_components'][1],flush=True)
    payload={'method':'Threshold the saved exact Euclidean MST: connected components equal those of the complete Euclidean Rips graph at every radius. No PH rerun.',
             'normalization':'Each radius divided by median of the same 100000 seeded off-diagonal pair draws for its layer.',
             'baseline':{'random_pair_same_lead':float(sum(n*(n-1) for n in np.bincount(LEAD))/ (10000*9999)),
                         'random_pair_same_digit_length':float(sum(n*(n-1) for n in np.bincount(DIGITS))/(10000*9999)),
                         'random_pair_mean_absolute_numeric_difference':10001/3,
                         'random_pair_consecutive_probability':2/10000,
                         'adjusted_rand_index':'Chance-adjusted partition agreement; 0 expected for random partitions with fixed sizes, 1 identical.'},
             'cautions':['MST edges depend on tie choices; all threshold partitions include ties and are invariant to the chosen valid MST.',
                         'A chosen k=9 cut is a diagnostic, not an unsupervised discovery of nine clusters. Actual k recorded when tied radii skip a count.',
                         'Largest gap is selected post hoc from the H0 curve; it is not a significance test.',
                         'Leading digit, digit length and numeric value are related; these associations cannot establish a causal tokenizer explanation.',
                         'Uniform random-pair edge comparisons are descriptive baselines, not fitted null models or p-values.'],
             'layers':allrows}
    (OUT/'all_layers_h0_groups.json').write_text(json.dumps(payload,indent=2))
    flat=[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in allrows]
    with (OUT/'layer_summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    render_outputs(allrows)
    print('COMPLETE',len(allrows),flush=True)
