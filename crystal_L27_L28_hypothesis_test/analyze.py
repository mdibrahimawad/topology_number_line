"""Saved-vector analysis only. No model imports, inference, or hidden-state generation.

Run from the project root: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
MPLCONFIGDIR=/private/tmp/crystal-mpl .venv/bin/python crystal_L27_L28_hypothesis_test/analyze.py
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/crystal-mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
RUN = ROOT / 'results/main_final/main/llm360-crystal/run_00'
sys.path.insert(0, str(OUT / 'python_deps'))
import numpy as np
import pandas as pd
import scipy
from scipy.linalg import subspace_angles, orthogonal_procrustes
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr, pearsonr, rankdata
import sklearn
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, silhouette_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import tokenizers
from tokenizers import Tokenizer

LOG = (OUT / 'analysis_details.txt').open('w')
def emit(*values):
    message = ' '.join(str(x) for x in values)
    print(message, flush=True)
    LOG.write(message + '\n')
    LOG.flush()

def save(name, rows, columns=None):
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows, columns=columns)
    df.to_csv(OUT / name, index=False, float_format='%.12g')
    return df

def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, default=lambda v: v.item() if isinstance(v, np.generic) else v.tolist()) + '\n')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def correlations(a, b):
    return {'pearson': float(pearsonr(a, b).statistic), 'spearman': float(spearmanr(a, b).statistic)}

def describe(x):
    return {k: float(f(x)) for k, f in [('mean', np.mean), ('median', np.median), ('min', np.min), ('max', np.max)]}

data = np.load(RUN / 'hidden_states.npz', allow_pickle=False)
records = json.loads((RUN / 'prompts.json').read_text())
emit('TASK 1: NPZ keys:', data.files)
assert isinstance(records, list), 'prompts.json must contain a record list'
for key, field in [('point_ids', 'point_id'), ('targets', 'target'), ('group_ids', 'group')]:
    assert np.array_equal(data[key], np.array([r[field] for r in records])), f'STOP: metadata mismatch in {key}'
H, targets, groups, ids = (data[k] for k in ['hidden_states', 'targets', 'group_ids', 'point_ids'])
n = len(ids)
assert H.shape == (33, 120, 4096), f'Unexpected hidden shape: {H.shape}'
assert len(set(ids.tolist())) == n and np.isfinite(H[[27, 28]]).all()
assert np.all(targets == np.floor(targets)) and np.all(targets > 0)
targets = targets.astype(int)
assert all(r['prompt'].endswith(f",{int(r['target'])}=") for r in records)
emit('Shape:', H.shape, 'dtype:', H.dtype, '; points:', n, '; exact positional metadata alignment: PASS')
group_ranges = []
for g in sorted(set(groups)):
    t = targets[groups == g]
    group_ranges.append({'group': f'G{g}', 'n': len(t), 'min': int(t.min()), 'max': int(t.max()), 'unique_targets': len(set(t))})
emit('Group ranges:', group_ranges)
emit('First five records:', json.dumps(records[:5], indent=2))
meta = pd.DataFrame(records)
meta['target'] = targets
meta['group'] = [f'G{g}' for g in groups]
meta['leading_decimal_digit'] = [str(t)[0] for t in targets]
centers = {1: 10, 2: 100, 3: 1000, 4: 10000}
meta['boundary_side'] = ['below' if t < centers[g] else 'above' if t > centers[g] else 'exact' for t, g in zip(targets, groups)]
save('verified_prompt_metadata.csv', meta)
diag = json.loads((RUN / 'extraction_diagnostics.json').read_text())
provenance = {'input_directory': str(RUN), 'input_sha256': {f: digest(RUN/f) for f in ['hidden_states.npz', 'prompts.json', 'extraction_diagnostics.json', 'pca_layer_metrics.csv']},
              'shape': list(H.shape), 'dtype': str(H.dtype), 'metadata_exact_positional_match': True,
              'groups': group_ranges, 'versions': {m.__name__: m.__version__ for m in [np, scipy, sklearn, pd, matplotlib, tokenizers]},
              'layer_indexing': 'Direct saved indices 27 and 28; index 0 is embedding output.',
              'pca': 'PCA(n_components=2, svd_solver=full) on unscaled saved float32 X; deterministic full SVD. Each PC1 sign oriented independently to nonnegative Spearman.',
              'knn': 'NearestNeighbors(n_neighbors=4, metric=euclidean).fit(X).kneighbors() with X omitted; four non-self neighbors. Original float32 input, no scaling.',
              'original_configuration': diag['model'], 'original_prompt_method': diag['prompt_method']}

# This is the actual fast-tokenizer backend used by the inspected wrapper.
# The wrapper defaults add_bos_token=False/add_eos_token=False and its postprocessor
# is exactly the identity template already saved in tokenizer.json.
tokdir = OUT / 'tokenizer'
config = json.loads((tokdir / 'tokenizer_config.json').read_text())
backend_config = json.loads((tokdir / 'tokenizer.json').read_text())
assert not config.get('add_bos_token', False) and not config.get('add_eos_token', False)
assert backend_config['post_processor']['single'] == [{'Sequence': {'id': 'A', 'type_id': 0}}]
assert not diag['prompt_method']['prepend_bos'] and diag['tokenizer_class'] == 'CrystalCoderTokenizerFast'
assert (tokdir / 'original_auto_map_wrapper.py').read_bytes() == (tokdir / 'tokenization_crystalcoder_fast.py').read_bytes()
tok = Tokenizer.from_file(str(tokdir / 'tokenizer.json'))
token_rows = []
for i, r in enumerate(records):
    prompt, text = r['prompt'], str(targets[i])
    prefix = tok.encode(prompt, add_special_tokens=True)
    complete = tok.encode(prompt + text, add_special_tokens=True)
    assert complete.ids[:len(prefix.ids)] == prefix.ids, f'Continuation prefix mismatch: point {ids[i]}'
    assert prefix.ids[-1] == 29922 and tok.decode([prefix.ids[-1]]) == '='
    cont = complete.ids[len(prefix.ids):]
    assert cont and tok.decode(cont) == text
    isolated = tok.encode(text, add_special_tokens=True).ids
    token_rows.append({'point_id': int(ids[i]), 'target': int(targets[i]), 'group': f'G{groups[i]}', 'prompt': prompt,
                       'target_text': text, 'first_decimal_digit': text[0], 'first_completion_token_id': cont[0],
                       'first_completion_token_text': tok.decode([cont[0]]), 'number_of_completion_tokens': len(cont),
                       'full_completion_token_ids': json.dumps(cont), 'full_completion_token_texts': json.dumps([tok.decode([x]) for x in cont]),
                       'full_completion_token_pieces': json.dumps(complete.tokens[len(prefix.ids):]),
                       'prefix_alignment_valid': True, 'prompt_token_ids': json.dumps(prefix.ids),
                       'isolated_target_token_ids': json.dumps(isolated), 'isolated_matches_contextual': isolated == cont})
token_df = save('crystal_target_tokenization.csv', token_rows)
for sample in diag['diagnostic_samples']:
    actual = tok.encode(sample['prompt']).ids[-1]
    assert actual == sample['last_token_id'], 'Saved tokenizer diagnostic mismatch'
provenance['tokenizer'] = {'repository': json.loads((tokdir/'repository_metadata.json').read_text())['id'],
    'revision': json.loads((tokdir/'repository_metadata.json').read_text())['sha'],
    'auto_map': config['auto_map'], 'wrapper_revision': json.loads((tokdir/'wrapper_repository_metadata.json').read_text())['sha'],
    'sha256': {f.name: digest(f) for f in tokdir.iterdir() if f.is_file()},
    'implementation': 'tokenizers.Tokenizer backend; inspected CrystalCoderTokenizerFast wrapper has no extra text preprocessing and adds neither BOS nor EOS. Backend identity postprocessor verified.',
    'historical_limit': 'Experiment revision is null; tokenizer hash was not archived. Retrieved pinned assets match the recorded class and all 12 saved final-token diagnostics, but historical byte identity cannot be proven.',
    'prefix_alignment_passes': n, 'saved_diagnostic_passes': len(diag['diagnostic_samples']),
    'isolated_contextual_matches': int(token_df.isolated_matches_contextual.sum())}
meta['first_completion_token_id'] = token_df.first_completion_token_id
meta['number_of_completion_tokens'] = token_df.number_of_completion_tokens
emit('TASK 6: contextual tokenizer test:', token_df[['point_id','target','first_completion_token_id','full_completion_token_ids']].to_string(index=False))
emit('First token mapping:', token_df[['first_decimal_digit','first_completion_token_id','first_completion_token_text']].drop_duplicates().sort_values('first_decimal_digit').to_dict('records'))

states, metrics, branches, feature_metrics, contingencies, connectivity = {}, [], [], [], [], []
color = {1: '#4469a1', 2: '#db9130', 3: '#44946e', 4: '#b15482'}
original_metrics = pd.read_csv(RUN / 'pca_layer_metrics.csv').set_index('layer')

def edge_record(L, i, j, D, z):
    hd, low = float(D[i,j]), float(np.linalg.norm(z[i].astype(float)-z[j].astype(float)))
    return {'layer': L, 'source_point_id': int(ids[i]), 'source_target': int(targets[i]), 'source_group': f'G{groups[i]}',
            'destination_point_id': int(ids[j]), 'destination_target': int(targets[j]), 'destination_group': f'G{groups[j]}',
            'euclidean_distance_4096d': hd, 'euclidean_distance_PCA2': low,
            'source_leading_digit': str(targets[i])[0], 'destination_leading_digit': str(targets[j])[0],
            'PCA2_to_4096d_distance_ratio': low/hd, 'distance_4096d_to_PCA2_ratio': hd/low if low else np.nan,
            'group_pair': f'G{min(groups[i],groups[j])}-G{max(groups[i],groups[j])}'}

for L in [27, 28]:
    X = H[L]
    pca = PCA(n_components=2, svd_solver='full')
    z = pca.fit_transform(X)
    if spearmanr(targets, z[:,0]).statistic < 0:
        z[:,0] *= -1
        pca.components_[0] *= -1
    row = {'layer': L, 'explained_variance_PC1': pca.explained_variance_ratio_[0], 'explained_variance_PC2': pca.explained_variance_ratio_[1],
           'Spearman_target_PC1': spearmanr(targets,z[:,0]).statistic, 'Spearman_target_PC2': spearmanr(targets,z[:,1]).statistic,
           'Pearson_target_PC1': pearsonr(targets,z[:,0]).statistic, 'Pearson_target_PC2': pearsonr(targets,z[:,1]).statistic,
           'Pearson_log10target_PC1': pearsonr(np.log10(targets),z[:,0]).statistic, 'Pearson_log10target_PC2': pearsonr(np.log10(targets),z[:,1]).statistic,
           'archived_rho_abs': original_metrics.loc[L,'rho_abs'], 'archived_ev_pc1': original_metrics.loc[L,'ev_pc1']}
    assert abs(row['Spearman_target_PC1'] - row['archived_rho_abs']) < 1e-10
    assert abs(row['explained_variance_PC1'] - row['archived_ev_pc1']) < 2e-6
    metrics.append(row)
    nodes = meta.copy()
    nodes.insert(0,'layer',L)
    nodes['PC1'], nodes['PC2'] = z[:,0], z[:,1]
    nodes['numerical_rank'], nodes['PC1_rank'] = rankdata(targets), rankdata(z[:,0])
    save(f'crystal_L{L}_nodes.csv',nodes)
    emit(f'TASK 2: L{L} PCA metrics:', row)
    emit(f'TASK 5, BEFORE CLUSTERING: L{L} raw coordinates:')
    emit(nodes[['point_id','target','group','PC1','PC2']].sort_values(['group','target','point_id']).to_string(index=False))

    nn = NearestNeighbors(n_neighbors=4, metric='euclidean').fit(X)
    distances, neighbors = nn.kneighbors()
    D = squareform(pdist(X.astype(np.float64), metric='euclidean'))
    # Independent direct float64 distances verify graph identity and float32 distances.
    Dc = D.copy(); np.fill_diagonal(Dc, np.inf)
    direct_neighbors = np.argsort(Dc, axis=1)[:,:4]
    assert np.array_equal(neighbors, direct_neighbors), f'Float precision changes neighbor ranks at L{L}'
    assert np.allclose(distances, np.take_along_axis(D,neighbors,axis=1),rtol=2e-6,atol=1e-5)
    assert all(i not in neighbors[i] and len(set(neighbors[i])) == 4 for i in range(n))
    edge_set = {tuple(sorted((i,int(j)))) for i in range(n) for j in neighbors[i]}
    directed = []
    for i in range(n):
        for rank, j in enumerate(neighbors[i], 1):
            r = edge_record(L,i,j,D,z)
            r.update(neighbor_rank=rank, euclidean_distance_4096d=float(distances[i,rank-1]))
            directed.append(r)
    assert len(directed) == 480
    save(f'crystal_L{L}_knn_directed.csv',directed)
    undirected = []
    for i,j in sorted(edge_set):
        r = edge_record(L,i,j,D,z)
        r.update(source_to_destination=bool(j in neighbors[i]), destination_to_source=bool(i in neighbors[j]),
                 source_neighbor_rank=int(np.where(neighbors[i]==j)[0][0]+1) if j in neighbors[i] else None,
                 destination_neighbor_rank=int(np.where(neighbors[j]==i)[0][0]+1) if i in neighbors[j] else None)
        undirected.append(r)
    edges = save(f'crystal_L{L}_knn_undirected.csv',undirected)
    cross = save(f'crystal_L{L}_cross_group_edges.csv',edges[edges.source_group != edges.destination_group])
    adjacency = np.zeros((n,n),dtype=bool)
    for i,j in edge_set: adjacency[i,j] = adjacency[j,i] = True
    communicators = []
    for i in range(n):
        js = np.flatnonzero(adjacency[i] & (groups != groups[i]))
        if len(js):
            communicators.append({'layer': L,'point_id':int(ids[i]),'target':int(targets[i]),'group':f'G{groups[i]}',
                'leading_digit':str(targets[i])[0],'cross_group_degree':len(js),
                'exact_connected_nodes':json.dumps([{'point_id':int(ids[j]),'target':int(targets[j]),'group':f'G{groups[j]}'} for j in js])})
    comm = pd.DataFrame(communicators).sort_values(['cross_group_degree','point_id'],ascending=[False,True])
    save(f'crystal_L{L}_communicators.csv',comm)
    emit(f'TASK 4: L{L} all cross-group edges (point IDs disambiguate repeat targets):')
    for a,b in itertools.combinations(range(1,5),2):
        pair = cross[cross.group_pair == f'G{a}-G{b}']
        emit(f'G{a} <-> G{b}: {len(pair)} edges'); emit(pair.to_string(index=False))
    emit(f'L{L} communicators ranked by undirected cross-group degree:'); emit(comm.to_string(index=False))

    cluster_ids = np.full(n,-1)
    for g in [2,3,4]:
        idx = np.flatnonzero(groups==g)
        km = KMeans(n_clusters=2,random_state=0,n_init=50).fit(z[idx])
        labels = km.labels_
        # Canonical labels: branch 0 has lower mean numerical target, solely for readability.
        if targets[idx][labels==0].mean() > targets[idx][labels==1].mean(): labels = 1-labels
        cluster_ids[idx] = labels
        for c in [0,1]:
            members = sorted(idx[labels==c],key=lambda i:(targets[i],ids[i]))
            branches.append({'layer':L,'group':f'G{g}','cluster_id':c,'n':len(members),
                'targets':json.dumps(targets[members].tolist()),'point_ids':json.dumps(ids[members].tolist()),
                'target_point_id_pairs':json.dumps([[int(targets[i]),int(ids[i])] for i in members])})
            emit(f'L{L} G{g} KMeans branch {c}:',[(int(targets[i]),int(ids[i])) for i in members])
        A = adjacency[np.ix_(idx,idx)]
        nc, component = connected_components(A,directed=False)
        directed_A = np.zeros((n,n),dtype=bool)
        for i in range(n): directed_A[i,neighbors[i]] = True
        ns, _ = connected_components(directed_A[np.ix_(idx,idx)],directed=True,connection='strong')
        local = {'layer':L,'group':f'G{g}','weak_components':int(nc),'strong_components':int(ns),
                 'within_group_edges':int(A.sum()//2),'edges_between_kmeans_branches':int(np.sum(A & (labels[:,None]!=labels[None,:]))//2),
                 'silhouette_PCA2':float(silhouette_score(z[idx],labels)),
                 'silhouette_4096d_same_labels':float(silhouette_score(X[idx].astype(float),labels)),
                 'component_targets':json.dumps([sorted(targets[idx][component==c].tolist()) for c in range(nc)]),
                 'component_point_ids':json.dumps([ids[idx][component==c].tolist() for c in range(nc)])}
        connectivity.append(local); emit('Within-group induced graph:',local)
        for feature in ['leading_decimal_digit','boundary_side','first_completion_token_id','number_of_completion_tokens']:
            values = meta.iloc[idx][feature].astype(str).to_numpy()
            ct = pd.crosstab(pd.Series(labels,name='cluster_id'),pd.Series(values,name=feature))
            # Standard cluster purity asks cluster -> feature. Reverse majority mapping
            # is the explicitly requested feature -> cluster descriptive accuracy.
            purity = float(ct.max(axis=1).sum()/len(idx))
            accuracy = float(ct.max(axis=0).sum()/len(idx))
            feature_metrics.append({'layer':L,'group':f'G{g}','feature':feature,'cluster_purity':purity,
                'adjusted_mutual_information':adjusted_mutual_info_score(labels,values),
                'feature_to_cluster_majority_accuracy':accuracy,'majority_cluster_baseline':np.bincount(labels).max()/len(idx)})
            for c in ct.index:
                for value in ct.columns:
                    contingencies.append({'layer':L,'group':f'G{g}','feature':feature,'cluster_id':c,'feature_value':value,'count':int(ct.loc[c,value])})
            emit(f'L{L} G{g} cluster vs {feature}:\n{ct.to_string()}')
    nodes['KMeans_branch_within_group'] = np.where(cluster_ids<0,np.nan,cluster_ids)
    save(f'crystal_L{L}_branch_assignments.csv',nodes)
    states[L] = {'pca':pca,'z':z,'neighbors':neighbors,'edges':edge_set,'adjacency':adjacency,'D':D,
                 'nodes':nodes,'cross':cross,'communicators':comm,'clusters':cluster_ids}

    # Group zooms place every label on a spaced rail with a leader to its point.
    for g in [None,1,2,3,4]:
        fig,ax = plt.subplots(figsize=(14,10) if g is None else (12,9))
        idx = np.arange(n) if g is None else np.flatnonzero(groups==g)
        ax.add_collection(LineCollection([(z[i],z[j]) for i,j in edge_set if i in idx and j in idx],colors='#939ba8',linewidths=0.8,alpha=0.45))
        for gg in sorted(set(groups[idx])):
            js = idx[groups[idx]==gg]
            if g in [2,3,4]:
                for c,marker in [(0,'o'),(1,'^')]:
                    jj = js[cluster_ids[js]==c]
                    ax.scatter(*z[jj].T,s=48,c=color[gg],marker=marker,label=f'Branch {c}',zorder=3)
            else: ax.scatter(*z[js].T,s=35,c=color[gg],label=f'G{gg}',zorder=3)
        if g is None:
            for i in idx: ax.annotate(f'{targets[i]} [{ids[i]}]',z[i],xytext=(3,3),textcoords='offset points',fontsize=6,alpha=0.85)
        else:
            xmin,ymin = z[idx].min(axis=0); xmax,ymax = z[idx].max(axis=0)
            dx,dy = max(xmax-xmin,1),max(ymax-ymin,1)
            ordered = idx[np.argsort(z[idx,0])]
            for side, js in [(-1,ordered[:len(idx)//2]),(1,ordered[len(idx)//2:])]:
                js = js[np.argsort(z[js,1])]
                xpos = xmin-0.19*dx if side<0 else xmax+0.19*dx
                for i,ypos in zip(js,np.linspace(ymin-0.06*dy,ymax+0.06*dy,len(js))):
                    ax.annotate(f'{targets[i]} [id {ids[i]}]',xy=z[i],xytext=(xpos,ypos),fontsize=9,
                        ha='right' if side<0 else 'left',va='center',arrowprops={'arrowstyle':'-','color':'#a6a6a6','lw':0.6},zorder=4)
            ax.set_xlim(xmin-0.55*dx,xmax+0.55*dx);ax.set_ylim(ymin-0.15*dy,ymax+0.15*dy)
        ax.set_xlabel('PC1 (oriented to positive Spearman with target)')
        ax.set_ylabel('PC2 (SVD sign; arbitrary)')
        suffix = 'all_groups' if g is None else f'G{g}'
        ax.set_title(f'Crystal run_00 · L{L} · {suffix.replace("_"," ")}\nPCA display; edges are 4-NN in 4096-D · labels: target [point ID]')
        ax.grid(alpha=.15);ax.legend(loc='best');fig.tight_layout()
        fig.savefig(OUT/f'crystal_L{L}_PCA_{suffix}.png',dpi=160)
        fig.savefig(OUT/f'crystal_L{L}_PCA_{suffix}.svg')
        plt.close(fig)

save('crystal_pca_metrics.csv',metrics)
save('crystal_branch_membership.csv',branches)
save('crystal_branch_feature_metrics.csv',feature_metrics)
save('crystal_branch_contingency_tables.csv',contingencies)
save('crystal_within_group_connectivity.csv',connectivity)
emit('Feature metrics:',pd.DataFrame(feature_metrics).to_string(index=False))

# Exact rank calculations retain all repeated-target observations.
a,b = states[27],states[28]
r27,r28,rt = rankdata(a['z'][:,0]),rankdata(b['z'][:,0]),rankdata(targets)
rank_rows = meta[['point_id','target','group']].copy()
rank_rows['PC1_L27'],rank_rows['PC1_rank_L27'] = a['z'][:,0],r27
rank_rows['PC1_L28'],rank_rows['PC1_rank_L28'] = b['z'][:,0],r28
rank_rows['rank_change'] = r28-r27
rank_rows['absolute_rank_change'] = abs(r28-r27)
rank_rows['numerical_rank'] = rt
# Exact covariance decomposition works with target ties (unlike 1-6 sum d²/n(n²-1)).
denom = np.sqrt(np.sum((rt-rt.mean())**2)*np.sum((r27-r27.mean())**2))
assert np.array_equal(np.sort(r27),np.sort(r28))
rank_rows['contribution_to_rho_drop'] = (rt-rt.mean())*(r27-r28)/denom
assert np.isclose(rank_rows.contribution_to_rho_drop.sum(),metrics[0]['Spearman_target_PC1']-metrics[1]['Spearman_target_PC1'])
save('crystal_L27_L28_pc1_rank_changes.csv',rank_rows)
emit('TASK 7: all point rank changes:\n',rank_rows.to_string(index=False))
top30 = rank_rows.sort_values(['absolute_rank_change','point_id'],ascending=[False,True]).head(30)
save('crystal_L27_L28_top30_rank_changes.csv',top30)
emit('Top 30 absolute PC1 rank changes:\n',top30.to_string(index=False))
flips, pair_summary = [], []
pair_counts = {}
ties = {'equal_target_pairs':0,'PC1_ties_L27':0,'PC1_ties_L28':0}
for i,j in itertools.combinations(range(n),2):
    pair = f'G{min(groups[i],groups[j])}-G{max(groups[i],groups[j])}'
    count = pair_counts.setdefault(pair,{'group_pair':pair,'total_pairs':0,'flips':0,'equal_target_pairs':0,'equal_target_flips':0,'concordant_to_discordant':0,'discordant_to_concordant':0})
    count['total_pairs'] += 1
    equal = targets[i]==targets[j]
    ties['equal_target_pairs'] += int(equal);count['equal_target_pairs'] += int(equal)
    d27,d28 = r27[i]-r27[j],r28[i]-r28[j]
    ties['PC1_ties_L27'] += int(d27==0);ties['PC1_ties_L28'] += int(d28==0)
    if d27*d28 < 0:
        count['flips'] += 1;count['equal_target_flips'] += int(equal)
        c27,c28 = int(np.sign((targets[i]-targets[j])*d27)),int(np.sign((targets[i]-targets[j])*d28))
        if c27==1 and c28==-1: count['concordant_to_discordant']+=1
        if c27==-1 and c28==1: count['discordant_to_concordant']+=1
        flips.append({'point_id_A':int(ids[i]),'target_A':int(targets[i]),'group_A':f'G{groups[i]}',
                      'point_id_B':int(ids[j]),'target_B':int(targets[j]),'group_B':f'G{groups[j]}','group_pair':pair,
                      'rank_A_L27':r27[i],'rank_B_L27':r27[j],'rank_A_L28':r28[i],'rank_B_L28':r28[j],
                      'signed_rank_gap_L27':d27,'signed_rank_gap_L28':d28,'rank_gap_reorganization':abs(d28-d27),
                      'same_numerical_target':bool(equal),'numerical_concordance_L27':c27,'numerical_concordance_L28':c28})
flip_df = save('crystal_L27_L28_pairwise_rank_flips.csv',pd.DataFrame(flips).sort_values('rank_gap_reorganization',ascending=False))
for c in pair_counts.values():
    c['flip_percent'] = 100*c['flips']/c['total_pairs'];pair_summary.append(c)
save('crystal_L27_L28_rank_flip_summary.csv',pair_summary)
emit('Pair flip breakdown:',pd.DataFrame(pair_summary).to_string(index=False))
emit('Largest pair reorganizations:',flip_df.head(20).to_string(index=False))

retention=[]
for i in range(n):
    row={'point_id':int(ids[i]),'target':int(targets[i]),'group':f'G{groups[i]}'}
    for kind in ['undirected','directed_outgoing']:
        na,nb = (set(np.flatnonzero(a['adjacency'][i])),set(np.flatnonzero(b['adjacency'][i]))) if kind=='undirected' else (set(a['neighbors'][i]),set(b['neighbors'][i]))
        row.update({f'{kind}_degree_L27':len(na),f'{kind}_degree_L28':len(nb),f'{kind}_intersection':len(na&nb),
                    f'{kind}_union':len(na|nb),f'{kind}_jaccard':len(na&nb)/len(na|nb),
                    f'{kind}_lost_point_ids':json.dumps(sorted(int(ids[j]) for j in na-nb)),
                    f'{kind}_gained_point_ids':json.dumps(sorted(int(ids[j]) for j in nb-na))})
    retention.append(row)
ret = save('crystal_L27_L28_neighbor_retention.csv',retention)
intersection,union = a['edges']&b['edges'], a['edges']|b['edges']
graph={'edges_L27':len(a['edges']),'edges_L28':len(b['edges']),'intersection':len(intersection),'union':len(union),
       'jaccard':len(intersection)/len(union),'fraction_L27_edges_retained':len(intersection)/len(a['edges']),
       'lost_edges':len(a['edges']-b['edges']),'gained_edges':len(b['edges']-a['edges']),
       'undirected_node_jaccard':describe(ret.undirected_jaccard),'directed_outgoing_node_jaccard':describe(ret.directed_outgoing_jaccard)}
changed=[]
for status,es in [('lost',a['edges']-b['edges']),('gained',b['edges']-a['edges'])]:
    for i,j in sorted(es): changed.append({'status':status,'point_id_A':int(ids[i]),'target_A':int(targets[i]),'group_A':f'G{groups[i]}','point_id_B':int(ids[j]),'target_B':int(targets[j]),'group_B':f'G{groups[j]}',
        'distance_4096d_L27':a['D'][i,j],'distance_4096d_L28':b['D'][i,j]})
save('crystal_L27_L28_changed_edges.csv',changed)
emit('TASK 8: graph comparison:',graph)
emit('Most changed undirected neighborhoods:\n',ret.sort_values('undirected_jaccard').head(20).to_string(index=False))

z27,z28 = a['z'].astype(float),b['z'].astype(float)
cosines = a['pca'].components_.astype(float) @ b['pca'].components_.astype(float).T
za,zb,disparity = procrustes(z27,z28)
# Mapping z28 into z27 coordinate frame, allowing a global scale and reflection.
R,scale_sum = orthogonal_procrustes(z28-z28.mean(0),z27-z27.mean(0))
scale=scale_sum/np.sum((z28-z28.mean(0))**2)
aligned=(z28-z28.mean(0)) @ R*scale+z27.mean(0)
transport27=(H[28].astype(float)-H[28].mean(0)) @ a['pca'].components_.astype(float).T
transport28=(H[27].astype(float)-H[27].mean(0)) @ b['pca'].components_.astype(float).T
orientation={'component_cosine_matrix_rows_L27_cols_L28':cosines.tolist(),
    'component_unsigned_angles_degrees':np.degrees(np.arccos(np.clip(abs(cosines),0,1))).tolist(),
    'PCA_2D_subspace_principal_angles_degrees':np.degrees(subspace_angles(a['pca'].components_.T,b['pca'].components_.T)).tolist(),
    'score_correlations_rows_L27_cols_L28':[[correlations(z27[:,i],z28[:,j]) for j in range(2)] for i in range(2)],
    'pairwise_PCA2_distance_correlations':correlations(pdist(z27),pdist(z28)),
    'pairwise_4096d_distance_correlations':correlations(squareform(a['D']),squareform(b['D'])),
    'procrustes_disparity':float(disparity),'procrustes_similarity_1_minus_disparity':float(1-disparity),
    'procrustes_normalized_RMSE':float(np.sqrt(disparity)),
    'z28_to_z27_orthogonal_matrix':R.tolist(),'z28_to_z27_scale':float(scale),'orthogonal_determinant':float(np.linalg.det(R)),
    'aligned_axis_correlations':[correlations(z27[:,i],aligned[:,i]) for i in range(2)],
    'aligned_L28_axis1_target_spearman':float(spearmanr(targets,aligned[:,0]).statistic),
    'L28_projected_onto_L27_PC1_target_spearman':float(spearmanr(targets,transport27[:,0]).statistic),
    'L27_projected_onto_L28_PC2_target_spearman':float(spearmanr(targets,transport28[:,1]).statistic)}
for g in [2,3,4]:
    idx=groups==g
    orientation[f'G{g}_branch_ARI_between_layers']=float(adjusted_rand_score(a['clusters'][idx],b['clusters'][idx]))
emit('TASK 9: orientation:',json.dumps(orientation,indent=2))
save('crystal_L27_L28_procrustes_coordinates.csv',pd.DataFrame({'point_id':ids,'target':targets,'group':meta.group,
     'PC1_L27':z27[:,0],'PC2_L27':z27[:,1],'PC1_L28_aligned':aligned[:,0],'PC2_L28_aligned':aligned[:,1]}))

fig,axes=plt.subplots(1,3,figsize=(19,6))
for g in [1,2,3,4]:
    idx=groups==g
    axes[0].scatter(r27[idx],r28[idx],c=color[g],label=f'G{g}',s=24)
    axes[1].scatter(z27[idx,0],-z28[idx,1],c=color[g],s=24)
    axes[2].scatter(z27[idx,0],z27[idx,1],c=color[g],s=24,label=f'G{g} L27')
    axes[2].scatter(aligned[idx,0],aligned[idx,1],edgecolors=color[g],facecolors='none',s=24)
axes[0].plot([1,120],[1,120],c='gray',ls='--');axes[0].set(xlabel='PC1 rank L27',ylabel='PC1 rank L28',title='PC1 order changes (sign corrected)');axes[0].legend()
axes[1].set(xlabel='L27 PC1',ylabel='L28 −PC2',title='Magnitude direction moves into PC2')
axes[2].add_collection(LineCollection([(z27[i],aligned[i]) for i in range(n)],color='gray',alpha=.2,linewidth=.5))
axes[2].set(xlabel='L27 PC1 coordinate frame',ylabel='L27 PC2 coordinate frame',title='Procrustes: filled L27 / open aligned L28')
for ax in axes:ax.grid(alpha=.15)
fig.tight_layout();fig.savefig(OUT/'crystal_L27_L28_orientation_comparison.png',dpi=170);fig.savefig(OUT/'crystal_L27_L28_orientation_comparison.svg');plt.close(fig)

distance_summary=[]
for L in [27,28]:
    for pair,df in states[L]['cross'].groupby('group_pair'):
        distance_summary.append({'layer':L,'group_pair':pair,'edge_count':len(df),
            **{f'{col}_{key}':v for col in ['euclidean_distance_4096d','euclidean_distance_PCA2','distance_4096d_to_PCA2_ratio'] for key,v in describe(df[col]).items()}})
save('crystal_cross_group_distance_summary.csv',distance_summary)
dump('provenance.json',provenance)
dump('analysis_metrics.json',{'pca':metrics,'graph':graph,'orientation':orientation,'rank_flips':{'total_pairs':n*(n-1)//2,'flips':len(flips),
    'flip_percentage':100*len(flips)/(n*(n-1)//2),'within_group_flips':sum(x['group_A']==x['group_B'] for x in flips),
    'between_group_flips':sum(x['group_A']!=x['group_B'] for x in flips),'ties':ties,'by_group_pair':pair_summary},
    'cross_group_distances':distance_summary})
emit('VALIDATION: metadata, prefix alignment, saved token diagnostics, 480 directed edges/layer, non-self neighbors, independent float64 kNN identity, archived PCA metrics, exact Spearman-drop decomposition: PASS.')
LOG.close()
