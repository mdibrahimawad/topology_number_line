from __future__ import annotations

import resource
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from .dataset import load_dataset, package_versions
from .extract import inventory, read_layer
from .storage import digest, fingerprint, read_json, write_json, write_npz


def exact_knn(x, point_ids, k=4, block_size=64):
    """All candidates, direct float64 Euclidean differences (no Gram-matrix cancellation).

    Only [block_size, N] distances are materialized. Stable sort preserves the
    ascending point-ID order for ties, including ties at the k-th boundary.
    """
    x = np.asarray(x, dtype=np.float64)
    ids = np.asarray(point_ids)
    if x.ndim != 2 or len(x) != len(ids) or len(x) <= k or not np.isfinite(x).all():
        raise ValueError('Invalid kNN input')
    if not np.all(np.diff(ids) > 0):
        raise ValueError('point_ids must be unique and in ascending canonical order')
    neighbors = np.empty((len(x), k), dtype=np.int32)
    distances = np.empty((len(x), k), dtype=np.float64)
    for start in range(0, len(x), block_size):
        stop = min(start + block_size, len(x))
        ds = cdist(x[start:stop], x, metric='euclidean')
        ds[np.arange(stop - start), np.arange(start, stop)] = np.inf
        order = np.argsort(ds, axis=1, kind='stable')[:, :k]
        neighbors[start:stop] = ids[order]
        distances[start:stop] = np.take_along_axis(ds, order, axis=1)
    return neighbors, distances


def union_graph(ids, neighbors):
    directed = {(int(i), int(j)) for i, row in zip(ids, neighbors) for j in row}
    edges = np.array(sorted({tuple(sorted(pair)) for pair in directed}), dtype=np.int32).reshape(-1, 2)
    forward = np.array([(int(a), int(b)) in directed for a, b in edges], dtype=bool)
    backward = np.array([(int(b), int(a)) in directed for a, b in edges], dtype=bool)
    return dict(union_edges=edges, low_to_high=forward, high_to_low=backward, mutual=forward & backward)


def distribution(values):
    a = np.asarray(values)
    return {name: float(value) for name, value in zip(['min', 'q25', 'median', 'q75', 'max', 'mean'],
                [*np.quantile(a, [0, .25, .5, .75, 1]), np.mean(a)])}


def graph_metrics(rows, neighbors, distances, previous=None):
    ids = np.array([r['point_id'] for r in rows], dtype=np.int32)
    target = np.array([r['target'] for r in rows], dtype=np.int32)
    digit = np.array([r['leading_digit'] for r in rows])
    length = np.array([r['digit_count'] for r in rows])
    index = np.searchsorted(ids, neighbors)
    if np.any(index >= len(ids)) or not np.array_equal(ids[index], neighbors):
        raise ValueError('Unknown neighbor point ID')
    same = digit[:, None] == digit[index]
    cross = length[:, None] != length[index]
    counts = np.zeros((9, 9), dtype=np.int64)
    np.add.at(counts, (np.repeat(digit - 1, neighbors.shape[1]), digit[index].ravel() - 1), 1)
    denominator = counts.sum(axis=1)
    fractions = np.divide(counts, denominator[:, None], out=np.full((9, 9), np.nan), where=denominator[:, None] != 0)
    union = union_graph(ids, neighbors)
    edge_idx = np.searchsorted(ids, union['union_edges'])
    graph = coo_matrix((np.ones(len(edge_idx)), (edge_idx[:, 0], edge_idx[:, 1])), shape=(len(ids), len(ids)))
    n_components, labels = connected_components(graph, directed=False)
    gaps = np.abs(target[:, None] - target[index])
    log_gaps = np.abs(np.log10(target[:, None]) - np.log10(target[index]))
    indegree = np.bincount(index.ravel(), minlength=len(ids))
    degree = np.bincount(edge_idx.ravel(), minlength=len(ids))
    overlaps = None if previous is None else np.array([len(set(a) & set(b)) for a, b in zip(previous, neighbors)])
    metrics = dict(point_count=len(ids), directed_relation_count=int(neighbors.size),
        union_edge_count=len(edge_idx), mutual_neighbor_fraction=float(2 * union['mutual'].sum() / neighbors.size),
        mutual_definition='fraction of directed relations whose reverse relation also exists; 2*mutual_union_edges/(N*k)',
        weak_component_count=int(n_components), largest_component_fraction=float(np.bincount(labels).max() / len(ids)),
        neighbor_distances=distribution(distances), same_leading_digit_fraction=float(same.mean()),
        same_digit_length_fraction=float((~cross).mean()), cross_digit_length_count=int(cross.sum()),
        cross_digit_length_fraction=float(cross.mean()),
        same_leading_digit_cross_length_count=int((same & cross).sum()),
        same_leading_digit_given_cross_length=float(same[cross].mean()) if cross.any() else None,
        conditional_denominator=int(cross.sum()),
        same_leading_digit_by_source=[float(fractions[i, i]) if denominator[i] else None for i in range(9)],
        absolute_numerical_gaps=distribution(gaps), absolute_log10_gaps=distribution(log_gaps),
        preceding_valid_layer_overlap_count=int(overlaps.sum()) if overlaps is not None else None,
        preceding_valid_layer_overlap_denominator=int(neighbors.size) if overlaps is not None else None,
        preceding_valid_layer_overlap_fraction=float(overlaps.sum() / neighbors.size) if overlaps is not None else None)
    arrays = dict(point_ids=ids, neighbors=neighbors, ranks=np.arange(1, neighbors.shape[1] + 1, dtype=np.int8),
        distances=distances, digit_counts=counts, digit_fractions=fractions, numerical_gaps=gaps, log10_gaps=log_gaps,
        in_degree=indegree, union_degree=degree, component=labels,
        same_leading_digit_per_point=same.mean(axis=1), cross_length_count_per_point=cross.sum(axis=1),
        previous_neighbor_overlap_per_point=overlaps if overlaps is not None else np.full(len(ids), -1), **union)
    return metrics, arrays


def degeneration(x):
    x = np.asarray(x, dtype=np.float64)
    centered = x - x.mean(axis=0)
    spread = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    reference = float(np.sqrt(np.mean(np.sum(x * x, axis=1))))
    threshold = 1e-7 * max(1.0, reference)
    return dict(degenerate=spread <= threshold, rms_centered_norm=spread, rms_vector_norm=reference,
                threshold=threshold, criterion='RMS centered L2 norm <= 1e-7 * max(1, RMS vector L2 norm)')


def projection(x, targets):
    x = np.asarray(x, dtype=np.float64)
    pca = PCA(n_components=2, svd_solver='full', whiten=False)
    pca.fit(x)
    components = pca.components_.copy()
    scores = (x - pca.mean_) @ components.T
    rho = []
    for c in range(2):
        value = float(spearmanr(targets, scores[:, c]).statistic)
        if np.isfinite(value) and value < 0:
            scores[:, c] *= -1
            components[c] *= -1
            value *= -1
        rho.append(value if np.isfinite(value) else None)
    arrays = dict(scores=scores, mean=pca.mean_, components=components,
                  explained_variance=pca.explained_variance_, explained_variance_ratio=pca.explained_variance_ratio_)
    return arrays, dict(solver='full SVD', dtype='float64', whiten=False, center=True, standardize=False,
        normalize=False, random_seed=None, sign_convention='each score/loading pair oriented to nonnegative target Spearman',
        spearman_signed=rho, spearman_absolute=[abs(v) if v is not None else None for v in rho],
        explained_variance_ratio=pca.explained_variance_ratio_.tolist())


# Exact kernels above are unchanged from the serial implementation. Retaining its
# identity preserves compatible saved analysis, figures and adjacent-layer hashes.
SCIENTIFIC_SOURCE = 'd887a1241f00b89773a8e10bd638e510b79a3d0d8fb86dce871e2b24264dd510'
# A future scientific-kernel edit invalidates the compatibility alias automatically.
import ast
_KERNEL_NAMES = ['degeneration', 'distribution', 'exact_knn', 'graph_metrics', 'projection', 'union_graph']
_KERNEL_HASH = fingerprint([ast.dump(n, include_attributes=False) for n in ast.parse(Path(__file__).read_text()).body
                            if isinstance(n, ast.FunctionDef) and n.name in _KERNEL_NAMES])
if _KERNEL_HASH != '8fdc29ea8a14c36c1d40118b858ad1180737ecd8a01958119c8d513a339bf6ea':
    SCIENTIFIC_SOURCE = _KERNEL_HASH



def analysis_dependency(store):
    return fingerprint(dict(dataset=store.artifact_fingerprint('dataset'),
        chunks={k: v['files'] for k, v in store.manifest['artifacts'].items() if k.startswith('chunk/')},
        config=store.manifest['config_fingerprint'], source=SCIENTIFIC_SOURCE,
        packages=package_versions(['numpy', 'scipy', 'scikit-learn'])))


def compute_layer(store, layer, rows=None, x=None):
    """One independent layer, ALL canonical points; no preceding-layer dependency."""
    rows = load_dataset(store) if rows is None else rows
    ids = np.array([r['point_id'] for r in rows], dtype=np.int32)
    targets = np.array([r['target'] for r in rows])
    dep = analysis_dependency(store)
    key = f'layer/{layer:02d}'
    if hasattr(store, 'assigned') and key not in store.assigned:
        raise ValueError('Unassigned layer')
    if layer_ready(store, key, dep, rows):
        store.log(f'[SKIP] {key}: full-point kNN/PCA checkpoint valid')
        return
    tic = time.perf_counter()
    x = read_layer(store, layer, rows) if x is None else x
    diagnostic = degeneration(x)
    metric = dict(layer=layer, status='degenerate' if diagnostic['degenerate'] else 'valid',
                  diagnostic=diagnostic, point_count=len(rows),
                  analysis_source_sha256=SCIENTIFIC_SOURCE,
                  packages=package_versions(['numpy', 'scipy', 'scikit-learn']),
                  distance_method=store.manifest['config']['distance'])
    arrays = dict(point_ids=ids)
    if not diagnostic['degenerate']:
        neighbors, distances = exact_knn(x, ids, store.manifest['config']['k'])
        knn_seconds = time.perf_counter() - tic
        metrics, graphs = graph_metrics(rows, neighbors, distances)
        ptic = time.perf_counter()
        pca, pca_meta = projection(x, targets)
        metric.update(metrics, pca=pca_meta, preceding_valid_layer=None,
                      knn_seconds=knn_seconds, pca_seconds=time.perf_counter() - ptic)
        arrays.update(graphs, **pca)
    else:
        metric['interpretation'] = 'PCA and neighborhoods unavailable; excluded from substantive trends'
    metric['seconds'] = time.perf_counter() - tic
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    metric['process_peak_rss_bytes'] = int(rss if sys.platform == 'darwin' else rss * 1024)
    metric['dimension'] = x.shape[1]
    del x

    base = store.root / 'layers' / f'layer_{layer:02d}'
    write_npz(base.with_suffix('.npz'), **arrays)
    write_json(base.with_suffix('.json'), metric)
    # Validate serialized values before writing any completion receipt.
    with np.load(base.with_suffix('.npz'), allow_pickle=False) as saved:
        if set(saved.files) != set(arrays) or any(not np.array_equal(saved[k], v, equal_nan=True) for k, v in arrays.items()):
            raise IOError('Layer serialization failed')
    store.finish(key, [base.with_suffix('.npz'), base.with_suffix('.json')], dep)


def layer_ready(store, key, dependency, rows=None):
    if not store.valid(key, dependency):
        return False
    try:
        validate_layer(store, key, rows)
    except (ValueError, OSError, KeyError):
        return False
    return True


def validate_layer(store, key, rows=None):
    layer = int(key.split('/')[1])
    rows = load_dataset(store) if rows is None else rows
    ids = np.array([r['point_id'] for r in rows], dtype=np.int32)
    base = store.root / 'layers' / f'layer_{layer:02d}'
    metric = read_json(base.with_suffix('.json'))
    if metric['layer'] != layer or metric['point_count'] != len(ids) or metric['status'] not in ('valid', 'degenerate'):
        raise ValueError('Invalid layer metadata')
    with np.load(base.with_suffix('.npz'), allow_pickle=False) as arrays:
        if not np.array_equal(arrays['point_ids'], ids):
            raise ValueError('Layer omitted or reordered points')
        if metric['status'] == 'valid':
            nb, ds = arrays['neighbors'], arrays['distances']
            if (nb.shape != (len(ids), store.manifest['config']['k']) or ds.shape != nb.shape
                or not np.isin(nb, ids).all() or np.any(nb == ids[:, None])
                or not np.isfinite(ds).all() or np.any(ds < 0)
                or arrays['scores'].shape != (len(ids), 2) or not np.isfinite(arrays['scores']).all()):
                raise ValueError('Invalid full-point graph or PCA')


def pending_layers(store):
    """Coordinator: adopt compatible serial geometry without changing old outputs.

    Final comparison hashes can change when an earlier layer is repaired. Copying
    valid geometry into independent layer checkpoints prevents needless kNN/PCA
    recomputation of downstream layers from the original serial implementation.
    """
    rows = load_dataset(store)
    _, _, shape = inventory(store, rows, require_complete=True)
    dep = analysis_dependency(store)
    preceding = None
    for layer in range(shape[0]):
        final_key, raw_key = f'analysis/{layer:02d}', f'layer/{layer:02d}'
        base = store.root / 'analysis' / f'layer_{layer:02d}'
        if not store.valid(final_key):
            continue
        metric = read_json(base.with_suffix('.json'))
        prior = metric.get('preceding_valid_layer', preceding)
        prior_key = None if prior is None else f'analysis/{prior:02d}'
        prior_hash = None if prior_key is None else store.artifact_fingerprint(prior_key)
        expected = fingerprint(dict(analysis=dep, previous_valid_graph=prior_hash))
        if store.valid(final_key, expected) and not layer_ready(store, raw_key, dep):
            with np.load(base.with_suffix('.npz'), allow_pickle=False) as data:
                arrays = {k: data[k].copy() for k in data.files}
            raw = store.root / 'layers' / f'layer_{layer:02d}'
            write_npz(raw.with_suffix('.npz'), **arrays)
            write_json(raw.with_suffix('.json'), metric)
            validate_layer(store, raw_key)
            store.finish(raw_key, [raw.with_suffix('.npz'), raw.with_suffix('.json')], dep)
            store.log(f'[ADOPT] {raw_key}: preserved compatible serial geometry')
        if metric['status'] == 'valid':
            preceding = layer
    missing = [f'layer/{layer:02d}' for layer in range(shape[0])
               if not layer_ready(store, f'layer/{layer:02d}', dep)]
    store.log(f'[LAYERS] completed={shape[0] - len(missing)} skipped={shape[0] - len(missing)} remaining={len(missing)}')
    return missing


def analyze_assignments(store, blas_threads=4):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=blas_threads):
        for index, key in enumerate(store.assigned):
            compute_layer(store, int(key.split('/')[1]))
            store.log(f'[ANALYSIS WORKER] completed={index + 1} remaining={len(store.assigned) - index - 1}')


def analyze(store, allow_compute=True, rows=None, validated_shape=None):
    rows = load_dataset(store) if rows is None else rows
    shape = validated_shape
    if shape is None:
        _, _, shape = inventory(store, rows, require_complete=True)
    dep = analysis_dependency(store)
    store.stage('analysis', 'running')
    ids = np.array([r['point_id'] for r in rows], dtype=np.int32)
    targets = np.array([r['target'] for r in rows])
    previous, previous_layer, previous_hash = None, None, None
    complete = sum(store.valid(f'analysis/{layer:02d}') for layer in range(shape[0]))
    store.log(f'[ANALYSIS] completed={complete} remaining={shape[0] - complete}; coordinator finalizes layers in order')
    for layer in range(shape[0]):
        key = f'analysis/{layer:02d}'
        layer_dep = fingerprint(dict(analysis=dep, previous_valid_graph=previous_hash))
        base = store.root / 'analysis' / f'layer_{layer:02d}'
        metric_path = base.with_suffix('.json')
        array_path = base.with_suffix('.npz')
        if store.valid(key, layer_dep):
            metric = read_json(metric_path)
            store.log(f'[SKIP] {key}: validated saved kNN and PCA')
        else:
            raw_key = f'layer/{layer:02d}'
            if not layer_ready(store, raw_key, dep, rows):
                if not allow_compute:
                    raise ValueError(f'Missing {raw_key}; resume analysis workers first')
                compute_layer(store, layer, rows)
            validate_layer(store, raw_key, rows)
            raw = store.root / 'layers' / f'layer_{layer:02d}'
            metric = read_json(raw.with_suffix('.json'))
            with np.load(raw.with_suffix('.npz'), allow_pickle=False) as saved:
                arrays = {name: saved[name].copy() for name in saved.files}
            if metric['status'] == 'valid':
                # Only adjacent-layer comparison depends on earlier outputs.
                overlaps = None if previous is None else np.array([len(set(a) & set(b)) for a, b in zip(previous, arrays['neighbors'])])
                arrays['previous_neighbor_overlap_per_point'] = overlaps if overlaps is not None else np.full(len(ids), -1)
                denominator = arrays['neighbors'].size
                metric.update(preceding_valid_layer=previous_layer,
                    preceding_valid_layer_overlap_count=int(overlaps.sum()) if overlaps is not None else None,
                    preceding_valid_layer_overlap_denominator=int(denominator) if overlaps is not None else None,
                    preceding_valid_layer_overlap_fraction=float(overlaps.sum() / denominator) if overlaps is not None else None)
            write_npz(array_path, **arrays)
            write_json(metric_path, metric)
            with np.load(array_path, allow_pickle=False) as saved:
                if not np.array_equal(saved['point_ids'], ids):
                    raise IOError('Analysis serialization failed')
            store.finish(key, [array_path, metric_path], layer_dep)
        if metric['status'] == 'valid':
            with np.load(array_path, allow_pickle=False) as saved:
                previous = saved['neighbors'].copy()
            previous_layer, previous_hash = layer, store.artifact_fingerprint(key)
        store.log(f'[ANALYSIS] completed={layer + 1}/{shape[0]} remaining={shape[0] - layer - 1}')
    if not store.valid('baselines', dep):
        nb, ds = exact_knn(targets[:, None], ids)
        metrics, arrays = graph_metrics(rows, nb, ds)
        counts = np.bincount([r['leading_digit'] for r in rows], minlength=10)[1:]
        probability = (counts - 1) / (len(rows) - 1)
        if not store.manifest['config']['smoke'] and counts.tolist() != [1112] + [1111] * 8:
            raise ValueError('Unexpected leading-digit category counts')
        write_npz(store.root / 'baselines.npz', **arrays)
        write_json(store.root / 'baselines.json', dict(numerical_distance=metrics,
            frequency=dict(counts=counts.tolist(), probability_same_digit_by_source=probability.tolist(),
                           weighted_same_digit_probability=float(np.sum(counts * probability) / len(rows))),
            interpretation='Descriptive only; numerical locality can already yield same-leading-digit edges'))
        store.finish('baselines', [store.root / 'baselines.npz', store.root / 'baselines.json'], dep)
    store.stage('analysis', 'complete')
    return shape[0]
