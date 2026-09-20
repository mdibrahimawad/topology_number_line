"""Independent task/context geometry from saved hidden states; no inference.

At 1,000 points an exact distance matrix is small (8 MB in float64). One matrix
serves the original-space kNN and optional H0/H1 computations. PCA is display
only. Labels describe the stimulus or expected answer, not observed accuracy.
"""
from __future__ import annotations

import hashlib
import io
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from numzig.fullrange import analysis as fullrange_analysis
from numzig.fullrange.analysis import degeneration, union_graph
from numzig.fullrange.dataset import package_versions
from numzig.fullrange.storage import Store, atomic, digest, fingerprint, read_json, write_json, write_npz


CAVEATS = [
    'Each PCA and PH cloud contains one model, task and demonstration context; contexts are never pooled.',
    'Cross-task comparisons must join numeric target IDs; global point IDs differ across tasks.',
    'Independent PCA frames may rotate or reflect. Two- or three-dimensional shape is a projection.',
    'Expected output labels do not establish that the model performs the task; consult behavior results.',
    'For written numbers, decimal digit labels refer to the underlying numeric value, not the input spelling.',
    'Digit-position variance fractions are separate one-way associations, not causal effects or an additive decomposition.',
    'PH uses all points and original Euclidean distances divided by this cloud\'s median pair distance.',
    'Betti numbers depend on connection distance; H0/H1 do not identify a unique geometric shape.',
]


def _array_hash(x):
    h = hashlib.sha256()
    h.update(str((x.shape, x.dtype.str)).encode())
    for layer in x:
        h.update(np.ascontiguousarray(layer).view(np.uint8))
    return h.hexdigest()


def _labels(rows):
    targets = np.array([int(r['target']) for r in rows], dtype=np.int64)
    first_input = np.array([str(int(r['target']))[0] for r in rows])
    outputs = [str(r['expected_output']).strip() for r in rows]
    if any(not output for output in outputs):
        raise ValueError('expected_output cannot be empty')
    numeric_output = all(output.isdigit() for output in outputs)
    first_output = np.array([s[0] if numeric_output else s.split()[0] for s in outputs])
    token_counts = np.array([int(r.get('token_count', r.get('prompt_token_count',
        len(r.get('input_token_ids', r.get('token_ids', [])))))) for r in rows], dtype=np.int32)
    source = np.full((len(rows), 4), -1, dtype=np.int8)
    for i, row in enumerate(rows):
        values = row.get('source_digits', list(str(int(row['target']))))
        if values is None:
            values = list(str(int(row['target'])))
        for j, value in enumerate(values[:4]):
            source[i, j] = int(value)
    next_token = np.array([str(r.get('first_continuation_token', r.get('expected_first_token_id',
        r.get('continuation_token_ids', [None])[0] if r.get('continuation_token_ids') else None))) for r in rows])
    return dict(targets=targets, input_first_digit=first_input, output_first_label=first_output,
                token_counts=token_counts, source_digits=source, expected_next_token=next_token,
                output_label_kind='first decimal output digit' if numeric_output else 'first output word')


def group_variance(x, labels, valid=None):
    """Fraction of centered original-space energy explained by category means."""
    valid = np.ones(len(x), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    a = np.asarray(x[valid], dtype=np.float64)
    if not len(a):
        return dict(fraction=None, point_count=0, category_count=0)
    labels = np.asarray(labels)[valid]
    unique, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    centered = a - a.mean(axis=0)
    total = float(np.sum(centered * centered))
    if not total:
        return dict(fraction=None, point_count=len(a), category_count=len(unique))
    sums = np.zeros((len(unique), a.shape[1]), dtype=np.float64)
    np.add.at(sums, inverse, centered)
    between = float(np.sum(sums * sums / counts[:, None]))
    fraction = between / total
    if not -1e-12 <= fraction <= 1 + 1e-12:
        raise ArithmeticError('Invalid between-category variance fraction')
    return dict(fraction=float(np.clip(fraction, 0, 1)), point_count=len(a), category_count=len(unique))


def geometry(x, ids, targets):
    """Three PCs and exact Euclidean k=4; stable ties use ascending point IDs."""
    a = np.asarray(x, dtype=np.float64)
    diagnostic = degeneration(a)
    diagnostic['exactly_constant'] = bool(np.all(a == a[0]))
    if diagnostic['degenerate']:
        return dict(point_ids=ids, targets=targets), dict(status='degenerate', diagnostic=diagnostic), None
    distances = squareform(pdist(a, metric='euclidean'))
    if not np.isfinite(distances).all():
        raise ArithmeticError('Nonfinite original-space distance')
    np.fill_diagonal(distances, np.inf)
    indices = np.argsort(distances, axis=1, kind='stable')[:, :4]
    neighbor_distances = np.take_along_axis(distances, indices, axis=1)
    np.fill_diagonal(distances, 0)
    neighbors = ids[indices]
    # ARPACK computes just the requested components, preserving the exact cloud.
    solver = 'arpack' if min(a.shape) > 3 else 'full'
    pca = PCA(n_components=min(3, *a.shape), svd_solver=solver, tol=1e-10, random_state=42)
    pca.fit(a)
    components = pca.components_.copy()
    scores = (a - pca.mean_) @ components.T
    rho = []
    for axis in range(scores.shape[1]):
        value = float(spearmanr(targets, scores[:, axis]).statistic) if np.ptp(scores[:, axis]) else np.nan
        if np.isfinite(value) and value < 0:
            components[axis] *= -1
            scores[:, axis] *= -1
            value *= -1
        rho.append(value if np.isfinite(value) else None)
    arrays = dict(point_ids=ids, targets=targets, scores=scores, scores3=scores,
                  components=components, mean=pca.mean_, explained_variance=pca.explained_variance_,
                  explained_variance_ratio=pca.explained_variance_ratio_, neighbors=neighbors,
                  distances=neighbor_distances, **union_graph(ids, neighbors))
    metadata = dict(status='valid', diagnostic=diagnostic, point_count=len(a), dimension=a.shape[1],
        pca=dict(solver=solver, tolerance=1e-10, random_seed=42, centered=True, whiten=False,
                 standardized=False, component_count=scores.shape[1],
                 sign_convention='each PC oriented to nonnegative numeric target Spearman rho',
                 spearman_absolute=rho, explained_variance_ratio=pca.explained_variance_ratio_.tolist()),
        knn=dict(k=4, metric='original-space float64 Euclidean', ties='ascending canonical point_id',
                 directed_relations=int(neighbors.size), union_edges=len(arrays['union_edges'])))
    return arrays, metadata, distances


def persistence(distances):
    """Exact full-cloud Rips H0/H1, with median pair distance = one."""
    positive_scale = float(np.median(distances[np.triu_indices(len(distances), 1)]))
    if not positive_scale > 0:
        return {}, dict(status='unavailable', reason='Median pair distance is zero; normalization is undefined')
    normalized = distances / positive_scale
    try:
        from gph import ripser_parallel
        result = ripser_parallel(normalized, metric='precomputed', maxdim=1, n_threads=2)
        backend = 'giotto-ph.ripser_parallel'
    except ImportError:
        from ripser import ripser
        result = ripser(normalized, distance_matrix=True, maxdim=1)
        backend = 'ripser'
    diagrams = [np.asarray(d, dtype=np.float64) for d in result['dgms']]
    h0, h1 = diagrams[:2]
    if not 1 <= len(h0) <= len(distances) or np.isinf(h0[:, 1]).sum() != 1:
        raise ArithmeticError('Unexpected complete-filtration H0 diagram')
    finite_deaths = np.concatenate([d[np.isfinite(d[:, 1]), 1] for d in diagrams])
    upper = float(finite_deaths.max()) if len(finite_deaths) else 1.
    grid = np.linspace(0, max(upper * 1.05, 1e-9), 301)
    curves = [np.array([np.sum((d[:, 0] <= epsilon) & (epsilon < d[:, 1])) for epsilon in grid], dtype=np.int64)
              for d in diagrams]
    max_h1 = float(np.max(h1[:, 1] - h1[:, 0])) if len(h1) else 0.
    arrays = dict(h0=h0, h1=h1, betti_epsilon=grid, betti_h0=curves[0], betti_h1=curves[1])
    return arrays, dict(status='complete', backend=backend, max_homology_dimension=1,
        distance_normalization='all off-diagonal unordered-pair median', median_pair_distance=positive_scale,
        coefficient_field=2, h0_count=len(h0), h1_count=len(h1), max_h1_lifetime=max_h1,
        zero_lifetime_h0_omitted=len(distances) - len(h0),
        betti_convention='birth <= epsilon < death; one essential H0 class included',
        betti_at_one=[int(np.sum((d[:, 0] <= 1) & (1 < d[:, 1]))) for d in diagrams])


def _save_figure(path, fig):
    try:
        content = io.BytesIO()
        fig.savefig(content, format='png', dpi=150, bbox_inches='tight')
        atomic(path, lambda stream: stream.write(content.getvalue()))
    finally:
        plt.close(fig)


def _categorical(ax, scores, labels, title, order):
    categories = sorted(set(labels.tolist()), key=lambda s: (len(str(s)), str(s)))
    colors = plt.get_cmap('tab10')
    for i, category in enumerate(categories):
        selected = order[labels[order] == category]
        color = ('#333333' if str(category) == '0' else colors((int(category) - 1) % 10)) if str(category).isdigit() else colors(i % 10)
        ax.scatter(scores[selected, 0], scores[selected, 1], s=8, alpha=.65,
                   color=color, label=str(category), rasterized=True)
    ax.set_title(title)
    if len(categories) <= 12:
        ax.legend(ncol=min(5, len(categories)), fontsize=7, loc='best')


def plots(base, arrays, metric, labels):
    title = f'{metric["model"]} · {metric["task"]} · context {metric["context_id"]} · level {metric["layer"]}'
    paths = []
    if metric['status'] == 'degenerate':
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.text(.5, .5, title + '\nConstant / numerically degenerate cloud: geometry unavailable',
                ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        path = base.with_name(base.name + '_pca.png')
        _save_figure(path, fig)
        return [path]
    scores = arrays['scores']
    evr = arrays['explained_variance_ratio']
    order = np.random.default_rng(42).permutation(len(scores))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), layout='constrained')
    _categorical(axes[0], scores, labels['input_first_digit'], 'Input value: first decimal digit', order)
    _categorical(axes[1], scores, labels['output_first_label'], 'Expected ' + labels['output_label_kind'], order)
    values = labels['token_counts']
    im = axes[2].scatter(scores[order, 0], scores[order, 1], c=values[order], s=8, alpha=.65, cmap='viridis')
    axes[2].set_title('Prompt token count' if np.all(values > 0) else 'Prompt token count (0 = unavailable)')
    fig.colorbar(im, ax=axes[2], label='Tokens')
    for ax in axes:
        ax.set_xlabel(f'PC1 ({evr[0]:.1%} variance)')
        ax.set_ylabel(f'PC2 ({evr[1]:.1%} variance)')
        ax.grid(alpha=.15)
    fig.suptitle(title + '\nSame coordinates in all three panels; labels are predefined')
    path = base.with_name(base.name + '_pca.png')
    _save_figure(path, fig)
    paths.append(path)
    fig, ax = plt.subplots(figsize=(10, 7), layout='constrained')
    edge_indices = np.searchsorted(arrays['point_ids'], arrays['union_edges'])
    ax.add_collection(LineCollection(scores[edge_indices, :2], colors='#59636b', linewidths=.3, alpha=.15))
    _categorical(ax, scores, labels['input_first_digit'], title, order)
    ax.set_xlabel(f'PC1 ({evr[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({evr[1]:.1%} variance)')
    fig.suptitle('Original-space Euclidean k=4 union graph; PCA only positions the drawing')
    path = base.with_name(base.name + '_graph.png')
    _save_figure(path, fig)
    paths.append(path)
    if 'h0' in arrays:
        fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout='constrained')
        upper = float(arrays['betti_epsilon'][-1])
        for dimension, color in [(0, '#277da1'), (1, '#f3722c')]:
            diagram = arrays[f'h{dimension}']
            finite = diagram[np.isfinite(diagram[:, 1])]
            if len(finite):
                axes[0].scatter(finite[:, 0], finite[:, 1], s=9, alpha=.6, color=color, label=f'H{dimension}')
            axes[dimension + 1].step(arrays['betti_epsilon'], arrays[f'betti_h{dimension}'], where='post', color=color)
            axes[dimension + 1].set(xlabel='Normalized connection distance', ylabel=f'β{dimension}', title=f'Betti {dimension}')
        axes[0].plot([0, upper], [0, upper], color='#777777', linewidth=.7)
        axes[0].set(xlabel='Birth', ylabel='Death', title='Persistence (essential H0 omitted)')
        if axes[0].get_legend_handles_labels()[0]:
            axes[0].legend()
        fig.suptitle(title + '\nAll original-space points; median pair distance normalized to 1')
        path = base.with_name(base.name + '_ph.png')
        _save_figure(path, fig)
        paths.append(path)
    return paths


def analyze_group(x, rows, outdir, modelkey, *, do_ph=True, layer_ids=None, commit=lambda: None):
    """Analyze one task/context; durable per-layer checkpoints resume independently.

    x is [levels, points, hidden dimensions], corresponding exactly to rows.
    Distinct inputs or code require another outdir; corrupt files in this same
    run are repaired. commit is the Modal Volume.commit callback when remote.
    """
    x = np.asarray(x)
    rows = list(rows)
    if x.ndim != 3 or x.dtype != np.float32 or x.shape[1] != len(rows) or len(rows) < 5 or x.shape[2] < 3:
        raise ValueError('Expected float32 [levels, >=5 points, >=3 dimensions] matching rows')
    if not np.isfinite(x).all():
        raise ValueError('Hidden states must be finite')
    task_values = {str(r.get('task', r.get('condition', 'unknown'))) for r in rows}
    context_values = {str(r.get('context_id', r.get('context', 'unknown'))) for r in rows}
    if len(task_values) != 1 or len(context_values) != 1:
        raise ValueError('Analyze exactly one task and one context; do not pool contexts')
    ids = np.array([int(r.get('point_id', i)) for i, r in enumerate(rows)], dtype=np.int64)
    targets = [int(r['target']) for r in rows]
    if len(set(ids)) != len(ids) or len(set(targets)) != len(targets):
        raise ValueError('point_id and target must each be unique within a task/context group')
    if not np.all(np.diff(ids) > 0):
        raise ValueError('Rows and x must be in ascending point_id order')
    layer_ids = list(range(x.shape[0])) if layer_ids is None else list(layer_ids)
    if len(layer_ids) != x.shape[0] or len(set(layer_ids)) != len(layer_ids):
        raise ValueError('layer_ids must uniquely label every input level')
    outdir = Path(outdir)
    labels = _labels(rows)
    config = dict(schema=1, model=str(modelkey), task=next(iter(task_values)), context_id=next(iter(context_values)),
        row_hash=fingerprint(rows), input_hash=_array_hash(x), shape=list(x.shape), do_ph=bool(do_ph),
        layers=layer_ids, source=digest(Path(__file__)),
        shared_analysis_source=digest(Path(fullrange_analysis.__file__)),
        packages=package_versions(['numpy', 'scipy', 'scikit-learn', 'matplotlib', 'ripser', 'giotto-ph']))
    store = Store(outdir, config=config, commit=commit)
    layers = []
    with threadpool_limits(limits=2):
        for position, layer in enumerate(layer_ids):
            key = f'layer/{layer:02d}'
            dependency = fingerprint(dict(config=config, layer=layer))
            base = outdir / 'layers' / f'layer_{layer:02d}'
            metric_path = base.with_suffix('.json')
            if store.valid(key, dependency):
                layers.append(read_json(metric_path))
                continue
            started = time.perf_counter()
            arrays, metric, distances = geometry(x[position], ids, labels['targets'])
            metric.update(model=str(modelkey), task=config['task'], context_id=config['context_id'],
                          layer=int(layer), point_count=len(rows), dimension=int(x.shape[2]))
            arrays.update(input_first_digit=labels['input_first_digit'], output_first_label=labels['output_first_label'],
                          source_digits=labels['source_digits'], token_counts=labels['token_counts'])
            metric['ph'] = dict(status='not_requested' if not do_ph else 'unavailable',
                                reason=None if not do_ph else 'Degenerate cloud')
            if metric['status'] == 'valid':
                a = x[position]
                metric['variance_by_label'] = dict(
                    input_first_digit=group_variance(a, labels['input_first_digit']),
                    expected_output_first_label=group_variance(a, labels['output_first_label']),
                    token_count=group_variance(a, labels['token_counts'], labels['token_counts'] > 0),
                    expected_next_token=group_variance(a, labels['expected_next_token'], labels['expected_next_token'] != 'None'),
                    source_positions=[group_variance(a, labels['source_digits'][:, j], labels['source_digits'][:, j] >= 0)
                                      for j in range(4)])
                if do_ph:
                    ph_arrays, ph_metric = persistence(distances)
                    arrays.update(ph_arrays)
                    metric['ph'] = ph_metric
            metric['output_label_kind'] = labels['output_label_kind']
            figure_paths = plots(base, arrays, metric, labels)
            metric['figures'] = [str(p.relative_to(outdir)) for p in figure_paths]
            metric['arrays'] = str(base.with_suffix('.npz').relative_to(outdir))
            metric['seconds'] = time.perf_counter() - started
            write_npz(base.with_suffix('.npz'), **arrays)
            write_json(metric_path, metric)
            store.finish(key, [base.with_suffix('.npz'), metric_path, *figure_paths], dependency)
            layers.append(metric)
    summary = dict(model=str(modelkey), task=config['task'], context_id=config['context_id'],
        status='complete', layer_count=len(layers), point_count=len(rows), layers=layers,
        caveats=CAVEATS, input_hash=config['input_hash'], row_hash=config['row_hash'])
    summary_path = outdir / 'summary.json'
    if not summary_path.exists() or read_json(summary_path) != summary:
        write_json(summary_path, summary)
        commit()
    return summary
