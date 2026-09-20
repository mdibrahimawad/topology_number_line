from __future__ import annotations

import io
from pathlib import Path
import tarfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

from .dataset import load_dataset, package_versions
from .storage import atomic, digest, fingerprint, read_json, write_json

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#17becf']
LIMITATIONS = '''Each target has one randomized context. Prompt length varies with target length.
Demonstration position and digit length are coupled by the fixed 1/2/3/4-digit order.
Chance demonstration–target matches are recorded. Same-digit connectivity can arise from numerical locality.
Cross-length conditional fractions require their saved denominators; undefined fractions are missing.
10000 is the only five-digit target. PCA captures only part of the variance; visual overlap or separation
alone does not establish original-space graph structure. Independently fitted PCA signs and axes do not
form a shared coordinate frame across layers. This experiment does not prove that next-token preparation
causes the geometry. The stimulus design differs from the previous experiment, so differences cannot be
attributed only to denser target coverage. Comparisons are descriptive, not causal tests.'''


def figure_file(store, key, path, dep, draw):
    if store.valid(key, dep):
        store.log(f'[SKIP] {key}: checksum valid')
        return
    fig = draw()
    try:
        content = io.BytesIO()
        fig.savefig(content, format='png', dpi=240, bbox_inches='tight')
        content.seek(0)
        with Image.open(content) as image:
            image.verify()
        atomic(path, lambda f: f.write(content.getvalue()))
    finally:
        plt.close(fig)
    store.finish(key, [path], dep)


def axes_labels(ax, data):
    ev = data['explained_variance_ratio'] * 100
    ax.set_xlabel(f'PC1 ({ev[0]:.2f}% variance)')
    ax.set_ylabel(f'PC2 ({ev[1]:.2f}% variance)')


def draw_graph(data, metric, rows, order, mode='graph'):
    label = metric.get('display_label', 'Crystal')
    if metric['status'] != 'valid':
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.text(.5, .5, f'{label} layer {metric["layer"]}: degenerate\nPCA / neighborhoods unavailable',
                ha='center', va='center', transform=ax.transAxes, fontsize=18)
        ax.axis('off')
        return fig
    z = data['scores']
    digit = np.array([r['leading_digit'] for r in rows])
    target = np.array([r['target'] for r in rows])
    if mode == 'digits':
        fig, axes = plt.subplots(3, 3, figsize=(13, 12), sharex=True, sharey=True, layout='constrained')
        limits = [(z[:, i].min(), z[:, i].max()) for i in range(2)]
        for d, ax in enumerate(axes.flat, 1):
            ax.scatter(z[order, 0], z[order, 1], c='#bbbbbb', s=2, alpha=.15, rasterized=True)
            ix = order[digit[order] == d]
            ax.scatter(z[ix, 0], z[ix, 1], c=COLORS[d - 1], s=4, alpha=.8, rasterized=True)
            ax.set_title(f'Leading digit {d} · n={len(ix):,}')
            for i, setter in enumerate([ax.set_xlim, ax.set_ylim]):
                low, high = limits[i]
                pad = max((high - low) * .04, 1e-8)
                setter(low - pad, high + pad)
            axes_labels(ax, data)
        fig.suptitle(f'{label} layer {metric["layer"]} · one global PCA shared across all nine panels')
        return fig
    fig, ax = plt.subplots(figsize=(10, 7), layout='constrained')
    if mode == 'heatmap':
        im = ax.imshow(data['digit_fractions'], vmin=0, vmax=1, cmap='viridis')
        ax.set_xticks(range(9), range(1, 10)); ax.set_yticks(range(9), range(1, 10))
        ax.set_xlabel('Destination leading digit'); ax.set_ylabel('Source leading digit')
        fig.colorbar(im, ax=ax, label='Fraction of source category outgoing relations')
        ax.set_title(f'{label} layer {metric["layer"]} · original-space directed k=4')
        return fig
    if mode == 'graph':
        edges = np.searchsorted(data['point_ids'], data['union_edges'])
        ax.add_collection(LineCollection(z[edges], colors='#455a64', linewidths=.22, alpha=.10, rasterized=True))
        ax.scatter(z[order, 0], z[order, 1], c=np.array(COLORS)[digit[order] - 1], s=4, alpha=.8, rasterized=True)
        handles = [Line2D([], [], linestyle='', marker='o', color=c, label=str(i)) for i, c in enumerate(COLORS, 1)]
        ax.legend(handles=handles, title='Leading decimal digit', ncol=9, loc='upper center', bbox_to_anchor=(.5, -.1))
    else:
        im = ax.scatter(z[order, 0], z[order, 1], c=target[order], cmap='viridis', vmin=1, vmax=10000,
                        s=4, alpha=.8, rasterized=True)
        fig.colorbar(im, ax=ax, label='Numerical target (linear, fixed 1–10000)')
    axes_labels(ax, data)
    ax.set_title(f'{label} layer {metric["layer"]} · k=4 · {len(rows):,} points\n'
                 'Neighbors selected in original hidden space; PCA is display only')
    return fig


def draw_contact(paths, title):
    cols = 2 if len(paths) <= 6 else 4
    nrows = (len(paths) + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(7 * cols, 5.3 * nrows), squeeze=False, layout='constrained')
    for ax, path in zip(axes.flat, paths):
        with Image.open(path) as im:
            ax.imshow(np.asarray(im))
        ax.axis('off')
    for ax in list(axes.flat)[len(paths):]:
        ax.axis('off')
    fig.suptitle(title)
    return fig


def draw_depth(metrics, baseline):
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), layout='constrained')
    layers = [m['layer'] for m in metrics]
    specs = [('same_leading_digit_fraction', 'Same-leading-digit fraction'),
             ('cross_digit_length_fraction', 'Cross-digit-length fraction'),
             ('cross_digit_length_count', 'Cross-digit-length directed count'),
             ('same_leading_digit_given_cross_length', 'Same-leading-digit given cross length'),
             ('preceding_valid_layer_overlap_fraction', 'Neighbor overlap with preceding valid layer')]
    for ax, (key, title) in zip(axes.flat, specs):
        values = [m.get(key) if m['status'] == 'valid' else None for m in metrics]
        ax.plot(layers, [np.nan if v is None else v for v in values], marker='o', ms=3)
        if key in baseline['numerical_distance'] and baseline['numerical_distance'][key] is not None:
            ax.axhline(baseline['numerical_distance'][key], color='#444444', ls='--', label='Numerical-distance baseline')
            ax.legend(fontsize=8)
        if key == 'same_leading_digit_fraction':
            ax.axhline(baseline['frequency']['weighted_same_digit_probability'], color='#999999', ls=':', label='Random-other baseline')
            ax.legend(fontsize=8)
        ax.set_title(title); ax.set_xlabel('Returned hidden-state level'); ax.grid(alpha=.2)
        if key != 'cross_digit_length_count':
            ax.set_ylim(0, 1)
    ax = axes.flat[-1]
    for d in range(9):
        ax.plot(layers, [m['same_leading_digit_by_source'][d] if m['status'] == 'valid' else np.nan for m in metrics],
                label=str(d + 1), color=COLORS[d])
    ax.set_ylim(0, 1); ax.legend(ncol=3, title='Source digit', fontsize=8)
    ax.set_title('Same-leading-digit fraction by source'); ax.set_xlabel('Returned hidden-state level')
    from matplotlib.ticker import MaxNLocator
    for ax in axes.flat:
        ax.set_xlim(min(layers) - .3, max(layers) + .3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle('Original-space directed k=4 · degenerate / undefined values omitted')
    return fig


def render(store):
    rows = load_dataset(store)
    if store.manifest['stages'].get('analysis') != 'complete':
        raise ValueError('Analysis is incomplete; plotting never starts inference or analysis')
    keys = sorted(k for k in store.manifest['artifacts'] if k.startswith('analysis/'))
    if not keys:
        raise ValueError('No saved layer analyses')
    for key in keys:
        store.require(key)
    store.require('baselines')
    store.stage('plots', 'running')
    plot_runtime = dict(source_sha256=digest(Path(__file__)), viewer_sha256=digest(Path(__file__).with_name('viewer.html')),
                        packages=package_versions(['matplotlib', 'plotly', 'numpy']))
    source_dep = fingerprint(plot_runtime)
    dep = fingerprint(dict(source=source_dep, layers=[store.artifact_fingerprint(k) for k in keys],
                           baselines=store.artifact_fingerprint('baselines')))
    order = np.random.default_rng(42).permutation(len(rows))
    style_path = store.root / 'display.json'
    if not store.valid('display', dep):
        write_json(style_path, dict(leading_digit_colors={str(i): c for i, c in enumerate(COLORS, 1)},
            drawing_order_point_ids=[rows[i]['point_id'] for i in order], drawing_seed=42,
            target_color_limits=[1, 10000], log10_color_limits=[0, 4], axis_warning='Independent PCA frames across layers'))
        write_json(store.root / 'plot_runtime.json', plot_runtime)
        store.finish('display', [style_path, store.root / 'plot_runtime.json'], dep)
    metrics, graph_paths = [], []
    for position, key in enumerate(keys):
        layer = int(key.split('/')[1])
        layer_dep = fingerprint(dict(source=source_dep, analysis=store.artifact_fingerprint(key)))
        base = store.root / 'analysis' / f'layer_{layer:02d}'
        metric = read_json(base.with_suffix('.json'))
        synthetic = read_json(store.root / 'tokenizer_provenance.json').get('synthetic', False)
        metric['display_label'] = 'SYNTHETIC VALIDATION' if synthetic else ('Crystal SMOKE' if store.manifest['config']['smoke'] else 'Crystal')
        with np.load(base.with_suffix('.npz'), allow_pickle=False) as saved:
            data = {k: saved[k] for k in saved.files}
        metrics.append(metric)
        for mode in ['graph', 'digits', 'heatmap', 'magnitude']:
            path = store.root / 'figures' / f'layer_{layer:02d}_{mode}.png'
            figure_file(store, f'figure/{layer:02d}/{mode}', path, layer_dep,
                        lambda mode=mode: draw_graph(data, metric, rows, order, mode))
            if mode == 'graph':
                graph_paths.append(path)
        viewer_path = store.root / 'viewer' / f'layer_{layer:02d}.json'
        if not store.valid(f'viewer/{layer:02d}', layer_dep):
            payload = dict(layer=layer, status=metric['status'])
            if metric['status'] == 'valid':
                payload.update(scores=data['scores'].tolist(), neighbors=data['neighbors'].tolist(),
                    distances=data['distances'].tolist(), union_edges=data['union_edges'].tolist(),
                    variance=data['explained_variance_ratio'].tolist())
            write_json(viewer_path, payload)
            store.finish(f'viewer/{layer:02d}', [viewer_path], layer_dep)
        store.log(f'[PLOTS] completed={position + 1}/{len(keys)} remaining={len(keys) - position - 1}')
    figure_file(store, 'figure/overview', store.root / 'figures/overview.png', dep,
                lambda: draw_contact(graph_paths, 'Crystal · all returned hidden-state levels · independent PCA frames'))
    for page, start in enumerate(range(0, len(graph_paths), 6), 1):
        figure_file(store, f'figure/contact/{page}', store.root / 'figures' / f'contact_{page:02d}.png', dep,
                    lambda start=start: draw_contact(graph_paths[start:start + 6], 'Original-space kNN projected into global per-layer PCA'))
    baseline = read_json(store.root / 'baselines.json')
    figure_file(store, 'figure/depth', store.root / 'figures/depth.png', dep, lambda: draw_depth(metrics, baseline))
    if not store.valid('viewer/index', dep):
        from plotly.offline import get_plotlyjs
        viewer = store.root / 'viewer'
        index = viewer / 'index.html'
        atomic(index, lambda f: f.write(Path(__file__).with_name('viewer.html').read_bytes()))
        atomic(viewer / 'plotly.min.js', lambda f: f.write(get_plotlyjs().encode()))
        write_json(viewer / 'index.json', dict(experiment=store.manifest['config']['experiment'],
            synthetic=read_json(store.root / 'tokenizer_provenance.json').get('synthetic', False),
            layers=[dict(layer=m['layer'], status=m['status']) for m in metrics],
            rows=[{k: r[k] for k in ['point_id', 'target', 'leading_digit', 'digit_count']} for r in rows],
            colors=COLORS, drawing_order=order.tolist()))
        store.finish('viewer/index', [index, viewer / 'plotly.min.js', viewer / 'index.json'], dep)
    if not store.valid('report', dep):
        report = store.root / 'SUMMARY.md'
        synthetic = read_json(store.root / 'tokenizer_provenance.json').get('synthetic', False)
        text = f'# {store.manifest["config"]["experiment"]}\n\n'
        text += '**SYNTHETIC VALIDATION ONLY — no model inference.**\n\n' if synthetic else ''
        text += f'{len(rows):,} points; {len(metrics)} returned levels. k=4 in original hidden space.\n\n'
        text += 'Union edges include either direction; union degree can exceed four. Mutual fraction uses directed relations.\n\n'
        text += '| Level | Status | Same digit | Cross length | Same digit given cross length | Conditional denominator | PC1+2 variance |\n|---|---|---|---|---|---|---|\n'
        for m in metrics:
            def value(k):
                v = m.get(k)
                return 'unavailable' if v is None else f'{v:.5g}'
            text += f'| {m["layer"]} | {m["status"]} | {value("same_leading_digit_fraction")} | {value("cross_digit_length_fraction")} | {value("same_leading_digit_given_cross_length")} | {value("conditional_denominator")} | '
            text += (f'{sum(m["pca"]["explained_variance_ratio"]):.5g}' if 'pca' in m else 'unavailable') + ' |\n'
        text += '\nNumerical-distance baseline same-digit fraction: ' + str(baseline['numerical_distance']['same_leading_digit_fraction'])
        text += '\n\nRandom-other-point weighted same-digit probability: ' + str(baseline['frequency']['weighted_same_digit_probability'])
        text += '\n\n' + LIMITATIONS + '\n\nSee `source/CRYSTAL_FULLRANGE.md` for reproduction, precision and checkpoint semantics.\n'
        atomic(report, lambda f: f.write(text.encode()))
        write_json(store.root / 'layer_metrics.json', metrics)
        store.finish('report', [report, store.root / 'layer_metrics.json'], dep)
    store.stage('plots', 'complete')
    package(store, dep)


def package(store, dependency):
    if store.valid('lightweight_export', dependency):
        store.log('[SKIP] lightweight export valid')
        return
    path = store.root / 'lightweight.tar.gz'
    def write(f):
        with tarfile.open(fileobj=f, mode='w:gz') as archive:
            for item in sorted(store.root.rglob('*')):
                relative = item.relative_to(store.root)
                if (item.is_file() and relative.parts[0] != 'hidden' and item != path
                    and not item.name.endswith('.partial')):
                    archive.add(item, arcname=str(relative), recursive=False)
    atomic(path, write)
    store.finish('lightweight_export', [path], dependency,
                 {'hidden_states_included': False, 'complete_download': 'download experiment directory'})
