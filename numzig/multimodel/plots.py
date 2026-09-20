"""Private per-artifact plotting receipts; saved-results-only final products."""
from pathlib import Path
import html
import tarfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from numzig.fullrange.storage import Store, digest, fingerprint, read_json, write_json, atomic
from numzig.fullrange.parallel import merge_receipts
from numzig.fullrange.plots import COLORS, LIMITATIONS, draw_graph, draw_contact, draw_depth, figure_file
from numzig.fullrange.dataset import package_versions
from . import MODELS
from .data import load

MODES = ('graph', 'digits', 'heatmap', 'magnitude')


def dependency(store, artifact=None):
    if store.manifest['stages'].get('analysis') != 'complete':
        raise ValueError('Plot-only requires completed analysis; inference is never implicit')
    keys = ([f'analysis/{artifact.split("/")[1]}'] if artifact is not None else
            sorted(k for k in store.manifest['artifacts'] if k.startswith('analysis/')))
    for key in keys:
        store.require(key)
    store.require('baselines')
    return fingerprint(dict(source=digest(__file__), primitives=digest(Path(__file__).parents[1] / 'fullrange/plots.py'),
        template=digest(Path(__file__).parents[1] / 'fullrange/viewer.html'),
        analysis={k: store.artifact_fingerprint(k) for k in keys},
        packages=package_versions(['matplotlib', 'plotly', 'numpy']), config=store.manifest['config_fingerprint']))


def validate_output(store, key):
    from numzig.fullrange.parallel import expected_paths
    path = store.root / next(iter(expected_paths(key)))
    if key.startswith('figure/'):
        with Image.open(path) as im:
            im.verify()
    else:
        payload = read_json(path)
        if payload['layer'] != int(key.split('/')[1]):
            raise ValueError('Viewer layer identity mismatch')
        if payload['status'] == 'valid' and len(payload['scores']) != len(load(store)):
            raise ValueError('Viewer point coverage mismatch')


def recover(store):
    dependency(store)  # Validate the completed analysis barrier.
    dependencies = {}
    def expected(key):
        layer = key.split('/')[1]
        if layer not in dependencies:
            dependencies[layer] = dependency(store, key)
        return dependencies[layer]
    for kind in ('figure', 'viewer'):
        merge_receipts(store, kind, expected, lambda key: validate_output(store, key), allow_incompatible=True)
    levels = read_json(store.root / 'tokenizer_provenance.json')['expected_levels']
    keys = [key for i in range(levels) for key in [*(f'figure/{i:02d}/{m}' for m in MODES), f'viewer/{i:02d}']]
    missing = []
    for key in keys:
        try:
            if not store.valid(key, expected(key)):
                missing.append(key)
            else:
                validate_output(store, key)
        except (ValueError, OSError):
            missing.append(key)
    store.log(f'[PLOTS] completed={len(keys)-len(missing)} skipped={len(keys)-len(missing)} remaining={len(missing)}')
    return missing


def heatmap(data, metric):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), layout='constrained')
    denom = data['digit_counts'].sum(axis=1)
    for ax, array, title in zip(axes, [data['digit_counts'], data['digit_fractions']], ['Directed counts', 'Row-normalized fractions']):
        is_fraction = title.startswith('Row')
        im = ax.imshow(array, cmap='viridis', vmin=0, vmax=1 if is_fraction else max(1, 4 * 1112))
        ax.set_xticks(range(9), range(1, 10))
        ax.set_yticks(range(9), [f'{i+1} (n={d:,})' for i, d in enumerate(denom)])
        ax.set_xlabel('Destination leading digit')
        ax.set_ylabel('Source digit (outgoing denominator)')
        for i in range(9):
            for j in range(9):
                value = array[i, j]
                ax.text(j, i, f'{value:.2f}' if is_fraction else str(value), ha='center', va='center', fontsize=7,
                        color='white' if (not is_fraction or value < .5) else 'black')
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    fig.suptitle(f'{metric["display_label"]} · L{metric["layer"]} · original-space k=4')
    return fig


def worker(store):
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    rows = load(store)
    order = np.random.default_rng(42).permutation(len(rows))
    label = MODELS[store.manifest['config']['model_key']]['label']
    if read_json(store.root / 'tokenizer_provenance.json').get('synthetic'):
        label += ' · SYNTHETIC VALIDATION'
    for key in store.assigned:
        dep = store.plan.get('dependencies', {}).get(key, store.plan['dependency'])
        if store.valid(key, dep):
            store.log(f'[SKIP] {key}')
            continue
        layer = int(key.split('/')[1])
        store.require(f'analysis/{layer:02d}')
        base = store.root / 'analysis' / f'layer_{layer:02d}'
        metric = read_json(base.with_suffix('.json')) | dict(display_label=label)
        with np.load(base.with_suffix('.npz'), allow_pickle=False) as saved:
            data = {k: saved[k] for k in saved.files}
        if key.startswith('figure/'):
            mode = key.split('/')[2]
            path = store.root / 'figures' / f'layer_{layer:02d}_{mode}.png'
            figure_file(store, key, path, dep,
                lambda: heatmap(data, metric) if mode == 'heatmap' and metric['status'] == 'valid' else draw_graph(data, metric, rows, order, mode))
        else:
            path = store.root / 'viewer' / f'layer_{layer:02d}.json'
            payload = dict(layer=layer, status=metric['status'], model=label)
            if metric['status'] == 'valid':
                payload.update({k: data[k].tolist() for k in ('scores', 'neighbors', 'distances', 'union_edges')})
                payload['variance'] = data['explained_variance_ratio'].tolist()
            write_json(path, payload)
            validate_output(store, key)
            store.finish(key, [path], dep)


def pca_depth(metrics, title):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    x = [m['layer'] for m in metrics]
    for i in range(2):
        for ax, field in zip(axes, ('explained_variance_ratio', 'spearman_absolute')):
            values = [m.get('pca', {}).get(field, [None, None])[i] for m in metrics]
            ax.plot(x, [np.nan if v is None else v for v in values], marker='.', label=f'PC{i+1}')
    for ax, label in zip(axes, ('Explained variance fraction', 'Absolute target Spearman correlation')):
        ax.set(xlabel='Actual returned level (L0 embedding; final level normalized)', ylabel=label, ylim=(0, 1))
        ax.legend(); ax.grid(alpha=.2)
    fig.suptitle(title + ' · independent global full-SVD PCA at each level')
    return fig


def finalize(store):
    if recover(store):
        raise ValueError('Plot barrier incomplete; completed figures retained')
    dep = dependency(store)
    rows = load(store)
    meta = read_json(store.root / 'tokenizer_provenance.json')
    label = MODELS[store.manifest['config']['model_key']]['label']
    if meta.get('synthetic'):
        label += ' · SYNTHETIC VALIDATION'
    metrics = [read_json(store.root / 'analysis' / f'layer_{i:02d}.json') for i in range(meta['expected_levels'])]
    paths = [store.root / 'figures' / f'layer_{i:02d}_graph.png' for i in range(len(metrics))]
    figure_file(store, 'figure/overview', store.root / 'figures/overview.png', dep,
                lambda: draw_contact(paths, label + ' · all returned states · independent PCA frames'))
    for page, start in enumerate(range(0, len(paths), 6), 1):
        figure_file(store, f'figure/contact/{page}', store.root / 'figures' / f'contact_{page:02d}.png', dep,
                    lambda start=start: draw_contact(paths[start:start+6], label + ' · global per-layer PCA'))
    def depth():
        fig = draw_depth(metrics, read_json(store.root / 'baselines.json'))
        fig.suptitle(label + ' · original-space directed k=4; undefined/degenerate omitted')
        return fig
    figure_file(store, 'figure/depth', store.root / 'figures/depth.png', dep, depth)
    figure_file(store, 'figure/pca_depth', store.root / 'figures/pca_depth.png', dep, lambda: pca_depth(metrics, label))
    if not store.valid('viewer/index', dep):
        from plotly.offline import get_plotlyjs
        viewer = store.root / 'viewer'
        template = (Path(__file__).parents[1] / 'fullrange/viewer.html').read_text().replace('Crystal', html.escape(label))
        atomic(viewer / 'index.html', lambda f: f.write(template.encode()))
        atomic(viewer / 'plotly.min.js', lambda f: f.write(get_plotlyjs().encode()))
        write_json(viewer / 'index.json', dict(experiment=store.manifest['config']['experiment'], model=label,
            synthetic=meta.get('synthetic', False), layers=[dict(layer=m['layer'], status=m['status']) for m in metrics],
            rows=[{k: r[k] for k in ('point_id', 'target', 'leading_digit', 'digit_count')} for r in rows],
            colors=COLORS, drawing_order=np.random.default_rng(42).permutation(len(rows)).tolist(), semantics=meta['levels']))
        store.finish('viewer/index', [viewer / p for p in ('index.html', 'index.json', 'plotly.min.js')], dep)
    if not store.valid('report', dep):
        write_json(store.root / 'layer_metrics.json', metrics)
        text = f'# {label}\n\n{len(rows)} targets, {len(metrics)} returned states.\n\n{meta["hidden_state_semantics"]}\n\n'
        text += 'All geometry is descriptive. Paper tokenization controls have not been reproduced. Models and layers have independent PCA frames.\n\n'
        text += LIMITATIONS + '\n\nSee layer_metrics.json, baselines.json and dataset_validation.json for full metrics and denominators.\n'
        atomic(store.root / 'SUMMARY.md', lambda f: f.write(text.encode()))
        store.finish('report', [store.root / 'SUMMARY.md', store.root / 'layer_metrics.json'], dep)
    store.stage('plots', 'complete')


def package(store):
    if store.manifest['stages'].get('plots') != 'complete':
        raise ValueError('Package requires completed plotting')
    dep = fingerprint({k: v['files'] for k, v in store.manifest['artifacts'].items()
                       if k not in ('lightweight_export',) and not k.startswith(('execution/', 'plan/', 'worker/'))})
    if store.valid('lightweight_export', dep):
        return
    path = store.root / 'lightweight.tar.gz'
    def write(f):
        with tarfile.open(fileobj=f, mode='w:gz') as archive:
            for item in sorted(store.root.rglob('*')):
                relative = item.relative_to(store.root)
                if item.is_file() and relative.parts[0] not in ('hidden', 'analysis_cache') and item != path and not item.name.endswith('.partial'):
                    archive.add(item, arcname=str(relative), recursive=False)
    atomic(path, write)
    with tarfile.open(path) as archive:
        if 'viewer/index.html' not in archive.getnames():
            raise ValueError('Invalid export')
    store.finish('lightweight_export', [path], dep, dict(hidden_states_included=False, cache_included=False,
        complete_download='download entire experiment directory; contains original float32 vectors and derived layer cache'))
    store.stage('package', 'complete')


def read_summary(root):
    """Read-only even for Crystal: never instantiate a recovering/writing Store."""
    root = Path(root)
    manifest = read_json(root / 'manifest.json')
    keys = ['dataset', 'baselines'] + sorted(k for k in manifest['artifacts'] if k.startswith('analysis/'))
    for key in keys:
        item = manifest['artifacts'][key]
        if item['status'] != 'complete' or any(digest(root / p) != h for p, h in item['files'].items()):
            raise ValueError('Comparison source incomplete/corrupt')
    rows = read_json(root / 'dataset.json')['records']
    meta = read_json(root / 'tokenizer_provenance.json')
    from . import RAW_FIELDS
    from collections import Counter
    metrics = [read_json(root / 'analysis' / f'layer_{int(k.split("/")[1]):02d}.json') for k in keys if k.startswith('analysis/')]
    return dict(config=manifest['config'], metadata=meta, metrics=metrics, baselines=read_json(root / 'baselines.json'),
        token_counts=dict(Counter(str(r['token_count']) for r in rows)),
        selected_text_sha256=fingerprint([{k: r[k] for k in RAW_FIELDS} for r in rows]),
        source_artifacts={k: manifest['artifacts'][k]['files'] for k in keys})


def comparison(roots, output):
    summaries = [read_summary(root) for root in roots]
    synthetic = any(s['metadata'].get('synthetic', False) for s in summaries)
    # Smoke compare selects the same subset from Crystal summaries only when all
    # sources have the same targets; the default full comparison requires all 10000.
    if len({s['selected_text_sha256'] for s in summaries}) != 1:
        raise ValueError('Cross-model text datasets differ; use full results for all three models')
    config = dict(experiment='numerical_model_comparison', sources=[s['config']['model_id'] for s in summaries])
    store = Store(output, config)
    dep = fingerprint(dict(sources=[s['source_artifacts'] for s in summaries], code=digest(__file__)))
    if not store.valid('comparison/data', dep):
        write_json(store.root / 'comparison.json', dict(models=summaries, synthetic=synthetic,
            caveats='Independent PCA frames; no absolute coordinate or raw-distance comparison; paper controls not reproduced'))
        store.finish('comparison/data', [store.root / 'comparison.json'], dep)
    def draw():
        fig, axes = plt.subplots(2, 3, figsize=(17, 9), layout='constrained')
        for summary in summaries:
            metrics = summary['metrics']; blocks = len(metrics) - 1
            label = summary['config']['model_id']
            x = [m['layer'] / blocks for m in metrics]
            series = [[sum(m.get('pca', {}).get('explained_variance_ratio', [np.nan])) for m in metrics],
                      [m.get('pca', {}).get('spearman_absolute', [np.nan, np.nan])[0] for m in metrics],
                      [m.get('pca', {}).get('spearman_absolute', [np.nan, np.nan])[1] for m in metrics]]
            series += [[m.get(k, np.nan) for m in metrics] for k in ('same_leading_digit_fraction', 'cross_digit_length_fraction', 'same_leading_digit_given_cross_length')]
            for ax, y in zip(axes.flat, series):
                ax.plot(x, [np.nan if v is None else v for v in y], marker='.', label=label)
            baseline = summary['baselines']['numerical_distance']
            for ax, key in zip(list(axes.flat)[3:], ('same_leading_digit_fraction', 'cross_digit_length_fraction', 'same_leading_digit_given_cross_length')):
                if baseline[key] is not None:
                    ax.axhline(baseline[key], color='gray', linestyle='--', alpha=.4)
        titles = ['PC1+PC2 variance', 'PC1 |rho(target)|', 'PC2 |rho(target)|', 'Same leading digit', 'Cross digit length', 'Same digit given cross length']
        for ax, title in zip(axes.flat, titles):
            ax.set(title=title, xlabel='Normalized transformer depth (0 embedding; 1 final normalized)', ylim=(0, 1))
            ax.grid(alpha=.2)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(('SYNTHETIC VALIDATION · ' if synthetic else '') + 'Saved-result comparison · independent PCA frames · dashed numerical baselines')
        return fig
    figure_file(store, 'comparison/curves', store.root / 'depth_comparison.png', dep, draw)
    if not store.valid('comparison/index', dep):
        import os
        lines = ['<!doctype html><meta charset="utf-8"><title>Numerical model comparison</title>',
                 '<h1>' + ('SYNTHETIC VALIDATION · ' if synthetic else '') + 'Saved numerical-representation comparison</h1>',
                 '<p>Independent PCA frames. Descriptive associations; paper tokenization controls have not been reproduced.</p>',
                 '<img src="depth_comparison.png" style="max-width:100%" alt="Normalized-depth descriptive comparisons">',
                 '<table border="1" cellpadding="8"><tr><th>Model/viewer</th><th>Revision</th><th>Dimensions / levels</th><th>Precision / special tokens</th></tr>']
        for root, summary in zip(roots, summaries):
            cfg, meta = summary['config'], summary['metadata']
            link = os.path.relpath(Path(root) / 'viewer/index.html', store.root)
            policy = cfg.get('special_policy', f'add_special_tokens={cfg.get("add_special_tokens")}; prepend_bos={cfg.get("prepend_bos")}')
            lines.append(f'<tr><td><a href="{html.escape(link)}">{html.escape(cfg["model_id"])}</a></td>'
                f'<td>{html.escape(cfg["model_revision"])}</td><td>{meta["expected_hidden_dim"]} / {meta["expected_levels"]}</td>'
                f'<td>{html.escape(cfg["model_dtype"])} / {html.escape(policy)}</td></tr>')
        lines += ['</table><p>L0 is the embedding-level state; final level includes final normalization. Actual indices, state semantics, tokenizer revisions/counts and numerical/frequency baselines are retained in <a href="comparison.json">comparison.json</a>.</p>']
        atomic(store.root / 'index.html', lambda f: f.write('\n'.join(lines).encode()))
        store.finish('comparison/index', [store.root / 'index.html'], dep)
    store.stage('comparison', 'complete')
    return store
