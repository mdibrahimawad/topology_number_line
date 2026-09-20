"""Saved-data audit; this module cannot invoke inference."""
import numpy as np
from scipy.spatial.distance import cdist
from PIL import Image
from numzig.fullrange.storage import fingerprint, write_json, read_json, digest
from numzig.fullrange.extract import inventory, read_layer
from numzig.fullrange.analysis import graph_metrics
from .data import load


def audit(store):
    rows = load(store)
    _, _, shape = inventory(store, rows, require_complete=True)
    dep = fingerprint(dict(code=digest(__file__), artifacts={k: v['files'] for k, v in store.manifest['artifacts'].items()
                       if k.startswith(('chunk/', 'analysis/', 'figure/', 'viewer/')) or k in ('dataset','baselines')}))
    # A dependency hash describes expected bytes, not their present integrity.
    # Validate source files before adopting a previous audit.
    for key in store.manifest['artifacts']:
        if key.startswith(('analysis/', 'figure/', 'viewer/')) or key == 'baselines':
            store.require(key)
    if store.valid('validation', dep):
        store.log('[SKIP] validation dependency unchanged')
        return read_json(store.root / 'validation.json')
    ids = np.array([r['point_id'] for r in rows], dtype=np.int32)
    checked = []
    for layer in range(shape[0]):
        key = f'analysis/{layer:02d}'; store.require(key)
        base = store.root / 'analysis' / f'layer_{layer:02d}'
        metric = read_json(base.with_suffix('.json'))
        if metric['status'] == 'degenerate':
            checked.append(dict(layer=layer, status='degenerate')); continue
        with np.load(base.with_suffix('.npz'), allow_pickle=False) as f:
            data = {k: f[k] for k in f.files}
        if not np.array_equal(data['point_ids'], ids): raise ValueError('Point order mismatch')
        x = read_layer(store, layer, rows).astype(np.float64)
        selected = np.arange(len(rows)) if len(rows) <= 100 else np.unique(np.linspace(0, len(rows)-1, 16, dtype=int))
        direct = cdist(x[selected], x)
        direct[np.arange(len(selected)), selected] = np.inf
        order = np.argsort(direct, axis=1, kind='stable')[:, :4]
        if not np.array_equal(ids[order], data['neighbors'][selected]): raise ValueError('Exact-neighbor audit failed')
        np.testing.assert_allclose(np.take_along_axis(direct, order, axis=1), data['distances'][selected], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(x.mean(axis=0), data['mean'], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose((x-data['mean']) @ data['components'].T, data['scores'], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(data['components'] @ data['components'].T, np.eye(2), atol=1e-10)
        np.testing.assert_allclose(data['scores'].var(axis=0, ddof=1), data['explained_variance'], rtol=1e-9, atol=1e-10)
        metric_again, arrays = graph_metrics(rows, data['neighbors'], data['distances'])
        for name in ('digit_counts', 'digit_fractions', 'union_edges', 'low_to_high', 'high_to_low', 'mutual'):
            np.testing.assert_allclose(arrays[name], data[name], equal_nan=True)
        for name in ('conditional_denominator', 'directed_relation_count', 'cross_digit_length_count', 'same_leading_digit_fraction'):
            if metric_again[name] != metric[name]: raise ValueError('Metric denominator mismatch')
        if data['digit_counts'].sum() != len(rows)*4: raise ValueError('Heatmap lost directed relations')
        checked.append(dict(layer=layer, status='valid', exact_neighbor_rows_checked=len(selected), candidates_per_row=len(rows)))
    # All published figure and viewer hashes are checked, not only sampled plots.
    for key, item in list(store.manifest['artifacts'].items()):
        if key.startswith(('figure/', 'viewer/')):
            store.require(key)
            for path in item['files']:
                if path.endswith('.png'):
                    with Image.open(store.root/path) as im: im.verify()
    meta = read_json(store.root/'tokenizer_provenance.json')
    report = dict(passed=True, points=len(rows), shape=[shape[0], len(rows), shape[1]], layers=checked,
        synthetic=meta.get('synthetic', False), pretrained_equivalence_passed=False if meta.get('synthetic') else 'see compatible smoke gate',
        scope='All chunk hashes/IDs/finiteness; all layer metrics/PCA and images; full-candidate sampled kNN (all rows for smoke). Browser visual review remains separate.')
    write_json(store.root/'validation.json', report)
    store.finish('validation', [store.root/'validation.json'], dep)
    store.stage('validation', 'complete')
    return report
