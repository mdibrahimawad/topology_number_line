"""Read-only audit of downloaded artifacts, supplementing the remote numeric audit."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def audit(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    files = {}
    for record in manifest['artifacts'].values():
        assert record['status'] == 'complete'
        for name, expected in record['files'].items():
            assert files.setdefault(name, expected) == expected
    for name, expected in files.items():
        assert sha(root / name) == expected, name
    data = json.loads((root / 'dataset.json').read_text())
    rows = data['records']
    crystal = json.loads(Path('results/crystal_fullrange_modal_20260919T123307Z/full_complete/crystal_fullrange_1_10000_ctx1234_k4_seed42/dataset.json').read_text())['records']
    ids = np.array([r['point_id'] for r in rows])
    assert len(set(ids)) == len(ids)
    if not data['config']['smoke']:
        assert np.array_equal(ids, np.arange(10000))
    for row in rows:
        assert row['target'] == row['point_id'] + 1
        for key in ('prompt', 'demonstrations', 'seed'):
            assert row[key] == crystal[row['point_id']][key]
    chunks = sorted((root / 'hidden').glob('*_ids.npy'))
    assert np.array_equal(np.concatenate([np.load(p) for p in chunks]), ids)
    vectors = [np.load(str(p).replace('_ids.npy', '.npy'), mmap_mode='r') for p in chunks]
    meta = json.loads((root / 'tokenizer_provenance.json').read_text())
    for v, p in zip(vectors, chunks):
        assert v.shape == (meta['expected_levels'], len(np.load(p)), meta['expected_hidden_dim'])
        assert v.dtype == np.float32 and np.isfinite(v).all()
    layers = []
    for layer in range(meta['expected_levels']):
        metric = json.loads((root / f'analysis/layer_{layer:02d}.json').read_text())
        x = np.concatenate([v[layer] for v in vectors]).astype(np.float64)
        # Independent degeneration measurement, including learned-position L0.
        centered_rms = float(np.sqrt(np.mean(np.sum((x - x.mean(axis=0)) ** 2, axis=1))))
        rms = float(np.sqrt(np.mean(np.sum(x ** 2, axis=1))))
        degenerate = centered_rms <= 1e-7 * max(1, rms)
        assert degenerate == (metric['status'] == 'degenerate')
        item = dict(layer=layer, status=metric['status'], centered_rms=centered_rms)
        if not degenerate:
            with np.load(root / f'analysis/layer_{layer:02d}.npz') as a:
                nb = a['neighbors']
                assert nb.shape == (len(ids), 4) and np.isin(nb, ids).all()
                assert (nb != ids[:, None]).all()
                assert (np.diff(np.sort(nb, axis=1), axis=1) != 0).all()
                assert nb.size == len(ids) * 4 == metric['directed_relation_count']
                # Verify every saved edge distance; sample only the global rank search.
                positions = np.searchsorted(ids, nb)
                for start in range(0, len(ids), 256):
                    delta = x[positions[start:start+256]] - x[start:start+256, None, :]
                    expected_distances = np.sqrt(np.sum(delta * delta, axis=2))
                    np.testing.assert_allclose(a['distances'][start:start+256], expected_distances, atol=1e-12, rtol=1e-12)
                directed = {(int(i), int(j)) for i, neighbors in zip(ids, nb) for j in neighbors}
                edges = sorted({tuple(sorted(pair)) for pair in directed})
                np.testing.assert_array_equal(a['union_edges'], edges)
                np.testing.assert_array_equal(a['low_to_high'], [(i,j) in directed for i,j in edges])
                np.testing.assert_array_equal(a['high_to_low'], [(j,i) in directed for i,j in edges])
                np.testing.assert_array_equal(a['mutual'], [(i,j) in directed and (j,i) in directed for i,j in edges])
                selected = np.unique(np.linspace(0, len(ids)-1, min(16, len(ids)), dtype=int))
                distances = cdist(x[selected], x)
                distances[np.arange(len(selected)), selected] = np.inf
                order = np.argsort(distances, axis=1, kind='stable')[:, :4]
                assert np.array_equal(nb[selected], ids[order])
                np.testing.assert_allclose(a['distances'][selected], np.take_along_axis(distances, order, axis=1), atol=1e-12, rtol=1e-12)
                viewer = json.loads((root / f'viewer/layer_{layer:02d}.json').read_text())
                np.testing.assert_array_equal(viewer['scores'], a['scores'])
                item.update(directed_relations=int(nb.size), all_edge_distances_checked=int(nb.size),
                            union_edges_checked=len(edges), sampled_rows=len(selected), candidates=len(ids))
        layers.append(item)
    assert json.loads((root / 'validation.json').read_text())['passed']
    report = dict(root=str(root), passed=True, points=len(ids), files_checked=len(files), layers=layers,
                  manifest_sha256=sha(root / 'manifest.json'), prompt_match=True)
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    audit(Path(sys.argv[1]))
