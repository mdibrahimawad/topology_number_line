import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

from numzig.fullrange.analysis import exact_knn
from numzig.fullrange.storage import digest
from numzig.taskrules.analyze import analyze_group, geometry, group_variance, persistence, _labels


def records(n):
    return [dict(point_id=i * 2, target=1000 + i, task='reverse', context_id=0,
                 expected_output=str(1000 + i)[::-1], source_digits=list(map(int, str(1000 + i))),
                 output_digits=list(map(int, str(1000 + i)[::-1])), token_count=24 + i % 2)
            for i in range(n)]


def test_knn_matches_existing_exact_kernel_and_pca_reconstruction():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(20, 9)).astype(np.float32)
    x[1] = x[0]
    ids = np.arange(20) * 2
    arrays, metric, distances = geometry(x, ids, np.arange(1000, 1020))
    neighbors, nd = exact_knn(x, ids)
    np.testing.assert_array_equal(arrays['neighbors'], neighbors)
    np.testing.assert_allclose(arrays['distances'], nd, atol=1e-13)
    np.testing.assert_allclose(distances, squareform(pdist(x.astype(float))))
    np.testing.assert_allclose(arrays['scores3'], (x - arrays['mean']) @ arrays['components'].T)
    np.testing.assert_allclose(arrays['components'] @ arrays['components'].T, np.eye(3), atol=1e-10)
    assert metric['knn']['k'] == 4
    assert len(metric['pca']['spearman_absolute']) == 3


def test_group_variance_and_constant_cloud():
    x = np.array([[0, 0, 1], [0, 0, 1], [5, 2, 1], [5, 2, 1]], dtype=float)
    assert group_variance(x, [0, 0, 1, 1])['fraction'] == pytest.approx(1)
    assert group_variance(x, [0, 1, 0, 1])['fraction'] == pytest.approx(0)
    assert group_variance(x, [0, 0, 0, 0], [False] * 4)['fraction'] is None
    arrays, meta, distances = geometry(np.ones((8, 3)), np.arange(8), np.arange(8))
    assert meta['status'] == 'degenerate' and distances is None and 'scores' not in arrays


def test_ph_circle_and_duplicate_zero_bars():
    angles = np.arange(24) * 2 * np.pi / 24
    x = np.column_stack([np.sin(angles), np.cos(angles)])
    x = np.concatenate([x, x[:1]])
    arrays, meta = persistence(squareform(pdist(x)))
    assert meta['status'] == 'complete'
    assert meta['zero_lifetime_h0_omitted'] == 1
    assert meta['max_h1_lifetime'] > .5
    assert np.isinf(arrays['h0'][:, 1]).sum() == 1
    assert arrays['betti_h0'][-1] == 1
    assert arrays['betti_h1'][-1] == 0


def test_word_labels_and_missing_source_positions():
    rows = [dict(target=1, expected_output='one', token_count=7),
            dict(target=234, expected_output='two hundred and thirty-four', input_token_ids=[1] * 11)]
    labels = _labels(rows)
    assert labels['output_first_label'].tolist() == ['one', 'two']
    assert labels['source_digits'].tolist() == [[1, -1, -1, -1], [2, 3, 4, -1]]
    assert labels['token_counts'].tolist() == [7, 11]
    assert labels['output_label_kind'] == 'first output word'


def test_resume_and_selective_corruption_repair(tmp_path, monkeypatch):
    from numzig.taskrules import analyze
    x = np.random.default_rng(11).normal(size=(2, 9, 7)).astype(np.float32)
    calls = []
    summary = analyze_group(x, records(9), tmp_path / 'result', 'synthetic', do_ph=False, commit=lambda: calls.append(1))
    assert summary['layer_count'] == 2 and calls
    files = list((tmp_path / 'result' / 'layers').iterdir())
    original = {p: (digest(p), p.stat().st_mtime_ns) for p in files}
    original_geometry = analyze.geometry
    monkeypatch.setattr(analyze, 'geometry', lambda *a: pytest.fail('Complete analysis must not repeat'))
    assert analyze_group(x, records(9), tmp_path / 'result', 'synthetic', do_ph=False) == summary
    assert all((digest(p), p.stat().st_mtime_ns) == value for p, value in original.items())
    monkeypatch.setattr(analyze, 'geometry', original_geometry)
    (tmp_path / 'result' / 'layers' / 'layer_01.npz').write_bytes(b'damaged')
    analyze_group(x, records(9), tmp_path / 'result', 'synthetic', do_ph=False)
    assert all((digest(p), p.stat().st_mtime_ns) == value for p, value in original.items() if 'layer_00' in p.name)
    changed = x.copy()
    changed[0, 0, 0] += 1
    with pytest.raises(ValueError, match='Incompatible'):
        analyze_group(changed, records(9), tmp_path / 'result', 'synthetic', do_ph=False)


def test_group_boundaries_and_target_alignment(tmp_path):
    x = np.zeros((1, 8, 5), dtype=np.float32)
    rows = records(8)
    rows[0]['context_id'] = 1
    with pytest.raises(ValueError, match='do not pool'):
        analyze_group(x, rows, tmp_path / 'mixed', 'synthetic')
    rows = records(8)
    rows[0]['target'] = rows[1]['target']
    with pytest.raises(ValueError, match='unique'):
        analyze_group(x, rows, tmp_path / 'duplicate', 'synthetic')


def test_degenerate_layer_and_ph_end_to_end(tmp_path):
    angles = np.arange(16) * 2 * np.pi / 16
    circle = np.column_stack([np.sin(angles), np.cos(angles), .1 * np.sin(angles * 2), np.zeros(16)]).astype(np.float32)
    x = np.stack([np.zeros_like(circle), circle])
    summary = analyze_group(x, records(16), tmp_path / 'result', 'synthetic')
    assert summary['layers'][0]['status'] == 'degenerate'
    assert summary['layers'][1]['ph']['status'] == 'complete'
    assert len(summary['layers'][0]['figures']) == 1
    assert len(summary['layers'][1]['figures']) == 3
    with np.load(tmp_path / 'result' / 'layers' / 'layer_01.npz') as saved:
        np.testing.assert_array_equal(saved['targets'], np.arange(1000, 1016))
        assert saved['scores3'].shape == (16, 3)
        assert 'h0' in saved and 'h1' in saved
