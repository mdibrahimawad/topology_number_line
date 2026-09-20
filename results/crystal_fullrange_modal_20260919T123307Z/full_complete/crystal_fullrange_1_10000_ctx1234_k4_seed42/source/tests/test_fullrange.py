import json
from pathlib import Path

import numpy as np
import pytest

from numzig.fullrange import configuration
from numzig.fullrange.__main__ import SyntheticTokenizer, synthetic_vector, dry_run
from numzig.fullrange.dataset import records, tokenize, prepare, load_dataset, SMOKE_TARGETS
from numzig.fullrange.storage import Store, digest, read_json, write_json
from numzig.fullrange.extract import extract_chunks, inventory, read_layer
from numzig.fullrange.analysis import (exact_knn, union_graph, graph_metrics, projection, degeneration, analyze)


def fixture_store(path, chunk_size=8):
    cfg = configuration(True)
    cfg['chunk_size'] = chunk_size
    store = Store(path, cfg)
    prepare(store, tokenizer=SyntheticTokenizer(), tokenizer_provenance={'synthetic': True})
    return store


def hashes(store, prefix):
    return {p: (digest(store.root / p), (store.root / p).stat().st_mtime_ns)
            for key, value in store.manifest['artifacts'].items() if key.startswith(prefix)
            for p in value['files']}


def test_full_dataset_and_context_policy():
    cfg = configuration()
    a = records(cfg)
    assert a == records(cfg)
    assert [r['target'] for r in a] == list(range(1, 10001))
    assert [r['point_id'] for r in a] == list(range(10000))
    assert np.bincount([r['leading_digit'] for r in a])[1:].tolist() == [1112] + [1111] * 8
    for r in a:
        demos = list(r['demonstrations'].values())
        assert all(lo <= d <= hi for d, lo, hi in zip(demos, [1, 10, 100, 1000], [9, 99, 999, 9999]))
        assert r['demonstration_digit_lengths'] == [1, 2, 3, 4]
        assert [len(str(d)) for d in demos] == [1, 2, 3, 4]
        assert r['prompt'].split(',') == [f'{d}={d}' for d in demos] + [f'{r["target"]}=']
        assert r['character_count'] == len(r['prompt'])
        assert r['digit_count'] == len(str(r['target']))
    assert len({tuple(r['demonstrations'].values()) for r in a}) > 9900
    assert {r['character_count'] for r in a} == {30, 31, 32, 33, 34}
    smoke = records(configuration(True))
    assert smoke == [r for r in a if r['target'] in SMOKE_TARGETS]
    assert {r['leading_digit'] for r in smoke} == set(range(1, 10))
    assert {r['digit_count'] for r in smoke} == set(range(1, 6))
    assert {9, 10, 99, 100, 999, 1000, 9999, 10000} <= {r['target'] for r in smoke}


def test_tokenization_metadata_and_invalid_prefix():
    cfg = configuration(True)
    rows = tokenize(records(cfg), SyntheticTokenizer())
    for r in rows:
        assert r['extraction_position'] == r['token_count'] - 1
        assert r['final_token_text'] == '='
        assert r['continuation_token_ids'] == [ord(c) for c in str(r['target'])]
        assert r['first_continuation_token'] == ord(str(r['target'])[0])
    class Merge(SyntheticTokenizer):
        def __call__(self, text, **kwargs):
            return {'input_ids': [ord(c) for c in text] if text.endswith('=') else [123]}
    bad = tokenize(records(cfg), Merge())
    assert all(not r['continuation_prefix_valid'] and r['continuation_token_ids'] is None for r in bad)
    class Bad(SyntheticTokenizer):
        def decode(self, ids, **kwargs):
            return 'pad'
    with pytest.raises(ValueError, match='final token'):
        tokenize(records(cfg), Bad())


def test_dataset_saved_reused_and_config_rejected(tmp_path, monkeypatch):
    s = fixture_store(tmp_path / 'run')
    before = hashes(s, 'dataset')
    def no_generation(*a, **kw):
        raise AssertionError('Saved prompts must never be regenerated')
    monkeypatch.setattr('numzig.fullrange.dataset.records', no_generation)
    prepare(s, tokenizer=SyntheticTokenizer())
    assert hashes(s, 'dataset') == before
    with pytest.raises(ValueError, match='Incompatible'):
        Store(s.root, {**s.manifest['config'], 'seed': 99})
    p = s.root / 'dataset.json'
    p.write_text(p.read_text() + ' ')
    with pytest.raises(ValueError, match='corrupt'):
        prepare(s, tokenizer=SyntheticTokenizer())


def test_interruption_recovery_matches_uninterrupted(tmp_path):
    s = fixture_store(tmp_path / 'resumed')
    call_ids = []
    def interrupt(row):
        call_ids.append(row['point_id'])
        if len(call_ids) == 12:
            raise RuntimeError('synthetic interruption inside second chunk')
        return synthetic_vector(row)
    with pytest.raises(RuntimeError, match='interruption'):
        extract_chunks(s, interrupt)
    assert len(inventory(s, load_dataset(s))[0]) == 1
    before = hashes(s, 'chunk/')
    resumed = Store(s.root, s.manifest['config'])
    retry_ids = []
    def retry(row):
        retry_ids.append(row['point_id'])
        return synthetic_vector(row)
    extract_chunks(resumed, retry)
    assert not set(call_ids[:8]) & set(retry_ids)
    assert all(hashes(resumed, 'chunk/')[p] == old for p, old in before.items())
    reference = fixture_store(tmp_path / 'reference')
    extract_chunks(reference, synthetic_vector)
    for layer in range(3):
        np.testing.assert_array_equal(read_layer(resumed, layer), read_layer(reference, layer))
    before_all = hashes(resumed, 'chunk/')
    extract_chunks(resumed, lambda row: pytest.fail('No completed inference may be repeated'))
    assert before_all == hashes(resumed, 'chunk/')


@pytest.mark.parametrize('failure_commit', [1, 2])
def test_recovery_at_each_chunk_commit(tmp_path, failure_commit):
    s = fixture_store(tmp_path / str(failure_commit))
    commits = []
    def commit():
        if 'chunk/00000' not in read_json(s.path)['artifacts']:
            return
        commits.append(read_json(s.path)['artifacts']['chunk/00000']['status'])
        if len(commits) == failure_commit:
            raise RuntimeError('commit interrupted')
    s.commit = commit
    with pytest.raises(RuntimeError, match='commit interrupted'):
        extract_chunks(s, synthetic_vector)
    assert commits[0] == 'prepared'
    before = hashes(s, 'chunk/')
    resumed = Store(s.root, s.manifest['config'])
    calls = []
    def infer(row):
        calls.append(row['point_id'])
        return synthetic_vector(row)
    extract_chunks(resumed, infer)
    assert not set(calls) & set(r['point_id'] for r in load_dataset(s)[:8])
    assert all(hashes(resumed, 'chunk/')[p] == value for p, value in before.items())


def test_partial_and_corrupt_chunks_only_redo_affected(tmp_path):
    s = fixture_store(tmp_path / 'run')
    extract_chunks(s, synthetic_vector)
    before = hashes(s, 'chunk/')
    (s.root / 'hidden/00000.npy.partial').write_bytes(b'incomplete')
    damaged = s.root / 'hidden/00001.npy'
    damaged.write_bytes(b'broken')
    calls = []
    def infer(row):
        calls.append(row['point_id'])
        return synthetic_vector(row)
    extract_chunks(Store(s.root), infer)
    assert calls == [r['point_id'] for r in load_dataset(s)[8:16]]
    after = hashes(Store(s.root), 'chunk/')
    for path, pair in before.items():
        if not path.startswith('hidden/00001'):
            assert after[path] == pair
    assert np.isfinite(read_layer(s, 1)).all()


def test_shape_dtype_finiteness_and_point_alignment(tmp_path):
    s = fixture_store(tmp_path / 'bad')
    with pytest.raises(ValueError, match='finite float32'):
        extract_chunks(s, lambda r: np.zeros((3, 8), dtype=np.float64))
    with pytest.raises(ValueError, match='finite float32'):
        extract_chunks(s, lambda r: np.full((3, 8), np.nan, dtype=np.float32))
    with pytest.raises(ValueError, match='expected'):
        extract_chunks(s, synthetic_vector, expected_shape=(33, 4096))
    assert not inventory(s, load_dataset(s))[0]


@pytest.mark.parametrize('offset', [0., 1e12])
def test_exact_knn_direct_reference_duplicates_cutoff_ties(offset):
    x = np.array([[0, 0], [0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [2, 0], [3, 0]], dtype=float) + offset
    ids = np.array([0, 2, 5, 10, 12, 30, 100, 9999], dtype=np.int32)
    n, d = exact_knn(x, ids, block_size=3)
    for i in range(len(x)):
        expected = sorted(((float(np.linalg.norm(x[i] - x[j])), int(ids[j])) for j in range(len(x)) if j != i))[:4]
        assert n[i].tolist() == [p[1] for p in expected]
        np.testing.assert_allclose(d[i], [p[0] for p in expected], rtol=1e-14, atol=1e-14)
        assert ids[i] not in n[i] and len(set(n[i])) == 4
    assert n[0].tolist() == [2, 5, 10, 12]
    union = union_graph(ids, n)
    assert len(union['union_edges']) == len({tuple(sorted((int(i), int(j)))) for i, row in zip(ids, n) for j in row})
    assert np.all(union['low_to_high'] | union['high_to_low'])
    assert np.array_equal(union['mutual'], union['low_to_high'] & union['high_to_low'])


def test_pca_scores_metrics_and_degenerate():
    rows = records(configuration(True))
    x = np.stack([synthetic_vector(r)[1] for r in rows])
    p, meta = projection(x, [r['target'] for r in rows])
    np.testing.assert_allclose((x.astype(float) - p['mean']) @ p['components'].T, p['scores'], atol=1e-12)
    assert all(v >= 0 for v in meta['spearman_signed'])
    assert degeneration(np.ones_like(x))['degenerate']
    assert not degeneration(x)['degenerate']
    ids = np.array([r['point_id'] for r in rows])
    n, d = exact_knn(x, ids)
    m, a = graph_metrics(rows, n, d, previous=n)
    np.testing.assert_allclose(a['digit_fractions'].sum(axis=1), 1)
    assert m['preceding_valid_layer_overlap_fraction'] == 1
    assert m['preceding_valid_layer_overlap_denominator'] == len(rows) * 4
    short = records(configuration())[:9]
    n, d = exact_knn(np.arange(9)[:, None], np.arange(9))
    m, a = graph_metrics(short, n, d)
    assert m['conditional_denominator'] == 0
    assert m['same_leading_digit_given_cross_length'] is None


def test_analysis_resume_and_no_pca_in_knn(tmp_path, monkeypatch):
    s = fixture_store(tmp_path / 'run')
    extract_chunks(s, synthetic_vector)
    import numzig.fullrange.analysis as module
    original = module.projection
    def fail_second(x, targets):
        if (s.root / 'analysis/layer_01.json').exists():
            raise RuntimeError('analysis interrupted')
        return original(x, targets)
    monkeypatch.setattr(module, 'projection', fail_second)
    with pytest.raises(RuntimeError, match='analysis interrupted'):
        analyze(s)
    before = hashes(s, 'analysis/')
    monkeypatch.setattr(module, 'projection', original)
    resumed = Store(s.root)
    analyze(resumed)
    assert all(hashes(resumed, 'analysis/')[p] == value for p, value in before.items())
    assert read_json(s.root / 'analysis/layer_00.json')['status'] == 'degenerate'
    with np.load(s.root / 'analysis/layer_00.npz') as data:
        assert 'neighbors' not in data and 'scores' not in data
    reference = fixture_store(tmp_path / 'uninterrupted')
    extract_chunks(reference, synthetic_vector)
    analyze(reference)
    for layer in range(3):
        with np.load(s.root / f'analysis/layer_{layer:02d}.npz') as actual, np.load(reference.root / f'analysis/layer_{layer:02d}.npz') as expected:
            assert set(actual.files) == set(expected.files)
            for field in actual.files:
                np.testing.assert_array_equal(actual[field], expected[field])
    monkeypatch.setattr(module, 'exact_knn', lambda *a, **k: pytest.fail('Completed kNN reran'))
    monkeypatch.setattr(module, 'projection', lambda *a, **k: pytest.fail('Completed PCA reran'))
    analyze(resumed)


def test_dry_run_no_model_or_modal_import():
    result = dry_run()
    assert result['prompt_count'] == 10000
    assert result['hidden_float32_bytes'] == 5406720000
    assert result['cloud_execution'] is False


def test_archived_real_fast_tokenizer_all_10000_targets():
    # Local tokenizer-only test: no weights, model object, network, or inference.
    from tokenizers import Tokenizer
    assets = Path(__file__).resolve().parents[1] / 'crystal_L27_L28_hypothesis_test/tokenizer'
    if not (assets / 'tokenizer.json').exists():
        pytest.skip('Archived tokenizer is not part of the source-only cloud upload')
    backend = Tokenizer.from_file(str(assets / 'tokenizer.json'))
    class SavedTokenizer:
        is_fast = True
        def __call__(self, text, **kwargs):
            return {'input_ids': backend.encode(text, add_special_tokens=True).ids}
        def decode(self, ids, **kwargs):
            return backend.decode(ids, skip_special_tokens=False)
    rows = tokenize(records(configuration()), SavedTokenizer())
    assert len(rows) == 10000
    assert all(r['final_token_text'] == '=' and r['continuation_prefix_valid'] for r in rows)
    assert {r['token_count'] for r in rows} == {31, 32, 33, 34, 35}
    assert all(r['continuation_length'] == len(str(r['target'])) for r in rows)


def test_numerical_baseline_full_coverage_and_frequency():
    rows = records(configuration())
    ids = np.arange(10000, dtype=np.int32)
    neighbors, distances = exact_knn((ids + 1)[:, None], ids, block_size=64)
    assert neighbors.shape == (10000, 4)
    assert neighbors[0].tolist() == [1, 2, 3, 4]
    assert neighbors[5000].tolist() == [4999, 5001, 4998, 5002]
    metrics, arrays = graph_metrics(rows, neighbors, distances)
    assert metrics['directed_relation_count'] == 40000
    assert metrics['weak_component_count'] == 1
    np.testing.assert_allclose(arrays['digit_fractions'].sum(axis=1), 1)
    counts = np.bincount([r['leading_digit'] for r in rows])[1:]
    expected = np.sum(counts * (counts - 1)) / (10000 * 9999)
    assert expected == pytest.approx(np.sum(counts * ((counts - 1) / 9999)) / 10000)


def test_plot_interruption_and_independent_artifact_repair(tmp_path, monkeypatch):
    from numzig.fullrange import plots
    import numzig.fullrange.analysis as geometry
    import numzig.fullrange.extract as extraction
    s = fixture_store(tmp_path / 'run')
    extract_chunks(s, synthetic_vector)
    analyze(s)
    def forbidden(*args, **kwargs):
        raise AssertionError('Plot-only must not infer, run kNN, or fit PCA')
    monkeypatch.setattr(extraction, 'extract_model', forbidden)
    monkeypatch.setattr(geometry, 'projection', forbidden)
    monkeypatch.setattr(geometry, 'exact_knn', forbidden)
    original = plots.draw_graph
    def interrupt(data, metric, *args, **kwargs):
        if metric['layer'] == 1:
            raise RuntimeError('plot interrupted')
        return original(data, metric, *args, **kwargs)
    monkeypatch.setattr(plots, 'draw_graph', interrupt)
    with pytest.raises(RuntimeError, match='plot interrupted'):
        plots.render(s)
    before = hashes(s, 'figure/')
    monkeypatch.setattr(plots, 'draw_graph', original)
    resumed = Store(s.root)
    plots.render(resumed)
    assert all(hashes(resumed, 'figure/')[p] == value for p, value in before.items())
    expected = ['figures/layer_01_magnitude.png', 'figures/layer_02_heatmap.png']
    (s.root / expected[0]).unlink()
    (s.root / expected[1]).write_bytes(b'corrupt png')
    before_repair = {
        p: (digest(s.root / p), (s.root / p).stat().st_mtime_ns)
        for v in resumed.manifest['artifacts'].values() for p in v['files']
        if p.startswith('figures/') and p not in expected}
    draws = []
    def track(data, metric, rows, order, mode):
        draws.append((metric['layer'], mode))
        return original(data, metric, rows, order, mode)
    monkeypatch.setattr(plots, 'draw_graph', track)
    plots.render(Store(s.root))
    assert draws == [(1, 'magnitude'), (2, 'heatmap')]
    for p, pair in before_repair.items():
        assert (digest(s.root / p), (s.root / p).stat().st_mtime_ns) == pair
    assert (s.root / 'viewer/index.html').exists()
    assert (s.root / 'lightweight.tar.gz').exists()
    assert 'SYNTHETIC' in (s.root / 'SUMMARY.md').read_text()
    with np.load(s.root / 'analysis/layer_01.npz') as data:
        viewer = read_json(s.root / 'viewer/layer_01.json')
        np.testing.assert_array_equal(viewer['scores'], data['scores'])
        assert len(viewer['scores']) == len(SMOKE_TARGETS)


@pytest.mark.parametrize('fail_at', [1, 2])
def test_restore_only_durably_committed_snapshot(tmp_path, fail_at):
    import shutil
    s = fixture_store(tmp_path / 'live')
    snapshot = tmp_path / 'persistent_volume'
    shutil.copytree(s.root, snapshot)
    chunk_commits = 0
    def commit():
        nonlocal chunk_commits
        if 'chunk/00000' in read_json(s.path)['artifacts']:
            chunk_commits += 1
            if chunk_commits == fail_at:
                raise RuntimeError('Simulated interrupted Modal commit')
        shutil.copytree(s.root, snapshot, dirs_exist_ok=True)
    s.commit = commit
    with pytest.raises(RuntimeError, match='interrupted Modal commit'):
        extract_chunks(s, synthetic_vector)
    # Simulate a NEW container: only a successful commit's snapshot is visible.
    restored = tmp_path / 'new_container'
    shutil.copytree(snapshot, restored)
    resume = Store(restored)
    calls = []
    def infer(row):
        calls.append(row['point_id'])
        return synthetic_vector(row)
    extract_chunks(resume, infer)
    ids = [r['point_id'] for r in load_dataset(resume)]
    assert calls == (ids if fail_at == 1 else ids[8:])
    for layer in range(3):
        np.testing.assert_array_equal(read_layer(resume, layer),
                                     np.stack([synthetic_vector(r)[layer] for r in load_dataset(resume)]))
