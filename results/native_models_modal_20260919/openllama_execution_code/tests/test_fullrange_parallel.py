"""Local synthetic concurrency and durable-volume recovery. No model/Modal jobs."""
from concurrent.futures import ThreadPoolExecutor
import shutil
import threading

import numpy as np
import pytest

from numzig.fullrange.__main__ import synthetic_vector
from numzig.fullrange.analysis import (analyze, analysis_dependency, compute_layer, pending_layers)
from numzig.fullrange.coordinator import recover_extraction, recover_analysis, drain_calls
from numzig.fullrange.dataset import load_dataset, records
from numzig.fullrange import configuration
from numzig.fullrange.extract import chunks, extract_chunks, extraction_dependency, read_layer
from numzig.fullrange.parallel import WorkerStore, make_plan, partition, execution_settings
from numzig.fullrange.storage import Store, digest, read_json, write_json
from test_fullrange import fixture_store, hashes


def run_workers(store, plan, action):
    with ThreadPoolExecutor(max_workers=len(plan['assignments'])) as pool:
        futures = [pool.submit(action, WorkerStore(store.root, plan['id'], slot), slot)
                   for slot in range(len(plan['assignments']))]
        errors = []
        for f in futures:
            try:
                f.result()
            except RuntimeError as exc:
                errors.append(exc)
    return errors


def test_full_disjoint_coverage_and_bounds():
    cfg = configuration()
    class S:
        manifest = {'config': cfg}
    parts = {key: part for _, part, key in chunks(S(), records(cfg))}
    for count in [1, 2, 7, 10]:
        assignments = partition(list(parts), count)
        assert len(assignments) == count
        assert sorted(k for a in assignments for k in a) == sorted(parts)
        targets = [r['target'] for a in assignments for k in a for r in parts[k]]
        assert sorted(targets) == list(range(1, 10001))
    assert partition([], 10) == []
    assert execution_settings()['aggregate_analysis_cpu_ceiling'] == 17
    for gpu, cpu in [(0, 4), (11, 4), (10, 0), (10, 13)]:
        with pytest.raises(ValueError):
            execution_settings(gpu, cpu)
    with pytest.raises(ValueError):
        partition(['chunk/00000'] * 2, 2)


def test_parallel_failure_merge_resume_different_count_and_serial_equivalence(tmp_path):
    s = fixture_store(tmp_path / 'parallel', chunk_size=4)
    dep = extraction_dependency(s)
    plan = make_plan(s, 'chunk', recover_extraction(s), 3, dep)
    manifest_before = s.path.read_bytes()
    barrier = threading.Barrier(3)
    seen = {i: [] for i in range(3)}
    def work(worker, slot):
        def infer(row):
            if not seen[slot]:
                barrier.wait(timeout=10)
            seen[slot].append(row['point_id'])
            if slot == 0 and len(seen[slot]) == 6:
                raise RuntimeError('worker interrupted inside second assigned chunk')
            return synthetic_vector(row)
        extract_chunks(worker, infer, assigned_keys=worker.assigned)
    assert len(run_workers(s, plan, work)) == 1
    assert s.path.read_bytes() == manifest_before  # No worker touched shared metadata.
    s = Store(s.root)
    missing = recover_extraction(s)
    assert missing and len(missing) < len(plan['assignments'][0])
    assert all(k in s.manifest['artifacts'] for k in plan['assignments'][1] + plan['assignments'][2])
    before = hashes(s, 'chunk/')
    completed_ids = {r['point_id'] for _, part, k in chunks(s, load_dataset(s)) if k not in missing for r in part}
    retry_ids = []
    resumed_plan = make_plan(s, 'chunk', missing, 2, dep)
    def resume(worker, slot):
        def infer(row):
            retry_ids.append(row['point_id'])
            return synthetic_vector(row)
        extract_chunks(worker, infer, assigned_keys=worker.assigned)
    assert not run_workers(s, resumed_plan, resume)
    s = Store(s.root)
    assert recover_extraction(s) == []
    assert not completed_ids & set(retry_ids)
    assert all(hashes(s, 'chunk/')[p] == old for p, old in before.items())
    assert len([k for k in s.manifest['artifacts'] if k.startswith('worker/')]) >= 4
    reference = fixture_store(tmp_path / 'serial', chunk_size=4)
    extract_chunks(reference, synthetic_vector)
    for layer in range(3):
        np.testing.assert_array_equal(read_layer(s, layer), read_layer(reference, layer))
    # Same input redelivered to the same worker also skips its durable receipts.
    worker = WorkerStore(s.root, resumed_plan['id'], 0)
    extract_chunks(worker, lambda row: pytest.fail('Completed representation recomputed'), assigned_keys=worker.assigned)


@pytest.mark.parametrize('fail_at', [1, 2])
def test_worker_durable_snapshots_at_both_commits(tmp_path, fail_at):
    s = fixture_store(tmp_path / 'live')
    plan = make_plan(s, 'chunk', recover_extraction(s), 2, extraction_dependency(s))
    snapshot = tmp_path / 'durable'
    shutil.copytree(s.root, snapshot)
    receipt = s.root / 'receipts/chunk/00000.json'
    count = 0
    def commit():
        nonlocal count
        if receipt.exists():
            count += 1
            if count == fail_at:
                raise RuntimeError('interrupted durable commit')
        shutil.copytree(s.root, snapshot, dirs_exist_ok=True)
    worker = WorkerStore(s.root, plan['id'], 0, commit)
    with pytest.raises(RuntimeError):
        extract_chunks(worker, synthetic_vector, assigned_keys=worker.assigned)
    restored = Store(snapshot)
    before = {p.name: (digest(p), p.stat().st_mtime_ns) for p in (snapshot / 'hidden').glob('*')} if (snapshot / 'hidden').exists() else {}
    missing = recover_extraction(restored)
    assert ('chunk/00000' in missing) == (fail_at == 1)
    next_plan = make_plan(restored, 'chunk', missing, 3, extraction_dependency(restored))
    assert not run_workers(restored, next_plan,
        lambda w, slot: extract_chunks(w, synthetic_vector, assigned_keys=w.assigned))
    assert recover_extraction(Store(snapshot)) == []
    if fail_at == 2:
        for name, pair in before.items():
            p = snapshot / 'hidden' / name
            assert (digest(p), p.stat().st_mtime_ns) == pair


def test_invalid_chunk_repair_and_assignment_guard(tmp_path):
    s = fixture_store(tmp_path / 'run')
    plan = make_plan(s, 'chunk', recover_extraction(s), 2, extraction_dependency(s))
    w = WorkerStore(s.root, plan['id'], 0)
    with pytest.raises(ValueError, match='Assignment'):
        extract_chunks(w, synthetic_vector, assigned_keys=plan['assignments'][1])
    run_workers(s, plan, lambda w, i: extract_chunks(w, synthetic_vector, assigned_keys=w.assigned))
    s = Store(s.root)
    assert recover_extraction(s) == []
    before = hashes(s, 'chunk/')
    (s.root / 'hidden/00001.npy').write_bytes(b'partial')
    assert recover_extraction(Store(s.root)) == ['chunk/00001']
    repair = make_plan(s, 'chunk', ['chunk/00001'], 10, extraction_dependency(s))
    run_workers(s, repair, lambda w, i: extract_chunks(w, synthetic_vector, assigned_keys=w.assigned))
    final = Store(s.root)
    assert recover_extraction(final) == []
    for p, old in before.items():
        if not p.startswith('hidden/00001'):
            assert hashes(final, 'chunk/')[p] == old


def test_parallel_full_layer_analysis_interruption_and_equivalence(tmp_path, monkeypatch):
    s = fixture_store(tmp_path / 'parallel')
    extract_chunks(s, synthetic_vector)
    dep = analysis_dependency(s)
    plan = make_plan(s, 'layer', recover_analysis(s), 2, dep)
    shared_before = s.path.read_bytes()
    barrier = threading.Barrier(2)
    def work(w, slot):
        barrier.wait(timeout=10)
        for key in w.assigned:
            if key == 'layer/02':
                raise RuntimeError('CPU worker interrupted')
            compute_layer(w, int(key.split('/')[1]))
    assert len(run_workers(s, plan, work)) == 1
    assert s.path.read_bytes() == shared_before
    s = Store(s.root)
    assert recover_analysis(s) == ['layer/02']
    before = hashes(s, 'layer/')
    resume = make_plan(s, 'layer', ['layer/02'], 3, dep)
    run_workers(s, resume, lambda w, slot: [compute_layer(w, int(k.split('/')[1])) for k in w.assigned])
    s = Store(s.root)
    assert recover_analysis(s) == []
    assert all(hashes(s, 'layer/')[p] == old for p, old in before.items())
    # Coordinator may only perform the scalar baseline and adjacent comparisons.
    import numzig.fullrange.analysis as module
    original = module.exact_knn
    def baseline_only(x, *args, **kwargs):
        assert x.shape[1] == 1
        return original(x, *args, **kwargs)
    with monkeypatch.context() as m:
        m.setattr(module, 'exact_knn', baseline_only)
        m.setattr(module, 'projection', lambda *a: pytest.fail('PCA repeated at barrier'))
        analyze(s, allow_compute=False)
    ref = fixture_store(tmp_path / 'serial')
    extract_chunks(ref, synthetic_vector)
    analyze(ref)
    for layer in range(3):
        with np.load(s.root / f'analysis/layer_{layer:02d}.npz') as actual, np.load(ref.root / f'analysis/layer_{layer:02d}.npz') as expected:
            assert set(actual.files) == set(expected.files)
            for key in actual.files:
                np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=1e-12)
        a = read_json(s.root / f'analysis/layer_{layer:02d}.json')
        b = read_json(ref.root / f'analysis/layer_{layer:02d}.json')
        for key in a:
            if key not in ('seconds', 'knn_seconds', 'pca_seconds', 'process_peak_rss_bytes'):
                assert a[key] == b[key]
        if a['status'] == 'valid':
            assert a['point_count'] == len(load_dataset(s))
            assert a['directed_relation_count'] == 4 * len(load_dataset(s))
    assert pending_layers(s) == []


def test_join_drains_peers_after_failure():
    seen = []
    class Call:
        def __init__(self, i):
            self.i = i
        def get(self):
            seen.append(self.i)
            if self.i == 0:
                raise RuntimeError('failed worker')
    errors = drain_calls([Call(0), Call(1), Call(2)])
    assert len(errors) == 1 and seen == [0, 1, 2]


def test_runtime_contract_rejects_incompatible_config(tmp_path, monkeypatch):
    from numzig.fullrange.runtime import ensure_extraction_contract
    s = fixture_store(tmp_path / 'run')
    ensure_extraction_contract(s)
    before = digest(s.root / 'extraction_contract.json')
    ensure_extraction_contract(Store(s.root))
    assert digest(s.root / 'extraction_contract.json') == before
    monkeypatch.setattr('numzig.fullrange.runtime.package_versions', lambda names: {'torch': 'incompatible'})
    with pytest.raises(ValueError, match='Incompatible'):
        ensure_extraction_contract(Store(s.root))


def test_ten_workers_complete_without_shared_writes(tmp_path):
    s = fixture_store(tmp_path / 'ten', chunk_size=4)
    plan = make_plan(s, 'chunk', recover_extraction(s), 10, extraction_dependency(s))
    before = s.path.read_bytes()
    barrier = threading.Barrier(10)
    def work(w, slot):
        barrier.wait(timeout=15)
        extract_chunks(w, synthetic_vector, assigned_keys=w.assigned)
    assert not run_workers(s, plan, work)
    assert s.path.read_bytes() == before
    s = Store(s.root)
    assert recover_extraction(s) == []
    assert len([k for k in s.manifest['artifacts'] if k.startswith('chunk/')]) == 10


def test_compatible_legacy_final_layers_preserved_without_raw_layers(tmp_path, monkeypatch):
    s = fixture_store(tmp_path / 'legacy')
    extract_chunks(s, synthetic_vector)
    analyze(s)
    before = hashes(s, 'analysis/')
    # Serial-format checkpoints predate the new independent layer/ records.
    for key in list(s.manifest['artifacts']):
        if key.startswith('layer/'):
            for path in s.manifest['artifacts'].pop(key)['files']:
                (s.root / path).unlink()
    s.save()
    monkeypatch.setattr('numzig.fullrange.analysis.exact_knn', lambda *a, **k: pytest.fail('Legacy kNN reran'))
    monkeypatch.setattr('numzig.fullrange.analysis.projection', lambda *a, **k: pytest.fail('Legacy PCA reran'))
    assert recover_analysis(s) == []
    analyze(s, allow_compute=False)
    assert hashes(s, 'analysis/') == before


def test_legacy_repair_reuses_downstream_geometry(tmp_path, monkeypatch):
    s = fixture_store(tmp_path / 'legacy_repair')
    extract_chunks(s, synthetic_vector)
    analyze(s)
    for key in list(s.manifest['artifacts']):
        if key.startswith('layer/'):
            for path in s.manifest['artifacts'].pop(key)['files']:
                (s.root / path).unlink()
    s.save()
    (s.root / 'analysis/layer_01.npz').write_bytes(b'corrupt')
    assert recover_analysis(s) == ['layer/01']
    downstream = hashes(s, 'layer/02')
    compute_layer(s, 1)
    monkeypatch.setattr('numzig.fullrange.analysis.exact_knn', lambda *a, **k: pytest.fail('Completed graph reran'))
    monkeypatch.setattr('numzig.fullrange.analysis.projection', lambda *a, **k: pytest.fail('Completed PCA reran'))
    analyze(s, allow_compute=False)
    assert hashes(s, 'layer/02') == downstream


def test_malformed_receipt_is_incomplete_not_a_completion_marker(tmp_path):
    s = fixture_store(tmp_path / 'malformed')
    plan = make_plan(s, 'chunk', recover_extraction(s), 2, extraction_dependency(s))
    run_workers(s, plan, lambda w, i: extract_chunks(w, synthetic_vector, assigned_keys=w.assigned))
    # Coordinator has not merged: an invalid receipt cannot certify its data.
    write_json(s.root / 'receipts/chunk/00000.json', [])
    assert recover_extraction(Store(s.root)) == ['chunk/00000']
    w = WorkerStore(s.root, plan['id'], 0)
    calls = []
    def infer(row):
        calls.append(row['point_id'])
        return synthetic_vector(row)
    extract_chunks(w, infer, assigned_keys=w.assigned)
    assert calls == [r['point_id'] for r in load_dataset(s)[:8]]
    assert recover_extraction(Store(s.root)) == []


def test_incompatible_receipt_is_rejected(tmp_path):
    s = fixture_store(tmp_path / 'incompatible')
    plan = make_plan(s, 'chunk', recover_extraction(s), 1, extraction_dependency(s))
    run_workers(s, plan, lambda w, i: extract_chunks(w, synthetic_vector, assigned_keys=w.assigned))
    path = s.root / 'receipts/chunk/00000.json'
    receipt = read_json(path)
    receipt['dependency'] = 'another-model-or-dataset'
    write_json(path, receipt)
    with pytest.raises(ValueError, match='Incompatible'):
        recover_extraction(Store(s.root))
