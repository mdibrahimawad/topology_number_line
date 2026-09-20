"""Coordinator barriers, durable layer cache and bounded worker entrypoints."""
from pathlib import Path
import os
import uuid
import time
import numpy as np
from numzig.fullrange.storage import Store, read_json, write_json, fingerprint, digest
from numzig.fullrange.extract import chunks, check_chunk, extraction_dependency, inventory
from numzig.fullrange.parallel import merge_receipts, merge_worker_metadata, make_plan, WorkerStore, partition
from numzig.fullrange.analysis import analysis_dependency, layer_ready, validate_layer, compute_layer, analyze
from .data import load


def recover_extraction(store):
    rows = load(store)
    parts = {key: part for _, part, key in chunks(store, rows)}
    checked = set()
    meta = read_json(store.root / 'tokenizer_provenance.json')
    expected_shape = (meta['expected_levels'], meta['expected_hidden_dim'])
    def validate(key):
        if key not in parts:
            raise ValueError('Unknown chunk')
        check_chunk(store, key, [r['point_id'] for r in parts[key]], expected_shape)
        checked.add(key)
    merge_receipts(store, 'chunk', extraction_dependency(store), validate)
    merge_worker_metadata(store)
    # Receipt merge already hashed and scientifically checked these immutable chunks.
    # Only records without a valid receipt need an additional manifest check.
    for key in parts.keys() - checked:
        if store.valid(key, extraction_dependency(store)):
            try:
                validate(key)
            except (ValueError, OSError, KeyError):
                pass
    valid, missing = sorted(checked), sorted(parts.keys() - checked)
    store.log(f'[EXTRACTION] completed={len(valid)} skipped={len(valid)} remaining={len(missing)}')
    if not missing:
        store.stage('extraction', 'complete')
    return missing


def saved_gate(store):
    merge_worker_metadata(store)
    for key, item in store.manifest['artifacts'].items():
        if key.startswith('worker/') and key.endswith('/extraction_runtime') and store.valid(key):
            report = read_json(store.root / next(iter(item['files'])))
            if report.get('passed') and report.get('gate', {}).get('passed'):
                return report
    return None


def layer_cache(store, validated_shape=None):
    """Coordinator validates source shards once, then transposes once to layer files.

    One extra float32 copy; excluded from lightweight export. Workers hash/read only
    their assigned layer. All mappings are closed before any Volume.commit.
    """
    rows = load(store)
    shape = validated_shape
    if shape is None:
        _, _, shape = inventory(store, rows, require_complete=True)
    dep = analysis_dependency(store)
    missing = [i for i in range(shape[0]) if not store.valid(f'cache/{i:02d}', dep)]
    store.log(f'[CACHE] completed={shape[0]-len(missing)} remaining={len(missing)}; additional_bytes={shape[0]*len(rows)*shape[1]*4}')
    maps, pending = {}, {}
    start = time.perf_counter()
    try:
        for layer in missing:
            path = store.root / 'analysis_cache' / f'layer_{layer:02d}.npy'
            path.parent.mkdir(exist_ok=True)
            temporary = path.with_suffix('.npy.partial')
            pending[layer] = (temporary, path)
            maps[layer] = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32, shape=(len(rows), shape[1]))
        for start_index, part, key in chunks(store, rows):
            if not missing:
                break
            x = np.load(store.root / 'hidden' / (key.split('/')[1] + '.npy'), mmap_mode='r', allow_pickle=False)
            try:
                for layer in missing:
                    maps[layer][start_index:start_index + len(part)] = x[layer]
            finally:
                x._mmap.close()
    finally:
        for value in maps.values():
            value.flush()
            value._mmap.close()
    for layer, (temporary, path) in pending.items():
        with temporary.open('rb') as f:
            os.fsync(f.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        x = np.load(path, mmap_mode='r', allow_pickle=False)
        try:
            if x.shape != (len(rows), shape[1]) or not np.isfinite(x).all():
                raise ValueError('Invalid transposed cache')
        finally:
            x._mmap.close()
        store.finish(f'cache/{layer:02d}', [path], dep, dict(shape=[len(rows), shape[1]], seconds=time.perf_counter()-start))
    return shape


def recover_analysis(store):
    rows = load(store)
    dep = analysis_dependency(store)
    merge_receipts(store, 'layer', dep, lambda key: validate_layer(store, key, rows))
    levels = read_json(store.root / 'tokenizer_provenance.json')['expected_levels']
    missing = [f'layer/{i:02d}' for i in range(levels) if not layer_ready(store, f'layer/{i:02d}', dep, rows)]
    store.log(f'[ANALYSIS] completed={levels-len(missing)} skipped={levels-len(missing)} remaining={len(missing)}')
    return missing


def analysis_worker(store):
    from threadpoolctl import threadpool_limits
    rows = load(store)
    dep = analysis_dependency(store)
    with threadpool_limits(limits=2):
        for key in store.assigned:
            if layer_ready(store, key, dep, rows):
                store.log(f'[SKIP] {key}')
                continue
            layer = int(key.split('/')[1])
            if not store.valid(f'cache/{layer:02d}', dep):
                raise ValueError('Missing/corrupt coordinator layer cache; resume analysis preparation')
            x = np.load(store.root / 'analysis_cache' / f'layer_{layer:02d}.npy', allow_pickle=False)
            compute_layer(store, layer, rows=rows, x=x)


def finalize_analysis(store):
    if recover_analysis(store):
        raise ValueError('Analysis barrier has missing layers')
    meta = read_json(store.root / 'tokenizer_provenance.json')
    return analyze(store, allow_compute=False, rows=load(store), validated_shape=(meta['expected_levels'], meta['expected_hidden_dim']))


def global_jobs(stores, missing, workers, kind, dependency):
    """One global bounded queue across models, with model-scoped immutable plans.

    Each assignment is one independent artifact; the caller bounds simultaneous
    calls instead of reserving idle workers. Plans are committed before submission.
    """
    jobs = []
    for store in stores:
        keys = missing(store)
        if not keys:
            continue
        # One plan slot per artifact; API's legacy worker ceiling is not used as a
        # concurrency limit here. Global executor + Modal container ceiling enforce it.
        plan = make_plan(store, kind, keys, len(keys), dependency(store), cpu_limit=max(len(keys), workers),
                         dependencies={key: dependency(store, key) for key in keys} if kind == 'plot' else None)
        jobs.extend((str(store.root), plan['id'], slot) for slot in range(len(plan['assignments'])))
    return jobs


def record_allocation(store, stage, settings, runnable, workers):
    path = store.root / 'execution' / f'{time.time_ns()}_{stage}.json'
    write_json(path, dict(stage=stage, settings=settings, runnable=runnable, requested_workers=min(workers, runnable),
        observed_workers=None, observed_note='Read Modal live stats during execution; request is not observed utilization'))
    store.finish(f'execution/{path.stem}', [path], fingerprint(settings))

def local_analysis(job):
    from numzig.fullrange.parallel import WorkerStore
    from .pipeline import analysis_worker
    analysis_worker(WorkerStore(*job))


def local_plot(job):
    from numzig.fullrange.parallel import WorkerStore
    from .plots import worker
    worker(WorkerStore(*job))

