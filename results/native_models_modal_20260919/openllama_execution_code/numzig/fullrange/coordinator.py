"""Shared state is owned here, outside worker fan-out and after the join barrier."""
from .dataset import load_dataset
from .extract import chunks, check_chunk, extraction_dependency, inventory
from .parallel import merge_receipts, merge_worker_metadata


def recover_extraction(store):
    rows = load_dataset(store)
    parts = {key: part for _, part, key in chunks(store, rows)}
    def validate(key):
        if key not in parts:
            raise ValueError('Unknown chunk')
        check_chunk(store, key, [r['point_id'] for r in parts[key]])
    merge_receipts(store, 'chunk', extraction_dependency(store), validate)
    merge_worker_metadata(store)
    valid, missing, shape = inventory(store, rows)
    store.log(f'[EXTRACTION COORDINATOR] completed={len(valid)} skipped={len(valid)} remaining={len(missing)}')
    if not missing:
        store.stage('extraction', 'complete')
    return missing


def recover_analysis(store):
    from .analysis import analysis_dependency, pending_layers, validate_layer
    merge_receipts(store, 'layer', analysis_dependency(store), lambda key: validate_layer(store, key))
    missing = pending_layers(store)
    store.log(f'[ANALYSIS COORDINATOR] remaining={len(missing)} independent layers')
    return missing


def drain_calls(calls):
    """Always join successful peers even when a worker fails; merge before raising."""
    failures = []
    for index, call in enumerate(calls):
        try:
            call.get()
        except Exception as exc:
            failures.append(f'worker {index}: {type(exc).__name__}: {exc}')
    return failures
