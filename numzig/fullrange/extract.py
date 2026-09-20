from __future__ import annotations

import platform
import time

import numpy as np

from numzig.config import resolve_models
from .dataset import load_dataset, package_versions
from .storage import atomic, digest, fingerprint, read_json, write_json


def extraction_dependency(store):
    return fingerprint(dict(config=store.manifest['config_fingerprint'], dataset=store.artifact_fingerprint('dataset')))


def chunks(store, rows):
    size = store.manifest['config']['chunk_size']
    for start in range(0, len(rows), size):
        yield start, rows[start:start + size], f'chunk/{start // size:05d}'


def check_chunk(store, key, expected_ids, shape=None):
    item = store.manifest['artifacts'][key]
    base = store.root / 'hidden' / key.split('/')[1]
    ids = np.load(str(base) + '_ids.npy', allow_pickle=False)
    x = np.load(str(base) + '.npy', mmap_mode='r', allow_pickle=False)
    try:
        if (ids.dtype != np.int32 or not np.array_equal(ids, expected_ids) or x.dtype != np.float32
            or x.ndim != 3 or x.shape[1] != len(ids) or not np.isfinite(x).all()
            or list(x.shape) != item['metadata']['shape']
            or (shape is not None and (x.shape[0], x.shape[2]) != tuple(shape))):
            raise ValueError(f'Invalid shape, alignment, or finite-value check for {key}')
        return (x.shape[0], x.shape[2])
    finally:
        x._mmap.close()


def inventory(store, rows, require_complete=False, assigned_keys=None):
    dep = extraction_dependency(store)
    valid, missing, shape = [], [], None
    for start, part, key in chunks(store, rows):
        if assigned_keys is not None and key not in assigned_keys:
            continue
        if store.valid(key, dep):
            try:
                checked_shape = check_chunk(store, key, [r['point_id'] for r in part], shape)
            except (ValueError, OSError, KeyError):
                missing.append(key)
                continue
            shape = checked_shape
            valid.append(key)
        else:
            missing.append(key)
    if require_complete and missing:
        raise ValueError(f'{len(missing)} hidden chunks missing/invalid; explicitly resume extraction first')
    return valid, missing, shape


def extract_chunks(store, infer_one, expected_shape=None, assigned_keys=None):
    """infer_one(row) -> float32 [returned_levels, hidden_dim]. Also used with synthetic data."""
    rows = load_dataset(store)
    if hasattr(store, 'assigned') and assigned_keys != store.assigned:
        raise ValueError('Assignment differs from committed worker plan')
    if assigned_keys is not None:
        known = {key for _, _, key in chunks(store, rows)}
        if len(assigned_keys) != len(set(assigned_keys)) or not set(assigned_keys) <= known:
            raise ValueError('Invalid chunk assignment')
    valid, missing, shape = inventory(store, rows, assigned_keys=assigned_keys)
    if shape is not None and expected_shape is not None and tuple(shape) != tuple(expected_shape):
        raise ValueError("Saved hidden shape differs from loaded model configuration")
    shape = shape or expected_shape
    total_chunks = len(valid) + len(missing)
    store.log(f'[EXTRACTION] completed={len(valid)} skipped={len(valid)} remaining={len(missing)}')
    dep = extraction_dependency(store)
    if missing:
        store.stage('extraction', 'running')
    started = time.perf_counter()
    done_points = 0
    for start, part, key in chunks(store, rows):
        if assigned_keys is not None and key not in assigned_keys:
            continue
        if key in valid:
            store.log(f'[SKIP] {key}: checksums, IDs, dtype, dimensions and finite values valid')
            continue
        tic = time.perf_counter()
        buffer = None
        for i, row in enumerate(part):
            vector = np.asarray(infer_one(row))
            if vector.ndim != 2 or vector.dtype != np.float32 or not np.isfinite(vector).all():
                raise ValueError('Inference must return finite float32 [levels, hidden_dim]')
            if shape is None:
                shape = vector.shape
            if vector.shape != tuple(shape):
                raise ValueError(f'Model returned {vector.shape}; expected {shape}')
            if buffer is None:
                buffer = np.empty((shape[0], len(part), shape[1]), dtype=np.float32)
            buffer[:, i, :] = vector
        base = store.root / 'hidden' / key.split('/')[1]
        data_path, ids_path = str(base) + '.npy', str(base) + '_ids.npy'
        ids = np.array([r['point_id'] for r in part], dtype=np.int32)
        atomic(data_path, lambda f: np.save(f, buffer, allow_pickle=False))
        atomic(ids_path, lambda f: np.save(f, ids, allow_pickle=False))
        # Validate serialized bytes before giving the checkpoint completion status.
        restored = np.load(data_path, allow_pickle=False)
        if not np.array_equal(restored, buffer) or not np.array_equal(np.load(ids_path), ids):
            raise IOError('Hidden-state write verification failed')
        metadata = dict(shape=list(buffer.shape), dtype='float32', axis_order=['level', 'point', 'feature'],
                        first_point_id=int(ids[0]), last_point_id=int(ids[-1]),
                        inference_and_write_seconds=time.perf_counter() - tic)
        store.finish(key, [data_path, ids_path], dep, metadata)
        done_points += len(part)
        missing.remove(key)
        store.log(f'[EXTRACTION] completed={total_chunks - len(missing)}/{total_chunks} remaining={len(missing)}')
        del buffer, restored
    store.stage('extraction', 'complete')
    if done_points:
        elapsed = time.perf_counter() - started
        report = store.output_path('extraction_benchmark.json')
        write_json(report, dict(points_this_invocation=done_points, seconds=elapsed,
            seconds_per_point=elapsed / done_points, estimated_10000_point_seconds=elapsed / done_points * 10000,
            estimate_excludes_weight_load=True, estimate_includes_chunk_commits=True,
            hidden_shape=[shape[0], len(rows), shape[1]], hidden_bytes=shape[0] * len(rows) * shape[1] * 4,
            note='Per-worker single-prompt rate; not total parallel wall time; excludes weights and includes commits'))
        store.finish('extraction_benchmark', [report], dep)
    return shape


def read_layer(store, layer, rows=None):
    """Assemble ONE layer, canonical point order. Close each mmap before another stage reloads."""
    rows = rows or load_dataset(store)
    out = None
    for start, part, key in chunks(store, rows):
        if key not in store.manifest['artifacts']:
            raise ValueError(f'Missing {key}')
        base = store.root / 'hidden' / key.split('/')[1]
        ids = np.load(str(base) + '_ids.npy', allow_pickle=False)
        if not np.array_equal(ids, [r['point_id'] for r in part]):
            raise ValueError('Point alignment mismatch')
        x = np.load(str(base) + '.npy', mmap_mode='r', allow_pickle=False)
        try:
            if out is None:
                out = np.empty((len(rows), x.shape[2]), dtype=np.float32)
            out[start:start + len(part)] = x[layer]
        finally:
            x._mmap.close()
    return out


def infer_final_equals(model, row, expected):
    import torch
    ids = torch.tensor([row['input_token_ids']], dtype=torch.long, device='cuda')
    if row['final_token_text'] != '=' or row['extraction_position'] != ids.shape[1] - 1:
        raise ValueError('Invalid final meaningful token position')
    with torch.inference_mode():
        result = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                       output_hidden_states=True, use_cache=False, return_dict=True)
        states = result.hidden_states
        if len(states) != expected[0]:
            raise ValueError('Unexpected number of returned hidden-state levels')
        return torch.stack([h[0, row['extraction_position'], :] for h in states]).float().cpu().numpy()


def extract_model(store, token=None, assigned_keys=None):
    rows = load_dataset(store)
    _, missing, shape = inventory(store, rows, assigned_keys=assigned_keys)
    if not missing:
        store.log('[SKIP] assigned extraction complete; model weights will not be loaded')
        return shape
    import inspect
    import torch
    from transformers import AutoModelForCausalLM
    from pathlib import Path
    from .runtime import ensure_extraction_contract

    ensure_extraction_contract(store, read_only=hasattr(store, 'assigned'))
    if not torch.cuda.is_available():
        raise RuntimeError('Extraction requires CUDA; use synthetic local tests')
    config = store.manifest['config']
    spec = resolve_models('llm360-crystal')[0]
    if spec.model_id != config['model_id'] or spec.dtype != config['model_dtype']:
        raise ValueError('Crystal ModelSpec mismatch')
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    # Cache was warmed and committed by the coordinator. Workers do no network/cache writes.
    model = AutoModelForCausalLM.from_pretrained(spec.model_id, revision=config['model_revision'],
        code_revision=config['model_revision'], trust_remote_code=True, torch_dtype=torch.bfloat16,
        token=token, local_files_only=True).to('cuda').eval()
    try:
        model.config.use_cache = False
        expected = (int(model.config.num_hidden_layers) + 1, int(model.config.hidden_size))
        meta = read_json(store.root / 'tokenizer_provenance.json')
        if expected != (meta['expected_levels'], meta['expected_hidden_dim']):
            raise ValueError('Loaded architecture differs from prepared configuration')
        if digest(Path(inspect.getfile(type(model)))) != meta['assets_sha256']['modeling_crystalcoder.py']:
            raise ValueError('Loaded model code differs from pinned source')
        runtime = store.output_path('extraction_runtime.json')
        write_json(runtime, dict(versions=package_versions(['torch', 'transformers', 'tokenizers', 'numpy']),
            model_class=type(model).__name__, model_revision=getattr(model.config, '_commit_hash', None),
            hardware=torch.cuda.get_device_name(0), platform=platform.platform(),
            expected_shape=expected, batch_size=1, weights_loads_this_worker=1,
            assigned_chunks=assigned_keys, eval=True, inference_mode=True, output_hidden_states=True,
            use_cache=False, hidden_state_semantics=meta['hidden_state_semantics']))
        store.finish('extraction_runtime', [runtime], store.manifest['artifacts']['extraction_contract']['dependency'])
        return extract_chunks(store, lambda row: infer_final_equals(model, row, expected), expected, assigned_keys)
    finally:
        del model
        torch.cuda.empty_cache()
