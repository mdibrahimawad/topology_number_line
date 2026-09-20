"""One native eager backend plus a single-prompt LM-wrapper correctness reference."""
from pathlib import Path
import time
import numpy as np
from scipy.spatial.distance import cdist
from numzig.fullrange.storage import digest, fingerprint, write_json, read_json, atomic
from numzig.fullrange.dataset import package_versions
from numzig.fullrange.extract import chunks, inventory, extraction_dependency
from . import BATCH_SIZES, TOLERANCES
from .data import load


def contract(store):
    cfg = store.manifest['config']
    import inspect
    from transformers.models.gpt_bigcode.modeling_gpt_bigcode import GPTBigCodeModel
    from transformers.models.llama.modeling_llama import LlamaModel
    native = GPTBigCodeModel if cfg['model_key'] == 'starcoderbase-3b' else LlamaModel
    return dict(native_implementation_sha256=digest(inspect.getfile(native)), configuration={k: v for k, v in cfg.items() if k not in ('smoke', 'experiment')},
        packages=package_versions(['torch', 'transformers', 'tokenizers', 'sentencepiece', 'numpy']),
        sources={n: digest(Path(__file__).with_name(n)) for n in ('backend.py', 'data.py', '__init__.py')})


def ensure_contract(store):
    current = contract(store)
    path = store.root / 'extraction_contract.json'
    if 'extraction_contract' in store.manifest['artifacts']:
        store.require('extraction_contract')
        if read_json(path) != current:
            raise ValueError('Incompatible native extraction runtime/configuration')
    else:
        if any(k.startswith('chunk/') for k in store.manifest['artifacts']):
            raise ValueError('Cannot attach new backend contract to existing vectors')
        write_json(path, current)
        store.finish('extraction_contract', [path], fingerprint(current))
    return fingerprint(current)


def padded_inputs(rows, pad_id, device, padding_length=None):
    import torch
    lengths = [len(r['input_token_ids']) for r in rows]
    width = max(lengths) if padding_length is None else padding_length
    if width < max(lengths):
        raise ValueError('Padding cannot truncate real input')
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, r in enumerate(rows):
        if r['extraction_position'] != lengths[i] - 1:
            raise ValueError('Final-real-token index mismatch')
        ids[i, :lengths[i]] = torch.tensor(r['input_token_ids'], device=device)
        mask[i, :lengths[i]] = 1
    positions = torch.arange(width, device=device)[None, :].expand_as(ids)
    return dict(input_ids=ids, attention_mask=mask, position_ids=positions)


def forward(model, rows, pad_id, backbone=True, padding_length=None):
    import torch
    model.eval()
    device = next(model.parameters()).device
    inputs = padded_inputs(rows, pad_id, device, padding_length)
    target = model.base_model if backbone else model
    with torch.inference_mode():
        result = target(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
        states = result.hidden_states
        batch = torch.arange(len(rows), device=device)
        ends = torch.tensor([r['extraction_position'] for r in rows], device=device)
        # Only final-token vectors cross the device boundary.
        out = torch.stack([s[batch, ends] for s in states]).float().cpu().numpy()
    expected = (model.config.num_hidden_layers + 1, len(rows), model.config.hidden_size)
    if out.shape != expected or not np.isfinite(out).all():
        raise ValueError('Invalid native returned states')
    return out


def reference(model, rows, pad_id):
    return np.concatenate([forward(model, [r], pad_id, backbone=False) for r in rows], axis=1)


def batched(model, rows, pad_id, batch_size=16, padding_length=None, on_backoff=None):
    import torch
    if batch_size not in BATCH_SIZES:
        raise ValueError('Batch size outside predeclared equivalence family')
    order = sorted(range(len(rows)), key=lambda i: (len(rows[i]['input_token_ids']), rows[i]['point_id']))
    output, cursor, size = None, 0, batch_size
    while cursor < len(order):
        indices = order[cursor:cursor + size]
        try:
            values = forward(model, [rows[i] for i in indices], pad_id, padding_length=padding_length)
        except torch.cuda.OutOfMemoryError:
            if size == 1:
                raise
            size //= 2  # finite halving, family tested by smoke, never change dtype/backend
            torch.cuda.empty_cache()
            if on_backoff:
                on_backoff(size)
            continue
        if output is None:
            output = np.empty((values.shape[0], len(rows), values.shape[2]), dtype=np.float32)
        output[:, indices] = values
        cursor += len(indices)
    return output


def compare(reference_values, candidate, ids, tolerances=TOLERANCES):
    if reference_values.shape != candidate.shape or reference_values.dtype != candidate.dtype:
        raise ValueError('Equivalence shape/dtype mismatch')
    reports = []
    for a, b in zip(reference_values.astype(np.float64), candidate.astype(np.float64)):
        diff = np.linalg.norm(a - b, axis=1)
        norms = np.linalg.norm(a, axis=1)
        relative = diff / np.maximum(norms, 1e-12)
        other = np.linalg.norm(b, axis=1)
        denom = norms * other
        coserr = np.where(denom > 0, 1 - np.sum(a * b, axis=1) / np.maximum(denom, 1e-300), (diff != 0).astype(float))
        da, db = cdist(a, a), cdist(b, b)
        distance_ok = np.allclose(da, db, atol=tolerances['distance_atol'], rtol=tolerances['distance_rtol'])
        np.fill_diagonal(da, np.inf)
        np.fill_diagonal(db, np.inf)
        # Stable distance then canonical ID, independent of caller order.
        na = np.array([np.lexsort((ids, row)) for row in da])
        nb = np.array([np.lexsort((ids, row)) for row in db])
        k = min(4, len(ids) - 1)
        matches = np.array_equal(na[:, :k], nb[:, :k])
        sorted_da = np.take_along_axis(da, na, axis=1)
        margin = sorted_da[:, k] - sorted_da[:, k - 1] if len(ids) > k + 1 else np.zeros(len(ids))
        within = np.allclose(a, b, atol=tolerances['vector_atol'], rtol=tolerances['vector_rtol'])
        reports.append(dict(max_absolute=float(np.max(np.abs(a - b))), max_relative_l2=float(relative.max()),
            max_cosine_error=float(coserr.max()), pairwise_distance_pass=bool(distance_ok), knn_order_identical=matches,
            min_k_boundary_margin=float(margin.min()), near_tie_points=int(np.sum(margin <= 2 * tolerances['distance_atol'])),
            passed=bool(within and relative.max() <= tolerances['relative_l2'] and coserr.max() <= tolerances['cosine_error'] and distance_ok and matches)))
    return dict(passed=all(r['passed'] for r in reports), levels=reports, tolerances=tolerances,
                tie_policy='Any ordered neighbor identity difference fails, including near ties')


def equivalence_gate(model, rows, pad_id, max_padding=128, batch_sizes=BATCH_SIZES):
    start = time.perf_counter()
    ref = reference(model, rows, pad_id)
    reference_seconds = time.perf_counter() - start
    ids = np.array([r['point_id'] for r in rows])
    checks = []
    canonical = None
    for size in batch_sizes:
        tic = time.perf_counter()
        # Gate must actually test each batch size: no OOM fallback during gate.
        def fail_backoff(new_size):
            raise ValueError(f'Gate OOM: requested family not validated (fallback {new_size})')
        out = batched(model, rows, pad_id, size, on_backoff=fail_backoff)
        checks.append(dict(batch_size=size, seconds=time.perf_counter() - tic, **compare(ref, out, ids)))
        if canonical is None or size == 16:
            canonical = out
    reverse = rows[::-1]
    padding_supported = max(batch_sizes) > 1
    padded = batched(model, reverse, pad_id, min(8, max(batch_sizes)),
                     padding_length=max_padding if padding_supported else None, on_backoff=fail_backoff)[:, ::-1]
    checks.append(dict(case='reverse composition + maximum right padding' if padding_supported else
                      'reverse composition; one unpadded prompt per forward', **compare(ref, padded, ids)))
    for workers in (2, 3):
        # Logical disjoint worker assignments alter batch composition; no extra models/GPU processes.
        out = np.empty_like(ref)
        for slot in range(workers):
            indices = list(range(slot, len(rows), workers))
            out[:, indices] = batched(model, [rows[i] for i in indices], pad_id, min(4, max(batch_sizes)), on_backoff=fail_backoff)
        checks.append(dict(case=f'{workers} disjoint assignments', **compare(ref, out, ids)))
    report = dict(passed=all(c['passed'] for c in checks), checks=checks, reference_seconds=reference_seconds,
                  points=len(rows), supported_batch_sizes=list(batch_sizes), padding_supported=padding_supported,
                  maximum_padded_length=max_padding if padding_supported else None,
                  scope='representative smoke prompts; not proof for all untested prompts or hardware')
    return report, canonical


def extract_assigned(store, infer, expected_shape):
    rows = load(store)
    valid, missing, _ = inventory(store, rows, assigned_keys=store.assigned)
    dep = extraction_dependency(store)
    store.log(f'[EXTRACTION] completed={len(valid)} skipped={len(valid)} remaining={len(missing)}')
    for _, part, key in chunks(store, rows):
        if key not in missing:
            continue
        tic = time.perf_counter()
        values = np.asarray(infer(part))
        if values.shape != (expected_shape[0], len(part), expected_shape[1]) or values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError('Invalid chunk vectors')
        base = store.root / 'hidden' / key.split('/')[1]
        ids = np.array([r['point_id'] for r in part], dtype=np.int32)
        paths = [Path(str(base) + '.npy'), Path(str(base) + '_ids.npy')]
        for path, value in zip(paths, (values, ids)):
            atomic(path, lambda f, value=value: np.save(f, value, allow_pickle=False))
            if not np.array_equal(np.load(path, allow_pickle=False), value):
                raise IOError('Chunk serialization mismatch')
        store.finish(key, paths, dep, dict(shape=list(values.shape), dtype='float32', seconds=time.perf_counter() - tic))
        missing.remove(key)
        store.log(f'[EXTRACTION] completed={len(store.assigned)-len(missing)} remaining={len(missing)}')


def extract_model(store, batch_size=16, smoke_gate=None):
    import torch
    from transformers import AutoModelForCausalLM
    from huggingface_hub import snapshot_download
    cfg = store.manifest['config']
    if batch_size not in cfg['allowed_batch_sizes']:
        raise ValueError(f'Unsupported batch size for {cfg["model_key"]}: {cfg["allowed_batch_sizes"]}')
    rows = load(store)
    _, missing, _ = inventory(store, rows, assigned_keys=store.assigned)
    if not missing:
        store.log('[SKIP] no missing assigned vectors; model not loaded')
        return
    store.require('extraction_contract')
    if read_json(store.root / 'extraction_contract.json') != contract(store):
        raise ValueError('Worker runtime differs from coordinator contract')
    if not cfg['smoke'] and (not smoke_gate or not smoke_gate['passed'] or smoke_gate['contract'] != fingerprint(contract(store))):
        raise ValueError('Full extraction requires compatible passed real smoke gate')
    if not torch.cuda.is_available():
        raise RuntimeError('Real extraction requires CUDA; local tests must use synthetic fixtures')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42)
    start = time.perf_counter()
    # Transformers 4.40 probes remote safetensors even with local_files_only.
    # An exact cached snapshot directory avoids that probe and auto-conversion.
    snapshot = snapshot_download(cfg['model_id'], revision=cfg['model_revision'], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(snapshot, revision=cfg['model_revision'],
        torch_dtype=torch.float32, trust_remote_code=False, attn_implementation='eager', local_files_only=True).to('cuda').eval()
    meta = read_json(store.root / 'tokenizer_provenance.json')
    expected = (meta['expected_levels'], meta['expected_hidden_dim'])
    if (model.config.num_hidden_layers + 1, model.config.hidden_size) != expected:
        raise ValueError('Loaded shape differs from pinned preparation')
    pad = meta['eos_token_id'] if meta['eos_token_id'] is not None else 0
    runtime = dict(model_loads=1, load_seconds=time.perf_counter() - start, gpu=torch.cuda.get_device_name(),
        contract=fingerprint(contract(store)), packages=contract(store)['packages'], actual_batch_size=batch_size,
        dtype=str(next(model.parameters()).dtype), expected_shape=expected, tf32=False,
        attention='eager', backend='native backbone', hardware_total_memory=torch.cuda.get_device_properties(0).total_memory)
    if smoke_gate and smoke_gate.get('gpu') != runtime['gpu']:
        raise ValueError('GPU differs from validated smoke hardware')
    cached = None
    if cfg['smoke']:
        if smoke_gate is not None or any(k.startswith('chunk/') for k in store.manifest['artifacts']):
            # Never re-infer a completed target on partial smoke resume. Use its durable gate.
            if not smoke_gate or not smoke_gate['passed'] or smoke_gate['contract'] != runtime['contract']:
                raise ValueError('Restore valid saved smoke gate before partial resume')
        else:
            gate, values = equivalence_gate(model, rows, pad, cfg['maximum_padded_length'], cfg['allowed_batch_sizes'])
            runtime.update(gate=gate, passed=gate['passed'])
            path = store.output_path('extraction_runtime.json')
            write_json(path, runtime)
            store.finish('extraction_runtime', [path], extraction_dependency(store))
            if not gate['passed']:
                raise ValueError('Real equivalence gate failed; no target checkpoint published. Inspect diagnostics before changing backend.')
            cached = {r['point_id']: values[:, i] for i, r in enumerate(rows)}
    if not runtime.get('gate'):
        runtime.update(passed=True, gate_source=smoke_gate)
        path = store.output_path('extraction_runtime.json')
        write_json(path, runtime)
        store.finish('extraction_runtime', [path], extraction_dependency(store))
    backoffs = []
    def infer(part):
        if cached is not None:
            return np.stack([cached[r['point_id']] for r in part], axis=1)
        return batched(model, part, pad, batch_size, on_backoff=lambda size: backoffs.append(size))
    tic = time.perf_counter()
    extract_assigned(store, infer, expected)
    benchmark = store.output_path('extraction_benchmark.json')
    write_json(benchmark, dict(seconds=time.perf_counter() - tic, points=sum(len(p) for _, p, k in chunks(store, rows) if k in missing),
        peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(), oom_backoff_sizes=backoffs, batch_size=batch_size))
    store.finish('extraction_benchmark', [benchmark], extraction_dependency(store))
