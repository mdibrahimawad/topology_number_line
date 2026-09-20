"""Resumable task-rule experiment storage; old result namespaces are never written."""
from pathlib import Path
import time
import numpy as np
from numzig.fullrange.storage import atomic, digest, fingerprint, read_json, write_json

MODELS = ('crystal', 'starcoderbase-3b', 'openllama-3b')
CHUNK = 32


def chunk_rows(rows):
    for start in range(0, len(rows), CHUNK):
        yield f'{start // CHUNK:05d}', rows[start:start + CHUNK]


def chunk_valid(root, key, rows, contract):
    root = Path(root)
    receipt = root / 'receipts' / f'{key}.json'
    if not receipt.exists():
        return False
    r = read_json(receipt)
    if r['contract'] != contract or r['ids'] != [v['point_id'] for v in rows]:
        raise ValueError('Checkpoint belongs to a different dataset/backend')
    if not all((root / p).exists() and digest(root / p) == h for p, h in r['files'].items()):
        return False
    with np.load(root / 'hidden' / f'{key}.npz', allow_pickle=False) as z:
        return (np.array_equal(z['point_ids'], r['ids']) and z['hidden'].dtype == np.float32
                and list(z['hidden'].shape) == r['shape'] and np.isfinite(z['hidden']).all())


def extract_shard(root, model_key, rows, keys, contract, commit=lambda: None):
    from .extract import load_model, infer_rows
    root = Path(root)
    started = time.monotonic()
    groups = [(key, part) for key, part in chunk_rows(rows) if key in keys]
    missing = [(key, part) for key, part in groups if not chunk_valid(root, key, part, contract)]
    if not missing:
        return dict(seconds=time.monotonic()-started, completed=len(groups), skipped=len(groups), model=model_key)
    model, tokenizer, meta = load_model(model_key)
    for key, part in missing:
        tic = time.monotonic()
        hidden, token_rows = infer_rows(model, tokenizer, part, model_key, batch_size=1)
        ids = np.array([r['point_id'] for r in part], dtype=np.int32)
        if hidden.ndim != 3 or hidden.shape[1] != len(part) or not np.isfinite(hidden).all():
            raise ValueError('Invalid extraction chunk')
        arrays = root / 'hidden' / f'{key}.npz'
        tokens = root / 'tokens' / f'{key}.json'
        atomic(arrays, lambda f: np.savez(f, hidden=hidden, point_ids=ids))
        write_json(tokens, token_rows)
        receipt = dict(contract=contract, ids=ids.tolist(), shape=list(hidden.shape), model=model_key,
                       files={str(p.relative_to(root)): digest(p) for p in (arrays, tokens)},
                       seconds=time.monotonic()-tic, model_metadata=meta)
        # Bytes are committed before publishing the completion receipt.
        commit()
        write_json(root / 'receipts' / f'{key}.json', receipt)
        commit()
        print(f'[CHUNK] {model_key}/{key} {len(part)} points {receipt["seconds"]:.2f}s', flush=True)
    return dict(seconds=time.monotonic()-started, completed=len(groups), skipped=len(groups)-len(missing), model=model_key)


def summarize_behavior(records):
    from collections import defaultdict
    groups = defaultdict(list)
    for r in records:
        groups[(r['task'], r['context_id'])].append(r)
    result = []
    for (task, ctx), items in sorted(groups.items()):
        successes = [bool(r.get('exact_match', r.get('correct', r.get('success', False)))) for r in items]
        result.append(dict(task=task, context_id=ctx, n=len(items), exact_accuracy=sum(successes)/len(items)))
    return result


def run_pilot(root, model_key, rows, contract, commit=lambda: None):
    from .extract import load_model, infer_rows, behavior, equivalence_gate
    root = Path(root)
    receipt = root / 'receipt.json'
    if receipt.exists():
        r = read_json(receipt)
        if r['contract'] != contract:
            raise ValueError('Pilot backend/dataset changed')
        if all(digest(root / p) == h for p,h in r['files'].items()):
            return dict(seconds=0., summary=r['summary'], model=model_key, skipped=True)
    tic = time.monotonic()
    model, tokenizer, meta = load_model(model_key)
    gate_rows = [r for i,r in enumerate(rows) if i % 32 == 0]
    gate = equivalence_gate(model, tokenizer, gate_rows, model_key)
    write_json(root/"gate.json", gate)
    hidden, token_rows = infer_rows(model, tokenizer, rows, model_key, batch_size=1)
    generated = behavior(model, tokenizer, rows, model_key)
    summary = summarize_behavior(generated)
    path = root/'hidden.npz'
    atomic(path, lambda f: np.savez(f, hidden=hidden, point_ids=np.array([r['point_id'] for r in rows],dtype=np.int32)))
    write_json(root/'tokens.json', token_rows)
    write_json(root/'behavior.json', generated)
    write_json(root/'model.json', meta)
    files = {str(p.relative_to(root)):digest(p) for p in (path,root/'tokens.json',root/'behavior.json',root/'model.json',root/'gate.json')}
    commit()
    result = dict(contract=contract, files=files, summary=summary, model=model_key,
                  seconds=time.monotonic()-tic, rows=len(rows), meta=meta)
    write_json(receipt, result)
    commit()
    print('[PILOT]', model_key, summary, flush=True)
    return result


def analyze_saved_group(root, model_key, rows, task, context_id, contract, commit=lambda: None):
    from .analyze import analyze_group
    root = Path(root)
    selected = [r for r in rows if r['task']==task and r['context_id']==context_id]
    ids = {r['point_id']: i for i,r in enumerate(selected)}
    x = None
    tokens = [None]*len(selected)
    seen = set()
    for key, part in chunk_rows(rows):
        if not any(r['point_id'] in ids for r in part):
            continue
        if not chunk_valid(root,key,part,contract):
            raise ValueError(f'Missing or invalid source chunk {key}')
        token_part = read_json(root/'tokens'/f'{key}.json')
        with np.load(root/'hidden'/f'{key}.npz',allow_pickle=False) as z:
            a = z['hidden']
            if x is None:
                x = np.empty((a.shape[0],len(selected),a.shape[2]),dtype=np.float32)
            for j,point_id in enumerate(z['point_ids']):
                if int(point_id) in ids:
                    i=ids[int(point_id)]
                    x[:,i]=a[:,j]
                    tokens[i]=token_part[j]
                    seen.add(int(point_id))
    if seen!=set(ids) or any(r is None for r in tokens):
        raise ValueError('Incomplete group coverage')
    return analyze_group(x,tokens,root/'analysis'/f'{task}_ctx{context_id}',model_key,do_ph=True,commit=commit)
