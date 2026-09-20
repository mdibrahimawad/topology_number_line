"""Derive a mixed-prompt pilot from two validated saved pilots, with no inference."""
from pathlib import Path
import time
import numpy as np

from numzig.fullrange.storage import atomic, digest, read_json, write_json
from numzig.taskrules import extract
from numzig.taskrules.data import pilot_records
from numzig.taskrules.runtime import summarize_behavior

FILES = ('hidden.npz', 'tokens.json', 'behavior.json', 'model.json', 'gate.json')
SCIENTIFIC_FIELDS = ('model_key', 'model_id', 'revision', 'model_source_sha256', 'dtype',
    'saved_dtype', 'tokenizer_class', 'is_fast', 'special_token_policy', 'expected_levels',
    'expected_hidden_dim', 'hidden_state_semantics', 'output_logits_scale')


def _files_valid(root, files):
    for name, expected in files.items():
        path = root / name
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Receipt contains a path outside its result directory')
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f'Missing or corrupt pilot artifact: {path}')


def _source(root, model_key):
    receipt_path = root / 'receipt.json'
    receipt = read_json(receipt_path)
    if receipt.get('model') != model_key or not set(FILES).issubset(receipt['files']):
        raise ValueError('Source pilot model or artifact inventory differs')
    _files_valid(root, receipt['files'])
    meta, gate = read_json(root/'model.json'), read_json(root/'gate.json')
    if receipt.get('meta') != meta or meta.get('model_key') != model_key:
        raise ValueError('Source model metadata differs from its completion receipt')
    if meta.get('inference_source_sha256') != digest(extract.__file__):
        raise ValueError('Source pilot used a different extraction implementation')
    if gate.get('passed') is not True:
        raise ValueError('Source pilot did not pass its technical gate')
    for field in SCIENTIFIC_FIELDS:
        if field not in meta:
            raise ValueError(f'Missing scientific metadata: {field}')
    tokens, behavior = read_json(root/'tokens.json'), read_json(root/'behavior.json')
    with np.load(root/'hidden.npz', allow_pickle=False) as z:
        ids, hidden = z['point_ids'], z['hidden']
    expected_shape = (meta['expected_levels'], len(ids), meta['expected_hidden_dim'])
    if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or len(np.unique(ids)) != len(ids)
        or hidden.shape != expected_shape or hidden.dtype != np.float32 or not np.isfinite(hidden).all()
        or receipt.get('rows') != len(ids)
        or [r['point_id'] for r in tokens] != ids.tolist()
        or [r['point_id'] for r in behavior] != ids.tolist()):
        raise ValueError('Source pilot hidden states, token records or behavior coverage differ')
    if receipt.get('summary') != summarize_behavior(behavior):
        raise ValueError('Source behavior summary differs from saved records')
    provenance = dict(root=str(root), receipt_path=str(receipt_path), receipt_sha256=digest(receipt_path),
        contract=receipt['contract'], artifact_sha256=receipt['files'], model_metadata=meta)
    return dict(hidden=hidden, tokens=tokens, behavior=behavior, meta=meta, gate=gate,
                index={int(point_id): i for i, point_id in enumerate(ids)}, provenance=provenance)


def adopt_pilot(source_root, target_root, model_key, rows, contract, commit=lambda: None):
    """Create target_root/pilot/model_key using exact matching saved pilot rows.

    Four-digit tasks use the instruction pilot; numeric/word copying uses the
    examples pilot. Completion receipts explicitly describe derived provenance.
    """
    started = time.monotonic()
    model_key = extract.key_for(model_key)
    source_root, target_root = Path(source_root), Path(target_root)
    root = target_root/'pilot'/model_key
    selected = pilot_records(rows)
    if not selected or len({r['point_id'] for r in selected}) != len(selected):
        raise ValueError('Selected pilot IDs must be nonempty and unique')
    receipt_path = root/'receipt.json'
    if receipt_path.exists():
        saved = read_json(receipt_path)
        if (saved.get('contract') != contract or saved.get('model') != model_key
            or saved.get('ids') != [r['point_id'] for r in selected] or not saved.get('adopted_without_inference')):
            raise ValueError('Existing target pilot belongs to another derivation')
        _files_valid(root, saved['files'])
        return dict(seconds=time.monotonic()-started, model=model_key, summary=saved['summary'], skipped=True)
    if source_root.resolve() == target_root.resolve():
        raise ValueError('Adoption needs a distinct output namespace')
    sources = {
        'examples': _source(source_root/'pilot'/model_key, model_key),
        'instruction': _source(source_root/'pilot_instruction'/model_key, model_key),
    }
    for field in SCIENTIFIC_FIELDS:
        if sources['examples']['meta'][field] != sources['instruction']['meta'][field]:
            raise ValueError(f'Source scientific model metadata differs: {field}')
    states, tokens, behavior, selection = [], [], [], []
    for row in selected:
        style = 'instruction' if row['task'].endswith('4') else 'examples'
        source = sources[style]
        if row.get('prompt_style') != style:
            raise ValueError('Requested row does not follow the declared mixed prompt policy')
        if row['point_id'] not in source['index']:
            raise ValueError('A selected target is missing from the source pilot')
        index = source['index'][row['point_id']]
        token, answer = source['tokens'][index], source['behavior'][index]
        for field, value in row.items():
            if field not in token or token[field] != value or field not in answer or answer[field] != value:
                raise ValueError(f'Saved stimulus differs at point {row["point_id"]}, field {field}')
        states.append(source['hidden'][:,index]); tokens.append(token); behavior.append(answer)
        selection.append(dict(point_id=row['point_id'], task=row['task'], context_id=row['context_id'],
            prompt_style=style, source_receipt_path=source['provenance']['receipt_path'], source_index=index))
    hidden = np.stack(states, axis=1)
    ids = np.array([r['point_id'] for r in selected], dtype=np.int32)
    summary = summarize_behavior(behavior)
    provenance = dict(adopted_without_inference=True, source_root=str(source_root),
        derivation='exact row selection: instruction four-digit tasks; examples numeric/word copying',
        sources={k:v['provenance'] for k,v in sources.items()}, selection=selection,
        adoption_source_sha256=digest(__file__))
    meta = dict(sources['examples']['meta'], model_loads=0, load_seconds=0.,
                adopted_without_inference=True, source_model_metadata={k:v['meta'] for k,v in sources.items()})
    gate = dict(passed=True, adopted_without_inference=True,
        scope='Inherited passed technical gates from hash-verified saved pilots; no new model evaluation',
        source_gates={k:dict(report=v['gate'], file_sha256=v['provenance']['artifact_sha256']['gate.json'],
                            receipt_path=v['provenance']['receipt_path']) for k,v in sources.items()})
    root.mkdir(parents=True, exist_ok=True)
    atomic(root/'hidden.npz', lambda f: np.savez(f, hidden=hidden, point_ids=ids))
    for name,value in (('tokens.json',tokens),('behavior.json',behavior),('model.json',meta),
                       ('gate.json',gate),('adoption.json',provenance)):
        write_json(root/name,value)
    files = {name:digest(root/name) for name in (*FILES,'adoption.json')}
    commit()
    receipt = dict(contract=contract,files=files,summary=summary,model=model_key,seconds=time.monotonic()-started,
        rows=len(selected),ids=ids.tolist(),meta=meta,adopted_without_inference=True,
        source_receipts={k:dict(path=v['provenance']['receipt_path'],sha256=v['provenance']['receipt_sha256']) for k,v in sources.items()})
    write_json(receipt_path,receipt); commit()
    return dict(seconds=time.monotonic()-started,model=model_key,summary=summary,rows=len(selected),adopted_without_inference=True)
