"""Import immutable Crystal text; prepare independent native tokenizer datasets."""
from collections import Counter
from pathlib import Path
import copy

from numzig.fullrange.storage import Store, digest, fingerprint, read_json, write_json, atomic
from numzig.fullrange.dataset import SMOKE_TARGETS, provenance, package_versions
from . import MODELS, RAW_FIELDS, CRYSTAL_DATASET_SHA256


def import_text(root):
    root = Path(root)
    manifest = read_json(root / 'manifest.json')
    expected = manifest['artifacts']['dataset']['files']['dataset.json']
    if expected != CRYSTAL_DATASET_SHA256 or digest(root / 'dataset.json') != expected:
        raise ValueError('Authoritative Crystal dataset fingerprint mismatch')
    rows = [{k: copy.deepcopy(r[k]) for k in RAW_FIELDS} for r in read_json(root / 'dataset.json')['records']]
    validate_raw(rows, smoke=False)
    return rows, fingerprint(rows)


def validate_raw(rows, smoke):
    targets = SMOKE_TARGETS if smoke else list(range(1, 10001))
    if [r['target'] for r in rows] != targets:
        raise ValueError('Target coverage/order mismatch')
    for r in rows:
        ds = [r['demonstrations'][k] for k in 'ABCD']
        if (r['point_id'] != r['target'] - 1 or r['seed'] != 42
            or [len(str(v)) for v in ds] != [1, 2, 3, 4]
            or r['prompt'] != ','.join([f'{v}={v}' for v in ds] + [f'{r["target"]}='])
            or r['character_count'] != len(r['prompt']) or r['target_string'] != str(r['target'])
            or r['digit_count'] != len(str(r['target'])) or r['leading_digit'] != int(str(r['target'])[0])
            or r['demonstration_equals_target'] != [v == r['target'] for v in ds]
            or r['demonstration_digit_lengths'] != [1, 2, 3, 4]):
            raise ValueError('Invalid raw stimulus')


def token_rows(raw, tokenizer, config):
    rows = copy.deepcopy(raw)
    special = set(tokenizer.all_special_ids)
    def encode(text):
        ids = tokenizer(text, add_special_tokens=False)['input_ids']
        if config['bos']:
            if tokenizer.bos_token_id is None or tokenizer.bos_token_id in ids:
                raise ValueError('Missing or duplicate BOS')
            ids = [tokenizer.bos_token_id] + ids
        return ids
    for r in rows:
        ids = encode(r['prompt'])
        last = tokenizer.decode([ids[-1]], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        decoded = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if ids[-1] in special or not last.endswith('=') or decoded.strip() != r['prompt']:
            raise ValueError('Tokenizer did not preserve final meaningful equals/prompt')
        if len(ids) > config['maximum_padded_length']:
            raise ValueError('Prompt exceeds validated padding envelope')
        full = encode(r['prompt'] + r['target_string'])
        prefix = full[:len(ids)] == ids
        continuation = full[len(ids):] if prefix else None
        r.update(input_token_ids=ids, token_pieces=tokenizer.convert_ids_to_tokens(ids),
            decoded_input=decoded, token_count=len(ids), special_token_policy=config['special_policy'],
            extraction_position=len(ids) - 1, final_token_id=ids[-1], final_token_text=last,
            final_equals_shares_token=(last != '='), continuation_prefix_valid=prefix,
            continuation_token_ids=continuation, first_continuation_token=continuation[0] if continuation else None,
            continuation_length=len(continuation) if continuation is not None else None,
            continuation_status='contextual prefix valid' if prefix else 'prefix mismatch; unavailable')
    return rows


def load(store):
    store.require('dataset')
    saved = read_json(store.root / 'dataset.json')
    if saved['config'] != store.manifest['config']:
        raise ValueError('Dataset configuration mismatch')
    rows = saved['records']
    validate_raw(rows, saved['config']['smoke'])
    for r in rows:
        ids = r['input_token_ids']
        if (r['token_count'] != len(ids) or len(r['token_pieces']) != len(ids)
            or r['extraction_position'] != len(ids) - 1 or r['final_token_id'] != ids[-1]
            or not r['final_token_text'].endswith('=')):
            raise ValueError('Invalid model token metadata')
    return rows


def semantics(model_type, blocks, dim):
    if model_type not in ('llama', 'gpt_bigcode'):
        raise ValueError('Unsupported architecture')
    embedding = ('token + learned absolute position embedding after eval dropout; L0 may vary with token length'
                 if model_type == 'gpt_bigcode' else 'token embedding only; RoPE enters attention blocks, not additive L0')
    final = 'ln_f LayerNorm' if model_type == 'gpt_bigcode' else 'RMSNorm'
    return dict(model_type=model_type, expected_levels=blocks + 1, expected_hidden_dim=dim,
        hidden_state_semantics=f'L0: {embedding}; L1..L{blocks-1}: block residual outputs; L{blocks}: final block after {final}',
        levels=[dict(level=i, normalized_transformer_depth=i / blocks,
                     state='embedding' if i == 0 else (f'final-normalized ({final})' if i == blocks else 'block residual'))
                for i in range(blocks + 1)])


def prepare(store, crystal_root, token=None, tokenizer=None, model_config=None, synthetic=False):
    cfg = store.manifest['config']
    if store.valid('dataset', store.manifest['config_fingerprint']):
        store.log('[SKIP] saved model dataset reused')
        save_sources(store)
        if store.manifest['stages'].get('dataset') != 'complete':
            store.stage('dataset', 'complete')
        return load(store)
    if 'dataset' in store.manifest['artifacts'] or any(k.startswith('chunk/') for k in store.manifest['artifacts']):
        raise ValueError('Corrupt saved dataset: restore from backup; do not regenerate')
    raw, shared_hash = import_text(crystal_root)
    if cfg['smoke']:
        raw = [r for r in raw if r['target'] in SMOKE_TARGETS]
    assets = store.root / 'tokenizer'
    if tokenizer is None:
        from transformers import AutoTokenizer, AutoConfig
        tokenizer = AutoTokenizer.from_pretrained(cfg['model_id'], revision=cfg['tokenizer_revision'],
            use_fast=cfg['use_fast'], trust_remote_code=False, token=token)
        model_config = AutoConfig.from_pretrained(cfg['model_id'], revision=cfg['model_revision'], trust_remote_code=False, token=token)
        if cfg['bos'] and tokenizer.is_fast:
            raise ValueError('OpenLLaMA requires authors’ slow tokenizer')
        tokenizer.save_pretrained(assets)
        model_config.save_pretrained(assets)
    if model_config.model_type != MODELS[cfg['model_key']]['model_type']:
        raise ValueError('Checkpoint architecture mismatch')
    meta = semantics(model_config.model_type, model_config.num_hidden_layers, model_config.hidden_size)
    meta.update(synthetic=synthetic, model_id=cfg['model_id'], model_revision=cfg['model_revision'],
        tokenizer_revision=cfg['tokenizer_revision'], tokenizer_class=type(tokenizer).__name__, is_fast=tokenizer.is_fast,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
        special_token_policy=cfg['special_policy'], shared_text_sha256=shared_hash,
        original_config_dtype=str(getattr(model_config, 'torch_dtype', None)), inference_dtype=cfg['model_dtype'],
        native_config=model_config.to_dict(), paper_checkpoint_match='unverified')
    rows = token_rows(raw, tokenizer, cfg)
    if not synthetic and any(tokenizer(r['prompt'], add_special_tokens=True)['input_ids'] != r['input_token_ids'] for r in rows):
        raise ValueError('Explicit special-token policy differs from native tokenizer defaults; inspect before extraction')
    write_json(store.root / 'dataset.json', dict(config=cfg, records=rows))
    write_json(store.root / 'tokenizer_provenance.json', meta)
    write_json(store.root / 'dataset_validation.json', dict(points=len(rows), shared_text_sha256=shared_hash,
        crystal_dataset_sha256=CRYSTAL_DATASET_SHA256, selected_text_sha256=fingerprint(raw),
        token_counts=dict(Counter(str(r['token_count']) for r in rows)),
        prefix_mismatches=sum(not r['continuation_prefix_valid'] for r in rows),
        merged_final_equals=sum(r['final_equals_shares_token'] for r in rows)))
    files = [store.root / n for n in ('dataset.json', 'tokenizer_provenance.json', 'dataset_validation.json')]
    files += sorted(p for p in assets.rglob('*') if p.is_file())
    store.finish('dataset', files, store.manifest['config_fingerprint'])
    save_sources(store)
    store.stage('dataset', 'complete')
    return load(store)


def save_sources(store):
    if store.valid('source_snapshot'):
        return
    # Code snapshots have their own checkpoint; implementation-only plotting fixes
    # must not alter the dataset identity or extraction dependencies.
    snapshot = provenance(store)
    project = Path(__file__).resolve().parents[2]
    for name in ('modal_multimodel_fullrange.py', 'MULTIMODEL_FULLRANGE.md', 'requirements-multimodel.txt'):
        src = project / name
        if src.exists():
            dest = store.root / 'source' / name
            atomic(dest, lambda f, src=src: f.write(src.read_bytes()))
            snapshot.append(dest)
    store.finish('source_snapshot', snapshot, fingerprint({str(p.relative_to(store.root)): digest(p) for p in snapshot}))
