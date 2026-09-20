from __future__ import annotations

from collections import Counter
import importlib.metadata
from pathlib import Path
import random
import subprocess

from numzig.prompts import _format_numeral_prompt
from . import MODEL_ID
from .storage import atomic, digest, fingerprint, read_json, write_json

SMOKE_TARGETS = sorted(set(range(1, 11)) | {d * p for p in (10, 100, 1000) for d in range(1, 10)}
                       | {99, 999, 9999, 10000})


def records(config):
    rng = random.Random(config['seed'])
    rows = []
    # Draw in production order, even in smoke, so smoke uses exactly the production stimuli.
    for target in range(1, 10001):
        demos = [rng.randint(lo, hi) for lo, hi in [(1, 9), (10, 99), (100, 999), (1000, 9999)]]
        if config['smoke'] and target not in SMOKE_TARGETS:
            continue
        prompt = _format_numeral_prompt([*demos, target])
        rows.append(dict(point_id=target - 1, target=target, target_string=str(target),
                         leading_digit=int(str(target)[0]), digit_count=len(str(target)),
                         demonstrations=dict(zip('ABCD', demos)), demonstration_digit_lengths=[1, 2, 3, 4],
                         demonstration_equals_target=[d == target for d in demos], prompt=prompt,
                         character_count=len(prompt), seed=config['seed']))
    return rows


def tokenize(rows, tokenizer):
    if not tokenizer.is_fast:
        raise ValueError('Crystal requires a fast tokenizer')
    for row in rows:
        ids = tokenizer(row['prompt'], add_special_tokens=True)['input_ids']
        last = tokenizer.decode([ids[-1]], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if last != '=' or not row['prompt'].endswith('='):
            raise ValueError(f'Point {row["point_id"]}: final token is {last!r}, expected exactly equals')
        combined = tokenizer(row['prompt'] + row['target_string'], add_special_tokens=True)['input_ids']
        prefix = combined[:len(ids)] == ids
        continuation = combined[len(ids):] if prefix else None
        row.update(input_token_ids=ids, token_count=len(ids), final_token_id=ids[-1],
                   final_token_text=last, extraction_position=len(ids) - 1,
                   continuation_prefix_valid=prefix, continuation_token_ids=continuation,
                   first_continuation_token=continuation[0] if continuation else None,
                   continuation_length=len(continuation) if continuation is not None else None)
    return rows


def validate_rows(rows, config):
    targets = SMOKE_TARGETS if config['smoke'] else list(range(1, 10001))
    if [r['target'] for r in rows] != targets:
        raise ValueError('Dataset target coverage/order mismatch')
    for r in rows:
        ds = list(r['demonstrations'].values())
        # JSON sorting can reorder keys; A/B/C/D order remains canonical.
        if (r['point_id'] != r['target'] - 1 or r['prompt'] != _format_numeral_prompt([*ds, r['target']])
            or [len(str(d)) for d in ds] != [1, 2, 3, 4]
            or r['final_token_text'] != '=' or r['extraction_position'] != len(r['input_token_ids']) - 1
            or r['token_count'] != len(r['input_token_ids']) or r['final_token_id'] != r['input_token_ids'][-1]):
            raise ValueError(f'Invalid stimulus metadata: point {r["point_id"]}')


def package_versions(names):
    out = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def source_files():
    root = Path(__file__).resolve().parents[2]
    return root, [*sorted((root / 'numzig').rglob('*.py')), *sorted((root / 'tests').glob('*.py')),
                  root / 'numzig/fullrange/viewer.html', root / 'modal_crystal_fullrange.py',
                  root / 'requirements-crystal-fullrange.txt', root / 'CRYSTAL_FULLRANGE.md']


def provenance(store):
    root, sources = source_files()
    saved = []
    hashes = {}
    for p in sources:
        if p.is_file():
            relative = p.relative_to(root)
            dest = store.root / 'source' / relative
            atomic(dest, lambda f, p=p: f.write(p.read_bytes()))
            saved.append(dest)
            hashes[str(relative)] = digest(dest)
    git = None
    try:
        top = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], cwd=root, stderr=subprocess.DEVNULL).decode().strip()
        if Path(top).resolve() == root:
            git = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    path = store.root / 'provenance.json'
    write_json(path, dict(source_hashes=hashes, git_revision=git,
                         git_note='Only use Git revision when this project is the repository root',
                         packages=package_versions(['numpy', 'scipy', 'scikit-learn', 'matplotlib',
                                                    'torch', 'transformers', 'tokenizers', 'modal', 'plotly'])))
    return [path, *saved]


def prepare(store, token=None, tokenizer=None, tokenizer_provenance=None):
    config = store.manifest['config']
    dep = store.manifest['config_fingerprint']
    if store.valid('dataset', dep):
        rows = load_dataset(store)
        store.log(f'[SKIP] dataset: reusing all {len(rows)} saved prompts and token IDs')
        return rows
    if 'dataset' in store.manifest['artifacts'] or any(k.startswith('chunk/') for k in store.manifest['artifacts']):
        raise ValueError('Saved dataset is corrupt/incompatible; restore it from backup, never regenerate contexts')
    files = []
    if tokenizer is None:
        from huggingface_hub import HfApi, hf_hub_download
        from transformers import AutoConfig
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        model_info = HfApi().model_info(MODEL_ID, revision=config['model_revision'], token=token)
        wrapper_info = HfApi().model_info(config['tokenizer_code_repo'], revision=config['tokenizer_code_revision'], token=token)
        if model_info.sha != config['model_revision'] or wrapper_info.sha != config['tokenizer_code_revision']:
            raise ValueError('Resolved revision mismatch')
        assets = store.root / 'tokenizer'
        assets.mkdir(exist_ok=True)
        for name in ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                     'configuration_crystalcoder.py', 'modeling_crystalcoder.py', 'config.json']:
            src = hf_hub_download(MODEL_ID, name, revision=config['model_revision'], token=token)
            dest = assets / name
            atomic(dest, lambda f, src=src: f.write(Path(src).read_bytes()))
            files.append(dest)
        wrapper_src = hf_hub_download(config['tokenizer_code_repo'], 'tokenization_crystalcoder_fast.py',
                                      revision=config['tokenizer_code_revision'], token=token)
        wrapper_dest = assets / 'tokenization_crystalcoder_fast.py'
        atomic(wrapper_dest, lambda f: f.write(Path(wrapper_src).read_bytes()))
        files.append(wrapper_dest)
        cls = get_class_from_dynamic_module('tokenization_crystalcoder_fast.CrystalCoderTokenizerFast',
                                           config['tokenizer_code_repo'], revision=config['tokenizer_code_revision'], token=token)
        tokenizer = cls.from_pretrained(MODEL_ID, revision=config['tokenizer_revision'], token=token)
        model_config = AutoConfig.from_pretrained(MODEL_ID, revision=config['model_revision'], trust_remote_code=True, token=token)
        source = (assets / 'modeling_crystalcoder.py').read_text()
        if 'hidden_states = self.ln_f(hidden_states)' not in source:
            raise ValueError('Unknown final normalization convention; inspect new model source')
        tokenizer_provenance = dict(requested_model=MODEL_ID, resolved_model=model_info.id,
            model_revision=model_info.sha, tokenizer_revision=config['tokenizer_revision'],
            requested_wrapper=config['tokenizer_code_repo'], resolved_wrapper=wrapper_info.id,
            wrapper_revision=wrapper_info.sha, tokenizer_class=type(tokenizer).__name__,
            is_fast=tokenizer.is_fast, backend_sha256=fingerprint(tokenizer.backend_tokenizer.to_str()),
            compatibility='Explicit pinned CrystalCoderTokenizerFast wrapper; same referenced code, no model substitution',
            expected_levels=int(model_config.num_hidden_layers) + 1, expected_hidden_dim=int(model_config.hidden_size),
            hidden_state_semantics='0: scaled token embedding after dropout (eval); 1..n-1: block residual outputs; n: final block after ln_f. Rotary positions, no additive positional embedding.',
            assets_sha256={p.name: digest(p) for p in files})
    rows = tokenize(records(config), tokenizer)
    validate_rows(rows, config)
    path = store.root / 'dataset.json'
    write_json(path, dict(config=config, records=rows))
    meta = store.root / 'tokenizer_provenance.json'
    write_json(meta, tokenizer_provenance or {'synthetic': True})
    counts = lambda field: {str(k): v for k, v in sorted(Counter(r[field] for r in rows).items())}
    report = store.root / 'dataset_validation.json'
    write_json(report, dict(points=len(rows), leading_digit_counts=counts('leading_digit'),
        character_count_distribution=counts('character_count'), token_count_distribution=counts('token_count'),
        chance_match_points=sum(any(r['demonstration_equals_target']) for r in rows),
        invalid_continuation_prefixes=sum(not r['continuation_prefix_valid'] for r in rows),
        actual_unpadded_lengths=True, prompt_coverage_valid=True))
    files.extend([path, meta, report, *provenance(store)])
    store.finish('dataset', files, dep)
    store.stage('dataset', 'complete')
    return rows


def load_dataset(store):
    store.require('dataset')
    payload = read_json(store.root / 'dataset.json')
    if payload['config'] != store.manifest['config']:
        raise ValueError('Dataset configuration mismatch')
    validate_rows(payload['records'], payload['config'])
    return payload['records']
