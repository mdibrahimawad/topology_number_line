"""Read-only audit of the exported task-rule gallery and its source artifacts."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

ROOT = Path('/Users/mohammed.awad/Documents/Codex/2026-09-19/c/outputs/taskrules/run')
OUT = ROOT.parent / 'gallery_validation.json'
NOTES = Path('/Users/mohammed.awad/Documents/Codex/2026-09-19/c/work/taskrules/gallery_validation.md')
MODELS = {'crystal': 33, 'starcoderbase-3b': 37, 'openllama-3b': 27}
TASKS = ('copy4', 'reverse4', 'swap_first4', 'swap_last4', 'numeric_copy', 'word_copy')
started = time.monotonic()
failures = []
checks = Counter()
referenced = set()
counts = Counter()


def load(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def check(condition, message):
    checks['assertions'] += 1
    if not bool(condition):
        failures.append(message)


def same(actual, expected, message):
    a, b = np.asarray(actual), np.asarray(expected)
    check(a.shape == b.shape and np.array_equal(a, b), message)
    checks['exact_array_comparisons'] += 1


def ref(relative):
    path = ROOT / relative
    check(path.resolve().is_relative_to(ROOT.resolve()), f'Unsafe reference: {relative}')
    check(path.is_file(), f'Missing referenced file: {relative}')
    referenced.add(str(relative))
    return path


dataset_path = ROOT / 'dataset.json'
manifest_path = ROOT / 'gallery/manifest.json'
initial_hashes = {str(p.relative_to(ROOT)): sha(p) for p in (dataset_path, manifest_path, ROOT / 'index.html')}
dataset, manifest = load(dataset_path), load(manifest_path)
by_id = {r['point_id']: r for r in dataset}
check(len(dataset) == 12000 and len(by_id) == 12000, 'Dataset point coverage is not 12,000 unique IDs')
by_group = defaultdict(list)
for row in dataset:
    by_group[(row['task'], str(row['context_id']))].append(row)
check(set(by_group) == {(task, str(ctx)) for task in TASKS for ctx in (0, 1)}, 'Dataset task/context coverage mismatch')
for key, rows in by_group.items():
    check(len(rows) == 1000 and len({r['target'] for r in rows}) == 1000, f'Dataset group coverage: {key}')

tokens = {}
behavior = {}
for model in MODELS:
    rows = []
    for path in sorted((ROOT / model / 'tokens').glob('*.json')):
        rows.extend(load(path))
    table = {r['point_id']: r for r in rows}
    check(len(rows) == 12000 and set(table) == set(by_id), f'{model}: token metadata coverage mismatch')
    tokens[model] = table
    counts['token_rows'] += len(rows)
    for row in rows:
        original = by_id[row['point_id']]
        for key in ('task', 'context_id', 'target', 'input_text', 'expected_output', 'prompt'):
            check(row[key] == original[key], f'{model} point {row["point_id"]}: token metadata {key} misaligned')
        check(row['token_count'] == len(row['input_token_ids']), f'{model} point {row["point_id"]}: token count mismatch')
    bpath = ref(f'pilot/{model}/behavior.json')
    groups = defaultdict(list)
    for row in load(bpath):
        groups[(row['task'], str(row['context_id']))].append(row)
    for key, entries in groups.items():
        n = len(entries)
        correct = sum(bool(row['exact_match']) for row in entries)
        behavior[(model, *key)] = (correct, n, correct / n)

check(manifest['status'] == 'complete', 'Gallery does not report complete')
check(not manifest['missing_groups'] and not manifest['missing_levels'] and not manifest['warnings'], 'Gallery reports missing groups or warnings')
group_keys = set()
degenerate = []
for group in manifest['groups']:
    model, task, ctx = group['model'], group['task'], str(group['context_id'])
    key = (model, task, ctx)
    check(key not in group_keys, f'Duplicate gallery group {key}')
    group_keys.add(key)
    rows = by_group[(task, ctx)]
    ids = [r['point_id'] for r in rows]
    targets = [r['target'] for r in rows]
    same([layer['layer'] for layer in group['layers']], np.arange(MODELS[model]), f'{key}: layer coverage/order')
    check(group['point_count'] == 1000, f'{key}: point count not 1000')
    summary_path = ref(group['summary'])
    summary = load(summary_path)
    check(summary['status'] == 'complete' and summary['layer_count'] == MODELS[model], f'{key}: incomplete source summary')
    saved_layers = {item['layer']: item for item in summary['layers']}
    correct, n, accuracy = behavior[key]
    b = group['behavior']
    check((b['correct'], b['n'], b['exact_accuracy']) == (correct, n, accuracy), f'{key}: pilot accuracy mismatch')
    check(bool(b['warning']) == (accuracy < .8), f'{key}: missing or spurious behavior warning')
    for layer in group['layers']:
        label = f'{model}/{task}/ctx{ctx}/L{layer["layer"]}'
        counts['layers'] += 1
        meta = saved_layers[layer['layer']]
        check(layer['status'] == meta['status'], f'{label}: status mismatch')
        check(layer['ph'] == meta['ph'], f'{label}: PH display metadata mismatch')
        archive_path = ref(layer['arrays'])
        expected_figures = {str((summary_path.parent / p).relative_to(ROOT)) for p in meta['figures']}
        check(set(layer['figures'].values()) == expected_figures, f'{label}: figure links mismatch source summary')
        for figure in layer['figures'].values():
            ref(figure)
            counts['layer_figures'] += 1
        with np.load(archive_path, allow_pickle=False) as archive:
            same(archive['point_ids'], ids, f'{label}: saved point IDs mismatch dataset')
            same(archive['targets'], targets, f'{label}: saved targets mismatch dataset')
            first_input = [str(r['leading_digit']) for r in rows]
            numeric = all(r['expected_output'].isdigit() for r in rows)
            first_output = [r['expected_output'][0] if numeric else r['expected_output'].split()[0] for r in rows]
            token_counts = [tokens[model][r['point_id']]['token_count'] for r in rows]
            same(archive['input_first_digit'], first_input, f'{label}: saved input digit labels mismatch dataset')
            same(archive['output_first_label'], first_output, f'{label}: saved expected-output labels mismatch dataset')
            same(archive['token_counts'], token_counts, f'{label}: saved token counts mismatch extraction metadata')
            source_digits = np.full((1000, 4), -1, dtype=np.int8)
            for i, row in enumerate(rows):
                source_digits[i, :len(row['source_digits'])] = row['source_digits']
            same(archive['source_digits'], source_digits, f'{label}: source digit positions mismatch dataset')
            if layer['status'] == 'degenerate':
                check('data3d' not in layer and 'scores3' not in archive, f'{label}: degenerate cloud has a 3D representation')
                degenerate.append(label)
                continue
            counts['data3d'] += 1
            data = load(ref(layer['data3d']))
            same(data['scores'], archive['scores3'], f'{label}: 3D JSON scores differ from saved PCA')
            same(data['point_ids'], archive['point_ids'], f'{label}: 3D point IDs mismatch')
            same(data['targets'], archive['targets'], f'{label}: 3D targets mismatch')
            same(data['input_first_digit'], archive['input_first_digit'], f'{label}: 3D input labels mismatch')
            same(data['output_first_label'], archive['output_first_label'], f'{label}: 3D output labels mismatch')
            same(data['token_counts'], archive['token_counts'], f'{label}: 3D token counts mismatch')
            same(data['explained_variance_ratio'], archive['explained_variance_ratio'], f'{label}: 3D PCA variance mismatch')
            same(layer['explained_variance_ratio'], archive['explained_variance_ratio'], f'{label}: gallery PCA variance mismatch')
            same(data['input_text'], [r['input_text'] for r in rows], f'{label}: input hover strings mismatch dataset')
            same(data['expected_output'], [r['expected_output'] for r in rows], f'{label}: output hover strings mismatch dataset')
            expected_kind = 'first decimal output digit' if numeric else 'first output word'
            check(data['output_label_kind'] == expected_kind, f'{label}: wrong lexical-versus-digit label description')
            scores = np.asarray(data['scores'])
            evr = np.asarray(data['explained_variance_ratio'])
            check(scores.shape == (1000, 3) and np.isfinite(scores).all(), f'{label}: nonfinite or malformed 3D scores')
            check(evr.shape == (3,) and np.isfinite(evr).all() and np.all(evr >= 0) and evr.sum() <= 1 + 1e-12,
                  f'{label}: nonfinite or invalid explained variance')
            counts['three_d_target_rows'] += len(scores)
            counts['three_d_coordinate_scalars'] += scores.size
    print(f'Validated {model} {task} context {ctx}', flush=True)

expected_groups = {(model, task, str(ctx)) for model in MODELS for task in TASKS for ctx in (0, 1)}
check(group_keys == expected_groups, 'Model/task/context group grid incomplete')
comparison_keys = set()
for comparison in manifest['comparisons']:
    model, ctx, family, layer = comparison['model'], comparison['context_id'], comparison['family'], comparison['layer']
    label = f'{model}/ctx{ctx}/{family}/L{layer}'
    key = (model, str(ctx), family, layer)
    check(key not in comparison_keys, f'Duplicate comparison {label}')
    comparison_keys.add(key)
    path = ref(comparison['arrays'])
    meta = load(path.with_suffix('.json'))
    for figure in comparison['figures'].values():
        ref(figure)
        counts['comparison_figures'] += 1
    with np.load(path, allow_pickle=False) as archive:
        same(archive['task_names'], comparison['tasks'], f'{label}: task ordering mismatch')
        same(archive['explained_variance_ratio'], comparison['explained_variance_ratio'], f'{label}: comparison variance mismatch')
        check(np.isfinite(archive['scores']).all(), f'{label}: nonfinite shared-PCA scores')
        check(archive['scores'].shape == (len(comparison['tasks']), 1000, 3), f'{label}: shared-PCA shape mismatch')
        for index, task in enumerate(comparison['tasks']):
            rows = by_group[(task, str(ctx))]
            same(archive['targets'], [r['target'] for r in rows], f'{label}/{task}: comparison target pairing mismatch')
            same(archive['input_leading_digit'], [r['leading_digit'] for r in rows], f'{label}/{task}: comparison input label mismatch')
            same(archive['output_leading_digit'][index], [r['output_digits'][0] for r in rows],
                 f'{label}/{task}: comparison output label mismatch')
    check(comparison['metrics'] == meta['metrics'], f'{label}: displayed comparison metrics mismatch')
    counts['comparisons'] += 1

expected_comparisons = {(model, str(ctx), family, layer) for model, levels in MODELS.items()
    for ctx in (0, 1) for family in ('permutations', 'notation') for layer in range(levels)}
check(comparison_keys == expected_comparisons, 'Shared-comparison grid incomplete')

# Check every file protected by a completed analysis or comparison receipt.
# Comparison receipt JSONs do not hash themselves; their declared NPZ/PNG files
# are included, while layer metric JSONs are covered by the group manifest.
checksums = {}
for model in MODELS:
    for path in sorted((ROOT / model / 'analysis').glob('*/manifest.json')):
        saved_manifest = load(path)
        for artifact, record in saved_manifest['artifacts'].items():
            check(record['status'] == 'complete', f'{path.relative_to(ROOT)}/{artifact}: incomplete receipt')
            for relative, expected_sha in record['files'].items():
                file = path.parent / relative
                name = str(file.relative_to(ROOT))
                check(name not in checksums, f'Duplicate protected analysis file: {name}')
                checksums[name] = expected_sha
    for path in sorted((ROOT / model / 'comparison').glob('ctx*/*/layer_*.json')):
        receipt = load(path)
        for relative, expected_sha in receipt['files'].items():
            file = path.parent.parent / relative
            name = str(file.relative_to(ROOT))
            check(name not in checksums, f'Duplicate protected comparison file: {name}')
            checksums[name] = expected_sha
for relative, expected_sha in checksums.items():
    path = ref(relative)
    check(path.is_file() and sha(path) == expected_sha, f'Artifact SHA-256 mismatch: {relative}')
counts['sha256_protected_artifacts'] = len(checksums)
check(len(checksums) == 6724, 'Expected 6,724 protected analysis/comparison artifacts')

for path in ('index.html', 'SUMMARY.md', 'gallery/manifest.json', 'gallery/plotly.min.js'):
    ref(path)

pngs = sorted(path for path in referenced if path.endswith('.png'))
for relative in pngs:
    try:
        with Image.open(ROOT / relative) as image:
            check(image.format == 'PNG' and min(image.size) >= 200, f'Malformed PNG {relative}')
            image.verify()
    except Exception as exc:
        failures.append(f'Unreadable PNG {relative}: {exc!r}')
check(len(list(ROOT.rglob('*.png'))) == len(pngs) == 4008, 'Static PNG count differs from referenced 4,008 figures')
check(len(list((ROOT / 'gallery/data').glob('*.json'))) == counts['data3d'] == 1132, '3D file count mismatch')
check(counts['layers'] == 1164, 'Layer count mismatch')
check(counts['comparisons'] == 388, 'Comparison count mismatch')
check(counts['layer_figures'] == manifest['figure_count'] == 3426, 'Per-layer figure count mismatch')
check(counts['comparison_figures'] == manifest['comparison_figure_count'] == 582, 'Comparison figure count mismatch')
check(len(degenerate) == 32, 'Degenerate saved-level count mismatch')
for relative, old in initial_hashes.items():
    check(sha(ROOT / relative) == old, f'Input artifact changed during audit: {relative}')

report = dict(status='passed' if not failures else 'failed',
    audited_at_utc=datetime.now(timezone.utc).isoformat(), root=str(ROOT), scope='Local read-only gallery and label-alignment audit',
    counts=dict(groups=len(group_keys), static_figures=len(pngs), referenced_files=len(referenced),
                degenerate_layers=len(degenerate), **dict(counts)),
    checks=dict(checks), exact_tolerance='JSON scores and variance must equal original NPZ values exactly; no rounding tolerance',
    input_sha256=initial_hashes, failures=failures, degenerate_layers=degenerate,
    refresh_context='Revalidated after canonical timing-only layer metadata restoration and the 3D digit-9 palette correction; no scientific arrays were changed.',
    metadata_repair_record=str(ROOT.parent / 'metadata_repair.json'),
    artifact_hash_scope='All files listed in 36 group analysis manifests and 388 comparison receipts (NPZs, per-layer JSON metrics, PNG figures).',
    seconds=time.monotonic()-started,
    excluded=['No model inference or cloud actions', 'No PH, kNN or PCA recomputation',
              'No browser or visual interpretation', 'Does not validate causality or unseen-target behavioral accuracy'])
OUT.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
NOTES.write_text('# Independent gallery audit\n\n' +
    f'**{report["status"].upper()}** — {len(failures)} failures.\n\n' +
    '- All 36 groups, 1,164 saved levels, 388 shared comparisons and 4,008 PNG figures are present.\n' +
    '- Every referenced file exists; every PNG passed integrity verification.\n' +
    '- All 6,724 analysis/comparison artifacts match their existing manifest or receipt SHA-256 checksums.\n' +
    '- All 1,132 interactive views match their NPZ scores, IDs, targets, labels and explained variance exactly.\n' +
    '- All 1,132,000 hover rows match the saved input/expected output; token counts match extraction metadata.\n' +
    '- All 36 pilot accuracy banners match saved generated-answer records; low-accuracy warnings are present where required.\n' +
    '- Thirty-two degenerate clouds are correctly omitted from interactive 3D views.\n' +
    '- No run artifact was written; no PH/inference was recomputed.\n\n' +
    'Refreshed after the parent restored one timing-only JSON field to its canonical saved value and aligned the 3D digit-9 color with static figures.\n\n' +
    ('Failures:\n' + '\n'.join('- ' + f for f in failures) if failures else 'No data-alignment or gallery-export defect found.\n'))
print(json.dumps({k: v for k, v in report.items() if k not in ('degenerate_layers', 'input_sha256')}, indent=2))
raise SystemExit(bool(failures))
