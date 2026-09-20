"""Independent saved-result checks; no model access or cloud calls."""
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist

from numzig.fullrange.storage import digest, fingerprint, read_json, write_json
from numzig.taskrules.runtime import chunk_rows, load_saved_group

SPECS = {'crystal': (33, 4096), 'starcoderbase-3b': (37, 2816), 'openllama-3b': (27, 3200)}
TASKS = ('copy4', 'reverse4', 'swap_first4', 'swap_last4', 'numeric_copy', 'word_copy')


def audit_run(root):
    root = Path(root)
    counts = dict(dataset_rows=0, hidden_receipts=0, analysis_layers=0, analysis_files_hashed=0,
                  comparison_layers=0, comparison_files_hashed=0, sampled_source_groups=0,
                  sampled_hidden_chunks_hashed=0, sampled_geometry_layers=0,
                  sampled_knn_queries=0, shared_reconstruction_layers=0)
    report = dict(passed=False, errors=[], counts=counts, scope={
        'hidden_sources': 'All receipt metadata and file existence; full bytes/shape/finite checks for copy4 context 0 and word_copy context 1 in each model.',
        'analysis': 'Every analysis manifest artifact is hashed; independent geometry checks at first, middle and final contextual levels in the six source groups.',
        'shared_pca': 'Every comparison artifact is hashed and target/shape coverage checked; saved coordinates reconstructed for the corresponding sampled sources.',
        'knn': '16 deterministic queries per sampled layer, exact Euclidean distances to all 1000 candidates, excluding self and sorting ties by canonical ID.',
        'not_tested': 'Hidden-file bytes outside the six selected source groups; model inference correctness; visual appearance.'})
    try:
        rows = read_json(root / 'dataset.json')
        contract = read_json(root / 'contract.json')
        assert len(rows) == 12000 and [r['point_id'] for r in rows] == list(range(12000)), 'Dataset IDs/coverage'
        assert fingerprint(rows) == contract['dataset_hash'], 'Dataset fingerprint mismatch'
        groups = {(task, ctx): [r for r in rows if r['task'] == task and r['context_id'] == ctx]
                  for task in TASKS for ctx in (0, 1)}
        assert sum(map(len, groups.values())) == len(rows), 'Unknown task/context rows'
        reference = [r['target'] for r in groups['copy4', 0]]
        leading_counts = np.bincount([int(str(n)[0]) for n in reference], minlength=10)[1:]
        assert leading_counts.max() - leading_counts.min() <= 1, 'Unbalanced four-digit sample'
        permutations = {'copy4': (0, 1, 2, 3), 'reverse4': (3, 2, 1, 0),
                        'swap_first4': (1, 0, 2, 3), 'swap_last4': (0, 1, 3, 2)}
        for (task, ctx), selected in groups.items():
            targets = [r['target'] for r in selected]
            assert len(selected) == len(set(targets)) == 1000, f'Coverage {task}/{ctx}'
            assert targets == (reference if task in permutations else list(range(1, 1001))), 'Unpaired targets'
            comparison = groups['copy4' if task in permutations else 'numeric_copy', ctx]
            for row, paired in zip(selected, comparison):
                assert row['demo_values'] == paired['demo_values'] and len(set(row['demo_values'])) == 8, 'Unpaired demos'
                assert row['prompt'].endswith(row['input_text'] + '='), 'Query boundary'
                assert row['source_digits'] == [int(c) for c in str(row['target'])] and row['leading_digit'] == int(str(row['target'])[0]), 'Source digit labels'
                if task in permutations:
                    assert 1000 <= row['target'] <= 9999 and row['target'] not in row['demo_values'], 'Four-digit target/demo collision'
                    assert row['input_text'] == str(row['target']), 'Numeric input mismatch'
                    assert row['expected_output'] == ''.join(row['input_text'][i] for i in permutations[task]), 'Permutation mismatch'
                    assert row['output_digits'] == [int(c) for c in row['expected_output']], 'Output digit labels'
                    demo_body = ','.join(str(n) + '=' + ''.join(str(n)[i] for i in permutations[task]) for n in row['demo_values'])
                    assert row['prompt'].rsplit('\n', 1)[-1] == demo_body + ',' + row['input_text'] + '=', 'Demonstration transformation'
                else:
                    assert row['expected_output'] == row['input_text'], 'Copy output mismatch'
                    if task == 'numeric_copy':
                        assert row['input_text'] == str(row['target']), 'Numeric-copy realization mismatch'
        counts['dataset_rows'] = len(rows)
        for model, (levels, dimensions) in SPECS.items():
            base = root / model
            expected_chunks = list(chunk_rows(rows))
            assert {p.stem for p in (base / 'receipts').glob('*.json')} == {key for key, _ in expected_chunks}, 'Hidden receipt coverage'
            for key, part in expected_chunks:
                receipt = read_json(base / 'receipts' / f'{key}.json')
                assert receipt['contract'] == contract['fingerprint'] and receipt['model'] == model, 'Hidden contract/model'
                assert receipt['ids'] == [r['point_id'] for r in part], 'Chunk ID alignment'
                assert receipt['shape'] == [levels, len(part), dimensions], 'Chunk shape'
                assert set(receipt['files']) == {f'hidden/{key}.npz', f'tokens/{key}.json'}, 'Chunk artifacts'
                assert all((base / name).is_file() for name in receipt['files']), 'Missing hidden artifacts'
                counts['hidden_receipts'] += 1
            for task, ctx in groups:
                directory = base / 'analysis' / f'{task}_ctx{ctx}'
                manifest = read_json(directory / 'manifest.json')
                assert fingerprint(manifest['config']) == manifest['config_fingerprint'], 'Analysis config fingerprint'
                assert manifest['config']['model'] == model and manifest['config']['task'] == task and str(manifest['config']['context_id']) == str(ctx), 'Analysis identity'
                assert manifest['config']['shape'] == [levels, 1000, dimensions], 'Analysis input shape'
                assert set(manifest['artifacts']) == {f'layer/{i:02d}' for i in range(levels)}, 'Analysis level coverage'
                for artifact in manifest['artifacts'].values():
                    assert artifact['status'] == 'complete', 'Incomplete analysis artifact'
                    for name, sha in artifact['files'].items():
                        assert digest(directory / name) == sha, f'Analysis hash: {directory}/{name}'
                        counts['analysis_files_hashed'] += 1
                    counts['analysis_layers'] += 1
                summary = read_json(directory / 'summary.json')
                assert summary['status'] == 'complete' and summary['layer_count'] == levels, 'Analysis summary incomplete'
            for ctx in (0, 1):
                for family, family_tasks in (('permutations', TASKS[:4]), ('notation', TASKS[4:])):
                    directory = base / 'comparison' / f'ctx{ctx}'
                    for layer in range(levels):
                        receipt = read_json(directory / family / f'layer_{layer:02d}.json')
                        assert receipt['model'] == model and receipt['context_id'] == ctx and receipt['layer'] == layer, 'Comparison identity'
                        assert receipt['tasks'] == list(family_tasks) and receipt['count_per_condition'] == 1000, 'Comparison conditions'
                        required = {f'{family}/layer_{layer:02d}.npz', f'{family}/layer_{layer:02d}_input.png'}
                        if family == 'permutations':
                            required.add(f'{family}/layer_{layer:02d}_output.png')
                        assert set(receipt['files']) == required, 'Comparison artifact coverage'
                        for name, sha in receipt['files'].items():
                            assert digest(directory / name) == sha, f'Comparison hash: {directory}/{name}'
                            counts['comparison_files_hashed'] += 1
                        with np.load(directory / family / f'layer_{layer:02d}.npz', allow_pickle=False) as z:
                            assert z['scores'].shape == (len(family_tasks), 1000, 3) and np.isfinite(z['scores']).all(), 'Comparison scores'
                            assert z['task_names'].tolist() == list(family_tasks), 'Comparison task order'
                            assert z['targets'].tolist() == [r['target'] for r in groups[family_tasks[0], ctx]], 'Comparison targets'
                        counts['comparison_layers'] += 1
            for task, ctx in (('copy4', 0), ('word_copy', 1)):
                x, token_rows = load_saved_group(base, model, rows, task, ctx, contract['fingerprint'])
                selected = groups[task, ctx]
                assert [r['point_id'] for r in token_rows] == [r['point_id'] for r in selected], 'Token rows/hidden IDs'
                assert [r['target'] for r in token_rows] == [r['target'] for r in selected], 'Token targets'
                counts['sampled_source_groups'] += 1
                selected_ids = {r['point_id'] for r in selected}
                counts['sampled_hidden_chunks_hashed'] += sum(any(r['point_id'] in selected_ids for r in part) for _, part in chunk_rows(rows))
                sampled = np.linspace(0, 999, 16, dtype=int)
                for layer in sorted({1, levels // 2, levels - 1}):
                    a = np.asarray(x[layer], dtype=np.float64)
                    archive = base / 'analysis' / f'{task}_ctx{ctx}' / 'layers' / f'layer_{layer:02d}.npz'
                    with np.load(archive, allow_pickle=False) as z:
                        ids = np.array([r['point_id'] for r in selected])
                        assert np.array_equal(z['point_ids'], ids), 'Analysis point IDs'
                        assert all(np.isfinite(z[name]).all() for name in ('mean', 'components', 'scores3', 'explained_variance', 'explained_variance_ratio', 'distances')), 'Nonfinite geometry'
                        np.testing.assert_allclose(z['mean'], a.mean(axis=0), atol=1e-9, rtol=1e-9)
                        np.testing.assert_allclose(z['components'] @ z['components'].T, np.eye(3), atol=1e-8)
                        np.testing.assert_allclose(z['scores3'][sampled], (a[sampled] - z['mean']) @ z['components'].T, atol=1e-7, rtol=1e-7)
                        score_variance = np.var(z['scores3'], axis=0, ddof=1)
                        np.testing.assert_allclose(score_variance, z['explained_variance'], atol=1e-7, rtol=1e-6)
                        np.testing.assert_allclose(score_variance / np.var(a, axis=0, ddof=1).sum(), z['explained_variance_ratio'], atol=1e-8, rtol=1e-6)
                        distances = cdist(a[sampled], a)
                        distances[np.arange(len(sampled)), sampled] = np.inf
                        nearest = np.argsort(distances, axis=1, kind='stable')[:, :4]
                        assert np.array_equal(z['neighbors'][sampled], ids[nearest]), 'Independent kNN identity mismatch'
                        np.testing.assert_allclose(z['distances'][sampled], np.take_along_axis(distances, nearest, axis=1), atol=1e-8, rtol=1e-8)
                        for dimension in (0, 1):
                            diagram = z[f'h{dimension}']
                            assert diagram.ndim == 2 and diagram.shape[1] == 2 and np.isfinite(diagram[:, 0]).all(), 'PH interval shape/births'
                            assert np.all(diagram[:, 1] >= diagram[:, 0]), 'Negative PH lifetime'
                            assert (np.isposinf(diagram[:, 1]).sum() == 1 if dimension == 0 else np.isfinite(diagram).all()), 'PH infinite deaths'
                        counts['sampled_geometry_layers'] += 1
                        counts['sampled_knn_queries'] += len(sampled)
                    family = 'permutations' if task == 'copy4' else 'notation'
                    with np.load(base / 'comparison' / f'ctx{ctx}' / family / f'layer_{layer:02d}.npz', allow_pickle=False) as z:
                        task_index = z['task_names'].tolist().index(task)
                        np.testing.assert_allclose(z['scores'][task_index, sampled], (a[sampled] - z['mean']) @ z['components'].T, atol=1e-7, rtol=1e-7)
                        counts['shared_reconstruction_layers'] += 1
                del x
        assert counts['hidden_receipts'] == 1125 and counts['analysis_layers'] == 1164 and counts['comparison_layers'] == 388, 'Final coverage counts'
        report['passed'] = True
    except Exception as error:
        report['errors'].append(f'{type(error).__name__}: {error}')
    write_json(root / 'validation.json', report)
    return report
