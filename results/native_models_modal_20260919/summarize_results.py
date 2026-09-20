"""Summarize measured saved outputs and monitoring snapshots; never run inference."""
from pathlib import Path
import hashlib
import json
import statistics
from collections import Counter
import numpy as np

B = Path(__file__).resolve().parent


def stats(values):
    return dict(min=min(values), median=statistics.median(values), max=max(values)) if values else None


def summarize():
    result = {}
    for model in ('starcoderbase-3b', 'openllama-3b'):
        root = B / 'full_complete' / (model + '_fullrange_1_10000_ctx1234_k4_seed42')
        manifest = json.loads((root / 'manifest.json').read_text())
        assert all(manifest['stages'].get(k) == 'complete' for k in ('extraction', 'analysis', 'plots', 'validation', 'package'))
        metrics = json.loads((root / 'layer_metrics.json').read_text())
        valid = [m for m in metrics if m['status'] == 'valid']
        rows = json.loads((root / 'dataset.json').read_text())['records']
        leading = np.array([r['leading_digit'] for r in rows])
        lengths = np.array([r['digit_count'] for r in rows])
        # Independently check every heatmap and conditional denominator from IDs.
        for m in valid:
            with np.load(root / f'analysis/layer_{m["layer"]:02d}.npz') as a:
                neighbors = a['neighbors']
                same = leading[:, None] == leading[neighbors]
                cross = lengths[:, None] != lengths[neighbors]
                counts = np.bincount(((leading[:, None]-1)*9 + leading[neighbors]-1).ravel(), minlength=81).reshape(9,9)
                np.testing.assert_array_equal(counts, a['digit_counts'])
                np.testing.assert_allclose(counts / counts.sum(axis=1, keepdims=True), a['digit_fractions'], rtol=0, atol=0)
                assert m['same_leading_digit_fraction'] == float(same.mean())
                assert m['cross_digit_length_fraction'] == float(cross.mean())
                assert m['conditional_denominator'] == int(cross.sum())
                assert m['same_leading_digit_given_cross_length'] == (float(same[cross].mean()) if cross.any() else None)
        runtimes = [json.loads(p.read_text()) for p in (root / 'workers').rglob('extraction_runtime.json')]
        benchmarks = [json.loads(p.read_text()) for p in (root / 'workers').rglob('extraction_benchmark.json')]
        assert all(r['passed'] and r['model_loads'] == 1 and r['dtype'] == 'torch.float32' for r in runtimes)
        prefix = 'star' if model.startswith('star') else 'open'
        logs = (['star_full_01'] if prefix == 'star' else
                ['open_full_01', 'open_recover_analyze_02', 'open_finish_03'])
        snapshots = []
        for log in logs:
            snapshots += [json.loads(s) for s in (B / (log + '_concurrency.jsonl')).read_text().splitlines() if s.strip()]
        functions = ('extraction', 'analysis_job', 'plot_job', 'pipeline')
        peak = {f: {kind: max(s.get('functions', {}).get(f, {}).get(kind, 0) for s in snapshots)
                    for kind in ('running', 'containers', 'backlog')} for f in functions}
        for path in (root / 'execution').glob('*_observed.json'):
            event = json.loads(path.read_text())
            function = dict(extraction='extraction', analysis='analysis_job', plot='plot_job')[event['stage']]
            for observation in event['observations']:
                for kind, field in [('running', 'running_inputs'), ('containers', 'containers'), ('backlog', 'backlog')]:
                    peak[function][kind] = max(peak[function][kind], observation[field])
        def cores(s, kind):
            return sum(s.get('functions', {}).get(f, {}).get(kind, 0) * (2 if f in ('extraction', 'analysis_job') else 1) for f in functions)
        selected = {str(i): {k: metrics[i].get(k) for k in ('status', 'pca', 'same_leading_digit_fraction',
            'cross_digit_length_fraction', 'same_leading_digit_given_cross_length', 'conditional_denominator',
            'largest_component_fraction', 'mutual_neighbor_fraction')} for i in (0, 1, (len(metrics)-1)//2, len(metrics)-1)}
        with (root / 'lightweight.tar.gz').open('rb') as stream:
            package_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
        result[model] = dict(points=len(rows), levels=len(metrics), valid_layers=len(valid),
            independent_heatmap_and_conditional_denominator_checks_passed=True,
            config=manifest['config'], token_count_distribution=dict(Counter(r['token_count'] for r in rows)),
            hidden_bytes=sum(p.stat().st_size for p in (root / 'hidden').glob('*.npy')),
            static_figures=len(list((root / 'figures').glob('*.png'))),
            manifest_sha256=hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest(),
            lightweight_bytes=(root / 'lightweight.tar.gz').stat().st_size, lightweight_sha256=package_hash,
            knn_seconds=stats([m['knn_seconds'] for m in valid]), pca_seconds=stats([m['pca_seconds'] for m in valid]),
            layer_process_peak_rss_bytes=stats([m['process_peak_rss_bytes'] for m in valid]),
            extraction_workers_with_runtime=len(runtimes), model_load_seconds=stats([r['load_seconds'] for r in runtimes]),
            workers_with_final_benchmark=len(benchmarks), extraction_worker_seconds=stats([v['seconds'] for v in benchmarks]),
            benchmark_points=sum(v['points'] for v in benchmarks),
            peak_cuda_allocated_bytes=max(v['peak_cuda_memory_bytes'] for v in benchmarks),
            observation_peaks=peak, peak_cores_allocated_to_running_inputs=max(cores(s, 'running') for s in snapshots),
            peak_cores_in_observed_containers=max(cores(s, 'containers') for s in snapshots), selected_layers=selected)
    (B / 'measured_results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({model: {k: r[k] for k in ('points', 'levels', 'static_figures', 'observation_peaks', 'knn_seconds', 'pca_seconds')} for model, r in result.items()}, indent=2))


if __name__ == '__main__':
    summarize()
