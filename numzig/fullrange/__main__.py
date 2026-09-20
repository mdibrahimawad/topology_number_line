"""Local-only entry points. This module never submits Modal work."""
import argparse
import json
from pathlib import Path

from . import NAME, configuration
from .dataset import SMOKE_TARGETS


def dry_run(smoke=False, stage='all', gpu_workers=10, cpu_workers=4):
    from .parallel import execution_settings
    settings = execution_settings(gpu_workers, cpu_workers)
    cfg = configuration(smoke)
    n = len(SMOKE_TARGETS) if smoke else 10000
    return dict(config=cfg, selected_stage=stage, cloud_execution=False,
        target_range=[1, 10000], prompt_count=n, targets='deterministic boundary-spanning subset' if smoke else 'every integer, ascending, exactly once',
        hardware='up to 10 independent L40S workers; bounded full-layer CPU workers', execution=settings,
        historical_expected_levels=33, historical_expected_hidden_dim=4096,
        dimensions='discovered and validated at load and every forward pass; historical values only estimate storage',
        hidden_float32_bytes=n * 33 * 4096 * 4, full_hidden_decimal_GB=10000 * 33 * 4096 * 4 / 1e9,
        chunk_size=cfg['chunk_size'], historical_max_chunk_bytes=cfg['chunk_size'] * 33 * 4096 * 4,
        pairwise_export='disabled; full float32 distance matrix would be 400 MB per layer',
        analysis_peak_array_estimate_bytes=4_000_000_000,
        analysis_memory_note='one full layer per CPU worker;  full float64 SVD workspace varies by BLAS; measured peak RSS recorded per layer',
        results_volume='numberline-zigzag-results', output_path='/artifacts/' + cfg['experiment'],
        runtime_estimate='GPU benchmark not run. Smoke saves measured extraction and per-layer analysis times; see runbook scaling assumptions.',
        revision=cfg['model_revision'])


class SyntheticTokenizer:
    """Character encoding for storage/interrupt tests. Never used by cloud stages."""
    is_fast = True
    def __call__(self, text, **kwargs):
        return {'input_ids': [ord(c) for c in text]}
    def decode(self, ids, **kwargs):
        return ''.join(chr(i) for i in ids)


def synthetic_vector(row):
    import numpy as np
    n = row['target']
    rng = np.random.default_rng(n)
    base = rng.normal(size=(3, 12)).astype(np.float32)
    base[0] = 1
    base[1, :2] = [np.log10(n), row['leading_digit']]
    base[2, :2] = [row['leading_digit'], np.log10(n)]
    return base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['dry-run', 'synthetic', 'analyze', 'plot'])
    parser.add_argument('--root', type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--stage', choices=['all', 'prepare', 'extract', 'analyze', 'plot'], default='all')
    parser.add_argument('--gpu-workers', type=int, default=10, help='GPU workers in dry-run cloud plan; local commands never infer')
    parser.add_argument('--cpu-workers', type=int, default=4, help='CPU workers in dry-run cloud plan; local saved-data analyze remains serial')
    args = parser.parse_args()
    if args.command == 'dry-run':
        print(json.dumps(dry_run(args.smoke, args.stage, args.gpu_workers, args.cpu_workers), indent=2))
        return
    if args.root is None:
        parser.error('--root is required for local saved-data commands')
    from .storage import Store
    if args.command == 'synthetic':
        from .dataset import prepare
        from .extract import extract_chunks
        from .analysis import analyze
        from .plots import render
        cfg = configuration(True)
        cfg['experiment'] += '_synthetic'
        store = Store(args.root, cfg)
        prepare(store, tokenizer=SyntheticTokenizer(), tokenizer_provenance={'synthetic': True})
        extract_chunks(store, synthetic_vector)
        analyze(store)
        render(store)
        print('SYNTHETIC validation only; no model loaded or inference run.')
    elif args.command == 'analyze':
        from .analysis import analyze
        analyze(Store(args.root))
    else:
        from .plots import render
        render(Store(args.root))


if __name__ == '__main__':
    main()
