"""Offline planning, saved-results stages, synthetic benchmark. Never loads weights."""
import argparse
from pathlib import Path
import json
import time
from . import MODELS, CRYSTAL_ROOT, configuration, resources
from .data import import_text
from .pipeline import local_analysis, local_plot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['dry-run', 'analyze', 'plot', 'validate', 'package', 'compare', 'benchmark'])
    p.add_argument('--root', type=Path)
    p.add_argument('--crystal-root', type=Path, default=CRYSTAL_ROOT)
    p.add_argument('--gpu-workers', type=int, default=10)
    p.add_argument('--cpu-workers', type=int, default=24)
    p.add_argument('--plot-workers', type=int, default=16)
    p.add_argument('--roots', nargs='+', type=Path)
    p.add_argument('--dimension', type=int, default=3200)
    args = p.parse_args()
    settings = resources(args.gpu_workers, args.cpu_workers, args.plot_workers)
    if args.stage == 'dry-run':
        rows, h = import_text(args.crystal_root)
        print(json.dumps(dict(mode='OFFLINE: no cloud, weights or pretrained inference', points_per_model=len(rows),
            total_new_observations=len(rows)*2, shared_text_sha256=h, models=MODELS, resources=settings,
            configs=[configuration(m) for m in MODELS]), indent=2))
        return
    if args.root is None:
        p.error('--root required for output/checkpoint location')
    if args.stage == 'benchmark':
        import resource
        import sys
        import numpy as np
        from threadpoolctl import threadpool_limits
        from numzig.fullrange.analysis import exact_knn, projection
        from numzig.fullrange.storage import write_json
        x = np.random.default_rng(42).normal(size=(10000, args.dimension)).astype(np.float32)
        with threadpool_limits(limits=2):
            tic = time.perf_counter(); exact_knn(x, np.arange(10000, dtype=np.int32)); knn = time.perf_counter()-tic
            tic = time.perf_counter(); projection(x, np.arange(1, 10001)); pca = time.perf_counter()-tic
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report = dict(synthetic=True, cloud=False, points=10000, dimension=args.dimension, blas_threads=2,
            knn_seconds=knn, pca_seconds=pca, peak_rss_bytes=rss if sys.platform == 'darwin' else rss*1024,
            limitation='Local host timing is not a Modal CPU benchmark; no pretrained weights used')
        write_json(args.root / 'synthetic_fullsize_benchmark.json', report); print(json.dumps(report, indent=2)); return
    from numzig.fullrange.storage import Store
    from numzig.fullrange.parallel import WorkerStore
    from numzig.fullrange.analysis import analysis_dependency
    from .pipeline import global_jobs, recover_analysis, analysis_worker, layer_cache, finalize_analysis
    from . import plots
    if args.stage == 'compare':
        if not args.roots:
            p.error('--roots requires saved Crystal and both new model directories')
        plots.comparison(args.roots, args.root); return
    # Local process isolation also avoids Matplotlib global-state races.
    from concurrent.futures import ProcessPoolExecutor
    store = Store(args.root)
    if args.stage == 'validate':
        from .validation import audit
        audit(store); return
    if args.stage == 'analyze':
        layer_cache(store)
        jobs = global_jobs([store], recover_analysis, args.cpu_workers, 'layer', analysis_dependency)
        function, limit = local_analysis, args.cpu_workers
    elif args.stage == 'plot':
        jobs = global_jobs([store], plots.recover, args.plot_workers, 'plot', plots.dependency)
        function, limit = local_plot, args.plot_workers
    else:
        plots.package(store); return
    errors = []
    with ProcessPoolExecutor(max_workers=min(limit, len(jobs)) or 1) as pool:
        futures = [pool.submit(function, job) for job in jobs]
        for future in futures:
            try:
                future.result()
            except Exception as exc:
                errors.append(str(exc))
    store = Store(args.root)
    missing = recover_analysis(store) if args.stage == 'analyze' else plots.recover(store)
    if errors or missing:
        raise RuntimeError(f'Completed peer checkpoints retained; remaining={missing}; errors={errors}')
    if args.stage == 'analyze':
        finalize_analysis(store)
    else:
        plots.finalize(store)



if __name__ == '__main__':
    main()
