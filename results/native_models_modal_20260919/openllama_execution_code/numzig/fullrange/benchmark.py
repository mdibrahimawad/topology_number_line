"""Local synthetic geometry benchmark; no tokenizer, model, or Modal dependencies.

python -m numzig.fullrange.benchmark --points 10000 --dimension 4096 --output /tmp/benchmark.json
"""
import argparse
import platform
import resource
import sys
import time

import numpy as np

from .analysis import exact_knn, projection
from .dataset import package_versions
from .storage import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--points', type=int, default=10000)
    p.add_argument('--dimension', type=int, default=4096)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if args.points <= 4 or args.dimension < 2:
        p.error('Need at least 5 points and 2 dimensions')
    rng = np.random.default_rng(42)
    x = rng.standard_normal((args.points, args.dimension), dtype=np.float32)
    print(f'SYNTHETIC geometry benchmark {x.shape}; no inference', flush=True)
    tic = time.perf_counter()
    neighbors, distances = exact_knn(x, np.arange(args.points, dtype=np.int32))
    knn = time.perf_counter() - tic
    print(f'Exact kNN finished in {knn:.3f} seconds; starting full SVD', flush=True)
    tic = time.perf_counter()
    arrays, _ = projection(x, np.arange(args.points))
    pca = time.perf_counter() - tic
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = dict(synthetic=True, inference=False, shape=list(x.shape), seed=42,
        platform=platform.platform(), machine=platform.machine(), knn_seconds=knn, pca_seconds=pca,
        peak_rss_bytes=int(rss if sys.platform == 'darwin' else rss * 1024),
        estimated_33_valid_layers_geometry_seconds=(knn + pca) * 33,
        estimate_scope='Same local hardware/dimensions only; excludes loading, metrics, commits, plots and inference. Not a cloud timing or cost estimate.',
        packages=package_versions(['numpy', 'scipy', 'scikit-learn']),
        neighbor_shape=list(neighbors.shape), pca_shape=list(arrays['scores'].shape),
        finite_distances=bool(np.isfinite(distances).all()))
    write_json(args.output, payload)
    print(payload, flush=True)


if __name__ == '__main__':
    main()
