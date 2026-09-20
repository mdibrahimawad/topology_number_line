import numpy as np
from numzig.geometry import fit_spacing_direct, compute_layer_pca_metrics, best_pca_layer


def test_direct_beta_exact_geometric_gaps():
    beta, scale, r2 = fit_spacing_direct([2.0, 4.0, 8.0])
    assert np.isclose(beta, 2.0, rtol=1e-3)
    assert np.isclose(scale, 2.0, rtol=1e-3)
    assert np.isclose(r2, 1.0, atol=1e-6)


def test_pca_metrics_shapes():
    rng = np.random.default_rng(0)
    targets = np.array([10,11,12,100,101,102,1000,1001,1002,10000,10001,10002], dtype=float)
    groups = np.repeat([1,2,3,4], 3)
    base = np.c_[np.log10(targets), rng.normal(size=(len(targets), 4))]
    hidden = np.stack([base * 0.0, base, base + rng.normal(scale=0.01, size=base.shape)], axis=0).astype(np.float32)
    rows = compute_layer_pca_metrics(hidden, targets, groups)
    assert len(rows) == 3
    assert best_pca_layer(rows) in (1,2)
