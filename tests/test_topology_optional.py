import importlib.util
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("dionysus") is None or importlib.util.find_spec("gudhi") is None,
    reason="dionysus/gudhi not installed in local lightweight environment",
)


def test_zigzag_smoke_small_cloud():
    from numzig.zigzag import compute_zigzag, effective_intervals
    theta = np.linspace(0, 2*np.pi, 12, endpoint=False)
    circle = np.c_[np.cos(theta), np.sin(theta)]
    reps = np.stack([circle, circle + 0.01, circle + 0.02], axis=0).astype(np.float32)
    out = compute_zigzag(reps, knn_k=2, max_simplex_dim=3)
    assert 0 in out
    if 1 in out:
        eff = effective_intervals(out[1])
        assert eff.shape[1] == 2
