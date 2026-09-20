"""Spacing estimators for number-line group means."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def fit_spacing_direct(gaps, *, compute_r2: bool = True) -> tuple[float, float, float]:
    """Fit d_i = scale * beta**i by least squares in the original gap space."""
    gaps = np.asarray(gaps, dtype=float)
    if len(gaps) < 2 or not np.all(np.isfinite(gaps)) or not np.any(gaps):
        return float("nan"), float("nan"), float("nan")

    i = np.arange(len(gaps), dtype=float)

    def loss(log_beta: float) -> float:
        basis = np.exp(log_beta * i)
        scale = np.dot(gaps, basis) / np.dot(basis, basis)
        return float(np.sum((gaps - scale * basis) ** 2))

    result = minimize_scalar(loss, bounds=(-8, 8), method="bounded")
    beta = float(np.exp(result.x))
    basis = beta**i
    scale = float(np.dot(gaps, basis) / np.dot(basis, basis))
    if not compute_r2:
        return beta, scale, float("nan")
    predicted = scale * basis
    ss_res = float(np.sum((gaps - predicted) ** 2))
    ss_tot = float(np.sum((gaps - gaps.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return beta, scale, r2


if __name__ == "__main__":
    beta, scale, r2 = fit_spacing_direct([2.0, 4.0, 8.0])
    assert np.isclose([beta, scale, r2], [2.0, 2.0, 1.0]).all()