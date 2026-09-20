from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr
from sklearn.decomposition import PCA


def fit_spacing_direct(gaps) -> tuple[float, float, float]:
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
    predicted = scale * basis
    ss_res = float(np.sum((gaps - predicted) ** 2))
    ss_tot = float(np.sum((gaps - gaps.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return beta, scale, r2


def compute_layer_pca_metrics(hidden: np.ndarray, targets: np.ndarray, group_ids: np.ndarray) -> list[dict]:
    """Exact PCA/rho/direct-beta branch used for comparison with the user's paper."""
    if hidden.ndim != 3:
        raise ValueError("hidden must have shape [layers, points, hidden_dim]")
    rows: list[dict] = []
    groups = sorted(np.unique(group_ids).tolist())
    for layer in range(hidden.shape[0]):
        X = np.asarray(hidden[layer], dtype=np.float32)
        if not np.isfinite(X).all():
            raise ValueError(f"non-finite hidden state at layer {layer}")
        try:
            pca = PCA(n_components=2)
            z2 = pca.fit_transform(X)
        except ValueError:
            rows.append({"layer": layer, "ev_pc1": math.nan, "rho_abs": math.nan, "beta": math.nan, "beta_scale": math.nan, "beta_r2": math.nan})
            continue
        pc1 = z2[:, 0]
        rho, _ = spearmanr(np.asarray(targets, dtype=float), pc1)
        means = np.asarray([pc1[group_ids == g].mean() for g in groups], dtype=float)
        beta, scale, r2 = fit_spacing_direct(np.abs(np.diff(means)))
        rows.append(
            {
                "layer": layer,
                "ev_pc1": float(pca.explained_variance_ratio_[0]),
                "rho_abs": float(abs(rho)) if np.isfinite(rho) else math.nan,
                "beta": beta,
                "beta_scale": scale,
                "beta_r2": r2,
            }
        )
    return rows


def best_pca_layer(rows: list[dict]) -> int:
    finite = [r for r in rows if np.isfinite(r["ev_pc1"])]
    if not finite:
        raise ValueError("no finite PCA layer")
    return int(max(finite, key=lambda r: r["ev_pc1"])["layer"])


def write_metrics_csv(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["layer", "ev_pc1", "rho_abs", "beta", "beta_scale", "beta_r2"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
