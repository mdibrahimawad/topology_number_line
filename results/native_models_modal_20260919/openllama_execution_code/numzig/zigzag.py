"""Zigzag persistence ported from the uploaded ZigZagLLMs code.

The construction is deliberately the same: kNN graph in the ORIGINAL hidden
space -> Gudhi clique expansion -> adjacent-layer intersection complexes ->
Dionysus zigzag persistence. PCA is never used to construct topology.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.neighbors import kneighbors_graph


def _contiguous_ranges(values: np.ndarray):
    if len(values) == 0:
        return []
    values = np.sort(np.asarray(values, dtype=int))
    # Exact time convention from ZigZagLLMs utils.fclaux.ranges:
    # a contiguous block [a..b] becomes insertion/deletion times (a+1, b+2).
    starts = [int(values[0]) + 1]
    ends: list[int] = []
    for prev, cur in zip(values[:-1], values[1:]):
        if cur != prev + 1:
            ends.append(int(prev) + 2)
            starts.append(int(cur) + 1)
    ends.append(int(values[-1]) + 2)
    return list(zip(starts, ends))


def build_zigzag_inputs(reps: np.ndarray, knn_k: int, max_simplex_dim: int):
    import dionysus as d
    import gudhi as gd

    if reps.ndim != 3:
        raise ValueError("reps must be [layers, points, hidden_dim]")
    if not (1 <= knn_k < reps.shape[1]):
        raise ValueError(f"knn_k must be in [1,{reps.shape[1]-1}], got {knn_k}")

    simplices: list[list[int]] = []
    simplex_id: dict[tuple[int, ...], int] = {}
    layer_simplex_ids: list[list[int]] = []

    for layer in range(reps.shape[0]):
        X = np.asarray(reps[layer], dtype=np.float32)
        G = kneighbors_graph(X, n_neighbors=knn_k, mode="connectivity", include_self=False)
        coo = G.tocoo()
        tree = gd.SimplexTree()
        for v in range(X.shape[0]):
            tree.insert([int(v)])
        for u, v in zip(coo.row, coo.col):
            tree.insert([int(u), int(v)])
        tree.expansion(max_simplex_dim)

        ids: list[int] = []
        for simplex, _ in tree.get_skeleton(max_simplex_dim):
            key = tuple(int(x) for x in simplex)
            sid = simplex_id.get(key)
            if sid is None:
                sid = len(simplices)
                simplex_id[key] = sid
                simplices.append(list(key))
            ids.append(sid)
        layer_simplex_ids.append(ids)

    layers: list[list[int]] = []
    for i in range(2 * len(layer_simplex_ids) - 1):
        if i % 2 == 0:
            layers.append(layer_simplex_ids[i // 2])
        else:
            a = set(layer_simplex_ids[i // 2])
            b = set(layer_simplex_ids[(i + 1) // 2])
            layers.append(list(a.intersection(b)))

    filtration = d.Filtration(simplices)
    appearance = np.zeros((len(layers), len(simplices)), dtype=np.uint8)
    for i, ids in enumerate(layers):
        appearance[i, ids] = 1
    times: list[list[int]] = []
    for sid in range(len(simplices)):
        ranges = _contiguous_ranges(np.where(appearance[:, sid] == 1)[0])
        flat: list[int] = []
        for start, stop in ranges:
            flat.extend([start, stop])
        times.append(flat)
    return filtration, times, len(simplices)


def compute_zigzag(reps: np.ndarray, knn_k: int, max_simplex_dim: int = 4) -> dict[int, np.ndarray]:
    import dionysus as d

    filtration, times, n_simplices = build_zigzag_inputs(reps, knn_k, max_simplex_dim)
    _, diagrams, _ = d.zigzag_homology_persistence(filtration, times, progress=False)
    out: dict[int, np.ndarray] = {}
    for dim, diagram in enumerate(diagrams):
        pairs = []
        for p in diagram:
            b = float(p.birth)
            death = float(p.death)
            pairs.append([b, death])
        out[dim] = np.asarray(pairs, dtype=float).reshape((-1, 2)) if pairs else np.empty((0, 2), dtype=float)
    out["_n_simplices"] = np.asarray([n_simplices], dtype=np.int64)  # type: ignore[index]
    return out


def effective_intervals(raw: np.ndarray) -> np.ndarray:
    """Paper demo mapping from zigzag indices to real model-layer indices: floor(index/2)."""
    if raw.size == 0:
        return np.empty((0, 2), dtype=int)
    if not np.isfinite(raw).all():
        raise ValueError("non-finite zigzag interval encountered")
    return (raw.astype(np.int64) // 2).astype(int)


def save_intervals_csv(intervals_by_dim: dict, out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"dimensions": {}}
    for dim, raw in intervals_by_dim.items():
        if not isinstance(dim, int):
            continue
        eff = effective_intervals(raw)
        path = out_dir / f"intervals_H{dim}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["raw_birth", "raw_death", "birth_layer", "death_layer", "lifetime_layers"])
            for (rb, rd), (b, d) in zip(raw, eff):
                w.writerow([rb, rd, int(b), int(d), int(d - b)])
        summary["dimensions"][str(dim)] = {
            "feature_count": int(len(eff)),
            "positive_lifetime_count": int(np.sum((eff[:, 1] - eff[:, 0]) > 0)) if len(eff) else 0,
            "mean_lifetime": float(np.mean(eff[:, 1] - eff[:, 0])) if len(eff) else 0.0,
        }
    summary["n_simplices"] = int(intervals_by_dim["_n_simplices"][0])
    return summary


def effective_persistence_matrix(eff: np.ndarray, n_layers: int) -> np.ndarray:
    """Count H_p features by effective death (row) and birth (column), like the paper."""
    mat = np.zeros((n_layers + 1, n_layers + 1), dtype=float)
    for b, d in eff:
        b = int(np.clip(b, 0, n_layers))
        d = int(np.clip(d, 0, n_layers))
        if d != b:
            mat[d, b] += 1.0
    return mat


def betti_curve(eff: np.ndarray, n_layers: int) -> np.ndarray:
    betti = np.zeros(n_layers, dtype=float)
    for b, d in eff:
        lo = max(0, int(b))
        hi = min(n_layers, int(d))
        if hi > lo:
            betti[lo:hi] += 1.0
    return betti


def interlayer_persistence_alpha0(eff: np.ndarray, n_layers: int) -> np.ndarray:
    """Paper's PHSim normalized by Betti, then alpha=0 weighted mean across layers."""
    betti = betti_curve(eff, n_layers)
    sim = np.zeros((n_layers, n_layers), dtype=float)
    for i in range(n_layers):
        alive_i = (eff[:, 0] <= i) & (eff[:, 1] > i) if len(eff) else np.zeros(0, dtype=bool)
        denom = float(np.sum(alive_i))
        if denom == 0:
            continue
        for j in range(n_layers):
            lo, hi = sorted((i, j))
            persistent = alive_i & (eff[:, 0] <= lo) & (eff[:, 1] > hi)
            sim[i, j] = float(np.sum(persistent)) / denom
    # alpha=0 => equal weights, including self-distance 0**0 == 1 as in numpy/Python.
    return sim.mean(axis=1)
