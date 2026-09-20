from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from .geometry import best_pca_layer
from .zigzag import effective_intervals, effective_persistence_matrix, interlayer_persistence_alpha0


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_layer_metrics(rows: list[dict], out: Path):
    layers = [r["layer"] for r in rows]
    best = best_pca_layer(rows)
    for key, label in [("ev_pc1", "PCA PC1 explained variance"), ("rho_abs", "|Spearman rho|"), ("beta", "Compression beta")]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(layers, [r[key] for r in rows], marker="o", markersize=3)
        ax.axvline(best, linestyle="--", linewidth=1)
        ax.set_xlabel("Layer (0 = embedding output)")
        ax.set_ylabel(label)
        ax.set_title(f"{label} across layers")
        ax.grid(alpha=0.25)
        _save(fig, out / f"layer_{key}.png")


def plot_pca_1d_best(hidden, targets, group_ids, rows, out: Path):
    layer = best_pca_layer(rows)
    pc1 = PCA(n_components=1).fit_transform(hidden[layer])[:, 0]
    fig, ax = plt.subplots(figsize=(8, 5))
    for g in sorted(np.unique(group_ids)):
        m = group_ids == g
        ax.scatter(targets[m], pc1[m], label=f"G{int(g)}", alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Numerical target (log scale)")
    ax.set_ylabel("PC1 coordinate")
    ax.set_title(f"Numerical number line at PCA-selected layer {layer}")
    ax.legend()
    ax.grid(alpha=0.25)
    _save(fig, out / "pca_1d_numberline_best_layer.png")


def _knn_edges_highdim(X, emb, k):
    # Match ZigZagLLMs demo exactly: with X=None sklearn excludes the query point itself.
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(X)
    inds = nn.kneighbors(return_distance=False)
    edges = set()
    segs = []
    for i in range(len(X)):
        for j in inds[i]:
            if i == j:
                continue
            key = tuple(sorted((int(i), int(j))))
            if key in edges:
                continue
            edges.add(key)
            segs.append((emb[i], emb[j]))
    return segs


def plot_pca_knn_all_layers(hidden, group_ids, knn_k: int, out: Path, ncols: int = 6):
    """Show every transformer layer in one PCA-2D grid.

    PCA is display-only. For each panel, the kNN edges are computed from the
    original full-dimensional hidden vectors for that same layer.
    """
    n_layers = int(hidden.shape[0])
    ncols = max(1, min(int(ncols), n_layers))
    nrows = int(np.ceil(n_layers / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.2 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for layer in range(n_layers):
        ax = axes_flat[layer]
        X = hidden[layer]
        emb = PCA(n_components=2).fit_transform(X)
        segs = _knn_edges_highdim(X, emb, knn_k)
        ax.add_collection(LineCollection(segs, linewidths=0.45, alpha=0.28))
        for g in sorted(np.unique(group_ids)):
            m = group_ids == g
            ax.scatter(emb[m, 0], emb[m, 1], s=10, alpha=0.85, label=f"G{int(g)}")
        ax.set_title(f"Layer {layer}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes_flat[n_layers:]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=8, frameon=False)
    fig.suptitle(
        f"All-layer numerical geometry: PCA-2D for display; kNN edges from full hidden space (k={knn_k})",
        y=0.995,
        fontsize=13,
    )
    fig.subplots_adjust(top=0.95, hspace=0.22, wspace=0.10)
    fig.savefig(out / "pca2d_knn_all_layers.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_barcode(eff: np.ndarray, n_layers: int, out: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    if len(eff):
        order = np.argsort(eff[:, 0] + 1e-3 * eff[:, 1])
        for y, idx in enumerate(order):
            b, d = eff[idx]
            ax.hlines(y, b, d, linewidth=1)
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("H1 feature")
    ax.set_title("H1 ZigZag barcode")
    ax.grid(axis="x", alpha=0.2)
    _save(fig, out / "barcode_H1.png")


def plot_birth_death(eff: np.ndarray, n_layers: int, out: Path):
    fig, ax = plt.subplots(figsize=(6, 6))
    if len(eff):
        ax.scatter(eff[:, 0], eff[:, 1], s=18, alpha=0.7)
    ax.plot([0, n_layers - 1], [0, n_layers - 1], linestyle="--", linewidth=1)
    ax.set_xlabel("Birth layer")
    ax.set_ylabel("Death layer")
    ax.set_title("H1 birth-death diagram")
    ax.grid(alpha=0.25)
    _save(fig, out / "birth_death_H1.png")


def plot_effective_persistence(eff: np.ndarray, n_layers: int, out: Path):
    # Paper-style display: x=birth, y=persistence=death-birth, color=count.
    maxlife = max(1, n_layers)
    hist = np.zeros((maxlife + 1, n_layers + 1), dtype=float)
    for b, d in eff:
        life = int(d - b)
        if life > 0 and 0 <= b <= n_layers and life <= maxlife:
            hist[life, int(b)] += 1
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(hist, origin="lower", aspect="auto", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="Number of H1 features")
    ax.set_xlabel("Birth layer")
    ax.set_ylabel("Persistence (death - birth)")
    ax.set_title("Effective H1 persistence image")
    _save(fig, out / "effective_persistence_H1.png")


def plot_interlayer(eff: np.ndarray, n_layers: int, out: Path):
    score = interlayer_persistence_alpha0(eff, n_layers)
    np.savetxt(out / "interlayer_persistence_H1_alpha0.csv", np.c_[np.arange(n_layers), score], delimiter=",", header="layer,score", comments="")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.arange(n_layers), score, marker="o", markersize=3)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Weighted inter-layer persistence (alpha=0)")
    ax.set_title("H1 inter-layer persistence")
    ax.grid(alpha=0.25)
    _save(fig, out / "interlayer_persistence_H1_alpha0.png")


def plot_combined(hidden, group_ids, eff, knn_k, selected_layers, n_layers, out: Path):
    n = len(selected_layers)
    fig = plt.figure(figsize=(5 * n, 8))
    for pos, layer in enumerate(selected_layers, start=1):
        ax = fig.add_subplot(2, n, pos)
        X = hidden[layer]
        emb = PCA(n_components=2).fit_transform(X)
        ax.add_collection(LineCollection(_knn_edges_highdim(X, emb, knn_k), linewidths=0.6, alpha=0.35))
        for g in sorted(np.unique(group_ids)):
            m = group_ids == g
            ax.scatter(emb[m, 0], emb[m, 1], s=18, label=f"G{int(g)}")
        ax.set_title(f"Layer {layer}")
        ax.set_xticks([]); ax.set_yticks([])
    axb = fig.add_subplot(2, 1, 2)
    if len(eff):
        order = np.argsort(eff[:, 0] + 1e-3 * eff[:, 1])
        for y, idx in enumerate(order):
            b, d = eff[idx]
            axb.hlines(y, b, d, linewidth=0.9)
    for layer in selected_layers:
        axb.axvline(layer, linestyle="--", linewidth=0.8)
    axb.set_xlim(-0.5, n_layers - 0.5)
    axb.set_xlabel("Transformer layer")
    axb.set_ylabel("H1 feature")
    axb.set_title("H1 ZigZag barcode")
    fig.suptitle(f"Full-space kNN topology with 2D PCA visualization (k={knn_k})")
    _save(fig, out / "combined_pca_knn_barcode_H1.png")


def render_all(hidden, targets, group_ids, pca_rows, raw_h1, knn_k: int, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    n_layers = hidden.shape[0]
    best = best_pca_layer(pca_rows)
    quartiles = [0, max(0, (n_layers - 1) // 3), max(0, 2 * (n_layers - 1) // 3), n_layers - 1]
    selected = []
    for x in quartiles + [best]:
        if x not in selected:
            selected.append(int(x))
    selected = selected[:5]
    eff = effective_intervals(raw_h1)
    plot_layer_metrics(pca_rows, out)
    # Main geometry visualization requested for this experiment: every layer, one figure.
    plot_pca_knn_all_layers(hidden, group_ids, knn_k, out)
    plot_barcode(eff, n_layers, out)
    plot_birth_death(eff, n_layers, out)
    plot_effective_persistence(eff, n_layers, out)
    plot_interlayer(eff, n_layers, out)
    # Keep a compact paper-style selected-layer + barcode figure as a supplementary output.
    plot_combined(hidden, group_ids, eff, knn_k, selected, n_layers, out)
    return {"all_layers_visualized": list(range(n_layers)), "selected_supplementary_layers": selected, "pca_best_layer_metric_only": best}
