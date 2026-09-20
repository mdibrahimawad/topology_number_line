from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .geometry import compute_layer_pca_metrics, write_metrics_csv
from .plots import render_all
from .zigzag import compute_zigzag, save_intervals_csv


def load_hidden(run_dir: str | Path):
    data = np.load(Path(run_dir) / "hidden_states.npz")
    return data["hidden_states"], data["targets"], data["group_ids"], data["point_ids"]


def scan_topology(run_dir: str | Path, *, knn_values: list[int], max_simplex_dim: int = 4) -> dict:
    run_dir = Path(run_dir)
    hidden, targets, group_ids, point_ids = load_hidden(run_dir)
    if len(set(point_ids.tolist())) != len(point_ids):
        raise ValueError("point IDs must be unique")
    if hidden.shape[1] != len(targets) or len(targets) != len(group_ids):
        raise ValueError("point metadata length mismatch")

    pca_rows = compute_layer_pca_metrics(hidden, targets, group_ids)
    write_metrics_csv(pca_rows, run_dir / "pca_layer_metrics.csv")

    scan = {"shape": list(hidden.shape), "knn": {}}
    topo_root = run_dir / "topology"
    for k in knn_values:
        print(f"[topology] k={k}", flush=True)
        diagrams = compute_zigzag(hidden, knn_k=k, max_simplex_dim=max_simplex_dim)
        summary = save_intervals_csv(diagrams, topo_root / f"k_{k:02d}")
        summary["knn_k"] = k
        scan["knn"][str(k)] = summary
    (run_dir / "topology_scan_summary.json").write_text(json.dumps(scan, indent=2), encoding="utf-8")
    return scan


def render_selected(run_dir: str | Path, *, knn_k: int) -> dict:
    run_dir = Path(run_dir)
    hidden, targets, group_ids, _ = load_hidden(run_dir)
    import pandas as pd
    rows = pd.read_csv(run_dir / "pca_layer_metrics.csv").to_dict(orient="records")
    h1_path = run_dir / "topology" / f"k_{knn_k:02d}" / "intervals_H1.csv"
    if not h1_path.exists():
        raise FileNotFoundError(h1_path)
    h1 = pd.read_csv(h1_path)
    raw = h1[["raw_birth", "raw_death"]].to_numpy(dtype=float) if len(h1) else np.empty((0, 2), dtype=float)
    fig_dir = run_dir / "figures"
    meta = render_all(hidden, targets, group_ids, rows, raw, knn_k, fig_dir)
    meta["knn_k"] = knn_k
    (run_dir / "selected_analysis.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def choose_global_k(run_dirs: list[str | Path]) -> dict:
    totals: dict[int, int] = {}
    used = []
    for d in run_dirs:
        path = Path(d) / "topology_scan_summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        used.append(str(d))
        for k, info in payload["knn"].items():
            count = int(info["dimensions"].get("1", {}).get("feature_count", 0))
            totals[int(k)] = totals.get(int(k), 0) + count
    if not totals:
        raise ValueError("no topology summaries found")
    selected = min((k for k, v in totals.items() if v == max(totals.values())))
    return {
        "selection_rule": "maximize total H1 feature count across all successful model/run point clouds; ties choose smaller k",
        "selected_knn_k": int(selected),
        "total_H1_features_by_k": {str(k): int(v) for k, v in sorted(totals.items())},
        "run_dirs": used,
    }


def aggregate_experiment(run_dirs: list[str | Path], *, knn_k: int, output_root: str | Path) -> dict:
    """Aggregate prompt-seed replicates per model, mirroring the existing multi-run reporting."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    output_root = Path(output_root)
    summary_root = output_root / "model_summaries"
    summary_root.mkdir(parents=True, exist_ok=True)
    by_model: dict[str, list[Path]] = {}
    for d in run_dirs:
        p = Path(d)
        alias = p.parent.name
        by_model.setdefault(alias, []).append(p)

    experiment = {"selected_knn_k": int(knn_k), "models": {}}
    for alias, dirs in sorted(by_model.items()):
        model_out = summary_root / alias
        model_out.mkdir(parents=True, exist_ok=True)
        pca_frames = []
        ilp_frames = []
        h1_counts = []
        h1_lifetimes = []
        shapes = []
        for d in sorted(dirs):
            diag = json.loads((d / "extraction_diagnostics.json").read_text(encoding="utf-8"))
            shapes.append(diag["shape"])
            pca = pd.read_csv(d / "pca_layer_metrics.csv")
            pca["run"] = d.name
            pca_frames.append(pca)
            ilp = pd.read_csv(d / "figures" / "interlayer_persistence_H1_alpha0.csv")
            ilp["run"] = d.name
            ilp_frames.append(ilp)
            h1 = pd.read_csv(d / "topology" / f"k_{knn_k:02d}" / "intervals_H1.csv")
            h1_counts.append(int(len(h1)))
            if len(h1):
                h1_lifetimes.extend(h1["lifetime_layers"].astype(float).tolist())

        # Within one model, architecture/layer count must match across prompt seeds.
        if len({tuple(s) for s in shapes}) != 1:
            raise ValueError(f"inconsistent hidden tensor shapes across runs for {alias}: {shapes}")

        pca_all = pd.concat(pca_frames, ignore_index=True)
        pca_agg = pca_all.groupby("layer")[["ev_pc1", "rho_abs", "beta", "beta_scale", "beta_r2"]].agg(["mean", "std"])
        pca_agg.columns = [f"{a}_{b}" for a, b in pca_agg.columns]
        pca_agg.reset_index().to_csv(model_out / "pca_layer_metrics_mean_std.csv", index=False)

        ilp_all = pd.concat(ilp_frames, ignore_index=True)
        ilp_agg = ilp_all.groupby("layer")["score"].agg(["mean", "std"]).reset_index()
        ilp_agg.to_csv(model_out / "interlayer_persistence_H1_mean_std.csv", index=False)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(ilp_agg["layer"], ilp_agg["mean"])
        std = ilp_agg["std"].fillna(0.0)
        ax.fill_between(ilp_agg["layer"], ilp_agg["mean"] - std, ilp_agg["mean"] + std, alpha=0.2)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Inter-layer persistence H1 (alpha=0)")
        ax.set_title(f"{alias}: mean +/- SD across prompt seeds")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(model_out / "interlayer_persistence_H1_mean_std.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        model_summary = {
            "runs": len(dirs),
            "hidden_shape_per_run": shapes[0],
            "H1_feature_count_mean": float(np.mean(h1_counts)),
            "H1_feature_count_std": float(np.std(h1_counts)),
            "H1_lifetime_mean": float(np.mean(h1_lifetimes)) if h1_lifetimes else 0.0,
            "H1_lifetime_std": float(np.std(h1_lifetimes)) if h1_lifetimes else 0.0,
        }
        (model_out / "topology_H1_summary.json").write_text(json.dumps(model_summary, indent=2), encoding="utf-8")
        experiment["models"][alias] = model_summary

    (output_root / "experiment_summary.json").write_text(json.dumps(experiment, indent=2), encoding="utf-8")
    return experiment
