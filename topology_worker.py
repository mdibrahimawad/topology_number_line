from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from numzig.zigzag import compute_zigzag, save_intervals_csv


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(
            "Usage: topology_worker.py RUN_DIR K MAX_SIMPLEX_DIM"
        )

    run_dir = Path(sys.argv[1])
    knn_k = int(sys.argv[2])
    max_simplex_dim = int(sys.argv[3])

    hidden_path = run_dir / "hidden_states.npz"

    if not hidden_path.exists():
        raise FileNotFoundError(
            f"Missing hidden states: {hidden_path}"
        )

    with np.load(hidden_path) as data:
        hidden = np.asarray(
            data["hidden_states"],
            dtype=np.float32,
        )

    print(
        f"[worker] {run_dir} k={knn_k} computing...",
        flush=True,
    )

    diagrams = compute_zigzag(
        hidden,
        knn_k=knn_k,
        max_simplex_dim=max_simplex_dim,
    )

    out_dir = (
        run_dir
        / "topology"
        / f"k_{knn_k:02d}"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = save_intervals_csv(
        diagrams,
        out_dir,
    )

    summary["knn_k"] = knn_k

    (
        out_dir
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"[worker] {run_dir} k={knn_k} FINISHED",
        flush=True,
    )


if __name__ == "__main__":
    main()