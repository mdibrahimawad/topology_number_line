"""
Resume the numerical ZigZag experiment from already-saved hidden states.

IMPORTANT
---------
- NO language models are run.
- NO GPU extraction is run.
- Existing completed topology results are skipped.
- Only missing (run, k) jobs are computed.
- Heavy ZigZag computation runs in a child subprocess so Modal's
  parent process can continue sending heartbeats.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal


# =====================================================================
# Configuration
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

REMOTE_PROJECT = "/root/numberline_zigzag_modal"

RESULTS_PATH = "/artifacts"

RESULTS_VOLUME = "numberline-zigzag-results"

APP_NAME = "numberline-zigzag-topology-final-resume"


# =====================================================================
# CPU-only Modal image
#
# We do NOT need Torch, Transformers, Hugging Face, or CUDA here.
# Hidden-state extraction is already finished.
# =====================================================================

image = (
    modal.Image.debian_slim(
        python_version="3.11"
    )
    .apt_install(
        "git",
        "build-essential",
    )
    .pip_install(
        "numpy>=1.26,<3",
        "scipy>=1.11,<2",
        "scikit-learn>=1.4,<2",
        "matplotlib>=3.8,<4",
        "pandas>=2.1,<3",
        "gudhi==3.9.0",
        "dionysus==2.0.10",
    )
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "MPLBACKEND": "Agg",
        }
    )
    .add_local_dir(
        PROJECT_ROOT,
        remote_path=REMOTE_PROJECT,
        ignore=[
            ".git",
            ".venv",
            "venv",
            "__pycache__",
            "*.pyc",
            "results",
            "*.zip",
        ],
    )
)


app = modal.App(
    APP_NAME,
    image=image,
)


results_volume = modal.Volume.from_name(
    RESULTS_VOLUME,
    create_if_missing=True,
)


# =====================================================================
# Stage 1
#
# One independent topology job for one (run, k).
#
# Existing completed jobs are SKIPPED.
#
# Heavy topology happens inside topology_worker.py as a subprocess.
# =====================================================================

@app.function(
    volumes={
        RESULTS_PATH: results_volume,
    },
    timeout=2 * 60 * 60,
    cpu=4,
    memory=16384,
    max_containers=25,
)
def topology_single_k(
    run_dir: str,
    knn_k: int,
    max_simplex_dim: int,
) -> dict:

    import subprocess
    import sys

    results_volume.reload()

    run_path = Path(run_dir)

    hidden_path = (
        run_path
        / "hidden_states.npz"
    )

    summary_path = (
        run_path
        / "topology"
        / f"k_{knn_k:02d}"
        / "summary.json"
    )

    # -------------------------------------------------------------
    # Already successfully calculated?
    # Do not waste CPU calculating it again.
    # -------------------------------------------------------------

    if summary_path.exists():

        print(
            f"[SKIP] {run_dir} "
            f"k={knn_k} already completed",
            flush=True,
        )

        return {
            "status": "skipped",
            "run_dir": run_dir,
            "knn_k": int(knn_k),
        }

    if not hidden_path.exists():

        raise FileNotFoundError(
            f"Missing saved hidden states: "
            f"{hidden_path}"
        )

    print(
        f"[SUBPROCESS] {run_dir} "
        f"k={knn_k} START",
        flush=True,
    )

    # -------------------------------------------------------------
    # Heavy Dionysus / ZigZag calculation happens in CHILD process.
    #
    # The Modal parent Python process remains free to send
    # heartbeats.
    # -------------------------------------------------------------

    subprocess.run(
        [
            sys.executable,
            f"{REMOTE_PROJECT}/topology_worker.py",
            run_dir,
            str(knn_k),
            str(max_simplex_dim),
        ],
        cwd=REMOTE_PROJECT,
        check=True,
    )

    # Worker should now have produced summary.json.

    if not summary_path.exists():

        raise RuntimeError(
            "Topology subprocess finished "
            "but summary.json was not created: "
            f"{summary_path}"
        )

    # Persist child-process output to Modal Volume.

    results_volume.commit()

    print(
        f"[SUBPROCESS] {run_dir} "
        f"k={knn_k} DONE",
        flush=True,
    )

    return {
        "status": "success",
        "run_dir": run_dir,
        "knn_k": int(knn_k),
    }


# =====================================================================
# Stage 2
#
# Once every k exists for a run, recreate its scan summary.
# =====================================================================

@app.function(
    volumes={
        RESULTS_PATH: results_volume,
    },
    timeout=60 * 60,
    cpu=4,
    memory=16384,
    max_containers=12,
)
def finalize_run_scan(
    run_dir: str,
    knn_values: list[int],
) -> dict:

    import sys

    import numpy as np

    sys.path.insert(
        0,
        REMOTE_PROJECT,
    )

    from numzig.geometry import (
        compute_layer_pca_metrics,
        write_metrics_csv,
    )

    results_volume.reload()

    run_path = Path(run_dir)

    hidden_path = (
        run_path
        / "hidden_states.npz"
    )

    if not hidden_path.exists():

        raise FileNotFoundError(
            hidden_path
        )

    with np.load(hidden_path) as data:

        hidden = np.asarray(
            data["hidden_states"]
        )

        targets = np.asarray(
            data["targets"]
        )

        group_ids = np.asarray(
            data["group_ids"]
        )

        point_ids = np.asarray(
            data["point_ids"]
        )

    # -------------------------------------------------------------
    # Basic integrity checks
    # -------------------------------------------------------------

    if (
        len(set(point_ids.tolist()))
        != len(point_ids)
    ):

        raise ValueError(
            f"Duplicate point IDs in {run_dir}"
        )

    if hidden.shape[1] != len(targets):

        raise ValueError(
            f"Hidden state / target mismatch "
            f"in {run_dir}"
        )

    if len(targets) != len(group_ids):

        raise ValueError(
            f"Target / group mismatch "
            f"in {run_dir}"
        )

    # -------------------------------------------------------------
    # Recreate PCA layer metrics
    # -------------------------------------------------------------

    pca_rows = compute_layer_pca_metrics(
        hidden,
        targets,
        group_ids,
    )

    write_metrics_csv(
        pca_rows,
        run_path
        / "pca_layer_metrics.csv",
    )

    # -------------------------------------------------------------
    # Rebuild topology_scan_summary.json using the independently
    # calculated k results.
    # -------------------------------------------------------------

    scan = {
        "shape": list(
            hidden.shape
        ),
        "knn": {},
    }

    for k in knn_values:

        summary_path = (
            run_path
            / "topology"
            / f"k_{k:02d}"
            / "summary.json"
        )

        if not summary_path.exists():

            raise FileNotFoundError(
                "Missing topology result "
                f"for {run_dir}, k={k}: "
                f"{summary_path}"
            )

        summary = json.loads(
            summary_path.read_text(
                encoding="utf-8"
            )
        )

        scan["knn"][str(k)] = summary

    (
        run_path
        / "topology_scan_summary.json"
    ).write_text(
        json.dumps(
            scan,
            indent=2,
        ),
        encoding="utf-8",
    )

    results_volume.commit()

    print(
        f"[FINALIZE] {run_dir} DONE",
        flush=True,
    )

    return {
        "status": "success",
        "run_dir": run_dir,
        "shape": list(
            hidden.shape
        ),
    }


# =====================================================================
# Stage 3
#
# Select the global k across all 9 full point clouds.
# =====================================================================

@app.function(
    volumes={
        RESULTS_PATH: results_volume,
    },
    timeout=60 * 60,
    cpu=2,
    memory=8192,
)
def select_global_k(
    run_dirs: list[str],
    experiment_name: str,
) -> dict:

    import sys

    sys.path.insert(
        0,
        REMOTE_PROJECT,
    )

    from numzig.pipeline import (
        choose_global_k,
    )

    results_volume.reload()

    payload = choose_global_k(
        run_dirs
    )

    root = (
        Path(RESULTS_PATH)
        / experiment_name
    )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        root
        / "global_knn_selection.json"
    ).write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    results_volume.commit()

    return payload


# =====================================================================
# Stage 4
#
# Generate final selected-k plots for every run.
# =====================================================================

@app.function(
    volumes={
        RESULTS_PATH: results_volume,
    },
    timeout=2 * 60 * 60,
    cpu=4,
    memory=16384,
    max_containers=12,
)
def render_run(
    run_dir: str,
    knn_k: int,
) -> dict:

    import sys

    sys.path.insert(
        0,
        REMOTE_PROJECT,
    )

    from numzig.pipeline import (
        render_selected,
    )

    results_volume.reload()

    meta = render_selected(
        run_dir,
        knn_k=knn_k,
    )

    results_volume.commit()

    print(
        f"[FIGURES] {run_dir} DONE",
        flush=True,
    )

    return {
        "status": "success",
        "run_dir": run_dir,
        **meta,
    }


# =====================================================================
# Stage 5
#
# Aggregate 3 runs / seeds for each model.
# =====================================================================

@app.function(
    volumes={
        RESULTS_PATH: results_volume,
    },
    timeout=60 * 60,
    cpu=4,
    memory=16384,
)
def aggregate_final(
    run_dirs: list[str],
    knn_k: int,
    experiment_name: str,
) -> dict:

    import sys

    sys.path.insert(
        0,
        REMOTE_PROJECT,
    )

    from numzig.pipeline import (
        aggregate_experiment,
    )

    results_volume.reload()

    root = (
        Path(RESULTS_PATH)
        / experiment_name
    )

    payload = aggregate_experiment(
        run_dirs,
        knn_k=knn_k,
        output_root=root,
    )

    results_volume.commit()

    return payload


# =====================================================================
# Main
# =====================================================================

@app.local_entrypoint()
def main(
    experiment_name: str = "main",
    knn_min: int = 1,
    knn_max: int = 15,
    max_simplex_dim: int = 4,
) -> None:

    # -------------------------------------------------------------
    # These are the three models we successfully extracted.
    # -------------------------------------------------------------

    models = [
        "falcon-rw-1b",
        "falcon-rw-7b",
        "llm360-crystal",
    ]

    runs = 3

    knn_values = list(
        range(
            knn_min,
            knn_max + 1,
        )
    )

    run_dirs = [

        (
            f"{RESULTS_PATH}/"
            f"{experiment_name}/"
            f"{model}/"
            f"run_{run_idx:02d}"
        )

        for model in models

        for run_idx in range(runs)
    ]

    print()
    print("=" * 80)

    print(
        "FINAL TOPOLOGY RESUME"
    )

    print("=" * 80)

    print(
        f"Experiment: "
        f"{experiment_name}"
    )

    print(
        f"Models: "
        f"{models}"
    )

    print(
        f"Saved point clouds: "
        f"{len(run_dirs)}"
    )

    print(
        f"k values: "
        f"{knn_values}"
    )

    print(
        f"Possible topology jobs: "
        f"{len(run_dirs) * len(knn_values)}"
    )

    print()

    print(
        "GPU extraction jobs: 0"
    )

    print(
        "LANGUAGE MODEL FORWARD PASSES: 0"
    )

    print(
        "EXISTING TOPOLOGY RESULTS: SKIPPED"
    )

    print(
        "MISSING TOPOLOGY RESULTS: SUBPROCESS"
    )

    print("=" * 80)

    # =============================================================
    # Stage 1
    #
    # Submit all possible jobs.
    #
    # Existing summary.json files return immediately as SKIPPED.
    # Only missing jobs perform computation.
    # =============================================================

    topology_jobs = []

    for run_dir in run_dirs:

        for k in knn_values:

            future = (
                topology_single_k.spawn(
                    run_dir,
                    k,
                    max_simplex_dim,
                )
            )

            topology_jobs.append(
                (
                    run_dir,
                    k,
                    future,
                )
            )

    print(
        f"[LOCAL] Submitted "
        f"{len(topology_jobs)} checks/jobs."
    )

    success_count = 0
    skipped_count = 0

    for (
        run_dir,
        k,
        future,
    ) in topology_jobs:

        result = future.get()

        status = result.get(
            "status"
        )

        if status == "success":

            success_count += 1

        elif status == "skipped":

            skipped_count += 1

        else:

            raise RuntimeError(
                "Topology job failed: "
                f"{run_dir}, "
                f"k={k}, "
                f"result={result}"
            )

        print(
            f"[LOCAL] "
            f"{status.upper()}: "
            f"{run_dir}, "
            f"k={k}"
        )

    print()
    print(
        f"[LOCAL] Previously completed / skipped: "
        f"{skipped_count}"
    )

    print(
        f"[LOCAL] Newly computed: "
        f"{success_count}"
    )

    # =============================================================
    # Stage 2
    #
    # Recreate complete scan summary for every run.
    # =============================================================

    print()
    print(
        "[LOCAL] Finalizing "
        "all 9 run summaries..."
    )

    finalize_jobs = []

    for run_dir in run_dirs:

        future = (
            finalize_run_scan.spawn(
                run_dir,
                knn_values,
            )
        )

        finalize_jobs.append(
            (
                run_dir,
                future,
            )
        )

    for (
        run_dir,
        future,
    ) in finalize_jobs:

        result = future.get()

        if (
            result.get("status")
            != "success"
        ):

            raise RuntimeError(
                f"Finalization failed: "
                f"{run_dir}"
            )

        print(
            f"[LOCAL] Scan finalized: "
            f"{run_dir}"
        )

    # =============================================================
    # Stage 3
    #
    # Choose ONE global k.
    # =============================================================

    print()
    print(
        "[LOCAL] Selecting global k..."
    )

    selection = select_global_k.remote(
        run_dirs,
        experiment_name,
    )

    selected_k = int(
        selection[
            "selected_knn_k"
        ]
    )

    print()
    print("=" * 80)

    print(
        f"GLOBAL SELECTED k = "
        f"{selected_k}"
    )

    print("=" * 80)

    # =============================================================
    # Stage 4
    #
    # Generate final plots using selected global k.
    # =============================================================

    print()
    print(
        "[LOCAL] Generating "
        "all final figures..."
    )

    render_jobs = []

    for run_dir in run_dirs:

        future = render_run.spawn(
            run_dir,
            selected_k,
        )

        render_jobs.append(
            (
                run_dir,
                future,
            )
        )

    for (
        run_dir,
        future,
    ) in render_jobs:

        result = future.get()

        if (
            result.get("status")
            != "success"
        ):

            raise RuntimeError(
                f"Rendering failed: "
                f"{run_dir}"
            )

        print(
            f"[LOCAL] Figures complete: "
            f"{run_dir}"
        )

    # =============================================================
    # Stage 5
    #
    # Aggregate runs/seeds.
    # =============================================================

    print()
    print(
        "[LOCAL] Aggregating "
        "3 runs per model..."
    )

    aggregate_final.remote(
        run_dirs,
        selected_k,
        experiment_name,
    )

    # =============================================================
    # Done
    # =============================================================

    print()
    print("=" * 80)

    print(
        "EXPERIMENT COMPLETE"
    )

    print("=" * 80)

    print(
        "GPU rerun: NO"
    )

    print(
        "Hidden-state extraction rerun: NO"
    )

    print(
        f"Final global k: "
        f"{selected_k}"
    )

    print(
        f"Results volume: "
        f"{RESULTS_VOLUME}"
    )

    print(
        f"Experiment folder: "
        f"/{experiment_name}"
    )

    print()
    print(
        "Download final results with:"
    )

    print()

    print(
        f"modal volume get "
        f"{RESULTS_VOLUME} "
        f"/{experiment_name} "
        f"results/main_final_download"
    )