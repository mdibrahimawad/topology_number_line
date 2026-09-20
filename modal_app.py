"""Modal orchestration for the four-model numerical ZigZag experiment."""
from __future__ import annotations

import json
import os
from pathlib import Path

import modal


PROJECT_ROOT = Path(__file__).resolve().parent
REMOTE_PROJECT = "/root/numberline_zigzag_modal"

CACHE_PATH = "/cache"
RESULTS_PATH = "/artifacts"

APP_NAME = "numberline-zigzag-four-models"

HF_SECRET_NAME = os.environ.get(
    "MODAL_HF_SECRET_NAME",
    "numberline-hf-token",
)

HF_CACHE_VOLUME = "numberline-zigzag-hf-cache"
RESULTS_VOLUME = "numberline-zigzag-results"


# ---------------------------------------------------------------------
# Modal image
# ---------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "build-essential",
    )
    .pip_install(
        "torch==2.6.0",
        "transformers>=4.57.0,<5",
        "accelerate>=1.2.0",
        "numpy>=1.26,<3",
        "scipy>=1.11,<2",
        "scikit-learn>=1.4,<2",
        "matplotlib>=3.8,<4",
        "pandas>=2.1,<3",
        "huggingface_hub>=0.27",
        "safetensors>=0.4.5",
        "sentencepiece>=0.2",
        "protobuf>=4.25",
        "tiktoken>=0.8",
        "hf_xet>=1.1",
        "gudhi==3.9.0",
        "dionysus==2.0.10",
    )
    .env(
        {
            "HF_HOME": f"{CACHE_PATH}/huggingface",
            "HF_HUB_CACHE": f"{CACHE_PATH}/huggingface/hub",
            "TRANSFORMERS_CACHE": f"{CACHE_PATH}/huggingface/transformers",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TOKENIZERS_PARALLELISM": "false",
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
            "__pycache__",
            "*.pyc",
            "results",
            "*.zip",
        ],
    )
)


# ---------------------------------------------------------------------
# Modal app / volumes / secrets
# ---------------------------------------------------------------------

app = modal.App(
    APP_NAME,
    image=image,
)

hf_cache = modal.Volume.from_name(
    HF_CACHE_VOLUME,
    create_if_missing=True,
)

results_volume = modal.Volume.from_name(
    RESULTS_VOLUME,
    create_if_missing=True,
)

hf_secret = modal.Secret.from_name(
    HF_SECRET_NAME,
)


# ---------------------------------------------------------------------
# Stage 1
# Hidden-state extraction on GPU
#
# Each extraction job gets one L40S.
# Modal can run up to 10 extraction containers at once.
# ---------------------------------------------------------------------

@app.function(
    gpu="L40S",
    secrets=[hf_secret],
    volumes={
        CACHE_PATH: hf_cache,
        RESULTS_PATH: results_volume,
    },
    timeout=4 * 60 * 60,
    memory=65536,
    max_containers=10,
    scaledown_window=60,
)
def extract_one(
    spec_dict: dict,
    run_idx: int,
    settings: dict,
) -> dict:

    import sys

    sys.path.insert(
        0,
        REMOTE_PROJECT,
    )

    from numzig.config import ModelSpec
    from numzig.extract import extract_hidden_states

    spec = ModelSpec.from_dict(
        spec_dict
    )

    run_seed = (
        int(settings["seed"])
        + int(run_idx)
    )

    out = (
        Path(RESULTS_PATH)
        / settings["experiment_name"]
        / spec.alias
        / f"run_{run_idx:02d}"
    )

    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )

    diagnostics = extract_hidden_states(
        spec,
        output_dir=out,
        seed=run_seed,
        samples_per_group=int(
            settings["samples_per_group"]
        ),
        num_examples=int(
            settings["num_examples"]
        ),
        context=settings["context"],
        upper_bound=int(
            settings["upper_bound"]
        ),
        prepend_bos=bool(
            settings["prepend_bos"]
        ),
        hf_token=token,
    )

    # Persist output files.
    results_volume.commit()

    # Persist any newly downloaded Hugging Face files.
    hf_cache.commit()

    return {
        "status": "success",
        "alias": spec.alias,
        "run_idx": run_idx,
        "run_dir": str(out),
        "shape": diagnostics["shape"],
    }


# ---------------------------------------------------------------------
# Stage 2
# Topology sweep on CPU
# ---------------------------------------------------------------------

@app.function(
    volumes={
        RESULTS_PATH: results_volume,
    },
    timeout=6 * 60 * 60,
    cpu=8,
    memory=32768,
    max_containers=16,
)
def topology_one(
    run_dir: str,
    knn_values: list[int],
    max_simplex_dim: int,
) -> dict:

    import sys

    sys.path.insert(
        0,
        REMOTE_PROJECT,
    )

    from numzig.pipeline import scan_topology

    results_volume.reload()

    scan = scan_topology(
        run_dir,
        knn_values=knn_values,
        max_simplex_dim=max_simplex_dim,
    )

    results_volume.commit()

    return {
        "status": "success",
        "run_dir": run_dir,
        "shape": scan["shape"],
    }


# ---------------------------------------------------------------------
# Stage 3
# Select one global k
# ---------------------------------------------------------------------

@app.function(
    volumes={
        RESULTS_PATH: results_volume,
    },
    timeout=60 * 60,
    cpu=2,
    memory=8192,
)
def select_k(
    run_dirs: list[str],
    experiment_name: str,
) -> dict:

    import sys

    sys.path.insert(
        0,
        REMOTE_PROJECT,
    )

    from numzig.pipeline import choose_global_k

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


# ---------------------------------------------------------------------
# Stage 4
# Visualization on CPU
# ---------------------------------------------------------------------

@app.function(
    volumes={
        RESULTS_PATH: results_volume,
    },
    timeout=2 * 60 * 60,
    cpu=4,
    memory=16384,
    max_containers=16,
)
def visualize_one(
    run_dir: str,
    knn_k: int,
) -> dict:

    import sys

    sys.path.insert(
        0,
        REMOTE_PROJECT,
    )

    from numzig.pipeline import render_selected

    results_volume.reload()

    meta = render_selected(
        run_dir,
        knn_k=knn_k,
    )

    results_volume.commit()

    return {
        "status": "success",
        "run_dir": run_dir,
        **meta,
    }


# ---------------------------------------------------------------------
# Stage 5
# Aggregate across seeds/models
# ---------------------------------------------------------------------

@app.function(
    volumes={
        RESULTS_PATH: results_volume,
    },
    timeout=60 * 60,
    cpu=4,
    memory=16384,
)
def aggregate_all(
    run_dirs: list[str],
    knn_k: int,
    experiment_name: str,
) -> dict:

    import sys

    sys.path.insert(
        0,
        REMOTE_PROJECT,
    )

    from numzig.pipeline import aggregate_experiment

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


# ---------------------------------------------------------------------
# Main local command
# ---------------------------------------------------------------------

@app.local_entrypoint()
def main(
    models: str = "all",
    experiment_name: str = "main",
    runs: int = 3,
    samples_per_group: int = 30,
    num_examples: int = 3,
    seed: int = 42,
    context: str = "random",
    upper_bound: int = 10000,
    prepend_bos: bool = False,
    knn_min: int = 1,
    knn_max: int = 15,
    max_simplex_dim: int = 4,
    smoke_test: bool = False,
    dry_run: bool = False,
) -> None:

    import sys

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

    from numzig.config import resolve_models

    selected = resolve_models(
        models
    )

    # -------------------------------------------------------------
    # Smoke-test configuration
    # -------------------------------------------------------------

    if smoke_test:
        runs = 1
        samples_per_group = 5

        knn_min = 2
        knn_max = 4

        max_simplex_dim = 3

        experiment_name = (
            f"{experiment_name}_smoke"
        )

    # -------------------------------------------------------------
    # Basic validation
    # -------------------------------------------------------------

    if runs < 1:
        raise ValueError(
            "runs must be >= 1"
        )

    if samples_per_group < 2:
        raise ValueError(
            "samples_per_group must be >= 2"
        )

    n_points = (
        4 * samples_per_group
    )

    if not (
        1
        <= knn_min
        <= knn_max
        < n_points
    ):
        raise ValueError(
            "Require "
            f"1 <= knn_min <= knn_max < {n_points}"
        )

    settings = {
        "experiment_name": experiment_name,
        "seed": seed,
        "samples_per_group": samples_per_group,
        "num_examples": num_examples,
        "context": context,
        "upper_bound": upper_bound,
        "prepend_bos": prepend_bos,
    }

    knn_values = list(
        range(
            knn_min,
            knn_max + 1,
        )
    )

    # -------------------------------------------------------------
    # Print experiment plan
    # -------------------------------------------------------------

    print()
    print("EXPERIMENT PLAN")
    print("=" * 80)

    print(
        "Models:",
        [m.alias for m in selected],
    )

    print(
        f"Runs/model: {runs}; "
        f"point cloud/run: {n_points} points"
    )

    print(
        f"kNN sweep: {knn_values}; "
        f"max simplex dimension: {max_simplex_dim}"
    )

    print(
        "GPU extraction jobs: "
        f"{len(selected) * runs}; "
        "Modal concurrency cap: 10"
    )

    print(
        "GPU type: L40S "
        "(one GPU per extraction container)"
    )

    print(
        "Topology and plotting run on CPU "
        "after hidden states are saved."
    )

    print("=" * 80)

    if dry_run:
        return

    # =============================================================
    # STAGE 1
    #
    # Spawn every model x seed extraction job.
    #
    # Full run:
    # 4 models x 3 seeds = 12 jobs.
    #
    # extract_one has max_containers=10, so Modal may execute
    # up to 10 of these at the same time.
    # =============================================================

    extract_jobs = []

    for spec in selected:

        for run_idx in range(runs):

            future = extract_one.spawn(
                spec.to_dict(),
                run_idx,
                settings,
            )

            extract_jobs.append(
                (
                    spec.alias,
                    run_idx,
                    future,
                )
            )

    run_dirs: list[str] = []

    # Wait for all extraction jobs.
    for (
        alias,
        run_idx,
        future,
    ) in extract_jobs:

        result = future.get()

        if (
            result.get("status")
            != "success"
        ):
            raise RuntimeError(
                "Extraction failed: "
                f"{alias} "
                f"run {run_idx}: "
                f"{result}"
            )

        run_dirs.append(
            result["run_dir"]
        )

        print(
            "[local] extraction done: "
            f"{alias} "
            f"run {run_idx}"
        )

    # =============================================================
    # STAGE 2
    #
    # Run ZigZag topology sweep on CPU.
    # =============================================================

    topology_jobs = []

    for run_dir in run_dirs:

        future = topology_one.spawn(
            run_dir,
            knn_values,
            max_simplex_dim,
        )

        topology_jobs.append(
            (
                run_dir,
                future,
            )
        )

    for (
        run_dir,
        future,
    ) in topology_jobs:

        result = future.get()

        if (
            result.get("status")
            != "success"
        ):
            raise RuntimeError(
                "Topology failed for "
                f"{run_dir}: {result}"
            )

        print(
            "[local] topology sweep done: "
            f"{run_dir}"
        )

    # =============================================================
    # STAGE 3
    #
    # Choose one global k for all models/runs.
    # =============================================================

    selection = select_k.remote(
        run_dirs,
        experiment_name,
    )

    selected_k = int(
        selection[
            "selected_knn_k"
        ]
    )

    print(
        "[local] selected global "
        f"kNN k={selected_k}"
    )

    # =============================================================
    # STAGE 4
    #
    # Produce PCA, kNN, barcode, birth/death,
    # persistence-image, etc. plots.
    # =============================================================

    visualization_jobs = []

    for run_dir in run_dirs:

        future = visualize_one.spawn(
            run_dir,
            selected_k,
        )

        visualization_jobs.append(
            (
                run_dir,
                future,
            )
        )

    for (
        run_dir,
        future,
    ) in visualization_jobs:

        result = future.get()

        if (
            result.get("status")
            != "success"
        ):
            raise RuntimeError(
                "Visualization failed for "
                f"{run_dir}: {result}"
            )

        print(
            "[local] figures done: "
            f"{run_dir}"
        )

    # =============================================================
    # STAGE 5
    #
    # Aggregate model/seed summaries.
    # =============================================================

    aggregate_all.remote(
        run_dirs,
        selected_k,
        experiment_name,
    )

    print(
        "[local] model-level "
        "mean/SD summaries done"
    )

    # -------------------------------------------------------------
    # Finished
    # -------------------------------------------------------------

    print()
    print("DONE")

    print(
        "Modal results volume:",
        RESULTS_VOLUME,
    )

    print(
        "Remote experiment folder:",
        f"/{experiment_name}",
    )

    print(
        "Download with:"
    )

    print(
        f"modal volume get "
        f"{RESULTS_VOLUME} "
        f"/{experiment_name} "
        f"results/{experiment_name}"
    )