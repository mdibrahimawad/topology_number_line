# Provenance and methodology lock

This repository is a bridge between two user-provided codebases. It deliberately avoids inventing a new PH construction.

## Numerical side copied/adapted from the user's project

- Prompt form and sampling: `utils/prompts.py` from `research_llm_project_code_2026-07-22.zip`.
- Default numerical groups: `(1,2,3,4)`.
- Default samples per group: `30`.
- Three context examples and random context sampling.
- Hidden-state extraction: `outputs.hidden_states[layer][0,-1,:]` at every hidden-state index, converted to float32.
- PCA metrics: one-dimensional PCA, absolute Spearman rho, direct least-squares spacing fit `d_i = scale * beta**i`.

Exact source snapshots are in `upstream_reference/`.

## Topological side copied/adapted from ZigZagLLMs

- k-nearest-neighbor graph built on the ORIGINAL high-dimensional representations at each layer.
- Gudhi clique expansion of each kNN graph.
- Adjacent-layer intersection complexes.
- Dionysus zigzag homology persistence (`dionysus==2.0.10` and `gudhi==3.9.0`, matching the uploaded ZigZagLLMs environment).
- Main sweep over `kNN = 1..15`.
- Main outputs focus on H1; H0-H3 are also saved when returned by the solver.
- k selection rule: maximize total H1 feature count. This repo applies the rule globally across all four model/run point clouds so the same k is used for fair cross-model comparison.
- 2D PCA is used only to draw the point cloud. kNN edges in those figures are computed in the original hidden space.
- Inter-layer persistence uses the paper notebook's alpha=0 equal-weight choice.

Exact source snapshots are in `upstream_reference/`.

## Intentional engineering choices (not scientific changes)

- Four models are hard-allowlisted.
- Model x seed extraction is parallelized on Modal with a 10-container GPU cap.
- Saved hidden states are float32 to match the original numerical collector.
- Topology is run on CPU after extraction so GPU time is not wasted.
- Figures use quartile-depth layers plus the PCA-selected layer for visualization only. This does not affect PH.

## FastZigZag

The uploaded ZigZagLLMs repository imports `pyfzz` for its fast large-dataset runner, but that external implementation is not bundled. With only ~120 points per numerical run, this repo uses the same repository's Dionysus zigzag implementation, which is also what their demo uses. No replacement FastZigZag algorithm was invented here.
