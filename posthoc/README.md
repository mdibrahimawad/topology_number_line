# Saved-data follow-up programs

`work/` preserves the authored programs from the local follow-up workspace, including its 3D PCA, PH, centroid, and geometric-template analyses. `SOURCE_SNAPSHOT.json` records their byte hashes when imported. These source copies do not include the generated `outputs/` tree, pretrained weights, or third-party dependency source.

The maintained extraction/task implementation is in the repository's `numzig/` and root `modal_*.py` files. The `work/taskrules*` folders retain earlier development and audit scripts for provenance; their launchers are not substitutes for the root v3 launcher. Some helper scripts download from Modal or launch benchmarks; this directory is not uniformly offline. Inspect a program before executing it.

## Programs and inputs

| Directory/program | Purpose | Required saved inputs |
|---|---|---|
| `pca3d/build.py`, `validate.py`, `benchmark.py` | Third PCA component, viewer, covariance/projection validation | Completed three-model full-range hidden states and saved PCA2/neighbor arrays |
| `pca3d_analysis/` | Projection fidelity, layer alignment, digit and magnitude fits | `outputs/pca3d/`; some checks also need original hidden vectors |
| `ph_analysis.py`, `ph_additional_checks.py` | Layer PH summaries and additional checks | Downloaded PH results, verified layer-selection metadata, model geometry |
| `ph_compare.py`, `ph_h0_groups.py` | Between-layer/model PH comparisons and leading-digit H0 checks | `outputs/ph_fullrange_overnight/results/results/` and companion metadata |
| `ph_audit_star_embedding_quotient.py` | Exact duplicate-quotient supplement for StarCoder L0 H1 | All StarCoder L0 chunks and archived H0/normalization |
| `build_ph_report.py`, `ph_gallery.py` | Assemble local PH reports and viewers | Completed analysis tables, figures and validated downloads |
| `check_final_layer_change.py` | Final normalization/layer geometry checks | Completed full-range model outputs |
| `check_digit_centroids.py`, `plot_digit_centroids.py`, `centroid_top4.py` | Digit ordering and centroid neighbors/plots | Saved full-range arrays and earlier centroid outputs |
| `geometric_shapes/` | Regular template fits, controls, validation and report | Original 3D PCA and task-rule PCA3 arrays, shared-PCA arrays, projection metadata |
| `taskrules/download_results.py`, `validate_gallery.py`, `rescore_pilot.py` | Retrieve/audit results and report behavioral pilots | v3 task outputs and saved pilot answers |

## Relocating the historical workspace

Scripts using `Path(__file__).resolve().parents[2]` inside a subdirectory resolve to `posthoc/`, preserving the original `work/` + `outputs/` layout:

```text
posthoc/
  work/
    pca3d/
    pca3d_analysis/
    geometric_shapes/
  outputs/                     # ignored by Git; restore downloaded results here
    pca3d/
    pca3d_analysis/
    taskrules/run/
    ph_fullrange_overnight/
    ph_analysis/
    geometric_shapes/
```

Several one-off scripts instead use absolute `ROOT`, `REPO`, `OUT`, `SOURCE`, or `DATA` constants from the original machine. In an **execution copy**, set those paths to your downloaded data/output location before running; check the definitions at the top of each file. Keep this archived source and its hashes if you need to identify the exact earlier calculation. Input paths in saved metadata may also need resolving to the downloaded tree. Do not rewrite scientific artifacts or their hashes just to relocate files.

`pca3d/build.py` takes `--workers` (default 2) and `--batch-size` (default 4). Its `ROOT` points to the full-range `results/` directory; its `OUT` is `posthoc/outputs/pca3d`. This is local CPU PCA processing, not model batching.

With all required saved inputs restored and paths configured:

```bash
# Local PCA3 only; does not run language models.
.venv/bin/python posthoc/work/pca3d/build.py --workers 2 --batch-size 4
.venv/bin/python posthoc/work/pca3d/validate.py

# Shape fitting only. Each pending view is independent; BLAS is limited to one thread.
.venv/bin/python posthoc/work/geometric_shapes/run.py --workers 10
.venv/bin/python posthoc/work/geometric_shapes/controls.py --workers 6 --replicates 3
.venv/bin/python posthoc/work/geometric_shapes/validate.py
.venv/bin/python posthoc/work/geometric_shapes/report.py
```

The shape runner also accepts `--limit` and `--only` for a selected subset. Such runs are partial and must not be reported as the entire inventory. Compatible saved fits are reused using input hashes and fit-version checks. The regular-template study has its own tests:

```bash
.venv/bin/python posthoc/work/geometric_shapes/test_templates.py
.venv/bin/python posthoc/work/geometric_shapes/test_fit.py
```

Fit parameters and template definitions live in `geometric_shapes/fit.py` and `templates.py`. These are scientific settings, not generic runtime switches. PCA/template descriptions concern the projected clouds; original-space neighbor retention and distortion diagnostics state how much geometry survives the projection. A template match is not proof that the full hidden representation is a regular solid.

## Historical coverage

The overnight PH job produced 96 full results. The separate StarCoder embedding supplement proves that five bitwise-identical-vector classes preserve the original 10,000-point Rips H1, checks two backends and an independent boundary calculation, and retains all original H0 zero merges. Its result completes 97-level coverage only when combined with the original archive. It never silently replaces the timed-out historical record.

The task-rule gallery covers six conditions, two contexts, and all 97 model levels, with explicit degenerate/undefined entries. Shared-PCA comparison views reuse those observations. They must not be counted as additional independent samples.
