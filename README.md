# Topology of the numerical number line in language models

Experiments on how language models represent numbers, copying rules, digit permutations, and written number words. The repository contains prompt generation, native hidden-state extraction, exact nearest-neighbor analysis, PCA visualizations, persistent homology, task-behavior checks, and resumable Modal execution.

**Start by choosing an experiment below.** There is no single configuration that runs every study. The original sampled zigzag experiment, the 10,000-target study, ordinary persistent homology, and the matched-task study use different datasets and constructions.

The code and historical runbooks are published here. Model weights and large result archives are not. A clone includes the immutable 10,000-prompt Crystal dataset and selected provenance fixtures needed for local tests, plus the frozen OpenLLaMA execution source. It does **not** include all completed hidden states or interactive result data.

## Contents

- [Choose an experiment](#choose-an-experiment)
- [Install and test locally](#install-and-test-locally)
- [Models and representation conventions](#models-and-representation-conventions)
- [Configure Modal](#configure-modal)
- [1. Original sampled number-line and zigzag experiment](#1-original-sampled-number-line-and-zigzag-experiment)
- [2. Crystal full-range experiment](#2-crystal-full-range-experiment)
- [3. StarCoderBase and OpenLLaMA full-range experiments](#3-starcoderbase-and-openllama-full-range-experiments)
- [4. Full-range persistent homology](#4-full-range-persistent-homology)
- [5. Digit rules and written-number copying](#5-digit-rules-and-written-number-copying)
- [6. Post-hoc geometry and visualization studies](#6-post-hoc-geometry-and-visualization-studies)
- [Checkpointing and safe resume](#checkpointing-and-safe-resume)
- [Outputs, downloads, and galleries](#outputs-downloads-and-galleries)
- [Changing the scientific configuration](#changing-the-scientific-configuration)
- [Code map and provenance](#code-map-and-provenance)

## Choose an experiment

| Study | Data and scientific question | Entry point | Compute |
|---|---|---|---|
| Original number-line / zigzag | Approximately 120 observations per model/seed; geometry near decimal boundaries and persistence **across layers** | `modal_app.py` | GPU extraction, CPU topology |
| Crystal full range | Every integer 1–10,000 once; leading digits, magnitude, and connectivity at every saved level | `modal_crystal_fullrange.py` | Up to 10 L40S workers; independent CPU layers |
| Native model comparison | Identical saved prompt text in StarCoderBase-3B and OpenLLaMA-3B | `modal_multimodel_fullrange.py` | Up to 10 L40S workers total; CPU layers and plots |
| CPU Rips PH baseline | H0/H1 **within each layer**, using all 10,000 hidden vectors | `modal_ph.py`, `python -m numzig.ph` | CPU, saved vectors only |
| GPU PH benchmark / bulk run | Same full-point Rips calculation using validated Ripser++ | `modal_ph_acceleration.py`, `modal_ph_overnight.py` | GPU topology; **no language-model inference** |
| Digit rules / number words | Six matched tasks × two contexts × three models | `modal_taskrules.py` | GPU extraction/behavioral pilot; CPU geometry and PH |
| Crystal L27–L28 follow-up | Whether a sharp PC1 correlation change reflects changed geometry or exchanged PCA axes | `crystal_L27_L28_hypothesis_test/analyze.py` | Local saved-vector analysis |
| 3D PCA, PH comparisons, digit centroids | Projection fidelity, layer comparisons, numerical/digit ordering, final normalization | `posthoc/work/` | Local saved-data analysis |
| Geometric template screening | Fixed regular shape templates fitted to saved 3D projections with controls | `posthoc/work/geometric_shapes/` | Bounded local CPU pool |

Recorded completion: the three full-range representation studies and the six-condition matched-task study completed. The bulk PH run completed 96/97 levels; a separately validated exact duplicate-quotient supplement supplies StarCoder embedding-level H1. Keep that distinction when reporting coverage. Historical runbooks describe intermediate failures as well as final results; dates and scope matter.

## Install and test locally

Python 3.11/3.12 environments were used. Modal builds its own pinned Linux images; the local environment is for launch tooling, synthetic tests, and saved-data analysis.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-multimodel.txt
# Needed by the legacy analysis and L27/L28 helper:
.venv/bin/python -m pip install 'pandas>=2.1,<3'

# Synthetic/fixture tests; no pretrained model downloads or Modal jobs.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg \
  .venv/bin/python -m pytest tests -q --ignore=tests/test_ph.py \
  --ignore=tests/test_ph_acceleration.py --ignore=tests/test_ph_overnight.py \
  --ignore=tests/test_taskrules_analyze.py
```

A few native-backend tests execute tiny randomly initialized local models. They never load the real checkpoints. Optional legacy topology tests skip without GUDHI/Dionysus. The prompt fixture is committed at its historical `results/.../dataset.json` path so a fresh clone can run the native-model tests without cloud access.

For PH and task-analysis tests, use the separate environment:

```bash
python3 -m venv .venv-ph
.venv-ph/bin/python -m pip install -r requirements-ph.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg \
  .venv-ph/bin/python -m pytest -q tests/test_ph.py tests/test_ph_acceleration.py \
  tests/test_ph_overnight.py tests/test_taskrules_analyze.py
```

Ripser++ CUDA integration is validated by its explicit cloud benchmark, not by these CPU tests. Pinned cloud requirements are in [requirements-crystal-fullrange.txt](requirements-crystal-fullrange.txt) and [requirements-ph.txt](requirements-ph.txt); [requirements-local.txt](requirements-local.txt) is the lighter legacy setup. `pytest.ini` limits ordinary test discovery to the maintained suite; historical working copies in `posthoc/` are not collected implicitly.

Safe local planning commands:

```bash
.venv/bin/python -m numzig.fullrange dry-run --gpu-workers 10 --cpu-workers 12
.venv/bin/python -m numzig.multimodel dry-run --gpu-workers 10 --cpu-workers 31 --plot-workers 16
.venv/bin/python -m numzig.fullrange synthetic --root results/local_synthetic
```

## Models and representation conventions

| Full-range model | Pinned checkpoint revision | Saved levels | Width | Inference / stored precision |
|---|---|---:|---:|---|
| `LLM360/Crystal` | `34fc9cd58acd87002560379a95b432147cc9135a` | 33 (0–32) | 4096 | bfloat16 / float32 |
| `bigcode/starcoderbase-3b` | `e1c5ef4ebb97afa0db09ec3e520f0487ca350bbe` | 37 (0–36) | 2816 | float32 / float32 |
| `openlm-research/open_llama_3b` | `141067009124b9c0aea62c76b3eb952174864057` | 27 (0–26) | 3200 | float32 / float32 |

Vectors are taken at the **final meaningful query equals sign**, before the answer. Level 0 is the native embedding output. The final returned level includes the native final normalization; it is not an additional transformer block. Cross-model comparisons use normalized depth where appropriate, but their dimensions, tokenizers, normalization, and inference precision differ.

Crystal uses the pinned CrystalCoder fast-tokenizer wrapper and remote model code. StarCoder uses the native fast GPT2 tokenizer with no inserted BOS/EOS; its learned absolute positions produce five distinct level-0 vectors for the full-range prompts. OpenLLaMA uses the authors' slow tokenizer with exactly one BOS and no trailing EOS; its level-0 query vectors are constant. Native models use eager attention with TF32 disabled.

**Graph edges come from original hidden-space distances.** PCA places points in a figure; it does not choose the full-range graph's edges or the Rips PH distances. Native full-range kNN uses exact float64 Euclidean direct differences, four non-self neighbors, and deterministic distance/point-ID tie handling. PCA uses centered full SVD without scaling or whitening. A projected loop or polyhedron is not proof of the hidden-space topology.

## Configure Modal

The recorded execution workspace/profile is `mdibrahimawad2`, environment `main`. These are historical identifiers, not a guarantee of access from your machine. Some task-specific launchers explicitly enforce this profile and fixed volume paths. Review those constants before adapting a **new** experiment to another workspace.

Expected resources:

- Results volume: `numberline-zigzag-results`.
- Model cache volume: `numberline-zigzag-hf-cache`.
- Existing secret: `numberline-hf-token`, providing `HF_TOKEN` to model preparation.
- Permission to access the exact model checkpoints, including gated StarCoder weights.

Authentication and any resource provisioning are separate setup steps. Never put token values in code, commands committed to Git, or result metadata. PH-only stages use saved vectors and do not need to load model weights.

Select the profile explicitly per shell and inspect existing apps before a launch:

```bash
export MODAL_PROFILE=mdibrahimawad2
.venv/bin/python -m modal app list --env main --json
```

**Every `modal run` command below launches cloud work and may incur charges.** Local `dry-run` commands above do not. Do not run two coordinators against the same result namespace. A replacement launch must wait until the previous app and all its workers are stopped. Quotas are workspace-wide; requesting 10 workers does not guarantee 10 will be scheduled.

## 1. Original sampled number-line and zigzag experiment

The legacy design samples four groups around decimal boundaries, normally 30 observations per group and three random demonstrations per prompt. Three seed runs are compared per model. Repeated numerical targets remain separate point IDs because their contexts can differ; this is not exhaustive 1–10,000 coverage.

The model aliases are `falcon-rw-1b`, `falcon-rw-7b`, `btlm-3b-8k`, and `llm360-crystal`. They are an allowlist, not arbitrary Hugging Face IDs. The completed topology-resume helper specifically targets the three successfully extracted Falcon/Crystal models.

Each layer's original-space kNN graph is expanded into a clique complex. Adjacent-layer intersections define the zigzag sequence. The default sweep is k=1…15; one global k maximizes the total H1 feature count across the model/run collection. H0–H3 intervals are retained where computed, with H1 emphasized in the plots. PCA magnitude correlation and spacing fits are separate metrics.

```bash
# Cloud smoke, then full legacy study:
.venv/bin/python -m modal run --env main modal_app.py --smoke-test
.venv/bin/python -m modal run --env main modal_app.py \
  --models all --experiment-name main --runs 3 --samples-per-group 30 \
  --num-examples 3 --seed 42 --context random --knn-min 1 --knn-max 15 \
  --max-simplex-dim 4

# Existing legacy vectors only: resume topology without GPU inference.
.venv/bin/python -m modal run --env main resume_topology.py --experiment-name main
```

Additional controls include `--upper-bound`, `--prepend-bos`, and `--dry-run`; inspect [modal_app.py](modal_app.py) for their defaults. Scientific changes should use a new experiment name. See [the legacy guide](LEGACY_EXPERIMENT.md), [provenance](PROVENANCE.md), and [audit](AUDIT_REPORT.md). The legacy summary-existence resume helper predates the stricter hash/receipt protocol of the full-range pipelines.

## 2. Crystal full-range experiment

For every target E=1…10,000, save one prompt:

```text
A=A,B=B,C=C,D=D,E=
```

A/B/C/D are independent uniform draws with 1/2/3/4 digits, respectively, using seed 42 in target order. Contexts vary by target, and demonstration/target collisions are recorded. The entire tokenized dataset is saved before inference and reused on resume. Smoke uses 40 selected rows from this same production dataset.

All 33 levels are extracted. Chunks hold 32 targets (313 chunks, the final chunk has 16). Each worker owns one complete Crystal model on one L40S and runs one prompt at a time. The coordinator prepares the dataset and shared cache before assigning disjoint missing chunks.

```bash
# Small cloud smoke:
.venv/bin/python -m modal run --env main --detach modal_crystal_fullrange.py \
  --stage all --smoke --gpu-workers 1 --cpu-workers 2

# Full run AND safe resume, after previous writers have stopped:
.venv/bin/python -m modal run --env main --detach modal_crystal_fullrange.py \
  --stage all --gpu-workers 10 --cpu-workers 12

# Analysis or figures from saved data; these stages never invoke extraction:
.venv/bin/python -m modal run --env main --detach modal_crystal_fullrange.py --stage analyze --cpu-workers 12
.venv/bin/python -m modal run --env main --detach modal_crystal_fullrange.py --stage plot
```

| Control | Allowed / default | Effect |
|---|---|---|
| `--stage` | `all`, `prepare`, `extract`, `analyze`, `plot`; default `all` | Stage selection |
| `--smoke` | Off by default | Separate 40-target namespace |
| `--gpu-workers` | 1–10; default 10 | One model/L40S per extraction worker |
| `--cpu-workers` | 1–12; default 4 | Four cores, 8 GiB, four BLAS threads per analysis worker |

At the maximum analysis setting the active analysis allocation is 48 worker cores plus one coordinator core. Every independent layer uses all 10,000 points. Adjacent-layer metrics run after their input layers are available. See [the full runbook](CRYSTAL_FULLRANGE.md) and [validation](CRYSTAL_FULLRANGE_VALIDATION.md).

## 3. StarCoderBase and OpenLLaMA full-range experiments

Both models import the **exact saved Crystal prompt text**, checking its pinned dataset hash. They tokenize independently under their native policies. Targets, contexts, k=4, extraction positions, saved-level conventions, and PCA settings remain fixed. Their checkpoints and output namespaces are independent.

StarCoder's larger batches failed the preregistered elementwise vector tolerance; it therefore runs one **unpadded** prompt per forward. OpenLLaMA passed batches 1/2/4/8/16/32 and padding/order/partition checks; batch 32 was selected in the recorded execution. Tolerances were not relaxed to enable batching.

```bash
# StarCoder smoke:
.venv/bin/python -m modal run --env main --detach modal_multimodel_fullrange.py \
  --stage all --smoke --models starcoderbase-3b --batch-size 1 \
  --gpu-workers 1 --cpu-workers 4 --plot-workers 4

# StarCoder full run / resume:
.venv/bin/python -m modal run --env main --detach modal_multimodel_fullrange.py \
  --stage all --models starcoderbase-3b --batch-size 1 \
  --gpu-workers 10 --cpu-workers 31 --plot-workers 16
```

**For the completed OpenLLaMA namespace, use its frozen source directory.** Its smoke contract predates the subsequent StarCoder backend restriction. Do not force a new backend fingerprint onto saved chunks.

```bash
# Run from the repository root; preserve an absolute interpreter path.
PROJECT_ROOT="$PWD"
cd results/native_models_modal_20260919/openllama_execution_code

# Smoke (omit this when merely resuming a validated completed smoke):
"$PROJECT_ROOT/.venv/bin/python" -m modal run --env main --detach modal_multimodel_fullrange.py \
  --stage all --smoke --models openllama-3b --batch-size 32 \
  --gpu-workers 1 --cpu-workers 4 --plot-workers 4

# Full run / resume from the same frozen directory:
"$PROJECT_ROOT/.venv/bin/python" -m modal run --env main --detach modal_multimodel_fullrange.py \
  --stage all --models openllama-3b --batch-size 32 \
  --gpu-workers 10 --cpu-workers 31 --plot-workers 16
cd "$PROJECT_ROOT"
```

Run the model extraction launches sequentially to keep the combined GPU total at ten. Within the native coordinator, model GPU phases are sequential; independent CPU model/layer jobs share a bounded pool.

| Control | Allowed / default | Allocation |
|---|---|---|
| `--models` | `starcoderbase-3b`, `openllama-3b`, `both`; default `both` | Model selection |
| `--gpu-workers` | 1–10 / 10 | L40S + 2 CPU cores + 32 GiB each |
| `--cpu-workers` | 1–31 / 24 | 2 cores + 6 GiB each; BLAS=2 |
| `--plot-workers` | 1–16 / 16 | 1 core + 4 GiB each |
| `--batch-size` | Default 1; model's saved gate must permit it | StarCoder must remain 1 |
| `--stage` | `all`, `prepare`, `extract`, `analyze`, `plot`, `validate`, `package`, `compare`, `finish` | `finish` runs plots, validation, packaging only |

At 31 analysis workers, the active analysis pool plus coordinator uses at most 63 cores. The conservative ceiling including residual GPU and plot containers is 99 cores, not a request for 100 idle cores. Fewer pending layers mean fewer jobs.

To repair figures, use `--stage finish` or `--stage plot` with the appropriate model and source directory. They do not rerun inference. For the saved three-model comparison, run from the root:

```bash
.venv/bin/python -m modal run --env main --detach modal_multimodel_fullrange.py --stage compare --models both
```

See [design/configuration](MULTIMODEL_FULLRANGE.md), [validation](MULTIMODEL_VALIDATION.md), and the [actual execution runbook](results/native_models_modal_20260919/EXECUTION_RUNBOOK.md). `modal_native_access.py` is a cloud checkpoint-access probe; `modal_native_reprepare.py` is a historical migration tool for failed preparations, not a routine resume command.

## 4. Full-range persistent homology

This study uses the already saved hidden vectors from all three models: 97 levels, each containing all 10,000 targets. It is **ordinary within-layer Vietoris–Rips H0/H1**, not the legacy across-layer zigzag and not PH of a four-neighbor graph or PCA projection.

- H0 uses a full-distance minimum spanning tree, retaining zero-distance merges.
- H1 uses the full Rips filtration over F2; backends evaluate float32 filtration values. Original distances are constructed using float64 direct differences.
- Raw and normalized intervals are retained. Normalization uses the median of 100,000 fixed seeded off-diagonal pair draws, with an explicitly recorded diameter fallback for zero medians.
- Constant clouds and duplicate-heavy embedding levels require separate interpretation.
- Full-scale runs have no cutoff. A deliberately selected local threshold produces censored H1 and must not be described as full-scale PH.

**CPU baseline:** a pilot gates the full launch. The original CPU pilot hit a hard-layer timeout; simply increasing its worker count does not make that individual reduction finish. The saved benchmark motivated the GPU PH implementation.

```bash
# One downloaded full layer, local CPU only:
.venv-ph/bin/python -m numzig.ph layer --source PATH_TO_SAVED_MODEL \
  --output results/ph_local/model/layer_07 --layer 7 --threads 4

# Historical CPU cloud pilot, bounded to four-core workers:
.venv-ph/bin/python -m modal run --env main --detach modal_ph.py \
  --stage pilot --workers 4 --seconds 900
```

`modal_ph.py` supports `--stage pilot|full`, `--workers 1..24`, `--seconds 60..3600`, and `--resume`; the full stage requires a passing pilot. Its persistent cost ledger is separate from the later overnight ledger.

**GPU PH:** Ripser++ is pinned to commit `30243c0c752de26d7fdf6e41f08bf7b840ca4744`. Full-layer CPU/GPU reference checks precede bulk work. This GPU use does not load language models or recompute representations.

```bash
# Explicit backend benchmark:
.venv-ph/bin/python -m modal run --env main --detach modal_ph_acceleration.py --efficient

# Bulk saved-data PH; use --resume for the existing interrupted namespace:
.venv-ph/bin/python -m modal run --env main --detach modal_ph_overnight.py --resume
```

The overnight launcher fixes its resource pool in code: up to eight preparation workers (4 cores, 8 GiB) and ten L40S workers (6 cores, 40 GiB), with up to four independent complete-layer processes per GPU. Maximum bulk CPU allocation is 92.25 cores including the coordinator. Process BLAS/OpenMP threads are bounded. There is no `--workers` flag on this launcher.

Distance blocks, H0, final H1, and figures are checkpointed separately. An interrupted **internal H1 reduction** must redo that layer's reduction; it reuses completed preprocessing and leaves other layers intact. The historical collector failed after 96 complete layers; direct downloads and local validation preserved their outputs. The missing StarCoder L0 H1 was subsequently computed locally on its five **exactly identical-vector equivalence classes**, with all 10,000 H0 points retained and independent checks. This exact quotient is not approximate subsampling.

The [CPU runbook](PH_FULLRANGE.md), [acceleration benchmark](PH_ACCELERATION.md), and [overnight runbook](PH_OVERNIGHT.md) preserve those stages. The supplement source is [ph_audit_star_embedding_quotient.py](posthoc/work/ph_audit_star_embedding_quotient.py). Historical compute allowances are conservative estimates, not current prices, account balances, or hard billing caps.

## 5. Digit rules and written-number copying

Three models × six conditions × two fixed contexts = **36,000 query representations**. Each condition/context is analyzed separately; repeated contexts are replications, not additional independent numbers.

| Task key | Desired behavior | Targets per context |
|---|---|---|
| `copy4` | `1234=1234` | 1,000 sampled four-digit numbers |
| `reverse4` | `1234=4321` | Same four-digit cohort |
| `swap_first4` | `1234=2134` | Same cohort |
| `swap_last4` | `1234=1243` | Same cohort; additional control |
| `numeric_copy` | `234=234` | Every integer 1–1,000 |
| `word_copy` | `two hundred and thirty-four=two hundred and thirty-four` | Same 1–1,000 numerical coverage |

Each context has eight demonstrations. Four-digit prompts use explicit instructions plus examples, selected after bounded behavioral pilots; numeric and word copying use examples only. British English spelling uses lowercase, “and”, and hyphenated tens. Permutations preserve every digit, including leading zeroes. Four-digit targets exclude demonstrations. Copying cohorts retain and flag demonstration collisions to preserve complete coverage.

The pilot greedily generates answers for 32 targets per task/context (384 prompts per model). Exact-answer accuracy and technical native-forward equivalence are distinct checks. A technically valid extraction does not establish successful reversal/swapping. Conditions below 80% pilot exact accuracy are flagged as attempted/failed tasks; pilot selection and varying instruction lengths remain limitations.

Per model/layer/context/condition, analysis includes original-space exact k=4 neighbors, PCA2/PCA3, label associations, and median-distance-normalized H0/H1 on each nondegenerate 1,000-point cloud. Matched-task comparisons also use **shared PCA bases**, because independently fitted axes can rotate or reflect. Full answers are generated for pilots; full-run next-token records are not full-answer accuracy measurements.

```bash
# Resume the recorded v3 study, after all prior writers have stopped:
.venv/bin/python -m modal run --env main --detach modal_taskrules.py --stage all

# Saved-data-only operations:
.venv/bin/python -m modal run --env main --detach modal_taskrules.py --stage analyze
.venv/bin/python -m modal run --env main --detach modal_taskrules.py --stage compare
.venv/bin/python -m modal run --env main --detach modal_taskrules.py --stage package
.venv/bin/python -m modal run --env main --detach modal_taskrules.py --stage audit
```

This launcher is specifically the **v3 continuation**, with namespace `taskrules_20260920_v3`. Its default `--stage pilot` **adopts hash-verified saved v2 example-only and instruction pilots**; it is not a fresh pilot generator on an empty workspace. `--stage instruction_check` performs the instruction pilot; `--stage extract` requires accepted pilots. `--stage package_fast` reuses an already valid gallery for bounded parallel export. The v2 artifacts are required to reproduce the v3 adoption step.

Worker limits are fixed in [modal_taskrules.py](modal_taskrules.py): ten L40S extraction workers total, 24 analysis workers with two cores/8 GiB each, and six comparison workers with two cores/12 GiB each. Stages have barriers. There are no model/task/worker CLI flags on this recorded launcher. The task cohort, contexts, prompt wording, and seed are configured in [data.py](numzig/taskrules/data.py); pipeline stage/model/resource constants live in the launcher. A new design needs a new namespace and new validated pilots, not edits to existing receipts.

The recorded study completed 4,008 static figures, 1,132 interactive 3D views, and 388 shared-PCA comparison levels. Undefined or degenerate cases are labeled, rather than filled with invented values. See [the full protocol](TASKRULES_PROTOCOL.md) and [resume/download runbook](TASKRULES.md).

## 6. Post-hoc geometry and visualization studies

The [post-hoc guide](posthoc/README.md) maps the preserved local programs to their required saved outputs. These are separate analysis programs, not new model inference pipelines.

- **Crystal L27–L28:** checks point IDs and original-space neighbors, aligns PCA views, measures edge retention, and tests decimal-boundary branch descriptions. In the recorded 120-point run, the large PC1 magnitude-correlation change largely coincides with exchanged dominant PCA directions. See [the report](crystal_L27_L28_hypothesis_test/SUMMARY.md). Its scripts require the original legacy NPZ and tokenizer assets, not the 10,000-point data.
- **3D PCA:** adds a verified third component to existing saved two-component views while preserving PC1/PC2 and original graph neighbors. Projection residuals, orthogonality, and variance ordering are checked. It does not rebuild kNN in 3D and relabel it as original-space kNN.
- **3D numerical/digit analysis:** measures projected versus hidden-space neighbor retention, digit-centroid geometry, numerical-order fits, and layer alignment. Some descriptions fit the projected coordinates; their scope must remain explicit.
- **PH comparisons:** H0/H1 summaries, Betti curves, normalized lifetime statistics, cross-layer/model comparisons, and leading-digit H0 component checks. Truncated comparison sketches carry approximation bounds; do not confuse them with full persistence diagrams.
- **Final-layer and centroid checks:** separate final normalization effects from digit-order/centroid patterns. These are descriptive saved-data analyses.
- **Geometric shape screening:** fits fixed-proportion templates to authoritative saved PCA3 arrays, including individual and shared-task views. It distinguishes vertices, frames, and surfaces, uses held-out fit checks and free-centroid/control comparisons, and records projection diagnostics. A fit in 3D is not proof of a regular solid in the full representation space. Shared overlays reuse observations and are not independent experiments.

Local post-hoc tools retain historical path assumptions; the guide explains where to set them. Their generated datasets and figures are not included in Git.

## Checkpointing and safe resume

The full-range and task pipelines save the complete prompt dataset before inference. Model/tokenizer revisions, token policies, source contracts, extraction settings, and dataset hashes must match on resume. **Never delete or bypass fingerprints to make an incompatible run continue.**

For parallel full-range extraction, workers receive disjoint missing chunk IDs. Each model is loaded once per worker. Hidden arrays and point IDs are written atomically, validated, hashed, and durably committed before completion is acknowledged. Workers publish independent receipts; a single coordinator validates and merges them after a worker barrier. Worker metadata and progress do not race on one shared manifest.

A failed worker does not invalidate another worker's committed chunks. Recovery inspects receipts and validates files, skipping completed compatible chunks; only missing/invalid work is rescheduled. Changing worker count does not change scientific chunk identity. A plotting failure is handled with the plotting/finish stage, never by rerunning GPU extraction.

Analysis checkpoints are per full layer (or per task/context/layer). Completed kNN and PCA results are reused; each layer still compares against its complete point cohort. Dependent adjacent-layer/shared-task metrics wait for their inputs. Valid figures are skipped independently.

**Resume procedure:** stop and verify the previous app has zero active tasks, keep the original source/configuration and output namespace, then repeat that study's full command (or `--resume` for PH). Use a saved-data-only stage when extraction is already complete. Local tests cover interrupted commits, corruption, disjoint coverage, serial/parallel equivalence, changed worker counts, and preservation of completed files.

## Outputs, downloads, and galleries

Typical full-range model directory:

```text
<experiment>/
  dataset.json                  # saved prompts, target IDs and native token metadata
  tokenizer_provenance.json
  manifest.json                 # configuration and committed artifact hashes
  hidden/                       # extraction chunks and matching point-ID arrays
  plans/  receipts/             # disjoint assignments and independent worker records
  layers/                       # original-space kNN, PCA and per-layer metrics
  figures/  viewer/             # static plots and interactive per-layer data
  source/                       # exact execution source provenance
```

Do not treat the small committed provenance subset under `results/` as a complete downloadable experiment. Manifests can describe files that remain only on the results volume. Generated arrays, caches, archives, and weights are ignored by Git.

Read-only retrieval examples (no compute launch):

```bash
.venv/bin/python -m modal volume get --env main numberline-zigzag-results \
  /crystal_fullrange_1_10000_ctx1234_k4_seed42 results/crystal_download
.venv/bin/python -m modal volume get --env main numberline-zigzag-results \
  /taskrules_20260920_v3/lightweight.tar.gz results/taskrules_lightweight.tar.gz
```

The task lightweight archive excludes full hidden vectors, which remain under `/taskrules_20260920_v3/<model>/hidden/`. PH uses `/ph_overnight_v1`. Historical paths and archived download commands are documented in each runbook. If an archive was never created because packaging failed, retrieve and validate the per-layer artifacts instead of rerunning inference.

Interactive viewers fetch sibling JSON/data files, so serve downloaded output over HTTP:

```bash
python3 -m http.server 8767 --bind 127.0.0.1 --directory PATH_TO_EXTRACTED_OUTPUTS
```

Open the corresponding `index.html` through that local server. Runbooks' localhost links only work when the matching downloaded directory is being served; they are not hosted public result pages.

## Changing the scientific configuration

| What you want to change | Where | Resume compatibility |
|---|---|---|
| Worker counts | Full-range CLI flags; PH/task launcher constants where no flag exists | Compatible if scientific inputs/contracts stay fixed; stop old writers first |
| Plot repair / packaging | `--stage plot`, native `finish`, task `package`/`package_fast` | Reuses completed extraction and analysis |
| Legacy sample sizes, seed, demos, k sweep | `modal_app.py` CLI | Use a new `--experiment-name` |
| Full-range seed, target policy, k, dtype, chunk/extraction settings | `numzig/fullrange/__init__.py`, `dataset.py` | New experiment namespace; do not overwrite the published design |
| Native checkpoint, tokenizer, BOS/batching policy | `numzig/multimodel/__init__.py`, `backend.py` | New scientific contract and passing equivalence gate |
| Task prompts/cohort/context | `numzig/taskrules/data.py` and launcher dataset construction | New namespace, saved dataset, pilots, extraction contract |
| PH normalization/backend/threshold | `numzig/ph.py`, `ph_acceleration.py`, `ph_overnight.py` | New compatible provenance contract/output root |
| Post-hoc data locations | Path constants in `posthoc/work/`; see its guide | Relocation is operational, but changed inputs must be revalidated |

Changing a scientific parameter is a new experiment, even if a directory name happens to be the same. Existing manifests fail closed on incompatible configurations. Do not claim precision equality across models or causal task effects from these descriptive comparisons.

## Code map and provenance

- `numzig/{prompts,extract,geometry,zigzag,pipeline,plots}.py`: legacy sampled experiment.
- `numzig/fullrange/`: saved dataset, atomic storage, independent receipts, exact geometry, Crystal pipeline/viewer.
- `numzig/multimodel/`: native tokenizer/backend gates, layer caches, CPU/plot pools, validation/comparison.
- `numzig/{ph,ph_acceleration,ph_overnight}.py`: saved-vector H0/H1 and bounded execution accounting.
- `numzig/taskrules/`: matched stimuli, native extraction, pilot adoption, per-condition analysis, shared PCA, audit, gallery/export.
- `modal_*.py`: cloud images, resource allocations, stage coordination, namespace/account guards.
- `tests/`: maintained synthetic and fixture-based tests.
- `posthoc/work/`: preserved local analysis sources, including historical working copies; maintained task code is under `numzig/taskrules/`.
- `results/native_models_modal_20260919/openllama_execution_code/`: frozen compatible OpenLLaMA source, intentionally tracked despite the general result exclusion.
- `upstream_reference/`: original source snapshots. [PROVENANCE.md](PROVENANCE.md) and [LICENSE_ZIGZAGLLMS.md](LICENSE_ZIGZAGLLMS.md) retain attribution and the upstream MIT notice.

The upstream MIT notice applies to the identified ZigZagLLMs material; it is not a blanket relicensing of every model, dataset, or third-party asset. Model access and license terms remain separate. Scientific modules and historical frozen contracts were preserved when publishing this repository; documentation and packaging do not authorize or launch cloud work.
