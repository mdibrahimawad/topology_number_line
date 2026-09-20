# Crystal full-range experiment: implementation and runbook

**Implementation only. No Modal jobs, model inference, or GPU smoke runs have been authorized or performed.** Confirm the Modal account/workspace with the user before executing any cloud command below. The existing experiments and results are untouched.

## Entry points and commands

Run from `/Users/mohammed.awad/Documents/numberline_zigzag_modal_FINAL`.

Local dependencies for tests and saved-data analysis: the existing `requirements-local.txt`, plus `plotly==5.24.1` and `tokenizers==0.19.1`. Model dependencies are unnecessary for these tests. `requirements-crystal-fullrange.txt` pins the separate cloud image, including Modal 1.5.4 (required for read-only cache mounts), Torch 2.6.0 and Transformers 4.40.2 (the remote model was authored against 4.40.1). It does not change legacy requirements.

```sh
# Local scientific/storage tests and existing regression tests; no inference.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/crystal-fullrange-mpl .venv/bin/python -m pytest tests -q

# Genuine dry run: no Modal import, connection, image build, GPU or weights.
.venv/bin/python -m numzig.fullrange dry-run --gpu-workers 10 --cpu-workers 4
.venv/bin/python -m numzig.fullrange dry-run --smoke --stage all

# All stages on synthetic vectors only, clearly labeled as synthetic.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/crystal-fullrange-mpl .venv/bin/python -m numzig.fullrange synthetic --root results/crystal_fullrange_synthetic_validation
```

The following commands are **provided, not executed** and are conditional on the user's confirmation of **profile `mdibrahimawad2` / workspace `mdibrahimawad2` / environment `main`**. All three were checked read-only. Confirming a different workspace requires verifying its resources first. Command-local `MODAL_PROFILE` does not persistently switch accounts. Use the project venv's Modal 1.5.4; global Modal 1.4.2 lacks the read-only mount method used here.

After confirmation, the sequence is **two-worker smoke, inspect it, then ten-worker full run**. Neither command has been executed. Smoke has 40 targets in two saved 32-point chunks, so at most two GPU workers are useful. The full run has 313 chunks assigned round-robin to ten workers (31 or 32 chunks each on a fresh run).

```sh
# Smoke, only after workspace confirmation and authorization.
MODAL_PROFILE=mdibrahimawad2 .venv/bin/python -m modal run --env main --detach modal_crystal_fullrange.py --stage all --smoke --gpu-workers 2 --cpu-workers 2

# Full experiment after the smoke passes: all 10,000 targets, up to ten L40S GPUs.
MODAL_PROFILE=mdibrahimawad2 .venv/bin/python -m modal run --env main --detach modal_crystal_fullrange.py --stage all --gpu-workers 10 --cpu-workers 4

# ONE RESUME COMMAND (identical to full launch). First verify the prior app has stopped.
MODAL_PROFILE=mdibrahimawad2 .venv/bin/python -m modal run --env main --detach modal_crystal_fullrange.py --stage all --gpu-workers 10 --cpu-workers 4

# Explicit stage-only commands, also requiring authorization.
MODAL_PROFILE=mdibrahimawad2 .venv/bin/python -m modal run --env main --detach modal_crystal_fullrange.py --stage prepare
MODAL_PROFILE=mdibrahimawad2 .venv/bin/python -m modal run --env main --detach modal_crystal_fullrange.py --stage extract --gpu-workers 10
MODAL_PROFILE=mdibrahimawad2 .venv/bin/python -m modal run --env main --detach modal_crystal_fullrange.py --stage analyze --cpu-workers 4
MODAL_PROFILE=mdibrahimawad2 .venv/bin/python -m modal run --env main --detach modal_crystal_fullrange.py --stage plot

# For the optional read-only download/view commands below, use the confirmed profile.
CONFIRMED_MODAL_PROFILE=mdibrahimawad2
MODAL_ENVIRONMENT=main
export MODAL_ENVIRONMENT

# Lightweight download: plots, offline viewer, metadata, metrics, graph/PCA arrays and source.
MODAL_PROFILE="$CONFIRMED_MODAL_PROFILE" .venv/bin/python -m modal volume get numberline-zigzag-results /crystal_fullrange_1_10000_ctx1234_k4_seed42/lightweight.tar.gz results/crystal_fullrange_lightweight.tar.gz
mkdir -p results/crystal_fullrange_review
tar -xzf results/crystal_fullrange_lightweight.tar.gz -C results/crystal_fullrange_review

# Complete download, including all hidden states. Use a new destination for each snapshot.
MODAL_PROFILE="$CONFIRMED_MODAL_PROFILE" .venv/bin/python -m modal volume get numberline-zigzag-results /crystal_fullrange_1_10000_ctx1234_k4_seed42 results/crystal_fullrange_complete

# Local viewer. Open http://localhost:8000/viewer/ in a browser.
.venv/bin/python -m http.server 8000 --bind 127.0.0.1 --directory results/crystal_fullrange_review

# Optional local analysis/plot regeneration after downloading complete results:
.venv/bin/python -m numzig.fullrange analyze --root results/crystal_fullrange_complete
.venv/bin/python -m numzig.fullrange plot --root results/crystal_fullrange_complete
```

To download smoke results, substitute `crystal_fullrange_1_10000_ctx1234_k4_seed42_smoke` for the remote experiment folder. To view synthetic checks, serve `results/crystal_fullrange_synthetic_validation` instead. `figures/overview.png`, `figures/contact_*.png`, and `figures/depth.png` are the primary static entry points.

`--gpu-workers` is an actual extraction fan-out control (1–10, default 10). `--cpu-workers` is an actual independent-layer fan-out control (1–12, default 4). Each GPU worker reserves one L40S, 2 CPU cores and 32 GiB host memory. Each CPU worker reserves/caps 4 CPUs, requests 8 GiB memory, and caps BLAS/OpenMP threads at 4. The coordinator reserves/caps 1 CPU and requests 8 GiB; it limits numerical threads to 1. Default analysis therefore requests **17 CPUs and 40 GiB total memory**, including the coordinator (workers: 16 CPUs / 32 GiB). Maximum supported analysis fan-out uses 49 CPUs including the coordinator, below the optional 100-CPU ceiling. GPU and CPU phases do not overlap. Worker count is outside the scientific fingerprint and may be changed on resume.

`--stage` accepts exactly `all`, `prepare`, `extract`, `analyze`, `plot`. `--smoke` selects the separate smoke dataset. Resume is always on; there is intentionally no overwrite or force-inference switch. `modal run --help` is SDK help, not this experiment's dry run. Use the local Python dry-run command above.

## Checkpoint and interruption contract

- `manifest.json` is atomically replaced and holds configuration/fingerprint, stage status and per-artifact file hashes, dependencies, shapes/dtypes and completion state. Unknown nonempty directories are rejected.
- Dataset preparation saves the complete prompts, actual unpadded IDs, final-token positions, contextual expected-answer tokenizations, length distributions, tokenizer provenance and source before extraction is possible. Resume reads this exact data. A damaged existing dataset causes a hard failure and must be restored from a backup; it is never silently regenerated.
- Extraction uses **32-point chunks**, at most about **17.3 MB** of float32 vectors at historical dimensions. Each chunk has an uncompressed `.npy` with axes **[returned level, point within chunk, feature]** and a separate int32 point-ID `.npy`. The in-memory allocation is discovered from the first returned state and checked against loaded configuration and every subsequent point/chunk.
- Write temporary files → flush/fsync → atomic rename → read/validate arrays → record SHA-256 and `prepared` state in an independent chunk/layer receipt → **results-volume commit** → mark `complete` → **second commit**. No complete marker can precede a successful data commit. No file handles remain open across a reload.
- Restart validates all saved chunks. Valid `prepared` chunks can be adopted after interrupted commits without recomputation; valid complete chunks are skipped. A crash before a complete chunk is saved can redo only that unfinished chunk. If an already completed file is subsequently corrupted, only that invalid chunk is recomputed. Configuration, pinned revisions, saved dataset, tokenizer assets, extraction source and package versions cannot silently mix.
- Extraction remains single-prompt reference inference: eval/inference mode, unpadded attention mask of ones, no manually prepended BOS, tokenizer's established `add_special_tokens=True`, `use_cache=False`, all `output_hidden_states`, final real `=` position, float32 save. No batching, no answer generation.
- `read_layer` assembles one layer in canonical point-ID order using closed per-shard memory maps. No all-layer list or all-layer array is retained.
- Analysis workers save independent kNN + PCA + metrics in `layers/` after **each layer**, with the same commit ordering and private receipts. Each worker loads one complete layer, all 10,000 points, and compares every point against all 10,000 candidates. Workers split layers, never candidate sets or graphs. After the barrier the coordinator computes preceding-valid-layer overlaps and writes final `analysis/` outputs. A failed comparison or plot never reruns valid layer geometry. Compatible older serial outputs are adopted into independent layer records without altering their bytes or rerunning kNN/PCA. The original scientific-kernel identity is retained only while its kernel AST hash is unchanged; package or scientific edits change the analysis dependency.
- Each figure and each viewer data file has a separate checksum and checkpoint. Valid outputs are skipped; missing/corrupt files alone are regenerated. A plotting exception stops plotting. It has **no inference path**. All-stage resume also checks extraction before submitting a GPU function, so a completed extraction incurs no GPU allocation.
- The coordinator prepares/reuses the saved dataset, checks an immutable extraction runtime contract, warms the pinned model cache **once before GPU fan-out**, and commits it. GPU workers mount that cache read-only; dynamic module imports go to a private `/tmp/hf_modules` directory. One model copy is loaded once per worker invocation and used for all its assigned prompts/chunks; no model is split across GPUs.
- Only the coordinator writes `manifest.json`, shared stage status and root `progress.log`. It commits immutable disjoint work plans. Workers write assigned data and `receipts/chunk/` or `receipts/layer/` entries plus private `workers/<plan>/<slot>/` runtime metadata, benchmark, status and progress. No shared runtime/benchmark file is overwritten. Coordinator joins **all** submitted peers even after one fails, reloads the volume, validates/adopts durable receipts and merges once, then reports any failure. Completed peer work survives. If the coordinator itself dies, resume recovers receipts before assigning missing work. A restarted worker also reads its assigned durable receipts, even before a coordinator merge.
- Each interrupted worker can redo at most its own unfinished chunk. Completed chunks are immutable on compatible resume; invalid/corrupted chunks alone are assigned again. Changing worker counts does not change chunk boundaries, point IDs, prompts, model settings or science dependencies. Two-phase prepared receipts are recoverable without recomputation.
- **Never overlap separate `modal run` apps for the same experiment.** One coordinator per app is not a distributed cross-app lock. Detached jobs survive terminal disconnection. Check `MODAL_PROFILE=mdibrahimawad2 .venv/bin/python -m modal app list --env main` (read-only) and confirm the old app has stopped before resuming. Do not invoke internal worker functions directly, or write locally against an active download.
- Progress includes `COMPLETE`, `RECOVER`, `SKIP`, `MERGE`, worker count and completed/skipped/remaining counts. Workers keep private progress logs; the coordinator keeps the shared progress log.

Atomic rename/fsync protects local file integrity. Durability across Modal containers uses explicit [volume commits and reloads](https://modal.com/docs/guide/volumes#volume-commits-and-reloads), not local fsync alone. The local tests simulate interrupts at both commit boundaries; real cloud durability remains a future authorized smoke check.

## Scientific choices

Crystal only: reuse `numzig.config.resolve_models('llm360-crystal')` and the existing `_format_numeral_prompt`. Do not call the legacy all-in-memory extractor, plotting functions that refit PCA, topology pipeline, k sweep or model fan-out.

Prompt generator: Python `random.Random(42)`, independently draw `randint(1,9)`, `(10,99)`, `(100,999)`, `(1000,9999)` for each ascending target 1 through 10000. Four identity demonstrations, no spaces, end immediately after target `=`. Point ID is `target - 1`; no target-dependent exclusions. Chance equality is recorded. Smoke uses the production draws for its selected targets. Character counts range from 30–34; actual token lengths are measured independently and saved. The locally validated pinned tokenizer yields 31–35 actual tokens (the extra token is the initial metaspace, not a manually added BOS). Contextual continuations compare prompt IDs with the prefix of `prompt + str(target)` IDs; invalid prefixes are explicitly missing, never replaced with isolated tokenization.

Public metadata checked during implementation resolves both `LLM360/Crystal` and the tokenizer auto-map reference `LLM360/CrystalCoder` to **`IFM/Crystal`**, revision **`34fc9cd58acd87002560379a95b432147cc9135a`**. Original requested IDs stay in the configuration; no model is substituted. Preflight resolves and records both identities at their pinned revisions. It explicitly imports the pinned `CrystalCoderTokenizerFast` wrapper referenced by the tokenizer config to avoid an unpinned cross-repository auto-map lookup, and saves original tokenizer/config/source assets with hashes. See [model source](https://huggingface.co/IFM/Crystal/blob/34fc9cd58acd87002560379a95b432147cc9135a/modeling_crystalcoder.py) and [tokenizer config](https://huggingface.co/IFM/Crystal/blob/34fc9cd58acd87002560379a95b432147cc9135a/tokenizer_config.json).

Source inspection shows that returned level 0 is the scaled token embedding after dropout (disabled in eval); levels 1 through n−1 are successive residual block outputs, and level n is the final block **after `ln_f`**. Rotary positions do not add a position vector to the embedding state. Expected history is 33 levels × 4096 features; the implementation discovers and validates the loaded configuration and returned shapes, and hashes the actual loaded model source. Actual CUDA model loading/inference is not locally verified.

Exact kNN uses SciPy `cdist(..., metric='euclidean')` with float64 direct differences, 64 source rows at a time against **every** target. It avoids the cancellation-prone Gram-matrix identity and never constructs an N×N×D tensor. Self-exclusion uses point identity, including duplicate-vector fixtures. Stable sorting in ascending point-ID order resolves every distance tie, including the fourth-neighbor boundary. Actual Euclidean distances are saved in float64. Directed neighbors are N×4, with ranks 1–4; the undirected union contains either direction, with direction flags and mutual flags. Union degree can exceed four. No PCA, standardization, normalization or approximation affects neighbor selection.

Degeneracy: RMS centered vector norm ≤ `1e-7 * max(1, RMS uncentered vector norm)`. Both measured norms and the threshold are saved. Hidden vectors remain available; no tie-generated graph or PCA is interpreted for a degenerate level. Overview plots include labeled placeholders; depth curves exclude unavailable values.

PCA: full SVD, float64, all points, centering only, two components, no whitening. This costs more CPU than randomized PCA but has no approximation/stability-seed issue. Scores are computed from the saved mean/loadings; each score/loading pair is oriented to nonnegative target Spearman. Signed and absolute correlations are saved after that documented sign convention. Every plot uses these saved coordinates; each layer has an independent PCA frame. Explained-variance values are fractions in files and percentages on axes.

Metrics use **directed relations** unless explicitly labeled union: digit and digit-length fractions, counts, conditional cross-length numerator/denominator, numerical and log10 gaps, distance summaries, weak components and largest fraction, reciprocal directed fraction, incoming/union degrees and preceding-valid-layer overlap (intersection count divided by N×4). Heatmaps save raw counts and row-normalized fractions, with missing rows represented by NaN in NumPy, not misleading zeroes. Undefined JSON metrics are `null`.

Numerical baseline: separate exact k=4 graph on the scalar target, same tie rules. Frequency baseline: `(count_d−1)/(N−1)`, weighted by actual category counts (1112 for leading 1, 1111 for each 2–9). The viewer's orange diamonds and list are exactly the four **outgoing** neighbors, distinct from all incident union edges. It reuses one active WebGL plot, fetches one saved layer at a time, retains selection across layers, and works offline after serving local files (Plotly JavaScript is bundled).

## Output layout and sizes

`/artifacts/crystal_fullrange_1_10000_ctx1234_k4_seed42/` contains:

- `manifest.json`, `dataset.json`, `dataset_validation.json`, `tokenizer_provenance.json`, `tokenizer/`, `provenance.json`, `source/`, `progress.log`.
- `hidden/00000.npy`, `hidden/00000_ids.npy`, …; full extraction is **5,406,720,000 bytes (5.40672 GB)** of float32 data at 33×10000×4096, plus small headers/IDs.
- `extraction_contract.json`, immutable `plans/`, independent `receipts/`, and per-worker `workers/<plan>/<slot>/{extraction_runtime.json,extraction_benchmark.json,status.json,progress.log}`. Any historical serial runtime/benchmark files are preserved.
- `layers/layer_XX.npz` and `.json`: durable independent full-layer geometry, before adjacent-layer comparison finalization.
- `analysis/layer_XX.npz`: point IDs, directed neighbors/ranks/distances, union edges/flags, PCA scores/mean/loadings/variance, heatmaps, pointwise graph metrics and gaps. `layer_XX.json`: diagnostics and descriptive metrics/timing/peak RSS.
- `baselines.json`, `baselines.npz`, `layer_metrics.json`, `display.json`, `SUMMARY.md`.
- `figures/layer_XX_{graph,digits,heatmap,magnitude}.png`, overview, paginated contact sheets and depth curves. High-resolution PNG; deterministic drawing order with all points, saved separately from canonical ordering.
- `viewer/index.html`, bundled `plotly.min.js`, `index.json` and per-layer compact JSON.
- `lightweight.tar.gz`: everything except hidden vectors. Full download is the experiment directory. The archive contains a manifest snapshot before its own export receipt; hidden entries are retained as provenance even though their files are excluded. Analyze requires the complete download; plot-only works with the lightweight package.

Source upload is restricted to `numzig/`, `tests/` and three named source/configuration/runbook files. No previous results, dependency trees, archives, virtual environments, credentials or unrelated analyses are uploaded. Project-root Git revision is used only when this folder is the actual repository root; otherwise source hashes are authoritative. The current directory is nested beneath an unrelated home-directory Git repository.

No pairwise-distance matrix is exported automatically. A full float32 10000×10000 matrix would add about **400 MB per layer**, or 13.2 GB for 33 levels. Arbitrary distances can be recomputed from the saved hidden vectors. No optional full-distance exporter is implemented.

## Benchmarks and remaining validation

No paid runtime or cost is claimed. No real model benchmark has run. A local full-size **synthetic** geometry benchmark completed exact kNN in **130.60 s** and full-SVD PCA in **11.53 s**, with **2,794,733,568 bytes** peak process RSS. A 33-valid-layer geometry-only extrapolation is **78.17 minutes on the same local platform**, excluding I/O, metrics, commits, plots and extraction; this is not a Modal runtime estimate. Raw measurements: `results/crystal_fullrange_synthetic_benchmark.json`. Future smoke writes measured single-prompt extraction seconds and a linear 10000-point estimate (excluding weight load, including chunk commits); target/token-length distribution and hardware can change it. It records analysis time, dimension and process peak RSS for every layer. The new full-size local parallel-preparation benchmark used four-thread environment limits: exact kNN **131.03 s**, full SVD **11.49 s**, peak RSS **2,845,556,736 bytes** (2.85 GB). Raw measurements: `results/crystal_fullrange_parallel_benchmark.json`. These are synthetic timings on macOS arm64, not Modal timings or evidence of ten-GPU quota. Four workers with 8 GiB each leave substantial room above measured per-process RSS. The direct-distance kNN loop is largely single-threaded; the four CPU/thread allowance also supports full SVD on the cloud BLAS implementation. Actual per-layer timings/RSS are recorded, so CPU count can be reduced on later launches if the cloud measurements favor it. No 100-CPU reservation is made.

Reproduce the synthetic geometry benchmark (no model code is loaded):

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 .venv/bin/python -m numzig.fullrange.benchmark --points 10000 --dimension 4096 --output results/crystal_fullrange_parallel_benchmark.json
```

For exact direct kNN, rough work scales as N²D; extrapolating 40 points to 10000 is very uncertain. Full SVD work scales approximately N·D·min(N,D), so do **not** extrapolate small smoke PCA linearly. Use a representative full-size synthetic benchmark or measured first full saved-vector layer before estimating the entire 33-layer analysis. Timings include implementation/platform effects; they are not cloud cost estimates.

Tests cover full target coverage and context policy; actual-token metadata; saved prompt reuse; partial writes, corruption and configuration mismatch; interruptions inside a chunk and at both commit boundaries; unchanged checksums/mtimes for completed chunks; equality to uninterrupted synthetic results; kNN ties/duplicates/self-exclusion/direct distances; union construction; PCA reconstruction; metrics/denominators/degeneracy; per-layer analysis recovery; and independent figure recovery. Synthetic figures and interactive features are inspected separately. See `CRYSTAL_FULLRANGE_VALIDATION.md` for checks actually performed and any limitations.

## Read-only account/resource verification (2026-09-19)

| Local profile | Verified workspace | Accessible environment |
|---|---|---|
| `mdibrahimawad` | `mdibrahimawad` | `main` (default) |
| **`mdibrahimawad2` (selected)** | **`mdibrahimawad2`** | **`main` (default)** |
| `new-workspace` | `runningalphas` | `main` (default) |

In `mdibrahimawad2/main`, read-only listings confirmed required secret **`numberline-hf-token`** and volumes **`numberline-zigzag-results`** / **`numberline-zigzag-hf-cache`**. Secret values were neither retrieved nor displayed. This verifies resource existence, not token validity, gated-model permissions, available disk space, or write access from a running container. The public Crystal checkpoint is cached by the future coordinator; workers do not receive secrets.

Modal's read-only `EnvironmentList` response reported current concurrent tasks **0**, current GPUs **0**, with `max_concurrent_tasks` and `max_concurrent_gpus` **unset**. Unset does **not** establish unlimited quota. Workspace GPU limits, L40S-specific availability, aggregate CPU/memory quotas and billing limits were not exposed by these checks and remain **unverified**. Up to ten simultaneous GPU workers are implemented, but actual scheduling depends on workspace quota and capacity. No account was switched or authenticated; no cloud jobs, image builds, inference, resource creation or secret-value reads occurred. Global Modal is 1.4.2; the project venv is 1.5.4.

## Interpretation limits

Each target has only one randomized context. Prompt length varies with target length. Demonstration position and digit length are coupled. Chance demonstration–target matches are possible and recorded. Same-digit connectivity can follow ordinary numerical locality. Cross-length fractions require denominators; 10000 is the only five-digit target. PCA captures only part of the variance. Visual overlap/separation alone does not establish original-space graph structure. This experiment cannot prove that next-token preparation causes geometry. The stimulus design also differs from the previous experiment, so differences cannot be attributed only to denser coverage. No extra context seeds, interventions, significance tests, topology, models, UMAP, t-SNE or force-directed layouts are run.
