# Implementation validation — 2026-09-19

**Execution update:** The user subsequently authorized cloud execution in `mdibrahimawad2/main`. Real smoke, completed-run resume, the full 10,000-target run, all downloads and final numerical/visual/browser validation passed. [Actual results](results/crystal_fullrange_modal_20260919T123307Z/RESULTS.md). See [execution record](results/crystal_fullrange_modal_20260919T123307Z/EXECUTION.md). Statements below about unperformed cloud checks describe the earlier implementation-only checkpoint, not current execution status.

**Ready for user-confirmed Modal smoke execution. No Modal jobs, image builds, cloud resources, model weights, model instances, forward passes, or answer generation were used during implementation. No scientific results from Crystal's new stimulus set exist yet.**

## Files added

- `numzig/fullrange/__init__.py`: isolated Crystal-only defaults and pinned revision.
- `numzig/fullrange/dataset.py`: full-range contexts, tokenizer metadata/provenance, immutable saved dataset and source snapshot.
- `numzig/fullrange/storage.py`: atomic files, SHA-256, two-phase durable completion and recovery.
- `numzig/fullrange/extract.py`: single-prompt extraction, 32-point shards, complete-chunk reuse, one-layer reader, runtime/configuration checks.
- `numzig/fullrange/analysis.py`: exact original-space kNN, deterministic ties, union graph, full-SVD PCA, metrics and baselines, per-layer resume.
- `numzig/fullrange/plots.py` and `viewer.html`: independently resumable PNGs, overview/contact sheets, digit panels, heatmaps/depth curves, magnitude views, offline WebGL explorer, summary and lightweight export.
- `numzig/fullrange/__main__.py`: local dry run, synthetic validation, saved-vector analysis and plot-only commands.
- `numzig/fullrange/benchmark.py`: local synthetic full-dimensional geometry benchmark.
- `modal_crystal_fullrange.py`: one coordinator, up to ten independent one-L40S workers, bounded full-layer CPU workers, read-only shared model cache, existing volumes/secret only, source-only upload.
- `numzig/fullrange/parallel.py`, `coordinator.py`, `runtime.py`: immutable disjoint plans, worker-private receipts/metadata, coordinator merge/recovery, scientific runtime compatibility.
- `tests/test_fullrange_parallel.py`: parallel ownership, interruption, durable receipt merge, worker-count changes, numerical equivalence and legacy checkpoint adoption.
- `requirements-crystal-fullrange.txt`: separate pinned cloud dependencies.
- `tests/test_fullrange.py`: scientific, storage, interruption, corruption and plot recovery tests.
- `CRYSTAL_FULLRANGE.md`: exact commands, methods, output schema, limits and resume contract.

Legacy experiments and saved result directories remain untouched. This update changes only the dedicated Crystal full-range implementation and its tests/docs, and adds a separate synthetic benchmark report. The working directory is nested under `/Users/mohammed.awad`'s unrelated Git repository, so no checkout, branch change, commit or staging was performed. Source hashes are recorded instead of claiming that repository's unrelated revision as experiment provenance.

## Tests actually run

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/crystal-fullrange-mpl .venv/bin/python -m pytest tests -q -rs
```

**37 passed, 1 skipped** (final run: 31.79 seconds). This includes all previous tests plus 13 parallel/recovery cases. The skip is the existing optional ZigZag test because `dionysus`/`gudhi` are absent; neither is needed by this experiment. Two warnings come from the existing legacy PCA test's deliberately constant layer. No new experiment test failed in the final run.

Coverage includes all 10000 targets and category counts; identity contexts; token/character lengths; saved-prompt reuse; invalid continuation prefixes; invalid final tokens; incompatible configuration; nonfinite vectors, dtype and shape errors; partial/corrupt chunk recovery; deterministic self-excluding kNN with duplicates and cutoff ties; direct Euclidean agreement including large offsets; union directions; PCA reconstruction; heatmaps and undefined denominators; degenerate layers; full numerical baseline; and viewer data equality to the saved PCA.

Interruption checks stop inside a chunk, at both commit boundaries, during per-layer analysis and during plotting. A simulated persistent-volume snapshot is copied into a new directory so recovery can see **only successfully committed files**, as a new container would. Completed chunk checksums and mtimes stay unchanged. Resumed hidden arrays and all saved per-layer NumPy analysis arrays match an uninterrupted run. Figure repair regenerates only the two deliberately missing/corrupt figures. The plot-only test replaces model extraction, kNN and PCA functions with failures to verify none are called.

Local CLI dry runs (production and smoke), Python compilation, `modal run modal_crystal_fullrange.py --help`, generic run help and volume-download help were checked. No launch command was executed. The implemented Modal interface exposes `--stage`, `--smoke / --no-smoke`, **`--gpu-workers`** and **`--cpu-workers`**; resume is automatic. Local module import and CLI help used project Modal 1.5.4 and performed no remote call.

## Parallel execution validation

Synthetic local tests exercised ten concurrent workers with disjoint ownership and verified that worker activity leaves the shared manifest byte-for-byte unchanged. Full production partition checks cover targets 1–10000 exactly once for 1, 2, 7 and 10 workers. Synthetic worker failure inside a chunk preserves peer completions; a coordinator restart merges their receipts and resumes missing chunks with a different worker count. Completed chunks keep both SHA-256 and modification time unchanged. A repeated invocation of the same plan skips its completed receipts before coordinator merge.

Durable-volume snapshots test failures at each chunk commit boundary. A fresh directory sees only successfully committed snapshots. Valid prepared receipts are adopted; a chunk whose data never committed is the only assigned work redone. Malformed receipts cannot certify completion, incompatible receipts are rejected, and corrupted chunk repair leaves other chunks unchanged. The join helper drains successful peers after a failure.

CPU workers split **layers**, with full canonical point coverage per layer. Concurrent synthetic layer outputs match serial outputs (integer graph/ID arrays and floating results within `1e-12`; exact extraction-array equality). JSON scientific metrics agree, excluding timing/RSS. Interrupted CPU workers resume only missing layers. Coordinator finalization cannot call PCA or hidden-space kNN. Compatible serial checkpoints are adopted without altering final files; repairing an earlier final output uses already completed downstream geometry rather than recomputing it. Original plot-only/no-inference and figure-repair regression tests still pass.

Cloud resource declarations import locally: extraction has `max_containers=10`, one L40S and 2 host CPUs per worker; analysis has a configurable 1–12 workers, each capped at 4 CPUs with 8 GiB requested memory. Default analysis is 4 workers (16 CPUs/32 GiB) plus a 1-CPU/8-GiB coordinator: **17 CPUs/40 GiB aggregate**. Worker BLAS/OpenMP limits are 4; the coordinator uses 1. Independent stages do not overlap. There is no model sharding, prompt batching, subset kNN graph or cloud execution in these tests.

The previous serial scientific kernels are unchanged. An AST checksum gates the compatibility alias for their original source identity; future kernel edits invalidate it. Worker counts are operational arguments and do not enter scientific dependencies. An immutable extraction contract checks model/dataset configuration, source and package versions before fan-out. The original serial runtime has an explicit, checked migration path.

## Tokenizer and source compatibility

Only public JSON/Python source metadata was downloaded, never model weights. Public model and wrapper references resolved to `IFM/Crystal`, commit `34fc9cd58acd87002560379a95b432147cc9135a`.

The pinned configuration, model and tokenizer Python modules imported successfully with local Torch 2.6.0 and Transformers 4.40.2. The configuration alone reports 32 transformer blocks, 33 returned levels and hidden dimension 4096. **No model class was instantiated.** Source inspection verifies final `ln_f` precedes the last returned hidden state.

The actual `CrystalCoderTokenizerFast` wrapper was instantiated from archived local tokenizer assets, using `local_files_only=True`. All **10000/10000** prompts passed the final-equals check and contextual continuation-prefix check. It adds neither BOS nor EOS. Character lengths are **30–34**; actual unpadded token lengths are **31–35**. The extra initial metaspace token explains the difference. An initial test mistakenly equated these distributions; it was corrected to the measured token lengths, leaving the prompt design unchanged.

## Synthetic artifacts and browser checks

All stages ran on 40 deterministic smoke targets using synthetic vectors, including a deliberately constant layer. The final visual sample is:

`results/crystal_fullrange_synthetic_validation_final/`

A preliminary sample remains separately at `results/crystal_fullrange_synthetic_validation/`; both are explicitly synthetic, with no model inference. Final static graphs are labeled `SYNTHETIC VALIDATION`. A second complete synthetic CLI invocation reused the saved dataset and skipped both extraction chunks, all three layer analyses, every valid figure and the lightweight export; all 30 artifact checksums validated.

Inspected graph, nine-panel, heatmap, magnitude, overview and depth PNGs for labels, limits, colors and clipping. The saved projection is shared across plot families. The overview includes the degenerate placeholder. Tests verify figure and per-layer viewer-file repair independently.

Browser checks on the local synthetic viewer passed:

- Target search for 1000 and point-click selection of 10000.
- Hover target, point ID, leading digit and digit length.
- Four highlighted outgoing neighbors and their saved original-space distances in the persistent side panel.
- Layer switching while preserving selected target, and appropriate degenerate-layer behavior.
- Selected-only, hidden and all-union edge modes.
- Leading-digit legend and fixed linear/log10 color scales.
- Drag zoom and pan, with visible axis changes.
- No browser console errors observed during these checks.

The browser sample uses 40 smoke points. Full 10000-point interactive browser performance and 33-level production figure rendering have not been measured; the implementation does not subsample either dataset.

## Full-sized local numerical benchmark

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m numzig.fullrange.benchmark --points 10000 --dimension 4096 --output results/crystal_fullrange_synthetic_benchmark.json
```

Synthetic float32 matrix, seed 42, macOS arm64; no model code loaded:

| Measurement | Result |
|---|---:|
| Exact k=4 original-space neighbors | 130.6019 seconds |
| Full-SVD global PCA | 11.5324 seconds |
| Peak process RSS | 2,794,733,568 bytes |
| Neighbor array | 10000 × 4 |
| PCA scores | 10000 × 2 |
| Finite distances | all |

The 33-valid-layer geometry-only extrapolation is 4690.43 seconds (~78.17 minutes) **on this local platform**, excluding I/O, metrics, commits, figures and model extraction. This is not a cloud timing or cost estimate. Local geometry packages: NumPy 2.5.2, SciPy 1.18.0, scikit-learn 1.9.0. The separate Linux cloud image pins NumPy 2.2.6, SciPy 1.15.3 and scikit-learn 1.6.1; its exact environment has not been built or tested remotely.

## Parallel-preparation full-size benchmark

A second local synthetic 10000×4096 benchmark ran with `OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4`. No model code was loaded.

| Measurement | Result |
|---|---:|
| Exact k=4 full-point neighbors | 131.0286 seconds |
| Full-SVD global PCA | 11.4905 seconds |
| Peak process RSS | 2,845,556,736 bytes |
| Neighbor / PCA shapes | 10000×4 / 10000×2 |
| Finite distances | all |

Raw report: `results/crystal_fullrange_parallel_benchmark.json`. This supports an 8-GiB per-worker memory request with headroom above the 2.85-GB observed local peak. Four CPU workers are the default; 100 CPUs are not allocated. The direct-distance kNN loop is largely single-threaded, while SVD can use the bounded BLAS allowance. Local timings are not Modal timing, CPU scaling or cost guarantees. The actual cloud BLAS implementation and container memory remain to be measured in authorized execution; per-layer outputs record timings and peak RSS.

## Environment and execution boundary

Read-only checks found global Modal **1.4.2**, project-venv Modal **1.5.4**. The latter is pinned in the dedicated requirements and used in launch commands because read-only volume mounts require its API.

| Local profile | Verified workspace | Environment |
|---|---|---|
| `mdibrahimawad` | `mdibrahimawad` | `main`, default |
| **`mdibrahimawad2`, selected** | **`mdibrahimawad2`** | **`main`, default** |
| `new-workspace` | `runningalphas` | `main`, default |

In **`mdibrahimawad2/main`**, secret-name listing confirms **`numberline-hf-token`**; volume-name listing confirms **`numberline-zigzag-results`** and **`numberline-zigzag-hf-cache`**. No secret values or credentials were retrieved/displayed. Existence is verified; token validity, runtime permissions and free storage are not. Other workspaces' resource inventories were not checked.

The read-only EnvironmentList API reports **0 current concurrent tasks**, **0 current concurrent GPUs**, and **unset** `max_concurrent_tasks` / `max_concurrent_gpus`. Unset is not proof of unlimited quota. Workspace-wide GPU/CPU/memory limits and live L40S capacity were not exposed; **ten-GPU scheduling capacity remains unverified**. No account switch, authentication flow, image build, cloud job, resource creation, weight download or model inference occurred. The selected workspace is information, not authorization.

Implementation and local verification are complete. Still intentionally unperformed: cloud image build; token validity and runtime volume access; pinned weight loading; CUDA/bfloat16 inference; real Modal parallel interruption/durability checks; GPU timing; full real-model analysis and scientific interpretation. After user confirmation and authorization, the documented sequence is **two-worker smoke → review → ten-worker full experiment**. Do not overlap launches for the same experiment; wait for the previous app to stop before resume. Exact commands are in `CRYSTAL_FULLRANGE.md`.

Public documentation used: [Modal volume commits/reloads](https://modal.com/docs/guide/volumes#volume-commits-and-reloads), [Modal CPU/memory resources](https://modal.com/docs/guide/resources), [Modal scaling](https://modal.com/docs/guide/scale) and [pinned Crystal source](https://huggingface.co/IFM/Crystal/blob/34fc9cd58acd87002560379a95b432147cc9135a/modeling_crystalcoder.py).

## Completed real Modal validation

The full app [ap-8gISuezjDj2MRrd3L6hmE4](https://modal.com/apps/mdibrahimawad2/main/ap-8gISuezjDj2MRrd3L6hmE4) ran in confirmed `mdibrahimawad2/main` from 13:00:16 to 13:58:49 UTC on 2026-09-19, then stopped with zero tasks. Actual concurrency reached ten one-L40S model workers and four full-layer CPU workers. All scientific configuration remained unchanged.

The real smoke passed; its completed-run resume skipped GPU submission and preserved all 254 artifact records. Full extraction completed 313 disjoint chunks, with 33×10000×4096 finite float32 states and exact point/target coverage. All 40 common smoke/full target vectors are bitwise equal. CPU receipts merged 33/33 with zero invalid. Layer 0 is exactly constant; 32 other layers have valid PCA and 40000 directed non-self kNN relations each.

Final local audit passed all **581 artifact hashes**, graph structures and union flags, PCA reconstruction/variance, heatmaps and metric denominators, adjacent overlaps, baselines and viewer arrays. Independent selected-query reference searches used ALL 10000 candidates at L1/L8/L16/L24/L27/L28/L32; selected Euclidean distances were recomputed from saved vectors. Every per-layer PNG was decoded/validated. Actual representative graphs and all plot families were visually inspected. The complete 10000-point viewer passed target search, selection retention across levels, four outgoing highlights, edge visibility and saved-neighbor equality, with zero browser console errors.

Both lightweight results (337526180 bytes, checksum verified) and the complete local experiment including hidden states (6262806007 bytes) are retained under `results/crystal_fullrange_modal_20260919T123307Z/`. Final manifest, source/provenance, receipts and raw arrays are preserved. See `full_audit.json`, `full_hidden_audit.json`, `visual_validation.json`, `runtime_summary.json`, `EXECUTION.md` and `RESULTS.md` there. The only runtime code adjustment was prompt GPU scale-down (`scaledown_window=2`); a CLI download-directory race was fixed by pre-creating the destination. No full-run inference or analysis was repeated. Real forced interruption was intentionally not manufactured; interruption/recovery remains covered by the local tests plus real completed-smoke resume.
