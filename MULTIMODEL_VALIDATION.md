# Native-model extension: local implementation validation

Historical implementation-only validation, recorded before the separately authorized real execution. Both real full experiments subsequently completed; see [the final results and validation](results/native_models_modal_20260919/RESULTS.md) and [execution runbook](results/native_models_modal_20260919/EXECUTION_RUNBOOK.md).

At this checkpoint, implementation was ready for a separately authorized real smoke in `mdibrahimawad2/main`. This is **not** a report of new pretrained scientific results. No Modal jobs/image builds, model-weight downloads, pretrained inference, license agreements, or account changes were performed in this turn.

## Actual tests

Final command:

```sh
.venv/bin/python -m pytest tests -q
```

**56 passed, 1 skipped, 2 warnings in 184.51 seconds.** The optional legacy topology test is skipped because its optional dependencies are absent. Both warnings are from the existing deliberately constant-input legacy PCA test. No new test failed in the final run.

Additional checks actually performed:

- Compile/import of new modules and cloud **definitions only**, without resolving lazy resources or running/building an app.
- Actual `python -m numzig.multimodel` subprocess commands exercised saved-data analysis, plot-only recovery for both models, audits and exports. Process entry points live in an importable module, including on macOS spawn.
- Offline dry run validated all 10000 saved Crystal raw records, both model configurations, shared-text hash, and global resource ceilings.
- Tiny locally initialized native GPTBigCode and LLaMA fixtures: reference-wrapper versus batched-backbone states passed every declared gate (all levels; batch sizes 1/2/4/8/16/32; reverse composition; 128-token padding; logical two/three-worker assignments). No pretrained parameters were used.
- Full synthetic extraction and analysis serial/parallel equivalence for both model identities, with different dynamic state counts. Completed chunks/layers retained both bytes and modification times after interrupted workers and changed worker counts.
- Both receipt commit-boundary interruptions recover durable data. Cross-model receipts/configurations are rejected. Worker writes leave the shared manifest unchanged until coordinator merge.
- A partial smoke test uses a mocked model loader/device and synthetic vectors: saved passed gate reused, completed targets not inferred again, one model load per resumed invocation. This is a scheduling/recovery test, not GPU validation.
- Bounded OOM halving and batch-family restrictions tested with synthetic CUDA-OOM exceptions; failure at batch size one propagates.
- Bounded mixed-model CPU scheduler tested using fake remote calls: global concurrency capped, failed peers drained, successful peers retained. The actual Modal app-list JSON schema is tested, including active/stopping app rejection and fail-closed unknown schemas.
- Plot interruption between figures, changed worker counts, isolated plotting processes, corruption of one PNG, and artifact-only repair all passed. No hidden representation changed. Package resume preserves a valid completed export. A saved audit cannot conceal subsequent file corruption; changing one analysis layer invalidates only that layer’s figures/viewer payload, preserving all unaffected figure checkpoints.
- Legacy exact-neighbor ties/duplicates/self-exclusion/distance checks, global full-SVD PCA, heatmap denominators, checkpoint migration and all previous regressions passed. The scientific kernel ASTs remain identical to the completed Crystal source snapshot.

A full-size synthetic 10000 × 3200 layer used two BLAS threads: exact kNN 105.526 seconds; full-SVD PCA 5.593 seconds; peak RSS 2,238,955,520 bytes. Regression tests were active concurrently. This is a local host measurement, not a cloud allocation/utilization or GPU performance claim. See [benchmark](results/multimodel_build_validation/synthetic_fullsize_benchmark.json).

## Completed Crystal preservation

The authoritative completed Crystal dataset was imported read-only. Its saved file SHA-256 and canonical raw-record hash are pinned in the extension. A full saved-artifact checksum audit found **581 artifact records / 1005 unique files / zero invalid files**. The final original manifest SHA-256 is `18f7f35e702ab2e043a9cba4ffb8937c798d6f2dd08afc39b76485297017635f`.

Crystal `__init__.py`, `dataset.py`, `extract.py`, `plots.py`, and `viewer.html` match their saved source bytes exactly. No completed Crystal artifact was rewritten. Its original extraction fingerprint and plotting/template dependencies remain intact. See [preservation audit](results/multimodel_build_validation/crystal_preservation.json).

## Output and visual validation

Persistent examples are under `results/multimodel_build_validation/`:

- `starcoderbase-3b_SYNTHETIC/`: 40 saved smoke prompts, synthetic 4-level × 12-dimensional vectors, all plot/viewer families, numerical audit, export; **54 checkpoint records, zero invalid artifacts**.
- `openllama-3b_SYNTHETIC/`: 40 saved smoke prompts, synthetic 3-level × 12-dimensional vectors, all plot/viewer families, numerical audit, export; **46 checkpoint records, zero invalid artifacts**.
- `comparison_SYNTHETIC/`: saved-result-only comparison of the two synthetic examples with the existing saved real Crystal smoke. It is clearly marked synthetic and is **only a comparison-pipeline check**, not evidence about the new pretrained models. Raw prompt alignment passed; existing Crystal files were read only.

The graph, directed count/fraction heatmap with explicit denominators, and comparison depth chart were visually inspected. Both actual browser viewers were exercised on localhost: correct model titles and dynamic layer counts; 40 retained smoke points; exact-number searches for 10000 and 1000; retained selection across layer changes; four outgoing neighbors matching saved JSON; leading-digit/logarithmic magnitude colors; all-union and hidden edge modes. No browser console errors were reported. The temporary local server was stopped after testing. Missing browser favicon is immaterial and does not affect viewer assets.

See [artifact audits](results/multimodel_build_validation/synthetic_artifacts_audit.json) and [visual checks](results/multimodel_build_validation/visual_validation.json). The numerical audit checks every hidden hash/ID/finite shape; all smoke neighbor identities against all smoke candidates; PCA means, loadings, scores and variance; union flags and metric/heatmap denominators; all published PNG/viewer hashes. For future full datasets it checks 16 selected sources per level against **all 10000 candidates**, while validating all saved graph/PCA structures. It does not claim to repeat every full pairwise computation.

## Implementation and reuse

Added:

- `numzig/multimodel/__init__.py`: exact pins, scientific configuration, global budgets, overlap guard.
- `data.py`: immutable Crystal text import, native per-model tokenization, special tokens, contextual continuations, state semantics and provenance.
- `backend.py`: single-prompt reference, batched backbone, numerical gates, OOM backoff, backend/source/runtime contracts, private durable extraction.
- `pipeline.py`: receipt recovery, one-pass layer-cache conversion, independent layer workers, global bounded job plans and allocations.
- `plots.py`: isolated per-artifact plotting, count/fraction heatmaps, PCA depth plots, model-aware viewers, saved-only comparison and resumable packages.
- `validation.py`: saved-vector numerical/figure audits with no inference path.
- `__main__.py`: offline dry run/benchmark and explicit saved-data analysis/plot/validate/package/compare commands.
- `modal_multimodel_fullrange.py`: separate future app, existing secret/volumes, sequential shared-GPU model scheduling, global CPU/plot pools, drain/merge barriers and requested/observed records.
- `requirements-multimodel.txt`, `MULTIMODEL_FULLRANGE.md`, this report, and `tests/test_multimodel.py` (19 added test cases, including parametrizations).

Small shared extensions:

- `numzig/fullrange/analysis.py`: optional model-specific rows and preloaded arrays/validated shape in orchestration only. All six scientific kernels unchanged; legacy defaults preserved.
- `numzig/fullrange/parallel.py`: additional whitelisted figure/viewer paths and recursive receipts, per-artifact plot dependencies, optional caller-managed CPU plan limit, and avoidance of a redundant same-file hash in conflict checking. Original Crystal worker limits/defaults unchanged.

Reused directly: atomic files, two-phase Store/WorkerStore completion, private receipts, chunk format and validators, exact kNN/graph/PCA kernels, adjacent-layer finalization/baselines, figure primitives, colors and bundled viewer template. The project is nested in an unrelated home-directory Git repository; no checkout, branch, staging, or commit was made.

## Pending real checks

Public metadata pinned both revisions. StarCoder config/tokenizer files returned unauthenticated HTTP 401; its dimensions/native token behavior remain dynamically checked at authorized preparation. Existing secret access is unverified. OpenLLaMA's public 3200-dimensional/26-block config and slow-tokenizer/BOS policy were verified. See [metadata record](results/multimodel_build_validation/metadata_access_check.json).

Before full execution: separately authorize the workspace; check current secret access and quotas; pass real L40S model/tokenizer loading and all batching/reference gates; measure GPU throughput/peak memory and real-layer CPU RSS/times; validate real smoke graphs/PCA, inspect its figures/viewers; verify a completed smoke resume submits no GPU work and retains hashes. The full extractor refuses a missing/incompatible gate or missing real smoke audit. Current workspace memory/container/GPU/CPU quotas and real-model speed are not claimed verified.

The planned default is up to **10 L40S GPUs total**, sequential models; **24 × 2-core / 6-GiB** analysis workers (49 CPUs and 152 GiB including coordinator); **16 × 1-core / 4-GiB** plotting workers. Residual warm-container accounting keeps the default aggregate CPU ceiling at 85 and the maximum exposed profile at 99. Worker counts shrink with pending work. No cloud allocation has been made.

[Runbook and exact offline, smoke, full, resume, stage-only, comparison and download commands](MULTIMODEL_FULLRANGE.md).
