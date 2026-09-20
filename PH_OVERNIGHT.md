# Authorized all-layer PH run

Scope: Crystal 33 saved levels, StarCoderBase-3B 37, OpenLLaMA-3B 27: **97 independent H0/H1 calculations, each with all 10,000 targets**. Level zero is included. No inference, model weights, PCA/kNN substitution, point sampling, H2, or interpretation. Raw and normalized diagrams and per-layer PNG/SVG diagrams, barcodes and Betti curves are saved.

The user authorized execution overnight on mdibrahimawad2/main with $24 remaining. This run has a new **$20 conservative compute allowance**, leaving $4 outside it. It is not a Modal account hard billing cap: image builds, storage and other apps are outside the estimate.

## Execution

```bash
MODAL_PROFILE=mdibrahimawad2 .venv-ph/bin/python -m modal run --env main --detach modal_ph_overnight.py
```

The coordinator is remote, so closing the laptop does not terminate PH. It first checks four full-point reference layers concurrently on one GPU, while building a reusable Crystal layer cache. That cache reads/hashes the original shards once, instead of rereading every shard for every layer. It then dispatches CPU preparation and GPU batches independently.

Resource ceilings: eight preparation containers at four CPU cores/8 GiB each; ten L40S containers at six CPU cores/40 GiB each. A GPU container hosts up to four independent layer processes, each with one OpenMP/BLAS thread. Processes keep full 10,000-point layers intact. Up to 40 PH layers can run concurrently; maximum bulk allocation is 92.25 CPU cores and 466 GiB including the coordinator, subject to workspace quotas and scheduling. Containers exist only while work is available. The initial cache worker does not overlap the bulk pool.

Every GPU layer passes synthetic and fixed-subset backend checks. The three layer-1 diagrams must match their full CPU references, and Crystal L7 must match its saved single-process GPU result, before the bulk GPU stage is admitted. Distances remain float64 direct differences; PH uses the tested pinned Ripser++ float32 filtration. CPU distance blocks are computed in bounded four-thread waves with one manifest writer. Identical-vector layers use a validated implicit all-zero matrix and retain every point in H0.

An incomplete layer gets at most one bounded retry. GPU retries are isolated to one layer per container and may receive up to 1,800 seconds. Numerical validation failures halt further admissions. Unknown/interrupted calls retain their full reservations. Finished calls settle to a conservative runtime-based allowance, including 120 seconds overhead and 25% margin; this is bookkeeping, not verified billing. New work is refused if settled allowances plus all live reservations would exceed $20. Graph collection is reserved before worker dispatch.

## Checkpoints and overnight outputs

Existing results remain unchanged. New volume root: `/ph_overnight_v1` on `numberline-zigzag-results`.

- `crystal_cache/`: immutable derived layer caches.
- `prepared/`: per-layer distance blocks, H0 and normalization.
- `results/`: per-layer H0/H1 arrays, provenance, metrics, manifests, PNG/SVG graphs.
- `progress.json`, `budget.json`, `report.json`: progress, resource allowance and completion status.
- `results/index.html`, `results/summary.json`: final gallery and explicit missing levels.
- `lightweight.tar.gz`, `package.json`: automatic downloadable results and archive checksum.
- `logs/`: worker logs, including errors and timeouts.

Graphs are generated inside each layer process and the final gallery/archive runs on Modal. Completed input/distance/H0/H1/plot stages are checked and reused. H1's internal matrix reduction cannot resume midway; that layer's PH calculation restarts using saved preprocessing. Partial layers stay explicit; a budget stop or failure is never called complete.

## Resume and download

Only resume after verifying the previous PH app is stopped with zero tasks. Keep the same scientific source files; never force checkpoint fingerprints.

```bash
MODAL_PROFILE=mdibrahimawad2 .venv-ph/bin/python -m modal run --env main --detach modal_ph_overnight.py --resume
```

After the final archive exists, this download performs no compute launch:

```bash
MODAL_PROFILE=mdibrahimawad2 .venv-ph/bin/python download_ph_overnight.py --output /Users/mohammed.awad/Documents/Codex/2026-09-19/c/outputs/ph_fullrange_overnight
```

The downloader checks the archive checksum, safe archive paths, all completed scientific artifact hashes, source configuration, all-point counts, H0 IDs, H1 finiteness and normalization, and model/layer coverage. The local output contains `results/index.html` and `validation.json`. No scientific interpretation is generated.

## Verification before launch

22 local tests passed, including threaded-distance equivalence, interruption/corruption recovery, unchanged completed checkpoints, zero-distance layers, all-ID Crystal caching, strict GPU-equivalence checks and budget exhaustion. A first startup attempt failed because of an unbundled launcher import before any PH work. The launcher was made standalone and its import was checked; the failed app was stopped before restarting. Cloud sharing and other-model GPU equivalence are checked by the first four-layer gate in the running app, not assumed from local tests.
