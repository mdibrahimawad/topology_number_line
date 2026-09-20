# Authorized native-model execution, 2026-09-19

Workspace/profile `mdibrahimawad2`, environment `main`. Existing HF secret and both volumes verified. Both exact pinned checkpoints are accessible. No account, agreement, billing or plan changes were made. The web Resource limits page verified 10 GPUs and 100 containers across all environments. A separate CPU-core or memory ceiling was not displayed. Observed concurrency is recorded separately from requests.

## Source versions and real-gate decisions

The initial cached loader triggered Transformers 4.40's remote safetensors probe despite `local_files_only=True`. Loading the pinned local snapshot directory corrected this without changing weights, precision, attention or prompts. Initial empty attempts were archived after checking they contained no representations. Their saved datasets were reused byte-for-byte.

OpenLLaMA passed the complete original batch family (1/2/4/8/16/32), maximum-padding/reverse-order and two/three-partition checks. Its passing model and scientific source is frozen in `openllama_execution_code/`, with original file hashes. The coordinator entrypoint alone received the restart guard and finish-stage patch described below; its original copy is retained. Use that directory for its full run and resumes. Batch 32 was fastest in the 40-point smoke measurement (0.338557 s); batch 16 took 0.346242 s. Peak CUDA allocated memory across the gate was 14,420,418,560 bytes. These are small-smoke measurements, not full-run sustained throughput.

StarCoder passed batch one but failed the original elementwise vector tolerance at larger batch sizes. All original distance and ordered-neighbor checks passed; those successes do not override the vector failures. Tolerances were not relaxed. The current repository explicitly supports only **one unpadded prompt per forward** for StarCoder. Its revised real smoke passed this restricted mode against the original single-prompt wrapper, reversed order and two/three disjoint worker assignments: every vector comparison had zero maximum absolute error. All 37 levels, saved geometry, figures and viewer checks passed. Repeating the smoke preserved 318 records and created zero extraction workers. No padded/batched mode is claimed validated. Its failed gate and old contract remain archived; no StarCoder target chunk was published under them. The revised dataset reuses the original saved records without re-tokenizing or redrawing prompts; only its top-level operational configuration changes.

The OpenLLaMA frozen model/scientific modules must remain unchanged. Its coordinator entrypoint may receive documented orchestration-only fixes without altering those model contracts. Later StarCoder backend changes are not forced onto OpenLLaMA's completed contracts. Crystal is read only throughout.

## Commands

Always check the previous native app is stopped with zero tasks before a replacement:

```sh
MODAL_PROFILE=mdibrahimawad2 .venv/bin/python -m modal app list --env main --json
```

OpenLLaMA smoke repeat from its frozen directory:

```sh
cd /Users/mohammed.awad/Documents/numberline_zigzag_modal_FINAL/results/native_models_modal_20260919/openllama_execution_code
MODAL_PROFILE=mdibrahimawad2 /Users/mohammed.awad/Documents/numberline_zigzag_modal_FINAL/.venv/bin/python -m modal run --env main --detach modal_multimodel_fullrange.py --stage all --smoke --models openllama-3b --gpu-workers 1 --cpu-workers 4 --plot-workers 4 --batch-size 16
```

OpenLLaMA full **and resume**, from that same frozen directory:

```sh
MODAL_PROFILE=mdibrahimawad2 /Users/mohammed.awad/Documents/numberline_zigzag_modal_FINAL/.venv/bin/python -m modal run --env main --detach modal_multimodel_fullrange.py --stage all --models openllama-3b --gpu-workers 10 --cpu-workers 31 --plot-workers 16 --batch-size 32
```

StarCoder restricted smoke, from the repository root:

```sh
MODAL_PROFILE=mdibrahimawad2 .venv/bin/python -m modal run --env main --detach modal_multimodel_fullrange.py --stage all --smoke --models starcoderbase-3b --gpu-workers 1 --cpu-workers 31 --plot-workers 16 --batch-size 1
```

StarCoder full **and resume**, from the repository root, only after its revised smoke passes:

```sh
MODAL_PROFILE=mdibrahimawad2 .venv/bin/python -m modal run --env main --detach modal_multimodel_fullrange.py --stage all --models starcoderbase-3b --gpu-workers 10 --cpu-workers 31 --plot-workers 16 --batch-size 1
```

Run full models sequentially to retain the global ten-GPU limit. Each GPU worker loads one complete model once and preserves each completed chunk. Analysis jobs compare each complete layer against all 10,000 points. Worker count shrinks to pending jobs: OpenLLaMA has at most 27 layers, StarCoder 37. At 31 analysis workers, active analysis allocation is at most 63 cores including coordinator. The conservative bound including all residual GPU and plot containers is 99 cores. Observed utilization is reported from snapshots; reservations are not assumed busy.

Use `--stage plot` for a plotting-only repair, with the same model-specific source directory. It never invokes extraction. Three-model comparison uses saved outputs only after both new full experiments finish:

```sh
MODAL_PROFILE=mdibrahimawad2 .venv/bin/python -m modal run --env main --detach modal_multimodel_fullrange.py --stage compare --models both
```

Lightweight and full downloads go to separate directories below this execution directory. `audit_download.py MODEL_DIRECTORY` is read only and supplements the remote numerical audit with all-file hashes, saved prompt comparison, full target/neighbor coverage, independent selected neighbor calculations against all saved vectors, embedding degeneracy and viewer coordinate checks. It passed against both synthetic fixtures before real downloads.

## Real interruption and recovery

Modal internally retried the full OpenLLaMA coordinator despite `retries=0` (the SDK retries internal failures/preemption independently). Its early volume snapshot contained only 141 receipts, while original workers later logged 312 completed chunks. The agent stopped the application before the replacement extraction assignments left the queue and verified stopped/zero tasks. A fresh read found all 313 receipts, including durable prepared chunk 00303. Analysis-only recovery validated all 313, adopted the prepared chunk and skipped every target representation. Pre-recovery receipt hashes are preserved in `open_receipts_after_stop/` and `open_recovery_receipt_inventory.json`.

The coordinator now commits a unique launch marker before any work. An automatic retry with the same launch ID fails closed; a deliberate new launch is allowed only after the existing stopped-app guard passes. A one-core, 8-GiB non-preemptible coordinator avoids ordinary spot preemption; Modal bills its CPU and memory at 3x list rates. Workers keep their original allocations. This fix changes only orchestration, not scientific modules or compatibility hashes. `coordinator_guard_tests.txt`: three targeted tests passed, including interruption during marker commit.

The analysis-only recovery and subsequent finish command both completed successfully. To repair downstream OpenLLaMA outputs without re-entering extraction or analysis using its frozen directory:

```sh
MODAL_PROFILE=mdibrahimawad2 /Users/mohammed.awad/Documents/numberline_zigzag_modal_FINAL/.venv/bin/python -m modal run --env main --detach modal_multimodel_fullrange.py --stage finish --models openllama-3b --gpu-workers 10 --cpu-workers 31 --plot-workers 16 --batch-size 32
```

`finish` means plot, validate and package. It requires saved completed analysis, has no inference path, and preserves valid figure checkpoints. Create the destination directory with `mkdir -p` before downloading a directory; otherwise the CLI can treat it as one file. For downloads into dedicated existing execution directories, use Modal's `volume get --force`: its concurrent downloader otherwise encountered directory-creation collisions. This only replaces the task's own local download copies, never remote results.

## Verified milestones

OpenLLaMA full results completed, downloaded and audited: 993 artifact files, all 10,000 targets, all 27 levels, all stored edge distances and union flags, and selected exact rankings against all 10,000 candidates per valid layer. All 626 hidden data/ID files match the pre-recovery hashes. Early/middle/final static graphs and the full viewer were inspected.

StarCoder full extraction completed all 313 chunks using ten simultaneously observed L40S workers. All 37 analyses, 158 PNGs, 37 viewer layers, numerical validation and packaging completed. The full download passed 1,096 artifact checks; the full viewer and common smoke/full prompt comparisons passed. See `RESULTS.md`, `star_full_01.log` and `execution.json`.

## Completion

Both full experiments and the saved-data three-model comparison are complete. All Modal apps are stopped with zero tasks. Full datasets, hidden states, metadata, plots, lightweight packages and viewers are downloaded. The 97-level gallery and comparison links were checked in the browser. Final evidence and measurements: [RESULTS.md](RESULTS.md).

To serve the downloaded viewers again from the repository root:

```sh
python3 -m http.server 8766 --bind 127.0.0.1 --directory results
```

Open `http://127.0.0.1:8766/native_models_modal_20260919/gallery/index.html`. This is saved-data viewing only and starts no Modal jobs.
