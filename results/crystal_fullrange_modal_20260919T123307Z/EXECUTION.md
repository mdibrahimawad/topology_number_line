# Crystal Modal execution record

User explicitly authorized smoke, fixes, full execution and downloads in profile/workspace mdibrahimawad2, environment main. Scientific configuration remains fixed.

Preflight: workspace/environment and required secret/volumes verified; no active apps returned. Environment max task/GPU limits unset; workspace limits/capacity unverified. Credentials not displayed.

## Smoke attempt 1

App: `ap-U26w9cjeehpHgN5KO1bBeb`
Dashboard: https://modal.com/apps/mdibrahimawad2/main/ap-U26w9cjeehpHgN5KO1bBeb

Command: `MODAL_PROFILE=mdibrahimawad2 .venv/bin/python -m modal run --env main --detach modal_crystal_fullrange.py --stage all --smoke --gpu-workers 2 --cpu-workers 2`

Local preflight: 37 passed, 1 optional skipped (46.72 s); production dry run verified. Local free disk: 591 GiB before downloads.

Smoke observations: cached pinned assets and weights prepared once; 2 concurrent extraction inputs/containers observed at 12:38 UTC. Both chunks committed and merged with zero invalid records. CPU analysis running with 2 workers (4 CPUs/8 GiB each). One completed extraction worker measured 8 targets in 7.7607 s including commits, excluding loading; this small-chunk rate is not a full-run wall-time estimate.

Operational adjustment for subsequent launches: GPU `scaledown_window=2` so idle extraction containers release promptly before CPU-only work. This changes only container lifetime, not inference, data, chunking or science fingerprints. The current smoke app remains untouched. Smoke L1 graph visually inspected: title, variance axes, categorical legend, points and graph edges render without clipping.

Smoke numerical diagnostics: level 0 has RMS centered norm exactly 0, status degenerate; excludes PCA/kNN interpretation. Main 32-target chunk took 11.9793 s (0.37435 s/target incl. commits, excl. weights). All 33 raw layers merged with zero invalid receipts. Smoke L16 nine-panel figure visually inspected: shared global axes/projection, correct category counts, readable titles/axes, consistent colors, no clipping.

Smoke app completed at 12:53:16 UTC, confirmed stopped with 0 tasks. All dataset/extraction/analysis/plot stages complete. Download attempt 1 encountered Modal CLI 1.5.4 directory-destination race (Errno 21): destination did not initially exist. SDK source inspection identifies per-entry is_dir classification before the first directory is created. Retry uses the now-existing empty task-owned directory; no cloud computation repeated. Single-file lightweight download also requested explicitly.

## Smoke acceptance passed

All 254 artifact records and checksums validated; hidden shape 33×40×4096 float32; exact target/ID/token coverage; 2 NVIDIA L40S runtimes; 32 valid layers and constant level 0; all valid-layer neighbors checked against all candidates and direct norm distances; PCA reconstruction/variance, metrics, directed/union graphs and viewer arrays validated. Static L1 graph, L16 nine-panel, L32 graph/heatmap, L27 magnitude, and depth chart visually inspected: readable and usable. Browser target search 1000, layer switching L1→L27→L28, four outgoing highlights, hidden edges and all union edges passed; no browser errors.

Completed-run resume app: https://modal.com/apps/mdibrahimawad2/main/ap-MtVMtrPGAll3LWEDHAvxSw . All 254 artifact records/hashes unchanged, all stages complete, explicit no-GPU-submission log and no GPU fan-out. No model inference repeated.

## Full experiment

App: `ap-8gISuezjDj2MRrd3L6hmE4`
Dashboard: https://modal.com/apps/mdibrahimawad2/main/ap-8gISuezjDj2MRrd3L6hmE4

Command: `MODAL_PROFILE=mdibrahimawad2 .venv/bin/python -m modal run --env main --detach modal_crystal_fullrange.py --stage all --gpu-workers 10 --cpu-workers 4`

Launched only after smoke and its completed-run resume passed and both apps stopped. Full dataset saved before fan-out. Actual concurrency is sampled read-only into `full_01_concurrency.jsonl`.

Full concurrency verified: read-only function statistics at 13:01:41 UTC report 10 simultaneous extraction inputs and 10 containers, backlog 0. No quota reduction required.

Full extraction: all 313 chunks saved, then coordinator merged 313 with zero invalid receipts and confirmed zero remaining. All ten worker runtimes report NVIDIA L40S and one model load. Nine worker extraction/commit times were 103.97–114.02 s; slower slot 00 took 269.18 s (weight loads excluded). All GPU containers were released before CPU analysis. Coordinator integrity merge completed approximately 13:10 UTC.

First real full-size valid-layer benchmark (L3): 358.474 s total, 318.298 s for data read/diagnostics/kNN, 39.988 s PCA, peak process RSS 2,391,674,880 bytes. The kNN timer starts before data read; do not describe it as isolated distance-kernel timing. Four analysis workers observed concurrently, zero GPU containers.

Local full hidden-state download completed during CPU analysis. 626 files / 313 chunks, 5.407 GB including headers/IDs. Every SHA-256 validated against the extraction-complete manifest snapshot; all 10000 IDs and 33×4096 float32 finite representations verified. All 40 smoke target vectors are bitwise identical to their full-run counterparts (maximum difference 0). Local free space before download: 590.11 GiB. Final complete local package will combine these immutable hidden files with the final lightweight archive, final manifest, archive receipt/file and final progress log.

All 33 independent layer checkpoints completed around 13:47 UTC; no failed worker or retry. Analysis began around 13:10:53 UTC with four active CPU workers. Coordinator merge/comparisons/plotting follow.

## Full acceptance passed

Full app stopped with zero tasks at 13:58:49 UTC; start 13:00:16 UTC, elapsed 58m33s. All dataset/extraction/analysis/plot/export stages complete. All 581 artifact records/hashes validated locally. Exactly 10000 targets, 33 levels, 32 valid graph/PCA levels and constant L0. Every valid layer has 40000 distinct directed non-self relations; selected all-candidate reference queries/direct distances passed at L1/L8/L16/L24/L27/L28/L32. All-layer PCA reconstruction/variance, heatmaps, metrics, denominators, adjacent overlap, baselines and viewer arrays passed.

Downloaded lightweight archive: 337526180 bytes, SHA-256 1b6ad5a64e6a316733609051d5788136b8f5110ef0c07432cc34edd503a631d7. Combined final archive metadata with already validated immutable hidden files, then restored final manifest/progress and retained archive in complete folder; the final 581-artifact hash audit passed. No cloud inference/analysis was repeated.

Visual checks passed for actual L1/L16/L27/L28/L32 graphs, L27 nine-panel, L28 heatmap, L32 magnitude, depth and contact_05. Full 10000-point viewer passed target search 1000/10000, layer switching L1/L27/L28/L32, selection retention, four outgoing markers/edges, hidden/all-union edges and saved-neighbor equality; console errors 0. Gallery and viewer opened for delivery. Full results/scientific interpretation and retrieval/resume commands: RESULTS.md.
