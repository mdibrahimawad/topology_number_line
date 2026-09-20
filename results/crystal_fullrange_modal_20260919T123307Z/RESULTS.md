# Crystal full-range results — 2026-09-19

**Complete and validated.** The real smoke, completed-smoke resume, full 10,000-target experiment, downloads, numerical audit and visual/browser review all passed. The full Modal app is stopped with zero tasks. All 581 artifact records and SHA-256 hashes validated.

[Open the local results gallery](http://127.0.0.1:8765/) · [Interactive viewer](http://127.0.0.1:8765/full_lightweight/viewer/) · [Modal full app](https://modal.com/apps/mdibrahimawad2/main/ap-8gISuezjDj2MRrd3L6hmE4)

## Observations

The original-dimensional k=4 graph shows a strong late-layer leading-digit association. It is weak at L1 (12.2325%), rises non-monotonically through intermediate layers, and reaches 99.9900% at L27 and 99.9950% at L28. The final returned level L32 remains high at 99.9050%. These are fractions of 40,000 directed relations per valid layer.

The random-other-point frequency baseline is 11.1022%, but scalar numerical-neighbor kNN already gives 99.4975% same-leading-digit relations. A large same-digit percentage alone therefore does not distinguish a category effect from numerical locality. Late layers also have predominantly same-leading-digit **cross-length** neighbors: L27 has 340/341, L28 has 350/350, and L32 has 1208/1218. These denominators represent only 0.8525%, 0.8750%, and 3.0450% of all directed relations, respectively. The scalar numerical baseline has 0/24 same-digit cross-length relations. These are descriptive comparisons, not causal or significance tests.

Visually, L1 is broadly color-mixed and L16 has a curved, overlapping distribution. L27/L28 show several separated digit-colored groups, with other categories still overlapping in PCA. L32 overlaps substantially more in the two-dimensional projection despite its high original-dimensional same-digit neighbor fraction. This is a direct example of why projected appearance and hidden-space neighborhood evidence must be distinguished.

The requested representative layers were selected before viewing the complete results: early L1, middle L16, L27, L28, and final L32. PCA accounts for only the variance shown below and does not determine any neighbor identities. L0 is exactly constant across targets and is excluded from geometric interpretation.

| Returned level | Same-leading-digit neighbors | Cross-length edges | Same digit among cross-length edges | PC1 + PC2 variance |
|---|---:|---:|---:|---:|
| L1 | 12.2325% | 5/40,000 | 0/5 | 38.32% |
| L16 | 57.3875% | 94/40,000 | 1/94 | 30.40% |
| L27 | 99.9900% | 341/40,000 | 340/341 | 18.65% |
| L28 | 99.9950% | 350/40,000 | 350/350 | 19.85% |
| L32 | 99.9050% | 1218/40,000 | 1208/1218 | 37.39% |

Each target has one randomized context; prompt length varies; demonstration position is coupled to digit length; and 10000 is the sole five-digit target. No intervention establishes that next-token preparation causes the observed geometry. No extra model, seed, context experiment or topology run was used.

## Configuration and resources

Pinned Crystal revision: `34fc9cd58acd87002560379a95b432147cc9135a`. Targets 1–10000, point ID target minus one, seed 42, four identity demonstrations in 1/2/3/4-digit order, final meaningful equals-token extraction, 33 returned levels × 4096 features, float32 saved states. Exact Euclidean k=4 in original space; global full-SVD PCA per layer for display only.

Full app wall time: **58 minutes 33 seconds**, 13:00:16–13:58:49 UTC (17:00:16–17:58:49 Dubai). This includes preparation, extraction, validation, analysis, plotting and packaging, and excludes smoke and local download/audit time.

Verified profile/workspace/environment: `mdibrahimawad2 / mdibrahimawad2 / main`. Required existing secret and volumes were verified without displaying credentials. No account, billing or plan changes were made.

Ten simultaneous one-L40S workers were actually observed. All ten runtime records confirm NVIDIA L40S and one model load each; assignments cover 313 disjoint chunks. Nine extraction/commit times were 103.97–114.02 seconds; the slower worker took 269.18 seconds. These exclude model loading. GPU containers were released before CPU analysis.

Four concurrent CPU workers each requested/capped four CPUs and requested 8 GiB, with BLAS limited to four threads. The coordinator requested/capped one CPU and requested 8 GiB: 17 aggregate CPU cores and 40 GiB memory requested during analysis. Valid-layer total times ranged from 193.36 to 405.90 seconds (median 237.73); PCA ranged from 31.01 to 41.42 seconds. Maximum recorded process RSS was 2,423,508,992 bytes. The `knn_seconds` field includes data reading and diagnostics before distance calculation. These measurements are not a billing report; billed cost was not retrieved. Workspace-wide quotas were not exposed, but actual ten-GPU and four-CPU-worker concurrency was observed.

## Validation and provenance

Local tests: 37 passed, one optional legacy topology test skipped. Real two-GPU smoke passed the full numerical/figure/browser audit. Completed smoke resume submitted no GPU work and preserved all 254 artifact records unchanged.

Full hidden-state audit passed: all 313 chunk hashes, exact 10000-point coverage, dtype/shapes and finite values. All 40 shared smoke targets have bitwise identical representations in the full run. Full layer receipts merged 33/33 with zero invalid records. The final audit passed all 581 artifact hashes, all-layer graph structure and PCA checks, metric denominators/heatmaps/adjacent overlaps, both baselines, all 132 per-layer PNGs, overview/depth/contact files, and viewer arrays. Selected neighbor identities were independently checked against ALL 10000 candidates at L1/L8/L16/L24/L27/L28/L32; selected distances were recomputed directly from saved vectors. This sampled distance audit does not claim an unnecessary complete rerun of all pairwise distances. The actual viewer passed target search (1000 and 10000), layer switching, retained selection, exact four outgoing highlights, hidden/all-union edges, and saved-neighbor equality, with zero browser console errors. Actual L1/L16/L27/L28/L32 graphs, L27 nine-panel, L28 heatmap, L32 magnitude, depth chart and a contact sheet were visually inspected. All plots retain the full dataset.

See [execution record](EXECUTION.md), [full audit](full_audit.json), [visual checks](visual_validation.json), [runtime measurements](runtime_summary.json), [hidden-state audit](full_hidden_audit.json), [smoke audit](smoke_audit.json), and [recorded concurrency](full_01_concurrency.jsonl). Source snapshots and model/tokenizer provenance are retained with the downloaded experiment. The only operational code change during execution was `scaledown_window=2` for GPU workers; no scientific code or settings changed. A smoke download initially hit the Modal CLI directory-destination race; pre-creating the destination fixed it without repeating cloud computation.

## Files and recovery

Lightweight package: `full_lightweight.tar.gz` (337,526,180 bytes, SHA-256 `1b6ad5a64e6a316733609051d5788136b8f5110ef0c07432cc34edd503a631d7`); unpacked viewer/figures: `full_lightweight/`.

The complete local package combines the immutable hidden download with the final archive, final remote manifest/progress log, and the archive file itself. Every artifact in the final remote manifest is present locally and checksum-valid. The lightweight archive deliberately contains its earlier manifest snapshot, before the archive itself was recorded; the complete local folder has the final manifest.

Complete local result: `full_complete/crystal_fullrange_1_10000_ctx1234_k4_seed42/`. Hidden files are under `hidden/`, with axes [returned level, point within chunk, feature]. Point IDs are in the paired `_ids.npy` files. Complete scientific arrays are under `analysis/`.

Durable remote location: volume `numberline-zigzag-results`, directory `/crystal_fullrange_1_10000_ctx1234_k4_seed42` in environment `main`.

Resume only after checking that the prior app has stopped:

```sh
MODAL_PROFILE=mdibrahimawad2 .venv/bin/python -m modal run --env main --detach modal_crystal_fullrange.py --stage all --gpu-workers 10 --cpu-workers 4
```

Retrieve a fresh full snapshot (pre-create a new destination):

```sh
mkdir -p results/crystal_fullrange_fresh_download
MODAL_PROFILE=mdibrahimawad2 .venv/bin/python -m modal volume get --env main numberline-zigzag-results /crystal_fullrange_1_10000_ctx1234_k4_seed42 results/crystal_fullrange_fresh_download
```

Reopen the local gallery/viewer from the repository directory:

```sh
.venv/bin/python -m http.server 8765 --bind 127.0.0.1 --directory results/crystal_fullrange_modal_20260919T123307Z
```
