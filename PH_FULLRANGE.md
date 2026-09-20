# Per-layer persistent homology: build and pilot runbook

This is an analysis-only extension. It reads existing vectors; it cannot load model weights or run inference. Original results are never written. The dedicated `.venv-ph` leaves `.venv` unchanged. Cloud pilot execution was authorized on 2026-09-19; execution records are in `results/ph_execution_20260919/`.

## Frozen first-stage scientific choices

- All 10,000 targets, canonical IDs 0–9999, for all 33 Crystal / 37 StarCoder / 27 OpenLLaMA saved levels. No point subsampling or approximation fallback is implicit.
- Euclidean distances on original vectors, calculated by float64 direct differences in 256-row resumable blocks. Neither kNN nor PCA is the PH input.
- H0 from an exact dense Prim minimum spanning tree, retaining zero-distance merges. H1 from full Vietoris–Rips over F2, default giotto-ph WITHOUT edge collapse. The pilot exposed severe preprocessing overhead: on local StarCoder L1 subsets, collapse took 0.389 vs 0.011 seconds at 256 points and 5.188 vs 0.049 at 512 points, with identical diagrams; at 1024 points direct PH took 0.248 seconds while collapse exceeded 60 seconds. Ripser is an independent local reference backend. Both PH backends round filtration values to float32; maximum conversion error is recorded. This is combinatorially full Rips with finite precision, not float64-exact H1.
- Each nondegenerate layer must pass a seeded 128-point giotto-ph versus Ripser H1 comparison before its full H1 calculation. The fixed tolerance is four float32 spacings at the subset's maximum distance. This is a sampled backend check, not proof of full-cloud equivalence. Raw target/prompt identities must match across all three models before dispatch.
- Full scale range by default. An explicit local `--threshold` is in normalized units, is included in the checkpoint configuration, and produces right-censored H1 bars. H0 remains full range. Censored deaths must not be interpreted as essential features.
- Normalization: median of 100,000 seeded off-diagonal pair draws, using the same index draws for every layer/model. Preserve raw diagrams and the scale. If this median is zero, use the diameter and label the exception. All-zero layers have trivial topology and no division by zero. Crystal/OpenLLaMA L0 are degenerate; StarCoder L0 is a positional control and can have many duplicate vectors.
- All complete per-layer diagrams remain saved. The comparison gallery uses longest-32-bar H1 sketches to bound comparison time, and saves lower/upper bottleneck bounds against the full saved diagrams using half the maximum omitted lifetime per diagram. This sketch affects only comparisons, never the primary PH computation. Exclude L0/degenerate/truncated levels from that comparison. Similar diagrams do not prove identical numerical organization.
- No RTD, H2, DTM, diffusion distance, landmark approximation, or GPU PH backend in this initial implementation. Evaluate extensions only after the CPU pilot establishes feasibility. No causal tokenizer claims.

## Local commands (no Modal jobs)

From `/Users/mohammed.awad/Documents/numberline_zigzag_modal_FINAL`:

```bash
.venv/bin/python -m venv .venv-ph
.venv-ph/bin/python -m pip install -r requirements-ph.txt
MPLCONFIGDIR=/tmp/numberline-ph-mpl .venv-ph/bin/python -m pytest tests/test_ph.py -q
.venv-ph/bin/python -m numzig.ph inventory
```

`inventory` validates committed dataset/provenance hashes and canonical target order without reading all hidden states. Actual layer ingestion verifies the selected layer cache hash, or all contributing hidden-shard hashes and IDs when no layer cache exists (Crystal). Source manifests are never opened through a mutating Store.

To compute a saved layer locally, explicitly provide separate source/output trees:

```bash
.venv-ph/bin/python -m numzig.ph layer --source PATH_TO_SAVED_EXPERIMENT --output PATH_TO_PH_OUTPUT/model/layer_07 --layer 7 --threads 4
```

Use `h0` instead of `layer` for H0-only, then the same `layer` command continues. Local scientific runs are not part of the build validation. Switching scientific settings/backend/packages/code requires a separate output tree. Worker counts and time limits can change on resume.

## Modal pilot — executed; performance gate did not pass

Expected workspace/profile `mdibrahimawad2`, environment `main`; existing volume `numberline-zigzag-results`. No HF secret or cache mount is needed. Confirm balance, workspace CPU/RAM quotas, and that old jobs are stopped before launch. The plan's advertised 100 containers is not a 100-core guarantee.

```bash
MODAL_PROFILE=mdibrahimawad2 .venv-ph/bin/python -m modal app list --env main --json
MODAL_PROFILE=mdibrahimawad2 .venv-ph/bin/python -m modal run --env main --detach modal_ph.py --stage pilot --workers 4 --seconds 900
```

Pilot: Crystal L1/L7/L16/L32; StarCoder L1/L18/L21/L36; OpenLLaMA L1/L8/L13/L26, interleaved by model. Every job uses all 10,000 targets. At most 4 workers, each 4 physical cores and 16 GiB hard memory limit, plus a 0.25-core/1-GiB coordinator. No GPUs. Each PH child has a 900-second timeout, with its container capped at 1020 seconds; startup allowance 120 seconds. The first failed wave stops further submission; there are no automatic retries.

The first 600-second attempt timed out: Crystal L1 was still preparing input/distances, while L7 had reached PH. Its 1200-second resume completed H0 for four layers but was stopped after diagnosing edge-collapse overhead. Both attempts remain charged to the same reservation ledger. Direct-Rips outputs live under `/ph_fullrange_v1/direct_rips/`, separate from the old results. Four complete v1 distance checkpoints can be reused read-only after checking the archived producer code hash, unchanged source identity, dependency versions, every file hash, shapes, symmetry and diagonal. New manifests record this dependency; old fingerprints are never rewritten. New layers create normal checkpoints in the direct-Rips tree.

Coordinator remains remote when the terminal disconnects. Each layer owns its manifest and commits checkpoints. Distance blocks are individually atomic and checksummed. H0 survives an H1 timeout. H1's internal reduction cannot resume midway: that layer retries its PH computation, reusing input/distances. Plot-only recovery never recomputes H1. Changing code invalidates reuse intentionally; don't bypass fingerprints.

## Budget behavior

- Persistent compute reservation ledger under `/ph_fullrange_v1/budget_ledger.json`.
- Pilot allowance $3; main allowance $19; leave $3 of the user's $25 outside these stage allowances.
- Reservations use standard Modal Functions rates checked 2026-09-19: $0.0000131/core-second and $0.00000222/GiB-second. Cost = 1.25 × (timeout + 240 seconds) × (CPU + RAM rates). Coordinator allowance is reserved too. No reservation refunds; interrupted/failed attempts consume allowance across resume.
- Entire worker waves are reserved durably before dispatch. Stop before admitting a wave over allowance. There are no GPUs, region premiums, nonpreemptible surcharges, or inference charges in this app.
- This is a conservative application compute allowance, NOT a guaranteed account billing cap. Image builds, storage, other applications, pricing changes, rare orchestration overhead, and the small lock-clear operation are outside this model. Check actual account usage. Never automatically purchase credits or increase a plan.
- Full run requires a passing pilot with the same source code/packages and a conservative measured cost estimate within the remaining full allowance. A passing pilot does not guarantee every other layer finishes within its timeout/memory limit. Failures preserve progress and require review; no automatic approximation or relaxed tolerance.

Pricing reference: https://modal.com/pricing

## Full run / resume after pilot review

Current status: the pilot did not pass (Crystal L7 H1 timeout). Three full diagrams and four H0 results are saved. Do not run the full command below on the current evidence; its passing-pilot gate intentionally blocks it. See PH_VALIDATION.md and results/ph_execution_20260919/.

These are later execution steps, not part of the implementation run:

```bash
MODAL_PROFILE=mdibrahimawad2 .venv-ph/bin/python -m modal run --env main --detach modal_ph.py --stage full --workers 8 --seconds 900
```

8 workers request 32 cores/128 GiB plus coordinator, subject to actual quota. Maximum configured workers is 24 (96 cores/384 GiB); do not choose that simply because the account allows many containers. Pilot measurements decide the useful allocation. Completed pilot layers are reused in the full run.

Resume with the same command after the old app is stopped with zero tasks. If an abruptly killed coordinator left a lock, add `--resume`; this explicitly clears it only after the app-list guard. A stopping app is still rejected. Do not overlap launches. The guard fails closed on an unfamiliar Modal CLI schema.

## Download and inspect

After jobs stop, download into a fresh destination:

```bash
mkdir -p results/ph_download
MODAL_PROFILE=mdibrahimawad2 .venv-ph/bin/python -m modal volume get --env main numberline-zigzag-results /ph_fullrange_v1 results/ph_download
```

Locate the downloaded directory containing `crystal/`, `starcoderbase-3b/`, `openllama-3b/`. Set `PH_ROOT` to that directory (CLI destination nesting can differ):

```bash
PH_ROOT='results/ph_download/ph_fullrange_v1/direct_rips'
MPLCONFIGDIR=/tmp/numberline-ph-mpl .venv-ph/bin/python -m numzig.ph collect --output "$PH_ROOT"
.venv-ph/bin/python -m http.server 8767 --bind 127.0.0.1 --directory "$PH_ROOT"
```

Open http://127.0.0.1:8767/. Collection validates PH artifact hashes, creates per-layer PNG/SVG figures, common-scale topology heatmaps, depth curves, bounded H1 diagram comparisons, a summary with explicitly missing levels, and a static gallery. Download contains large reusable distance blocks (~800 MB/layer) and input vectors; plan local disk accordingly (~78 GB distances for 97 levels plus vectors and other files). Do not delete remote checkpoints merely to reduce download size.

## Before interpreting results

Inspect numerical stability/backend equivalence in the real pilot, all-point counts, finite/censored distinctions, actual wall time/RAM and billing. Synthetic correctness tests do not establish real-cloud performance. Long lifetimes are descriptive, not p-values. One randomized context per target and differing tokenizers/precisions remain confounds. Numeric/log-numeric controls and additional context replication are follow-up scientific analyses; no conclusion about common numerical topology has yet been made.
