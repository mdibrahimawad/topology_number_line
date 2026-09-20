# StarCoderBase-3B and OpenLLaMA-3B: implementation and runbook

Status: cloud execution was separately authorized on 2026-09-19 in `mdibrahimawad2/main` and is in progress. Execution records are in `results/native_models_modal_20260919/`. Local synthetic validation remains distinct from real scientific results. No license acceptance or account/billing changes are authorized.

## Identities and access

| Model | Pinned model and tokenizer revision | Architecture / returned states | Public metadata check |
|---|---|---|---|
| `bigcode/starcoderbase-3b` | `e1c5ef4ebb97afa0db09ec3e520f0487ca350bbe` | Native `GPTBigCodeForCausalLM`; dimension and block count read from pinned config during preparation | Public API resolved revision; pinned config/tokenizer JSON returned HTTP 401 without credentials. Dimensions are deliberately not guessed. |
| `openlm-research/open_llama_3b` | `141067009124b9c0aea62c76b3eb952174864057` | Native `LlamaForCausalLM`, 3200 dimensions, 26 blocks, 27 returned states | Public pinned config verified; checkpoint config declares float16. |

Metadata checks were made on 2026-09-19. Both default IDs are retained; local repository/research-reference searches did not establish the precise paper checkpoints. Paper-checkpoint match remains **unverified**. Neither the paper's tokenization controls nor a causal tokenization explanation is claimed.

StarCoder access: visit <https://huggingface.co/bigcode/starcoderbase-3b> and, if needed, personally obtain access under its agreement. No agreement was accepted here. The existing Modal secret's access to this gated repository has **not** been tested; a public HTTP 401 does not establish whether that secret already has approval. Revision resolution succeeded despite the file gate. OpenLLaMA author guidance: <https://huggingface.co/openlm-research/open_llama_3b>.

Both models use native Transformers 4.40.2 / PyTorch 2.6.0 with `trust_remote_code=False`. `requirements-multimodel.txt` includes the pinned Crystal scientific/native dependencies; the new Modal app defines a separate image and does not modify the validated local environment or old app.

Authorized access preflight verified the existing secret can read both pinned checkpoints, including all weight shards. StarCoder's pinned configuration has 36 blocks, 37 returned levels and 2816 dimensions. During the first smoke, Transformers 4.40 attempted a remote safetensors probe despite `local_files_only=True`. The loader now resolves the exact pinned cached snapshot and loads its local directory, avoiding remote probes and automatic conversion. The failed attempts published zero hidden chunks; they were archived, and their saved datasets reused byte-for-byte under fresh contracts. No compatibility hash was forced onto existing vectors. See `empty_smoke_archival.json` and `loader_fix_tests.txt` in the execution directory.

## Exact text and tokenizer contract

Authoritative input:

`results/crystal_fullrange_modal_20260919T123307Z/full_complete/crystal_fullrange_1_10000_ctx1234_k4_seed42/dataset.json`

Its SHA-256 is `bf0616805c5f4740fb3e790b0f7d5e7735ac89639696c271a5e6e2a626c12016`. The imported canonical raw-record hash is `e50d8ed8d4f3136118393ba4e310dea2083d8810fbb8db90894ca8e2078790bd`. Import checks both the original manifest and the pinned file hash. It copies the complete saved prompt/context metadata; it never redraws demonstrations. All 10000 targets, IDs `target-1`, ordering, A/B/C/D, raw text, seed 42, digit metadata, and chance demonstration matches are validated. Smoke uses the exact 40 saved Crystal smoke target IDs. Dataset preparation is saved and validated before any extraction. A corrupted committed dataset must be restored, not regenerated.

StarCoder's intended policy is its native fast tokenizer without inserted BOS/EOS. OpenLLaMA uses the authors' slow tokenizer (`use_fast=False`) with exactly one explicit BOS and no EOS. Encoding first disables automatic special tokens, then inserts the single BOS only where required. Real preparation also requires these IDs to equal the tokenizer's native `add_special_tokens=True` behavior; a policy discrepancy stops before extraction. This StarCoder policy still requires gated-asset verification.

Each model saves input IDs, readable pieces, decoded input, actual length, special-token policy, extraction position, final token ID/text, and whether the final equals shares a token. An input token ending in `=` is allowed; standalone equals is not forced. Trailing special tokens are rejected. Contextual continuation IDs are saved only when encoding `prompt+target` retains the entire input-token prefix; otherwise the mismatch is explicit and continuation metadata is unavailable. No isolated-target tokenization is substituted. Saved tokenizer/config assets and provenance reproduce the inputs.

## Backend and predeclared gate

Both new models intentionally use **unquantized float32 inference and float32 saved vectors**. This is a native supported precision, chosen to make batching validation strict and transparent; it differs from Crystal's recorded bfloat16 inference and is identified in comparison metadata. No TF32, quantization, compilation, serving engine, remote code, generation, or KV cache is used. Attention is native eager. One L40S hosts one complete model, loaded once per assigned worker; there is no tensor parallelism.

The reference calls the full causal-LM wrapper on one prompt in eval/inference mode and reads all returned states. The optimized path calls the native backbone, avoiding vocabulary logits; only final-real-token vectors are transferred to CPU. Within each durable 32-point chunk, prompts are sorted by token length then canonical ID, right padded, masked, and assigned explicit arange position IDs. Vectors are restored to canonical point order. `--batch-size` accepts 1, 2, 4, 8, 16, or 32 (default 16). CUDA OOM halves within that family down to one, then stops if one fails. Backoff sizes and memory/timing are recorded; it never silently changes dtype or attention.

The real smoke gate compares the single-prompt wrapper against every permitted batch size, reversed composition, maximum padding length 128, and simulated two/three-worker partitions. It checks every level, shape/order, vector absolute error, relative L2 error, cosine error, pairwise distances, and exact ordered k=4 identities. Tolerances, declared before real weights:

- Vector elementwise `atol=rtol=2e-5`; maximum per-point relative L2 `2e-5`.
- Maximum cosine error `2e-6`.
- Pairwise distance `atol=rtol=2e-5`.
- **No ordered kNN identity differences permitted**, even near ties. Boundary margins and near-tie counts are reported, not used to excuse changes.

These are strict fp32 checks; passing cosine alone is insufficient. A failed gate stops before publishing target chunks. A gate OOM also fails because the entire advertised batch family was not tested. Inspect a failure and explicitly revise/test the backend before launching a scientifically different contract; no automatic relaxed tolerance or hidden fallback occurs. Tiny local gates passed, but real-weight gates remain pending. A representative smoke cannot prove equivalence for every untested input; full saved-data audits independently check graph/PCA correctness.

Smoke uses one GPU per model, sequentially, to validate all smoke prompts. The gate's already computed batch-16 vectors become smoke chunks without an extra target extraction. A saved passed gate is reused after interruption; completed target chunks are not inferred again. Full extraction requires the matching smoke contract, non-synthetic numerical audit, completed plots, and the same GPU type. Source/package/model/tokenizer/dtype/backend changes are rejected. Operational worker counts and a batch size within the validated family do not change the scientific identity.

Returned levels are discovered from native configs and verified against actual outputs. GPTBigCode L0 includes learned absolute position embeddings; its last state includes `ln_f`. LLaMA L0 is the token embedding without additive positions, RoPE enters attention, and its last state includes final RMSNorm. L1 through the penultimate returned level are block residual outputs. Every level is retained; degeneracy is measured with the existing criterion rather than assumed. Embedding-level and final-normalized states are explicitly labeled.

## Resource profile, measured bottlenecks, and I/O

The completed Crystal run used 10 L40S workers and four analysis workers × four cores plus one coordinator: 17 analysis cores, 40 GiB. Valid layers took 193–406 seconds (median 238); PCA took 31–41 seconds, peak process RSS 2.424 GB. Hashing every shard in workers and serial plotting were material overheads. The exact direct-distance kernel is largely single threaded, so independent layers are preferable to assuming four cores yield fourfold speed.

A local synthetic **10000 × 3200** full layer was benchmarked with two BLAS threads: exact float64 kNN **105.53 seconds**, full-SVD PCA **5.59 seconds**, peak RSS **2,238,955,520 bytes**. Local regression tests were running concurrently; these are host measurements, not an isolated benchmark, Modal utilization, or real-model throughput. Report: `results/multimodel_build_validation/synthetic_fullsize_benchmark.json`. Future smoke records GPU time/memory; the first completed real full-size layer records CPU time/RSS for reassessment.

Default high-throughput profile:

| Phase | Active worker ceiling | Each worker | Coordinator included total |
|---|---|---|---|
| Extraction | 10 **total**, sequential models | 1 L40S, 2 capped host CPUs, 32 GiB host RAM | 21 CPUs, 328 GiB host RAM plus GPU memory |
| Analysis | 24 independent model/layer jobs globally | 2 capped CPUs, 6 GiB RAM, BLAS 2 | 49 CPUs, 152 GiB RAM |
| Per-layer figures | 16 isolated containers/processes globally | 1 capped CPU, 4 GiB RAM, BLAS 1 | 17 CPUs, 72 GiB RAM |
| Finalization | one coordinator | 1 capped CPU, 8 GiB RAM, BLAS 1 | 1 CPU, 8 GiB RAM |

Global controls: `--gpu-workers 1..10`, `--cpu-workers 1..31`, `--plot-workers 1..16`. Default 24 analysis workers substantially raises layer concurrency from four. The optional 31-worker profile requests 63 analysis CPUs / 194 GiB. It does not reserve cores just to reach 100. Worker counts are capped by actual pending tasks. Analysis queues contain complete model/layer jobs: every kNN compares against **all 10000 points of that model**.

GPU models run sequentially with a shared ceiling of ten; each worker retains one model across many chunks. Both datasets and required weight caches are prepared before fan-out; only one available weight format is downloaded in the future. GPU/analysis/plot writing phases have join barriers and two-second worker scale-down settings. To cover residual idle containers during transitions, the conservative budget adds **all** worker types plus the coordinator: default 85 CPUs, maximum 99 CPUs. The deliberately conservative simultaneous memory sum is 536 GiB by default (normally phases do not overlap). This is a ceiling/accounting bound, not a request for that many active workers throughout the run.

Workspace-wide current GPU/CPU, memory and container quotas are **unverified**. The completed Crystal run demonstrates historical ten-L40S concurrency, not guaranteed present capacity. Modal may queue requests. Do not change billing or allocate idle workers to hit a number. If quota/memory/startup pressure makes the profile unsuitable, stop the old app, wait for `stopped` and zero tasks, then resume with e.g. `--gpu-workers 5 --cpu-workers 8 --plot-workers 4`. Successful peer checkpoints survive. Submitted allocation and sampled observed container/input counts are saved separately; samples may miss peak concurrency. There is no speculative automatic account/plan change.

One coordinator validates source hidden chunks, transposes them once into `analysis_cache/layer_NN.npy`, and durably commits each completed cache layer. Workers validate/read only the assigned cache layer, not every hidden shard. The cache is derived, one additional float32 copy, and excluded from lightweight exports. OpenLLaMA hidden storage is 3.456 GB decimal plus 3.456 GB for its cache; StarCoder storage is `returned_states × 10000 × hidden_dim × 4` for each copy, reported after gated config discovery. Cache conversion time/bytes are recorded. Interrupted conversion reuses valid complete cache layers and may redo unfinished cache preparation; original target representations remain untouched.

## Checkpoints and stage separation

The existing atomic fsync/replace and two-phase receipt/commit machinery is reused. Dataset, hidden chunks, raw layer analyses, final adjacent-layer metrics, individual figures/viewer layers, final products, validation, and exports have independent checkpoints. Workers never write the shared manifest/progress/runtime files. They own only disjoint assigned paths and independent receipts/private metadata. The coordinator joins all calls, reloads the volume, validates hashes and schemas, and merges successes **even if a peer failed**. Prepared durable receipts can be adopted without recomputation. Interrupted exports are never marked complete.

Each per-layer figure/viewer receipt depends on that layer alone; repairing another layer preserves it. Aggregate overviews/comparison products wait for all layers. Exactly compatible chunks/layers/figures are skipped with completed/skipped/remaining messages. A worker-count change repartitions only missing work. Model-specific directories/configuration hashes prevent collisions. Analysis-only, plot-only, validate-only, package-only, and comparison code have no implicit inference call. The app-list guard uses the installed Modal CLI's actual JSON schema and refuses another active/stopping native experiment app; unknown schemas fail closed. Always wait for the old app to be stopped with zero tasks before resuming. Do not bypass fingerprint or overlap guards.

The six Crystal scientific kernels are unchanged: float64 direct Euclidean k=4, stable distance/point-ID ties, self-exclusion, directed and deduplicated mutual union graphs, and centered unscaled/unwhitened full-SVD PCA. Optional orchestration arguments accept model-specific rows and preloaded layer arrays. Crystal's config/dataset/extractor files and plot/viewer files are byte-identical to its saved source snapshots. Its entry point and completed result artifacts are preserved.

## Outputs and interpretation

Separate directories (on the existing results volume):

- `starcoderbase-3b_fullrange_1_10000_ctx1234_k4_seed42` and suffix `_smoke`.
- `openllama-3b_fullrange_1_10000_ctx1234_k4_seed42` and suffix `_smoke`.
- `numerical_three_model_comparison` for saved full-results comparisons.

Each contains the text hash/provenance, tokenizer assets/metadata, hidden chunks and IDs, layer semantics, exact directed neighbors/distances, union direction/mutual flags, PCA mean/loadings/scores/variance/solver/sign convention, metrics/baselines, source/package snapshots, hardware/timing/RSS/concurrency records, hashes and checkpoints. Per-layer plots use Crystal's fixed nine-digit palette: graph, shared-global-PCA nine-panel view, directed count + normalized heatmap with outgoing denominators, and magnitude view. Dynamic overviews/contact sheets, existing graph depth metrics, and added PCA variance/rho curves follow the worker barrier. The bundled viewer has the correct model title, all points, exact-number search, four outgoing neighbors/distances, layer/color/edge controls, and local Plotly assets.

Comparison uses saved Crystal and both new **full** datasets only. It verifies identical raw text, reads their committed summaries, preserves actual indices/metadata, and plots normalized transformer depth with L0/final-normalized endpoints. PCA frames are independently fitted. Neither absolute coordinates nor raw hidden distances are compared as common-scale quantities. Numerical-distance and frequency baselines, token-count distributions, revisions, precision, and special policies remain in comparison data. No model is labeled tokenization-confounded/independent as a conclusion.

## Commands

Run from the repository. The first commands are local/offline and were used during implementation:

```sh
.venv/bin/python -m numzig.multimodel dry-run --gpu-workers 10 --cpu-workers 24 --plot-workers 16
.venv/bin/python -m pytest tests -q
.venv/bin/python -m numzig.multimodel benchmark --root results/multimodel_build_validation --dimension 3200
```

The following commands are prepared **but must not be executed until separately authorized**. Command-local `MODAL_PROFILE` does not persistently switch accounts. Existing secret: `numberline-hf-token`; results volume: `numberline-zigzag-results`; cache volume: `numberline-zigzag-hf-cache`. No resource is automatically created.

```sh
# Read-only status before launch/resume. Confirm prior native app stopped with zero tasks.
MODAL_PROFILE=mdibrahimawad2 .venv/bin/python -m modal app list --env main --json

# Real smoke: sequential one-GPU model gates, geometry, parallel plots and audit.
MODAL_PROFILE=mdibrahimawad2 .venv/bin/python -m modal run --env main --detach modal_multimodel_fullrange.py --stage all --smoke --models both --gpu-workers 1 --cpu-workers 4 --plot-workers 4 --batch-size 16

# If StarCoder access is pending, the independent OpenLLaMA smoke interface is:
MODAL_PROFILE=mdibrahimawad2 .venv/bin/python -m modal run --env main --detach modal_multimodel_fullrange.py --stage all --smoke --models openllama-3b --gpu-workers 1 --cpu-workers 4 --plot-workers 4 --batch-size 16

# FULL RUN — only after real smoke and visual/throughput review pass.
MODAL_PROFILE=mdibrahimawad2 .venv/bin/python -m modal run --env main --detach modal_multimodel_fullrange.py --stage all --models both --gpu-workers 10 --cpu-workers 24 --plot-workers 16 --batch-size 16

# ONE RESUME COMMAND: same full command, after previous app is stopped.
MODAL_PROFILE=mdibrahimawad2 .venv/bin/python -m modal run --env main --detach modal_multimodel_fullrange.py --stage all --models both --gpu-workers 10 --cpu-workers 24 --plot-workers 16 --batch-size 16

# Independent stages. None of analyze/plot/validate/package/compare invokes extraction.
MODAL_PROFILE=mdibrahimawad2 .venv/bin/python -m modal run --env main --detach modal_multimodel_fullrange.py --stage analyze --models both --cpu-workers 24
MODAL_PROFILE=mdibrahimawad2 .venv/bin/python -m modal run --env main --detach modal_multimodel_fullrange.py --stage plot --models both --plot-workers 16
MODAL_PROFILE=mdibrahimawad2 .venv/bin/python -m modal run --env main --detach modal_multimodel_fullrange.py --stage validate --models both
MODAL_PROFILE=mdibrahimawad2 .venv/bin/python -m modal run --env main --detach modal_multimodel_fullrange.py --stage package --models both
MODAL_PROFILE=mdibrahimawad2 .venv/bin/python -m modal run --env main --detach modal_multimodel_fullrange.py --stage compare --models both
```

Download alternatives after completion (pre-create directories to avoid the observed Modal CLI destination race):

```sh
mkdir -p results/native_download/lightweight results/native_download/complete results/native_download/comparison
for experiment in starcoderbase-3b_fullrange_1_10000_ctx1234_k4_seed42 openllama-3b_fullrange_1_10000_ctx1234_k4_seed42; do
  MODAL_PROFILE=mdibrahimawad2 .venv/bin/python -m modal volume get --env main numberline-zigzag-results /$experiment/lightweight.tar.gz results/native_download/lightweight/$experiment.tar.gz
  MODAL_PROFILE=mdibrahimawad2 .venv/bin/python -m modal volume get --env main numberline-zigzag-results /$experiment results/native_download/complete
 done
MODAL_PROFILE=mdibrahimawad2 .venv/bin/python -m modal volume get --env main numberline-zigzag-results /numerical_three_model_comparison results/native_download/comparison
```

The lightweight archives retain scientific graph/PCA arrays, metadata, figures, viewer and provenance; they omit hidden vectors and derived layer caches. Full directory download retains both. The archive contains the manifest snapshot before recording the archive itself, as in Crystal. Validate the current full manifest separately. Add `_smoke` to model directory names for smoke downloads.

Local saved-data commands (never load weights):

```sh
.venv/bin/python -m numzig.multimodel analyze --root results/native_download/complete/openllama-3b_fullrange_1_10000_ctx1234_k4_seed42 --cpu-workers 8
.venv/bin/python -m numzig.multimodel plot --root results/native_download/complete/openllama-3b_fullrange_1_10000_ctx1234_k4_seed42 --plot-workers 4
.venv/bin/python -m numzig.multimodel validate --root results/native_download/complete/openllama-3b_fullrange_1_10000_ctx1234_k4_seed42
.venv/bin/python -m numzig.multimodel package --root results/native_download/complete/openllama-3b_fullrange_1_10000_ctx1234_k4_seed42
.venv/bin/python -m numzig.multimodel compare --root results/native_download/local_comparison --roots results/crystal_fullrange_modal_20260919T123307Z/full_complete/crystal_fullrange_1_10000_ctx1234_k4_seed42 results/native_download/complete/starcoderbase-3b_fullrange_1_10000_ctx1234_k4_seed42 results/native_download/complete/openllama-3b_fullrange_1_10000_ctx1234_k4_seed42
.venv/bin/python -m http.server 8766 --bind 127.0.0.1 --directory results
```

The local comparison command writes viewer links for the actual local paths; use it after downloading instead of relying on the remote layout’s links. With the server rooted at `results`, open `/native_download/local_comparison/` or a model viewer under `/native_download/complete/`.

Local/cloud analysis runtime versions must match for checkpoint reuse. Do not force incompatible scientific fingerprints; the old Crystal Python-version-sensitive kernel alias remains unchanged. Rendering uses its own dependency hash and may legitimately regenerate if plotting versions change.

## Before full execution

1. Separate authorization for `mdibrahimawad2/main`; existing secret/volumes and capacity checked without credential disclosure.
2. Both pinned model/config/tokenizer assets accessible; native special-token policy and actual dimensions verified.
3. Real smoke gates pass every level and every advertised batch/padding/partition case on L40S; backend/runtime/hardware metadata saved.
4. Measure actual batch throughput, peak GPU memory, CPU times/RSS and concurrent allocations; reduce requested concurrency if workspace memory/container quotas require it.
5. Smoke graph/PCA/heatmap/figure audit passes; inspect representative real plots and exercise viewer search, four-neighbor highlighting and layer/color controls.
6. Stop/restart smoke, verify completed chunk hashes unchanged and no GPU work submitted for a completed resume. Exercise an intentional partial-stage interruption if required by the execution review.
7. Launch full only after these gates and review. A failed plot must resume plot only.

Actual local validation results and remaining limitations are in `MULTIMODEL_VALIDATION.md`.
