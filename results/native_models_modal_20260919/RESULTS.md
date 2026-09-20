# Completed numerical-representation experiments · 2026-09-19

![StarCoderBase-3B, final level 36](full_complete/starcoderbase-3b_fullrange_1_10000_ctx1234_k4_seed42/figures/layer_36_graph.png)

![OpenLLaMA-3B, final level 26](full_complete/openllama-3b_fullrange_1_10000_ctx1234_k4_seed42/figures/layer_26_graph.png)

Both new experiments are complete, downloaded, numerically audited, and visually inspected. Each uses exactly the saved 10,000 Crystal prompts, targets 1–10,000 once each, four identity demonstrations of digit lengths 1/2/3/4, seed 42, and extraction at the final real equals token. Crystal was preserved and never rerun.

## Open the results

- [Full gallery: all 97 returned levels across three models](http://127.0.0.1:8766/native_models_modal_20260919/gallery/index.html) ([portable HTML](gallery/index.html)).

- [StarCoder full viewer](http://127.0.0.1:8766/native_models_modal_20260919/full_complete/starcoderbase-3b_fullrange_1_10000_ctx1234_k4_seed42/viewer/index.html).

- [OpenLLaMA full viewer](http://127.0.0.1:8766/native_models_modal_20260919/full_complete/openllama-3b_fullrange_1_10000_ctx1234_k4_seed42/viewer/index.html).

- [Three-model comparison](http://127.0.0.1:8766/native_models_modal_20260919/comparison/index.html), [comparison curves](comparison/depth_comparison.png), and [complete comparison data](comparison/comparison.json).

- [StarCoder lightweight package](lightweight/starcoderbase-3b.tar.gz) and [OpenLLaMA lightweight package](lightweight/openllama-3b.tar.gz). These include plots, analysis and viewers; hidden vectors are in the full directories.

- Full hidden states and metadata: [StarCoder](full_complete/starcoderbase-3b_fullrange_1_10000_ctx1234_k4_seed42/) and [OpenLLaMA](full_complete/openllama-3b_fullrange_1_10000_ctx1234_k4_seed42/). Each contains `hidden/`, `dataset.json`, tokenizer provenance, runtime metadata, manifests, receipts, analysis, and source snapshots. The full downloads also retain the derived layer cache.

The localhost server is bound to 127.0.0.1. If it has stopped, run from the repository root: `python3 -m http.server 8766 --bind 127.0.0.1 --directory results`. The HTML gallery opens directly as a file; viewers require HTTP to load their saved JSON.

## Scientific contracts

| Model | Architecture and state semantics | Width / returned levels | Inference and tokenizer policy |
|---|---|---|---|
| StarCoderBase-3B | Native GPTBigCode; learned absolute positions; final LayerNorm | 2816 / 37 | float32 eager, TF32 off; GPT2TokenizerFast; no inserted BOS/EOS |
| OpenLLaMA-3B | Native Llama; RoPE; final RMSNorm | 3200 / 27 | float32 eager, TF32 off; authors’ slow Llama tokenizer; one explicit BOS, no EOS |
| Preserved Crystal | Pinned custom Crystal architecture; rotary positions; final ln_f | 4096 / 33 | Existing bfloat16 inference; CrystalCoderTokenizerFast; add_special_tokens=True, prepend_bos=False |

All saved hidden vectors are float32, including Crystal’s converted outputs. Inference precision is not the same across all models. L0 denotes embeddings and the last returned level includes final normalization. StarCoder L0 has exactly five distinct vectors, constant within each token-length group (30–34 tokens); it is not globally degenerate. OpenLLaMA L0 is constant and correctly has no meaningful PCA/kNN graph. Token lengths for OpenLLaMA are 32–36.

Pinned revisions: StarCoder `e1c5ef4ebb97afa0db09ec3e520f0487ca350bbe`; OpenLLaMA `141067009124b9c0aea62c76b3eb952174864057`; preserved Crystal `34fc9cd58acd87002560379a95b432147cc9135a`. Both new models use Transformers 4.40.2 and PyTorch 2.6.0 without quantization, generation, KV caching or remote model code.

StarCoder’s larger batches failed the original elementwise vector tolerance despite passing distance and neighbor checks. The supported mode was explicitly restricted to one unpadded prompt per forward and revalidated; no tolerance was relaxed. Its revised real smoke matched the single-prompt reference exactly at every level and passed reversed-order and partition checks. OpenLLaMA passed the original batch family 1/2/4/8/16/32, maximum-padding/reverse-order and partition checks. Batch 32 was fastest in its smoke and was used for its full run.

Each valid full layer uses exact float64 Euclidean kNN against all 10,000 candidates, stable point-ID tie breaking, four distinct non-self neighbors, and exactly 40,000 directed relations. PCA is centered, unscaled, unwhitened full SVD, fitted independently per layer; original-space neighbors are only displayed in PCA coordinates.

## Validation and preservation

- Remote audits passed full point/shape/finite-vector checks, PCA centering and projection consistency, orthonormal components, score variance, graph/union metrics, heatmap normalization, conditional denominators, and image checks.

- Local downloads passed every manifest artifact checksum: 1,096 StarCoder files and 993 OpenLLaMA files. Every saved directed-edge distance and union direction/mutual flag was independently checked. Exact neighbor ranks were independently recomputed for 16 selected queries per valid layer against all 10,000 candidates; this is a sampled rank audit, not a second complete all-query search.

- The independent metrics summary verifies every heatmap count/fraction and cross-length conditional denominator directly from saved neighbor IDs. All viewer coordinates exactly match saved PCA coordinates.

- Early, middle and final full-data graphs and final heatmaps were visually inspected. Both viewers passed target search, point-click selection, layer switching, all three color modes, edge modes and four-neighbor highlighting with no console errors. The gallery’s 510 links resolve. The locally linked comparison data equals the remote comparison data.

- Completed smoke repeats preserved 318 StarCoder and 236 OpenLLaMA checkpoint records and launched zero new extraction workers. The 40 common smoke/full prompts additionally passed the unchanged vector, cosine, distance and ordered-neighbor comparisons across one-worker and ten-worker execution. StarCoder was bitwise identical; OpenLLaMA maximum relative L2 difference was 2.684e-6. The neighbor comparison here uses the common 40-point candidate set; full graphs were audited separately.

- The actual interrupted OpenLLaMA full run recovered all 313 chunks, including one durable prepared receipt. All 626 hidden data/ID file hashes match the pre-recovery records. Recovery launched no GPU workers and recomputed zero target representations.

- Crystal’s manifest remains `18f7f35e702ab2e043a9cba4ffb8937c798d6f2dd08afc39b76485297017635f`; its source data and analysis were only read. The comparison used saved outputs.

Evidence: [StarCoder audit](star_full_download_audit.json), [OpenLLaMA audit](open_full_download_audit.json), [real smoke/full comparison](real_smoke_full_vector_comparison.json), [interruption recovery](open_actual_interruption_recovery_validation.json), [measured results](measured_results.json), and model-specific `*_visual_validation.json` files. Historical implementation validation recorded 56 passed and 1 optional skip; execution fixes passed 20 native-model tests, three restart-guard tests and three loader-focused tests, with logs in this directory. These test counts overlap and should not be added.

## Actual resources and timings

Verified profile/workspace `mdibrahimawad2`, environment `main`. Existing secret and both volumes were available and the secret could access both exact pinned checkpoints. The read-only workspace limits page showed 10 GPUs and 100 containers across environments. Separate aggregate CPU-core and memory quotas were not displayed. No account, license, billing or plan changes were made.

Both full runs used ten simultaneously observed L40S workers, each holding one complete model. Models ran sequentially, so total GPU concurrency never exceeded ten. Every worker’s saved runtime reports one model load. Analysis was capped at 31 workers × 2 cores, 6 GiB each, BLAS threads 2. Plotting used up to 16 workers × 1 core, 4 GiB each, BLAS threads 1. The coordinator used one core and 8 GiB; its non-preemptible CPU/memory are billed at 3× list rates. The conservative cap including every residual worker class is 99 cores; idle workers were not reserved to reach 100.

| Measurement | StarCoder | OpenLLaMA |
|---|---:|---:|
| Observed GPU workers | 10 | 10 |
| Peak reported analysis inputs in flight | 31 | 26 |
| Peak sampled analysis containers | 29 | 26 |
| Peak sampled cores in containers, including coordinator | 59 | 53 |
| Observed plot workers | 16 | 16 |
| Median layer kNN time | 124.22 s | 133.93 s |
| Median layer PCA time | 13.46 s | 17.43 s |
| Full-run app elapsed time | 27m 46s | 23m 56s across extraction/recovery/finish apps |
| Static PNG figures | 158 | 116 |

Input and container counters are asynchronous snapshots, not CPU-utilization measurements. StarCoder’s requested analysis allocation was 63 cores including coordinator; the 20-second samples observed up to 59 cores in containers. OpenLLaMA’s constant L0 finished quickly, leaving 26 nondegenerate layers. OpenLLaMA elapsed time sums its three full-run apps and excludes gaps between them; all times exclude initial model downloads and smoke debugging.

StarCoder model-load median: 7.46 s; extraction-plus-checkpoint worker median 101.79 s over all ten workers. OpenLLaMA model-load median: 5.40 s; worker median 81.46 s over nine completed benchmark records. Its interrupted tenth worker has no final benchmark, though its representations were recovered. These are per-worker times, not total parallel wall time.

Peak allocated CUDA memory was 11.53 GiB for StarCoder and 13.43 GiB for OpenLLaMA. Peak observed layer-process RSS was 1.55 and 1.73 GiB respectively, below the 6-GiB worker allocation. StarCoder L0 full-SVD PCA took 205.10 s, much longer than its typical layer. No scientific shortcut was substituted.

## Observations, not causal conclusions

| Final returned level | Same-leading-digit relations | Cross-digit-length relations | Same digit conditional on cross length | Conditional denominator |
|---|---:|---:|---:|---:|
| LLM360/Crystal L32 | 99.9050% | 3.0450% | 99.1790% | 1,218 |
| bigcode/starcoderbase-3b L36 | 99.9900% | 2.4675% | 100.0000% | 987 |
| openlm-research/open_llama_3b L26 | 99.7525% | 6.7400% | 99.9629% | 2,696 |

All three final layers strongly favor same-leading-digit neighbors. Their depth profiles and PCA geometry differ. The two displayed PCs capture only about 17.85% of StarCoder’s final variance and 17.95% of OpenLLaMA’s, so projected overlap does not imply weak original-space neighbor structure. Conditional fractions must be read alongside their denominators.

These associations do not isolate a causal effect of tokenization. Architecture, training, tokenizer policy, layer count, width and inference precision differ; absolute PCA coordinates and raw distance scales are not shared. The paper’s tokenization controls have not been reproduced, and exact paper-checkpoint correspondence remains unverified.

## Fixes, recovery, and commands

The cached-model loader was corrected to load the pinned local snapshot directory, avoiding Transformers’ remote safetensors probe. Empty failed attempts were archived without discarding representations; saved prompt records were reused. StarCoder’s unsupported larger-batch mode was narrowed and revalidated as described above. A real Modal internal coordinator retry was stopped before replacement extraction workers started; recovery adopted durable receipts. The coordinator now commits a launch marker so automatic retries fail closed, while explicit resume requires the prior app to be stopped. A separate `finish` stage can plot, validate and package without entering inference or analysis.

The download destination must already be a directory, and the Modal CLI uses `--force` only for these dedicated local copies. One comparison download was retried after the CLI treated a nonexistent destination as a file; remote outputs were unchanged.

Exact executed smoke, full, resume and plotting-only commands are in [EXECUTION_RUNBOOK.md](EXECUTION_RUNBOOK.md). OpenLLaMA must resume from its frozen execution source directory; StarCoder uses the repository root. No compatibility hash was changed to force mixed scientific outputs.

All experiment and comparison apps are stopped with zero tasks; [final app validation and elapsed times](final_app_validation.json). App IDs and exact arguments are in [execution.json](execution.json). Main dashboards:

- [StarCoder smoke · ap-mogeK0F6hhrGVJn05qFL3E](https://modal.com/apps/mdibrahimawad2/main/ap-mogeK0F6hhrGVJn05qFL3E)

- [StarCoder full · ap-4XWMguKifXLZzfeLsCrC6I](https://modal.com/apps/mdibrahimawad2/main/ap-4XWMguKifXLZzfeLsCrC6I)

- [OpenLLaMA smoke · ap-9MmbXStEXByhmqaKRcpT10](https://modal.com/apps/mdibrahimawad2/main/ap-9MmbXStEXByhmqaKRcpT10)

- [OpenLLaMA full extraction · ap-UiK2cqldByFcjQXNEX4UVL](https://modal.com/apps/mdibrahimawad2/main/ap-UiK2cqldByFcjQXNEX4UVL)

- [OpenLLaMA analysis recovery · ap-vicQl2hCNXpSQZBezS70PZ](https://modal.com/apps/mdibrahimawad2/main/ap-vicQl2hCNXpSQZBezS70PZ)

- [OpenLLaMA finish · ap-1VHFip8pHe2SboEb9s75oR](https://modal.com/apps/mdibrahimawad2/main/ap-1VHFip8pHe2SboEb9s75oR)

- [Saved comparison · ap-gzPgTqtLqoLgHlHWM44Fzu](https://modal.com/apps/mdibrahimawad2/main/ap-gzPgTqtLqoLgHlHWM44Fzu)
