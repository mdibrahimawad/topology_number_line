# PH acceleration — verified results

The new GPU-capable backend completes Crystal layer 7, which previously timed out. Every calculation still uses all 10,000 targets and the saved original-vector Euclidean distances.

| Calculation | Hardware | Measured PH time | Result |
|---|---|---:|---|
| Crystal L1, original giotto-ph path | 4 CPUs, 16 GiB | 146.9 seconds | Complete |
| Crystal L1, Ripser++ path | 1 L40S, 8 CPUs, 32 GiB | 52.6 seconds | All 12,634 bars exactly match the CPU result |
| Crystal L7, original path | 4 CPUs, 16 GiB | Worker timed out at 900 seconds | H0 saved; H1 unfinished |
| Crystal L7, Ripser++ path | 1 L40S, 8 CPUs, 32 GiB | 133.5 seconds | 24,712 finite H1 bars |
| Crystal L7, smaller allocation | 1 L40S, 4 CPUs, 12 GiB | **129.2 seconds** | All 24,712 bars exactly match the larger allocation |

The smaller L7 worker took 139.8 seconds including loading, validation and saving. Peak process memory was about 6.17 GiB. These are single-run measurements, not repeated benchmark medians; the small 133-to-129-second difference is not evidence that four CPUs are inherently faster than eight. It shows that this smaller allocation was sufficient in the tested run.

[Open the new Crystal L7 graph](/Users/mohammed.awad/Documents/Codex/2026-09-19/c/outputs/ph_acceleration/crystal_layer07_persistence.png) · [Gallery](/Users/mohammed.awad/Documents/Codex/2026-09-19/c/outputs/ph_acceleration/index.html) · [Validation](/Users/mohammed.awad/Documents/Codex/2026-09-19/c/outputs/ph_acceleration/validation.json)

## What changed

Added an isolated Ripser++ backend at revision `30243c0c752de26d7fdf6e41f08bf7b840ca4744`, compiled with CUDA 11.8 for L40S. The existing CPU pipeline and original model results remain unchanged. The adapter sends a contiguous float32 lower-triangle buffer to the upstream native entry point, avoiding the upstream Python binding's expansion of almost 50 million distances into Python objects. It preserves the same rounded distances.

The backend still calculates full-point, full-range Vietoris–Rips H1 over F2. No number was dropped; no PCA coordinates or kNN graph were substituted. The original float64-distance H0 for L7 is paired with the new H1 in the figure, using their identical layer scale and source identity.

18 local tests passed. Remote checks covered a circle and line, agreement with the official Python binding, independent Ripser checks on the fixed 128-point real-data subset, a full 10,000-point CPU/GPU comparison for L1, and full L7 equality across the two GPU resource settings. Downloaded artifacts were rehashed. There is no completed full CPU reference for L7, so its full CPU/GPU equality has not been established.

## Resource conclusion

For a future bulk run, **one L40S + four CPUs + 12 GiB per layer worker** is the measured starting point. Up to ten such workers would request ten GPUs, forty CPU cores and 120 GiB RAM, plus separate preparation/coordinator resources, subject to actual workspace quotas and availability. The ten-worker scheduling and throughput have not yet been implemented or tested in the bulk pipeline; the present launcher is a bounded single-GPU benchmark.

More CPU cores did not improve this tested layer materially. GPU activity was bursty: the first run's 71 samples, taken roughly three seconds apart, recorded peak utilization 14%, mean 0.45%, and peak GPU memory 3,503 MiB. Sampling can miss short peaks. The speed comparison changes both backend and hardware; it does not isolate acceleration caused by GPU hardware alone. A sensible next throughput test would check whether several CPU-heavy layer processes can share a GPU before allocating all ten. Do not promise linear tenfold scaling or maximum efficiency from this pilot.

[Ripser++ documentation](https://github.com/simonzhang00/ripser-plusplus/blob/30243c0c752de26d7fdf6e41f08bf7b840ca4744/README.md) explains that GPU work covers filtration construction and apparent-pair processing while residual reduction remains on CPU. [Modal pricing](https://modal.com/pricing) lists resource charges and the Starter plan's 100-container/10-GPU concurrency allowance; 100 containers is not a guarantee of 100 CPU cores.

## Run and budget status

Both GPU tests passed and stopped. All PH apps are verified stopped with zero active tasks. The full 97-level run and a ten-GPU batch have **not** been launched. Four distinct layers now have complete H0/H1 results: Crystal L1 and L7, StarCoder L1, and OpenLLaMA L1.

Cumulative pilot reservations, including the earlier CPU attempts, are **$2.984966875 of the $3 pilot allowance**. This is conservative reserved compute, not actual Modal billing; image builds, storage and other apps are excluded. No model inference, model-weight downloads, credit purchases or account changes were performed.

Evidence is saved under `ph_acceleration/`: both run reports, diagrams, metrics, manifests, GPU samples, final app statuses and exact comparison results. Large preprocessing checkpoints remain on the existing Modal volume. The original CPU pilot report remains historical evidence; its L7 timeout is now resolved by this GPU-backend follow-up.
