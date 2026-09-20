> Follow-up: GPU-backed PH now completed Crystal L7 in 129 seconds. See PH_ACCELERATION.md. The CPU pilot status below is historical.

# Persistent-homology validation and pilot status — 2026-09-19

The authorized CPU pilot has run and stopped. Performance gate: FAILED. Complete H0/H1 for Crystal L1, StarCoder L1 and OpenLLaMA L1, each with all 10,000 targets. Crystal L7 H0 is complete; H1 timed out after 900 seconds. The remaining eight pilot layers and the full 97-level run were not launched. All three pilot apps are stopped with zero tasks.

17 local tests passed. Completed diagrams, normalization, source identities and artifact hashes passed validation. Sampled independent backend comparisons had zero bottleneck distance. All four saved MSTs had every edge distance checked against original local vectors. The three completed figures were inspected visually. Full-point H1 was not independently recomputed with another backend.

Direct giotto-ph replaced optional edge collapse after real subset benchmarks demonstrated its overhead. New outputs use /ph_fullrange_v1/direct_rips, with whitelisted, fully hashed read-only reuse of four legacy input/distance checkpoints. Core source SHA256: 668849c59107f005ec1cf517c0b379790c9b75d4d65ca933908a5fcf3f39ec0e. No original inference output was changed. No new model inference, model weights or GPUs were used.

Cumulative reservations: $1.6204755 / $3 pilot allowance. This is not verified billing or an account hard cap. Current settings are not validated for a full run. Do not automatically launch all levels or change the scientific point set.

Execution evidence and downloaded diagrams: results/ph_execution_20260919/. Machine-readable final report: direct_lightweight/pilot_report.json; validation: direct_lightweight/validation.json; app status: final_apps.json. Full user-facing report and gallery: /Users/mohammed.awad/Documents/Codex/2026-09-19/c/outputs/PH_PILOT_RESULTS.md and outputs/ph_pilot/index.html.
