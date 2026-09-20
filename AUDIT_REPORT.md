# Security and quality audit

## Pass 1 — static security + repository integrity

Status: **PASS**.

Checks performed by `scripts/validate_repo.py`:

- scanned Python files for Hugging Face/AWS-style embedded secrets;
- rejected direct `eval()` / `exec()` usage;
- verified the fixed model allowlist exists before any `trust_remote_code=True` use;
- verified all required pipeline files are present;
- compiled every Python source file for syntax validity.

The generated machine-readable result is `VALIDATION_REPORT.json`.

## Pass 2 — methodology equivalence + unit tests

Status: **PASS for all tests runnable in this environment**.

Results: `6 passed, 1 skipped`.

The tests verify:

- deterministic numerical prompt generation;
- exact prompt-sampling equivalence against the uploaded `utils/prompts.py` snapshot;
- exact direct-beta fit equivalence against the uploaded `spacing_fit.py` snapshot;
- the ZigZag filtration-time offset convention used by the uploaded `fclaux.ranges` implementation;
- PCA layer metrics and best-layer logic;
- repository import/syntax behavior.

The one skipped test is the end-to-end synthetic Dionysus/Gudhi ZigZag smoke test because those two packages are not installed in this offline build environment. The Modal image pins the exact versions from the uploaded ZigZagLLMs environment (`gudhi==3.9.0`, `dionysus==2.0.10`).

## Runtime validity checks built into the experiment

The experiment additionally fails rather than silently continuing when hidden-state tensors have wrong rank, non-finite values, duplicated point IDs, inconsistent point metadata, invalid kNN settings, missing H1 outputs, or inconsistent tensor shapes across prompt-seed replicates.

## Remaining runtime check

Actual Hugging Face model loading, CUDA extraction, and Dionysus execution must be verified on Modal because this local environment has no internet/GPU dependency installation. Run the supplied `--smoke-test` first; only after it completes and produces the expected diagnostics/figures should the full 4-model run be launched.
