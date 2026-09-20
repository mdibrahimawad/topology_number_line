# Run, resume and retrieve this experiment

Repository: `/Users/mohammed.awad/Documents/numberline_zigzag_modal_FINAL`

The authorized workspace is `mdibrahimawad2`, environment `main`. Current full-run namespace: `taskrules_20260920_v3`. Existing Crystal/StarCoder/OpenLLaMA results are separate and untouched.

## Resume only after the previous app has stopped

```bash
cd /Users/mohammed.awad/Documents/numberline_zigzag_modal_FINAL
MODAL_PROFILE=mdibrahimawad2 .venv/bin/python -m modal app list --env main
MODAL_PROFILE=mdibrahimawad2 .venv/bin/python -m modal run --env main --detach modal_taskrules.py --stage all
```

The same command validates and skips completed extraction chunks and per-layer analyses. It preserves the compute ledger; restarting does not reset its allowance. Incompatible datasets/backends fail closed instead of overwriting old vectors.

CPU-only alternatives, which never trigger model inference:

```bash
MODAL_PROFILE=mdibrahimawad2 .venv/bin/python -m modal run --env main --detach modal_taskrules.py --stage analyze
MODAL_PROFILE=mdibrahimawad2 .venv/bin/python -m modal run --env main --detach modal_taskrules.py --stage compare
MODAL_PROFILE=mdibrahimawad2 .venv/bin/python -m modal run --env main --detach modal_taskrules.py --stage package
MODAL_PROFILE=mdibrahimawad2 .venv/bin/python -m modal run --env main --detach modal_taskrules.py --stage audit
```

If a complete gallery is already saved and only export was interrupted, use `--stage package_fast`. This checks every referenced output and exports it with 16 bounded file readers; it does not rebuild graphs or run inference. Use ordinary `--stage package` when the gallery needs rebuilding.

The audit stage validates all analysis/comparison file hashes and independently recomputes sampled original-space neighbors and PCA coordinates. It saves `validation.json` and never loads a model. Run it after analysis, comparisons and packaging have completed.

## Retrieve outputs

```bash
MODAL_PROFILE=mdibrahimawad2 .venv/bin/python -m modal volume get --env main numberline-zigzag-results /taskrules_20260920_v3/lightweight.tar.gz results/taskrules_lightweight.tar.gz
```

The lightweight archive contains plots, coordinates, PH intervals, metadata and behavior records. Full hidden vectors remain in the same Modal volume under `/taskrules_20260920_v3/<model>/hidden/`. Download the complete namespace when needed:

```bash
MODAL_PROFILE=mdibrahimawad2 .venv/bin/python -m modal volume get --env main numberline-zigzag-results /taskrules_20260920_v3 results/taskrules_complete
```

A local web server is needed for interactive gallery data loading. Serve the extracted directory and open its `index.html`.

## Prompt selection and pilots

`taskrules_20260920_v2/pilot` preserves the example-only behavioral pilot. `taskrules_20260920_v2/pilot_instruction` preserves the explicit-instruction pilot. V3 derives its matching pilot rows from those hash-verified artifacts without new inference: instruction prompts for the four-digit tasks and example-only prompts for numeric/word copying.

The main study has 1,000 sampled four-digit targets, not all 9,000 four-digit numbers. The word/numeric cohort includes every integer from 1 through 1,000. Two fixed contexts are analyzed separately. Poor behavioral performance remains visible; it is not relabeled as successful rule execution.

## Cost accounting

The $14.50 conservative compute allowance includes carried failed reservations, GPU/CPU/RAM rates, timeout reservations and startup margins. It is not an exact bill or an account-level hard limit. No billing settings changed. Rates were checked against [Modal pricing](https://modal.com/pricing): L40S $0.000542/second, physical CPU core $0.0000131/second and memory $0.00000222/GiB/second.
