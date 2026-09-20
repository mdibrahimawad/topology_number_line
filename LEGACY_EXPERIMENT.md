# Number-Line ZigZag Persistence on Modal

A self-contained experiment repo for four base LLMs:

- `falcon-rw-1b` -> `tiiuae/falcon-rw-1b`
- `falcon-rw-7b` -> `tiiuae/falcon-rw-7b`
- `btlm-3b-8k` -> `cerebras/btlm-3b-8k-base`
- `llm360-crystal` -> `LLM360/Crystal`

The scientific rule is simple: **numerical prompts and hidden-state extraction come from the user's number-line code; ZigZag PH comes from the uploaded ZigZagLLMs methodology.** PCA-2D is visualization only.

## What one run does

```text
sample 4 x 30 numerical targets
        |
        v
build the same n1=n1,...,n= prompts as the existing paper code
        |
        v
feed each prompt to one pretrained LLM
        |
        v
extract last-token hidden vector at EVERY hidden-state index
        |
        +------------------------------+
        |                              |
        v                              v
PCA branch                         topology branch
PC1 -> EV, |rho|, beta            full-dimensional vectors
PCA2 -> pictures                       |
                                        v
                              kNN graph at each layer
                                        |
                                        v
                              clique/simplicial complex
                                        |
                                        v
                           adjacent-layer intersections
                                        |
                                        v
                              ZigZag persistent homology
                                        |
                +-----------------------+--------------------+
                v                       v                    v
             barcode              birth/death        persistence image
                                        |
                                        v
                              inter-layer persistence
```

## Outputs per model and prompt seed

`hidden_states.npz` contains the raw float32 tensor `[layers, points, hidden_dim]` plus targets/groups. The repo also writes prompt metadata, tokenization diagnostics, PCA layer metrics, H0-H3 interval CSVs for every tested k, and final figures at one globally selected k.

The final experiment also writes `model_summaries/` with mean/SD across prompt seeds, matching the multi-run spirit of the original geometry analysis.

Final figures include:

- `pca2d_knn_all_layers.png` — **all transformer layers in one grid**; points are PCA-2D for display, while kNN edges come from the original full-dimensional hidden vectors.
- `combined_pca_knn_barcode_H1.png` — PCA/kNN snapshots above the H1 barcode.
- `barcode_H1.png`.
- `birth_death_H1.png`.
- `effective_persistence_H1.png`.
- `interlayer_persistence_H1_alpha0.png`.
- layer-wise EV, |rho|, and beta plots.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements-local.txt
modal setup
```

Create a Modal secret named `numberline-hf-token` containing your Hugging Face token under the key `HF_TOKEN`:

```bash
modal secret create numberline-hf-token HF_TOKEN="YOUR_HF_TOKEN"
```

Do not put the token in this repository.

## First run: dry plan

```bash
modal run modal_app.py --dry-run
```

## Recommended smoke test

This uses 5 points/group, one seed, and k=2..4:

```bash
modal run modal_app.py --smoke-test
```

Download the smoke outputs:

```bash
modal volume get numberline-zigzag-results /main_smoke results/main_smoke
```

Inspect `extraction_diagnostics.json`, `topology_scan_summary.json`, and the figures before launching the full run.

## Full four-model experiment

```bash
modal run modal_app.py \
  --models all \
  --experiment-name main \
  --runs 3 \
  --samples-per-group 30 \
  --num-examples 3 \
  --seed 42 \
  --context random \
  --knn-min 1 \
  --knn-max 15 \
  --max-simplex-dim 4
```

There are 4 models x 3 prompt seeds = 12 GPU extraction jobs. Modal is capped at 10 concurrent GPU containers, so up to ten can run at once. Each job uses one GPU; using ten GPUs inside one 7B inference container would not help this workload.

After extraction, topology is computed on CPU from the saved vectors. This lets you rerun PH/plots later without loading the LLM again.

Download everything:

```bash
modal volume get numberline-zigzag-results /main results/main
```

List remote results without downloading:

```bash
modal volume ls numberline-zigzag-results /main
```

## Run only selected models

```bash
modal run modal_app.py --models falcon-rw-1b,btlm-3b-8k --experiment-name pilot
```

Allowed aliases are fixed in `numzig/config.py`. Arbitrary model IDs are intentionally rejected because three of these models require `trust_remote_code=True`.

## Validation

```bash
python scripts/validate_repo.py
pytest -q
```

The repository ships with deterministic prompt tests, PCA/beta tests, and an optional synthetic ZigZag smoke test when `gudhi` and `dionysus` are installed locally.

See `PROVENANCE.md` for exactly what was copied, what was adapted, and what was an engineering-only choice.
