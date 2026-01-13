# RD-Agent → MLE-bench (native, no Docker)

This repo includes a **native** runner that executes RD-Agent’s DataScience loop directly on **MLE-bench prepared datasets**, with **AIDE-style observability** (tree HTML + selection-method submissions + grading).

## Prereqs

- A sibling RD-Agent checkout at: `/home/ka3094/RD-Agent`
- A conda env named `kaggle` with RD-Agent deps installed:
  - Python interpreter: `/home/ka3094/miniconda3/envs/kaggle/bin/python`
- MLE-bench dataset prepared under `data/competitions/<competition_id>/prepared/...`:
  - `mlebench prepare -c <competition_id> --data-dir data/competitions`

## Run (single competition)

```bash
source /home/ka3094/.aide_env
/home/ka3094/miniconda3/envs/kaggle/bin/python agents/rdagent_ds/run_native.py \
  --competition-id house_prices \
  --data-dir /home/ka3094/mle-bench/data/competitions \
  --variant bug-consultant \
  --loop-n 3
```

Variants (RD-Agent “3 experiments”):
- `post-search-on`
- `plan-constraints`
- `bug-consultant`

## Outputs (what to open)

Each run writes under `native_runs/rdagent_ds/<timestamp>_<competition>_<variant>/`.

Key files/folders:
- `log/tree_plot.html` (AIDE-like tree; includes per-node code/stdout/feedback)
- `log/console.log` (full captured stdout/stderr from the RD-Agent process)
- `submission/submission.csv` (the auto-selected submission copied from the chosen workspace)
- `submission/selections/selection_summary.html` (links to every selection method output)
- `submission/selections/*/submission.csv` (per-method submission CSVs)
- `submission/selections/grading.csv` (MLE-bench grading for each selection method)
- `rd_workspace/<uuid>/...` (all RD-Agent workspaces; contains `main.py`, `stdout.txt`, `scores.csv`, etc.)

Selection methods emitted (when possible):
- `auto`
- `best_valid`
- `maximin`
- `elite_maximin`
- `mean_minus_k_std`
- `maximin_no_filter`

## Run the 3 experiments over a competition list

```bash
source /home/ka3094/.aide_env
DATA_DIR=/home/ka3094/mle-bench/data/competitions \
LOOP_N=3 \
PYTHON_BIN=/home/ka3094/miniconda3/envs/kaggle/bin/python \
bash run_rdagent_native_experiments.sh experiments/splits/dev.txt
```

## VS Code debug

Use the debug configs in `/home/ka3094/.vscode/launch.json`:
- `MLE-Bench: RD-Agent native (house_prices, 3 loops, post-search-on)`
- `MLE-Bench: RD-Agent native (house_prices, 3 loops, plan-constraints)`
- `MLE-Bench: RD-Agent native (house_prices, 3 loops, bug-consultant)`

