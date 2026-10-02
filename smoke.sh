#!/bin/sh
# Smoke run of the official muPC script at the pinned commit: MNIST, H=8, 1 epoch, one lr pair, seed 0.
set -e
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$(dirname "$0")/../vendor/upstream_jpc"
SMOKE_RESULTS="${TMPDIR:-.}/mupc_smoke"
mkdir -p "$SMOKE_RESULTS"
START=$(date +%s)
../.venv-pinned/bin/python -m experiments.mupc_paper.train_pcn_no_metrics \
  --results_dir "$SMOKE_RESULTS" --datasets MNIST --n_hiddens 8 --act_fns relu \
  --param_lrs 1e-1 --activity_lrs 5e-1 --batch_size 64 --max_infer_iters 8 \
  --max_epochs 1 --test_every 300 --n_seeds 1 > ../logs/smoke.log 2>&1
echo "{\"smoke_seconds\": $(( $(date +%s) - START ))}" > ../results/smoke.json
