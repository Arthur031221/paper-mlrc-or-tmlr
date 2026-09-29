#!/bin/sh
# Official code, muPC at H = 64 (notebook cell, seeds 0 to 2), to test the
# reimplementation at a depth beyond the H <= 32 agreement check.
set -e
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$(dirname "$0")/../vendor/upstream_jpc"
PY=$(cd .. && pwd)/.venv-pinned/bin/python
$PY -m experiments.mupc_paper.train_pcn_no_metrics --results_dir ../../code/runs/fig1/pc --datasets MNIST \
  --n_hiddens 64 --act_fns relu --param_types mupc --param_lrs 1e-1 --activity_lrs 5e-1 \
  --batch_size 64 --max_infer_iters 64 --max_epochs 1 --test_every 300 --n_seeds 3 \
  >> ../../code/logs/official_h64.log 2>&1
touch ../../code/runs/fig1/H64_DONE
cd ../../code && $PY official_h64_summary.py >> logs/official_h64.log 2>&1
