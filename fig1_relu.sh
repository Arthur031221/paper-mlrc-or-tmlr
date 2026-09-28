#!/bin/sh
# Claim C1 (Fig. 1 right, ReLU): muPC and SP PC over depth, and BP with Depth-muP at H=128.
# MNIST, width 512, batch 64, T = H inference steps, 1 epoch, test every 300 iterations, seeds 0, 1, 2.
# Learning rates are the paper's selected values.
set -e
# JAX would otherwise take 75% of the card; the queue budgets memory per job.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$(dirname "$0")/../vendor/upstream_jpc"
PY=../.venv-pinned/bin/python
OUT=../runs/fig1
for H in 8 16 32 64 128; do
  for PT in mupc sp; do
    if [ "$PT" = mupc ]; then ALR=5e-1; else ALR=1e-2; fi
    # Resume: skip a configuration whose three seeds already finished.
    N=$(find $OUT/pc -path "*/${H}_n_hidden/*/${PT}_param/*" -name test_accs.npy 2>/dev/null | wc -l)
    [ "$N" -ge 3 ] && continue
    $PY -m experiments.mupc_paper.train_pcn_no_metrics --results_dir $OUT/pc --datasets MNIST \
      --n_hiddens $H --act_fns relu --param_types $PT --param_lrs 1e-1 --activity_lrs $ALR \
      --batch_size 64 --max_infer_iters $H --max_epochs 1 --test_every 300 --n_seeds 3 \
      >> ../logs/fig1.log 2>&1
  done
done
$PY -m experiments.mupc_paper.train_bpn --results_dir $OUT/bp --dataset MNIST --loss_id mse \
  --n_hidden 128 --act_fns relu --param_type depth_mup --lrs 5e-3 --batch_size 64 \
  --max_epochs 1 --test_every 300 --n_seeds 3 >> ../logs/fig1.log 2>&1
touch ../runs/fig1/DONE
