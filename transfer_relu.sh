#!/bin/sh
# Claim C3 (Fig. A.32, ReLU): minimum training loss over the paper's 4 x 11 grid of
# (weight lr, activity lr), 3 seeds, MNIST, one epoch; widths 64 to 1024 at H = 8
# (test every 300, as the notebook) and depths 8 to 128 at N = 512 (test every 100).
set -e
cd "$(dirname "$0")"
. ./env.sh
lean_ok || { echo "lean code not accepted (results/lean_vs_official.json)"; exit 1; }
PLR="5e-1 1e-1 5e-2 1e-2"
ALR="1e3 5e2 1e2 5e1 1e1 5 1 5e-1 1e-1 5e-2 1e-2"
for N in 64 128 256 512 1024; do
  $PY lean_mupc.py --out runs/transfer/width_N${N}_H8 --width $N --n_hidden 8 \
    --param_lr $PLR --activity_lr $ALR --test_every 300 >> logs/transfer.log 2>&1
done
for H in 8 16 32 64 128; do
  $PY lean_mupc.py --out runs/transfer/depth_N512_H${H} --width 512 --n_hidden $H \
    --param_lr $PLR --activity_lr $ALR --test_every 100 >> logs/transfer.log 2>&1
done
touch runs/transfer/DONE
