#!/bin/sh
# C1, standard parameterisation by the original's method (mupc paper A.4: "the same
# grid search over the weight and activity learning rates as used for muPC"):
# the 4 x 11 grid at H = 8 to 128, N = 512, seeds 0 to 2, MNIST, one epoch,
# selected by minimum training loss as for C3.
set -e
cd "$(dirname "$0")"
. ./env.sh
PLR="5e-1 1e-1 5e-2 1e-2"
ALR="1e3 5e2 1e2 5e1 1e1 5 1 5e-1 1e-1 5e-2 1e-2"
for H in 8 16 32 64 128; do
  $PY lean_mupc.py --out runs/sp_grid/depth_N512_H${H} --param_type sp --width 512 --n_hidden $H \
    --param_lr $PLR --activity_lr $ALR --test_every 100 >> logs/sp_grid.log 2>&1
done
touch runs/sp_grid/DONE
$PY sp_summary.py >> logs/sp_grid.log 2>&1
