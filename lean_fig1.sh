#!/bin/sh
# Claim C1 with lean_mupc.py: the same grid as fig1_relu.sh (muPC and SP PC,
# H 8 to 128, width 512, MNIST, 1 epoch, seeds 0 to 2), then the official BP
# Depth-muP baseline at H = 128. compare_lean.py decides whether the lean code
# is accepted; if it is, the C2 and C3 jobs are queued.
set -e
cd "$(dirname "$0")"
. ./env.sh
for H in 8 16 32 64 128; do
  for PT in mupc sp; do
    if [ "$PT" = mupc ]; then ALR=5e-1; else ALR=1e-2; fi
    $PY lean_mupc.py --out runs/lean_fig1/${PT}_H${H} --n_hidden $H --param_type $PT \
      --param_lr 1e-1 --activity_lr $ALR --seeds 0 1 2 >> logs/lean_fig1.log 2>&1
  done
done
touch runs/lean_fig1/DONE
if [ -z "$(find runs/fig1/bp -name test_accs.npy 2>/dev/null)" ]; then
  (cd "$JPC" && $PY $OLDPWD/run_bpn.py --results_dir $OLDPWD/runs/fig1/bp \
    --dataset MNIST --loss_id mse --n_hidden 128 --act_fns relu --param_type depth_mup \
    --lrs 5e-3 --batch_size 64 --max_epochs 1 --test_every 300 --n_seeds 3) >> logs/lean_fig1.log 2>&1
fi
$PY compare_lean.py >> logs/lean_fig1.log 2>&1
if lean_ok; then
  ./submit_next.sh added-value 4 300 added_value.sh >> logs/lean_fig1.log 2>&1
  ./submit_next.sh long-runs 4 300 long_runs.sh >> logs/lean_fig1.log 2>&1
  ./submit_next.sh transfer-relu 4 420 transfer_relu.sh >> logs/lean_fig1.log 2>&1
fi
