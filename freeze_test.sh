#!/bin/sh
# Partial-freeze test (post hoc; rule in freeze_summary.py, fixed before the runs).
# MNIST, H = 128, notebook cell, seeds 0 to 4, then the one-layer muPC control on
# the GPU (the first run was on the CPU), then CIFAR-10 at the C2 settings.
set -e
cd "$(dirname "$0")"
. ./env.sh
F=runs/freeze
M="--n_hidden 128 --param_lr 1e-1 --activity_lr 5e-1 --seeds 0 1 2 3 4 --log_weights --log_grads"
$PY lean_mupc.py --out $F/mupc_full $M >> logs/freeze.log 2>&1
$PY lean_mupc.py --out $F/mupc_frz1-121 --freeze_range 1 121 $M >> logs/freeze.log 2>&1
$PY lean_mupc.py --out $F/mupc_frz122-127 --freeze_range 122 127 $M >> logs/freeze.log 2>&1
cd "$JPC"
B="--dataset MNIST --loss_id mse --n_hidden 128 --act_fns relu --param_type depth_mup --lrs 5e-3 --batch_size 64 --max_epochs 1 --test_every 300 --n_seeds 5"
$PY $OLDPWD/run_bpn.py --results_dir $OLDPWD/$F/bp_full $B >> $OLDPWD/logs/freeze.log 2>&1
$PY $OLDPWD/run_bpn.py --results_dir $OLDPWD/$F/bp_frz1-121 --freeze_range 1 121 $B >> $OLDPWD/logs/freeze.log 2>&1
cd "$OLDPWD"
$PY lean_mupc.py --out runs/added/mupc_H1_gpu --n_hidden 1 --max_infer_iters 8 \
  --param_lr 1e-1 --activity_lr 1e-1 --seeds 0 1 2 >> logs/freeze.log 2>&1
touch $F/DONE
$PY freeze_summary.py results/freeze.json >> logs/freeze.log 2>&1
C=runs/freeze_cifar
$PY lean_mupc.py --out $C/mupc_frz1-121 --freeze_range 1 121 --dataset CIFAR10 --loss ce --n_hidden 128 \
  --param_lr 1e-1 --activity_lr 1e-2 --batch_size 128 --max_epochs 20 --test_every 389 \
  --seeds 0 1 2 --log_weights >> logs/freeze.log 2>&1
cd "$JPC"
$PY $OLDPWD/run_bpn.py --results_dir $OLDPWD/$C/bp_frz1-121 --freeze_range 1 121 --dataset CIFAR10 --loss_id ce \
  --n_hidden 128 --act_fns relu --param_type depth_mup --lrs 1e-2 --batch_size 128 --max_epochs 20 \
  --test_every 389 --n_seeds 3 >> $OLDPWD/logs/freeze.log 2>&1
cd "$OLDPWD"
touch $C/DONE
$PY freeze_summary.py results/freeze.json >> logs/freeze.log 2>&1
