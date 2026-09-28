#!/bin/sh
# Claim C2: 128-layer ReLU muPC nets against BP with Depth-muP (Figs. A.17 to A.19),
# with per-layer weight change logged at every test point (added-value axis).
# Settings from the official plotting notebook.
set -e
cd "$(dirname "$0")"
. ./env.sh
lean_ok || { echo "lean code not accepted (results/lean_vs_official.json)"; exit 1; }
L=runs/long
$PY lean_mupc.py --out $L/mupc_MNIST --dataset MNIST --n_hidden 128 --param_lr 1e-1 \
  --activity_lr 5e-1 --batch_size 64 --max_epochs 5 --test_every 300 --seeds 0 1 2 3 4 \
  --log_weights >> logs/long.log 2>&1
$PY lean_mupc.py --out $L/mupc_Fashion-MNIST --dataset Fashion-MNIST --n_hidden 128 \
  --param_lr 5e-2 --activity_lr 5e-1 --batch_size 64 --max_epochs 15 --test_every 900 \
  --seeds 0 1 2 --log_weights >> logs/long.log 2>&1
$PY lean_mupc.py --out $L/mupc_CIFAR10 --dataset CIFAR10 --loss ce --n_hidden 128 \
  --param_lr 1e-1 --activity_lr 1e-2 --batch_size 128 --max_epochs 20 --test_every 389 \
  --seeds 0 1 2 --log_weights >> logs/long.log 2>&1
cd "$JPC"
B=$OLDPWD/$L/bp
$PY $OLDPWD/run_bpn.py --results_dir $B --dataset MNIST --loss_id mse \
  --n_hidden 128 --lrs 5e-3 --batch_size 64 --max_epochs 5 --test_every 300 --n_seeds 5 >> $OLDPWD/logs/long.log 2>&1
$PY $OLDPWD/run_bpn.py --results_dir $B --dataset Fashion-MNIST --loss_id mse \
  --n_hidden 128 --lrs 5e-3 --batch_size 64 --max_epochs 15 --test_every 900 --n_seeds 3 >> $OLDPWD/logs/long.log 2>&1
$PY $OLDPWD/run_bpn.py --results_dir $B --dataset CIFAR10 --loss_id ce \
  --n_hidden 128 --lrs 1e-2 --batch_size 128 --max_epochs 20 --test_every 389 --n_seeds 3 >> $OLDPWD/logs/long.log 2>&1
touch $OLDPWD/$L/DONE
