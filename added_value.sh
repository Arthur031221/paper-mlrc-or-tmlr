#!/bin/sh
# Added-value arms (decision rule stated in the paper, fixed before any run): does muPC at
# H = 128 use its depth? MNIST, one epoch, batch 64, seeds 0 to 2.
#  (a) muPC H = 128 with the hidden weights frozen, and the BP Depth-muP
#      counterpart (lr 5e-3); the full-training runs come from lean_fig1.sh.
#  (b) muPC H = 128 with 4H = 512 inference steps instead of H.
#  (c) shallow control, H = 1 (784-512-10): muPC with T = 8 over the paper's
#      4 x 11 grid, BP over lr 1 to 1e-4; both selected by minimum training loss.
set -e
cd "$(dirname "$0")"
. ./env.sh
lean_ok || { echo "lean code not accepted (results/lean_vs_official.json)"; exit 1; }
A=runs/added
$PY lean_mupc.py --out $A/mupc_H128_frozen --n_hidden 128 --freeze_hidden \
  --param_lr 1e-1 --activity_lr 5e-1 --seeds 0 1 2 >> logs/added.log 2>&1
$PY lean_mupc.py --out $A/mupc_H1_grid --n_hidden 1 --max_infer_iters 8 \
  --param_lr 5e-1 1e-1 5e-2 1e-2 --activity_lr 1e3 5e2 1e2 5e1 1e1 5 1 5e-1 1e-1 5e-2 1e-2 \
  --seeds 0 1 2 >> logs/added.log 2>&1
cd "$JPC"
$PY $OLDPWD/run_bpn.py --results_dir $OLDPWD/$A/bp_H128_frozen --freeze_hidden --dataset MNIST \
  --loss_id mse --n_hidden 128 --act_fns relu --param_type depth_mup --lrs 5e-3 --batch_size 64 \
  --max_epochs 1 --test_every 300 --n_seeds 3 >> $OLDPWD/logs/added.log 2>&1
# The shallow grids are small; run on their own they may already be complete (9 lrs x 3 seeds).
if [ "$(find $OLDPWD/$A/bp_H1_grid -name test_accs.npy 2>/dev/null | wc -l)" -lt 27 ]; then
$PY $OLDPWD/run_bpn.py --results_dir $OLDPWD/$A/bp_H1_grid --dataset MNIST --loss_id mse \
  --n_hidden 1 --act_fns relu --param_type depth_mup --lrs 1 5e-1 1e-1 5e-2 1e-2 5e-3 1e-3 5e-4 1e-4 \
  --batch_size 64 --max_epochs 1 --test_every 300 --n_seeds 3 >> $OLDPWD/logs/added.log 2>&1
fi
cd "$OLDPWD"
$PY lean_mupc.py --out $A/mupc_H128_T512 --n_hidden 128 --max_infer_iters 512 \
  --param_lr 1e-1 --activity_lr 5e-1 --seeds 0 1 2 >> logs/added.log 2>&1
touch $A/DONE
