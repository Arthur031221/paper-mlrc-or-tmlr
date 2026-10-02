#!/bin/sh
set -e
cd "$(dirname "$0")"
. ./env.sh
lean_ok || { echo "lean code not accepted (results/lean_vs_official.json)"; exit 1; }
F=runs/tail_freeze
M="--n_hidden 128 --param_lr 5e-1 --activity_lr 5e-1 --seeds 5 6 7 8 9"
$PY lean_mupc.py --out "$F/mupc_full" $M >> logs/tail_freeze.log 2>&1
$PY lean_mupc.py --out "$F/mupc_frz1-121" --freeze_range 1 121 $M >> logs/tail_freeze.log 2>&1
$PY lean_mupc.py --out "$F/mupc_frz122-127" --freeze_range 122 127 $M >> logs/tail_freeze.log 2>&1
$PY tail_freeze_summary.py
