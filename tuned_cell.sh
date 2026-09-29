#!/bin/sh
# Partial-freeze test repeated at the H = 128 grid-best cell of C3 (results/c3.json,
# minimum training loss), seeds 0 to 4; same rule as freeze_test.sh (freeze_summary.py).
set -e
cd "$(dirname "$0")"
. ./env.sh
[ -e runs/transfer/DONE ] || { echo "C3 grids not finished"; exit 1; }
$PY c3_summary.py > /dev/null
set -- $($PY -c "import json; r=[r for r in json.load(open('results/c3.json'))['rows'] if r['axis']=='depth' and r['value']==128 and r['complete']][0]; print(*r['best_lr'])")
[ -n "$2" ] || { echo "H = 128 grid not complete in results/c3.json"; exit 1; }
F=runs/freeze_tuned
M="--n_hidden 128 --param_lr $1 --activity_lr $2 --seeds 0 1 2 3 4 --log_weights --log_grads"
$PY lean_mupc.py --out $F/mupc_full $M >> logs/freeze.log 2>&1
$PY lean_mupc.py --out $F/mupc_frz1-121 --freeze_range 1 121 $M >> logs/freeze.log 2>&1
$PY lean_mupc.py --out $F/mupc_frz122-127 --freeze_range 122 127 $M >> logs/freeze.log 2>&1
touch $F/DONE
$PY freeze_summary.py results/freeze.json >> logs/freeze.log 2>&1
