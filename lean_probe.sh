#!/bin/sh
# Timing and agreement probe for lean_mupc.py on the GPU: H = 8 and 16,
# muPC and SP, seeds 0 to 2, the Fig. 1 settings. The official code has
# finished all of these configurations, so compare_lean.py can judge them.
set -e
cd "$(dirname "$0")"
. ./env.sh
for H in 8 16; do
  for PT in mupc sp; do
    if [ "$PT" = mupc ]; then ALR=5e-1; else ALR=1e-2; fi
    /usr/bin/time -f "H=$H $PT wall %e s" $PY lean_mupc.py --out runs/lean_fig1/${PT}_H${H} \
      --n_hidden $H --param_type $PT --param_lr 1e-1 --activity_lr $ALR --seeds 0 1 2 \
      >> logs/lean_probe.log 2>&1
  done
done
