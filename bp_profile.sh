#!/bin/sh
# Per-layer weight change of the BP Depth-muP baseline at H=128 on MNIST (layer_profile.py counterpart).
set -e
cd "$(dirname "$0")"
. ./env.sh
cd "$JPC"
$PY $OLDPWD/bp_profile.py --seeds 0 1 2 3 4 >> $OLDPWD/logs/bp_profile.log 2>&1
