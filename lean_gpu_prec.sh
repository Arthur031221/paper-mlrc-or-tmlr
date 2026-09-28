#!/bin/sh
# lean_mupc.py reaches 96.8% on the CPU but stops at chance on the GPU
# (H = 8, muPC, seed 0). Run it on the GPU (H = 8 and 16, seeds 0 to 2) with float32 matmuls at the
# default and at the highest precision to test whether reduced-precision products are the cause.
# lean_mupc.py now defaults to highest; MATMUL_PRECISION selects the setting.
cd "$(dirname "$0")"
. ./env.sh
for H in 8 16; do
  for P in default highest; do
    MATMUL_PRECISION=$P $PY lean_mupc.py --out runs/gpu_prec/${P}_H$H --n_hidden $H \
      --param_type mupc --param_lr 1e-1 --activity_lr 5e-1 --seeds 0 1 2 >> logs/gpu_prec.log 2>&1
  done
done
echo "gpu_prec done" >> logs/gpu_prec.log
$PY gpu_prec_collect.py
