#!/bin/sh
# Stop the official Fig. 1 job once H = 32 is complete (3 muPC + 3 SP seeds).
# At about 1 h per H = 32 run on this machine, H = 64 and H = 128 with the
# official code would take about 60 h; lean_mupc.py covers those depths.
cd /home/levi/oss-engine/papers/mlrc-or-tmlr/code
while [ $(find runs/fig1/pc -path "*/32_n_hidden/*" -name test_accs.npy | wc -l) -lt 6 ]; do
  kill -0 2003179 2>/dev/null || exit 0
  sleep 120
done
kill -TERM 2003179; pkill -TERM -f 'experiments.mupc_paper.train_pcn_no_metrics --results_dir ../runs/fig1/pc'
echo "stopped official fig1 job after H=32 at $(date -u +%FT%TZ)" >> logs/fig1.log
