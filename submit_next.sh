#!/bin/sh
# Queue a follow-up GPU job from inside a finished one. The queue's tick can drop
# a submission made in the 20 s after it starts a job, so submit at second 38 to
# 45 of a minute and check the entry is still there a minute later.
# Usage: submit_next.sh NAME MEM_GB MINUTES SCRIPT
set -e
Q=/home/levi/oss-engine/scripts/gpu_queue.py
STATE=/home/levi/oss-engine/state/gpu_queue.json
HERE=$(cd "$(dirname "$0")" && pwd)
for try in 1 2 3; do
  while [ "$(date +%S)" -lt 38 ] || [ "$(date +%S)" -gt 45 ]; do sleep 1; done
  python3 $Q submit --item mlrc-or-tmlr --name "$1" --mem "$2" --minutes "$3" --cwd "$HERE" -- "$HERE/$4"
  sleep 60
  grep -q "\"name\": \"$1\"" $STATE && exit 0
done
echo "submission of $1 did not persist" >&2
exit 1
