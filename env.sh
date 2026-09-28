# Sourced by the job scripts: pinned interpreter and official code location.
# setup_upstream.sh puts both in ../vendor; older layouts kept them here.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
if [ -d ../vendor/upstream_jpc ]; then JPC=$(cd ../vendor/upstream_jpc && pwd); PY=$(cd ../vendor && pwd)/.venv-pinned/bin/python
else JPC=$(pwd)/upstream_jpc; PY=$(pwd)/.venv-pinned/bin/python; fi
export JPC_DIR=$JPC
lean_ok() {
  "$PY" -c "import json,sys; sys.exit(0 if json.load(open('results/lean_vs_official.json'))['accept_lean'] else 1)"
}
