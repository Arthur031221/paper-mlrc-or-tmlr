#!/bin/sh
# Fetch the official muPC code at the pinned commit and build the pinned
# environment beside this directory (../vendor), outside the published code.
# The first pin in our notes, 5fa5b1a, fails at import; 84f277b is the first
# later commit that imports, and its experiments/mupc_paper is identical to
# e007dca, the last change to that folder.
set -e
cd "$(dirname "$0")/.."
mkdir -p vendor
[ -d vendor/upstream_jpc ] || git clone https://github.com/thebuckleylab/jpc.git vendor/upstream_jpc
git -C vendor/upstream_jpc checkout 84f277b
uv venv --python 3.11 vendor/.venv-pinned
uv pip install --python vendor/.venv-pinned/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cpu -r code/requirements-pinned.txt
uv pip install --python vendor/.venv-pinned/bin/python --no-deps -e vendor/upstream_jpc
