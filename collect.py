"""Flatten the official scripts' result trees into one CSV.

The official scripts save test_accs.npy in a directory whose path encodes the
configuration (dataset/loss/width/depth/activation/.../seed). Each row of the
output is one test evaluation of one run.
Usage: python collect.py <results_root> <out.csv> <test_every>
"""
import csv
import sys
from pathlib import Path

import numpy as np

root, out, test_every = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
rows = []
for f in sorted(root.rglob("test_accs.npy")):
    parts = f.relative_to(root).parts[:-1]
    cfg = {"path": "/".join(parts[:-1]), "seed": int(parts[-1])}
    for p in parts:
        for key in ("n_hidden", "param", "max_infer_iters"):
            if p.endswith("_" + key):
                cfg[key] = p[: -len(key) - 1]
        for key in ("param_lr", "activity_lr"):
            if p.startswith(key + "_"):
                cfg[key] = p[len(key) + 1:]
    cfg["method"] = parts[0] if parts[0] in ("pc", "bp") else ""
    for i, acc in enumerate(np.load(f)):
        rows.append({**cfg, "iteration": (i + 1) * test_every, "test_acc": float(acc)})
out.parent.mkdir(exist_ok=True)
with out.open("w", newline="") as fh:
    keys = sorted({k for r in rows for k in r})
    w = csv.DictWriter(fh, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} rows from {root} -> {out}")
