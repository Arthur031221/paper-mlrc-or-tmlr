"""Claim C2 from long_runs.sh: 128-layer muPC against BP with Depth-muP.

Reads runs/long/mupc_<dataset> (lean_mupc.py, with weight change logged) and
runs/long/bp (official BP through run_bpn.py). Writes results/c2.json.

Criteria (main.tex, fixed before any run), on the seed mean of test accuracy
at the last test point:
  MNIST          muPC >= 97.0 and BP - muPC <= 1.0
  Fashion-MNIST  muPC >= 87.0 and BP - muPC <= 1.5
  CIFAR10        35.0 <= muPC <= 41.0 and muPC < BP
A run the training code stopped counts at its last recorded accuracy, or at
10% if it recorded none.
Weight change: per-layer ||W(t) - W(0)|| / ||W(0)|| at the last test point,
seed mean, reported for the input layer, the hidden layers (median, min,
max) and the readout.
Usage: python c2_summary.py
"""
import json
from pathlib import Path

import numpy as np
from criteria import C2_CRITERIA

HERE = Path(__file__).resolve().parent
L = HERE / "runs" / "long"
CHANCE = 10.0


def last(accs):
    return float(accs[-1]) if len(accs) else CHANCE


def stats(v):
    return {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None,
            "runs": [float(x) for x in v]}


out = {"datasets": {}}
for ds in C2_CRITERIA:
    recs = [json.loads(f.read_text()) for f in sorted((L / f"mupc_{ds}").glob("seed*.json"))]
    bp_files = sorted((L / "bp" / ds).rglob("test_accs.npy")) if (L / "bp" / ds).exists() else []
    if not recs or not bp_files:
        out["datasets"][ds] = None
        continue
    mu = stats([last(r["test_acc"]) for r in recs])
    bp = stats([last(np.load(f).tolist()) for f in bp_files])
    wc = np.mean([r["weight_change"][-1] for r in recs if r["weight_change"]], axis=0)
    hidden = wc[1:-1]
    row = {"mupc": mu, "bp": bp, "gap": bp["mean"] - mu["mean"],
           "stopped": [r["stop"] for r in recs],
           "weight_change": {"input": float(wc[0]), "readout": float(wc[-1]),
                             "hidden_median": float(np.median(hidden)),
                             "hidden_min": float(hidden.min()), "hidden_max": float(hidden.max()),
                             "per_layer": [float(x) for x in wc]}}
    if ds == "CIFAR10":
        lo, hi = C2_CRITERIA[ds]
        row["pass"] = lo <= mu["mean"] <= hi and mu["mean"] < bp["mean"]
    else:
        floor, gap = C2_CRITERIA[ds]
        row["pass"] = mu["mean"] >= floor and row["gap"] <= gap
    out["datasets"][ds] = row

out["complete"] = all(v is not None for v in out["datasets"].values())
if out["complete"]:
    out["pass"] = all(v["pass"] for v in out["datasets"].values())
(HERE / "results" / "c2.json").write_text(json.dumps(out, indent=1) + "\n")
for ds, v in out["datasets"].items():
    if v:
        print(f"{ds:14s} muPC {v['mupc']['mean']:.2f} BP {v['bp']['mean']:.2f} "
              f"hidden change median {v['weight_change']['hidden_median']:.3g} pass {v['pass']}")
    else:
        print(f"{ds:14s} missing")
print({k: v for k, v in out.items() if k != "datasets"})
