"""Apply the C1 criteria stated in the paper to results/fig1.csv.

C1 passes when muPC test accuracy at iteration 900 is at least 90% at every
depth, the spread of the per-depth seed means is at most 3 points, and SP PC
never exceeds 15% at H = 64 or H = 128.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader((HERE / "results" / "fig1.csv").open()))

acc = defaultdict(list)          # (method, param, H, iteration) -> per-seed accuracies
for r in rows:
    acc[(r["method"], r["param"], int(r["n_hidden"]), int(r["iteration"]))].append(float(r["test_acc"]))

summary = {}
for (method, param, h, it), v in sorted(acc.items()):
    summary[f"{method}/{param}/H{h}/it{it}"] = {
        "mean": float(np.mean(v)), "std": float(np.std(v)), "n_seeds": len(v)}

depths = [8, 16, 32, 64, 128]
mupc_900 = {h: acc.get(("pc", "mupc", h, 900), []) for h in depths}
means = [np.mean(v) for v in mupc_900.values() if v]
sp_deep = [a for (m, p, h, it), v in acc.items() if m == "pc" and p == "sp" and h in (64, 128) for a in v]
checks = {
    "mupc_all_depths_present": all(len(v) == 3 for v in mupc_900.values()),
    "mupc_min_acc_at_900_ge_90": bool(means) and min(means) >= 90.0,
    "mupc_depth_spread_le_3pp": bool(means) and max(means) - min(means) <= 3.0,
    "sp_deep_never_above_15": bool(sp_deep) and max(sp_deep) <= 15.0,
}
checks["C1_pass"] = all(checks.values())
out = {"checks": checks, "mupc_min_mean_at_900": float(min(means)) if means else None,
       "mupc_depth_spread_at_900": float(max(means) - min(means)) if means else None,
       "sp_deep_max": float(max(sp_deep)) if sp_deep else None, "summary": summary}
(HERE / "results" / "fig1_verdict.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(checks))
