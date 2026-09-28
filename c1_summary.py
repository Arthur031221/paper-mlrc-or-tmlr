"""Claim C1 from the Fig. 1 runs: per-depth accuracy and the pass/fail verdict.

Reads runs/fig1 (official scripts: PC under pc/, BP under bp/) and
runs/lean_fig1 (lean_mupc.py). Writes results/c1.json.

Criterion (IDEA.md, fixed before any run): muPC test accuracy at iteration
900 is at least 90% at every H with a spread across depths of at most 3 pp,
and SP PC at H in {64, 128} stays at or below 15%. A run that the training
code stopped (diverged, or below 15% at the first test point) counts as
failed at its last recorded accuracy, or at 10% if it recorded none.
The verdict uses the lean runs, which cover every depth; the official runs
at H <= 32 are listed beside them.
Usage: python c1_summary.py
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEPTHS = [8, 16, 32, 64, 128]
CHANCE = 10.0


def last(accs):
    return float(accs[-1]) if len(accs) else CHANCE


runs = defaultdict(dict)   # (source, method, H) -> seed -> accs
for f in (HERE / "runs" / "fig1").rglob("test_accs.npy"):
    parts = f.relative_to(HERE / "runs" / "fig1").parts
    h = next(int(p.split("_")[0]) for p in parts if p.endswith("_n_hidden"))
    if parts[0] == "pc":
        method = next(p[: -len("_param")] for p in parts if p.endswith("_param"))
    else:
        method = "bp"
    runs[("official", method, h)][int(parts[-2])] = np.load(f).tolist()
for f in (HERE / "runs" / "lean_fig1").glob("*_H*/seed*.json"):
    pt, h = f.parent.name.split("_H")
    r = json.loads(f.read_text())
    runs[("lean", pt, int(h))][r["seed"]] = r["test_acc"]

rows = []
for (src, method, h), seeds in sorted(runs.items()):
    finals = [last(seeds[s]) for s in sorted(seeds)]
    rows.append({"source": src, "method": method, "n_hidden": h, "seeds": sorted(seeds),
                 "stopped": [len(seeds[s]) < 3 for s in sorted(seeds)],
                 "final_accs": finals, "mean": float(np.mean(finals)),
                 "sd": float(np.std(finals, ddof=1)) if len(finals) > 1 else None})


def pick(src, method, h):
    return next((r for r in rows if (r["source"], r["method"], r["n_hidden"]) == (src, method, h)), None)


mupc = [pick("lean", "mupc", h) for h in DEPTHS]
sp_deep = [pick("lean", "sp", h) for h in (64, 128)]
complete = all(r and len(r["seeds"]) == 3 for r in mupc + sp_deep)
out = {"rows": rows, "complete": complete}
if complete:
    means = [r["mean"] for r in mupc]
    sp_max = max(max(r["final_accs"]) for r in sp_deep)
    out.update({
        "mupc_min_mean": min(means), "mupc_max_mean": max(means),
        "mupc_spread": max(means) - min(means),
        "mupc_min_run": min(min(r["final_accs"]) for r in mupc),
        "sp_deep_max": sp_max,
        "pass_mupc": min(means) >= 90.0 and max(means) - min(means) <= 3.0,
        "pass_sp": sp_max <= 15.0,
    })
    out["pass"] = out["pass_mupc"] and out["pass_sp"]
(HERE / "results" / "c1.json").write_text(json.dumps(out, indent=1) + "\n")
for r in rows:
    print(f"{r['source']:8s} {r['method']:9s} H={r['n_hidden']:3d} seeds {r['seeds']} "
          f"mean {r['mean']:.2f} stopped {sum(r['stopped'])}")
print({k: v for k, v in out.items() if k != "rows"})
