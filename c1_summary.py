"""Claim C1 from the Fig. 1 runs: per-depth accuracy and the outcome verdict.

Reads runs/fig1 (official scripts: PC under pc/, BP under bp/) and
runs/lean_fig1 (lean_mupc.py). Writes results/c1.json.

Criterion (stated in the paper, fixed before any run): muPC test accuracy at iteration
900 is at least 90% at every H with a spread across depths of at most 3 pp.
For SP PC at H in {64, 128}, a run either diverges before update 900 or has
test accuracy at or below 15%. A divergent run without a test point is a
training failure, not an accuracy measurement. final_accs keeps 10% as a
plotting coordinate only; measured_accs records missing values as null.
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
    return float(accs[-1]) if len(accs) else None


runs = defaultdict(dict)   # (source, method, H) -> seed -> test accuracies
stops = defaultdict(dict)  # (source, method, H) -> seed -> stop reason
iterations = defaultdict(dict)  # (source, method, H) -> seed -> final update
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
    key = ("lean", pt, int(h))
    runs[key][r["seed"]] = r["test_acc"]
    stops[key][r["seed"]] = r.get("stop")
    iterations[key][r["seed"]] = r.get("iterations")

rows = []
for (src, method, h), seeds in sorted(runs.items()):
    ordered_seeds = sorted(seeds)
    measured = [last(seeds[s]) for s in ordered_seeds]
    finals = [CHANCE if acc is None else acc for acc in measured]
    all_measured = all(acc is not None for acc in measured)
    rows.append({"source": src, "method": method, "n_hidden": h, "seeds": sorted(seeds),
                 "stopped": [len(seeds[s]) < 3 for s in ordered_seeds],
                 "stop_reasons": [stops[(src, method, h)].get(s) for s in ordered_seeds],
                 "iterations": [iterations[(src, method, h)].get(s) for s in ordered_seeds],
                 "final_accs": finals, "measured_accs": measured,
                 "mean": float(np.mean(measured)) if all_measured else None,
                 "sd": float(np.std(measured, ddof=1)) if all_measured and len(measured) > 1 else None})


def pick(src, method, h):
    return next((r for r in rows if (r["source"], r["method"], r["n_hidden"]) == (src, method, h)), None)


mupc = [pick("lean", "mupc", h) for h in DEPTHS]
sp_deep = [pick("lean", "sp", h) for h in (64, 128)]
complete = all(r and len(r["seeds"]) == 3 for r in mupc + sp_deep)
out = {"rows": rows, "complete": complete}
if complete:
    means = [r["mean"] for r in mupc]
    sp_runs = [(acc, stop, n) for r in sp_deep
               for acc, stop, n in zip(r["measured_accs"], r["stop_reasons"], r["iterations"])]
    sp_values = [acc for acc, _, _ in sp_runs if acc is not None]
    sp_outcomes = [(stop == "diverged" and n is not None and n < 900) or
                   (acc is not None and acc <= 15.0) for acc, stop, n in sp_runs]
    out.update({
        "mupc_min_mean": min(means), "mupc_max_mean": max(means),
        "mupc_spread": max(means) - min(means),
        "mupc_min_run": min(min(r["measured_accs"]) for r in mupc),
        "sp_deep_max": max(sp_values) if sp_values else None,
        "sp_accuracy_evaluable": all(acc is not None for acc, _, _ in sp_runs),
        "pass_mupc": min(means) >= 90.0 and max(means) - min(means) <= 3.0,
        "pass_sp": all(sp_outcomes),
    })
    out["pass"] = out["pass_mupc"] and out["pass_sp"]
(HERE / "results" / "c1.json").write_text(json.dumps(out, indent=1) + "\n")
for r in rows:
    mean = "unmeasured" if r["mean"] is None else f"{r['mean']:.2f}"
    print(f"{r['source']:8s} {r['method']:9s} H={r['n_hidden']:3d} seeds {r['seeds']} "
          f"mean {mean} stopped {sum(r['stopped'])}")
print({k: v for k, v in out.items() if k != "rows"})
