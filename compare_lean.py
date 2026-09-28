"""Compare lean_mupc.py with the official code on the Fig. 1 grid.

Rule, fixed before the lean runs (STATUS.md, 2026-09-29): the lean code is
used for the rest of the study if, at every (parameterisation, depth) that
both have finished, the lean seed mean at the last common test point lies
within the official seed range widened by 1 point on each side. Runs stopped
early (SP PC at chance) are compared at their last point; configurations where
every run of both codes diverged before the first test point count as agreeing.

Wall clock: the lean runs record their own time. The official scripts do not,
so their time per run is the gap between consecutive result files of one
configuration (the first seed of each configuration has no gap and is left out).
Usage: python compare_lean.py  (reads runs/fig1 and runs/lean_fig1)
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEST_EVERY = 300

official = defaultdict(dict)     # (param, H) -> seed -> (accs, mtime)
for f in (HERE / "runs" / "fig1" / "pc").rglob("test_accs.npy"):
    parts = f.relative_to(HERE / "runs" / "fig1" / "pc").parts
    h = next(int(p.split("_")[0]) for p in parts if p.endswith("_n_hidden"))
    pt = next(p[: -len("_param")] for p in parts if p.endswith("_param"))
    official[(pt, h)][int(parts[-2])] = (np.load(f).tolist(), f.stat().st_mtime)

lean = defaultdict(dict)
for f in (HERE / "runs" / "lean_fig1").glob("*_H*/seed*.json"):
    pt, h = f.parent.name.split("_H")
    r = json.loads(f.read_text())
    lean[(pt, int(h))][r["seed"]] = (r["test_acc"], r["seconds"])

rows, verdicts = [], []
for key in sorted(set(official) & set(lean)):
    o, l_ = official[key], lean[key]
    seeds = sorted(set(o) & set(l_))
    if not seeds:
        continue
    n = min(min(len(o[s][0]) for s in seeds), min(len(l_[s][0]) for s in seeds))
    if n == 0:
        # No test point at all: the runs diverged before iteration 300. The
        # two codes agree only if every seed of both stopped without one.
        ok = all(len(o[s][0]) == 0 for s in seeds) and all(len(l_[s][0]) == 0 for s in seeds)
        rows.append({"param": key[0], "n_hidden": key[1], "seeds": seeds, "iteration": 0,
                     "official_accs": [], "lean_accs": [], "official_mean": None, "lean_mean": None,
                     "outcome": "both diverged before the first test point" if ok else "mismatch",
                     "within_official_range_pm1": bool(ok),
                     "official_seconds_per_run": None,
                     "lean_seconds_per_run": float(np.mean([l_[s][1] for s in seeds]))})
        verdicts.append(ok)
        continue
    o_last = [o[s][0][n - 1] for s in seeds]
    l_last = [l_[s][0][n - 1] for s in seeds]
    lo, hi = min(o_last) - 1.0, max(o_last) + 1.0
    ok = lo <= float(np.mean(l_last)) <= hi
    times = [o[s][1] for s in sorted(o)]
    gaps = np.diff(times) if len(times) > 1 else []
    rows.append({
        "param": key[0], "n_hidden": key[1], "seeds": seeds, "iteration": n * TEST_EVERY,
        "official_accs": o_last, "lean_accs": l_last,
        "official_mean": float(np.mean(o_last)), "lean_mean": float(np.mean(l_last)),
        "within_official_range_pm1": bool(ok),
        "official_seconds_per_run": float(np.mean(gaps)) if len(gaps) else None,
        "lean_seconds_per_run": float(np.mean([l_[s][1] for s in seeds])),
    })
    verdicts.append(ok)

out = {"rule": "lean seed mean within official seed range +-1 pp at the last common test point",
       "configs_compared": len(rows), "accept_lean": bool(rows) and all(verdicts), "rows": rows}
(HERE / "results" / "lean_vs_official.json").write_text(json.dumps(out, indent=1) + "\n")
for r in rows:
    if r["official_mean"] is None:
        print(f"{r['param']:5s} H={r['n_hidden']:3d} {r['outcome']} ok {r['within_official_range_pm1']}")
        continue
    print(f"{r['param']:5s} H={r['n_hidden']:3d} it{r['iteration']} official {r['official_mean']:.2f} "
          f"lean {r['lean_mean']:.2f} ok {r['within_official_range_pm1']} "
          f"s/run {r['official_seconds_per_run']} vs {r['lean_seconds_per_run']:.0f}")
print("accept_lean", out["accept_lean"])
