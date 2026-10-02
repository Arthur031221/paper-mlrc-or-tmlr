"""C1, standard parameterisation over the 4 x 11 grid (sp_grid.sh).

Reads runs/sp_grid/depth_N512_H<H>/plr*_alr*/seed*.json, writes
results/sp_grid.json. Cells are scored as in c3_summary.py: seed mean of the
minimum training loss, a cell with a diverged or non-finite run is
ineligible, the best cell is the eligible cell with the lowest score.
Accuracy of a cell is the seed mean of test accuracy at update 900 when the
run reaches it. A run stopped before a test point has no measured accuracy.
C1's numerical accuracy condition cannot be evaluated at a depth with no
eligible cell. spMaxDeep, the highest cell accuracy over all eligible cells
at H >= 64, is descriptive.
Usage: python sp_summary.py
"""
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
G = HERE / "runs" / "sp_grid"
PLR = [5e-1, 1e-1, 5e-2, 1e-2]
ALR = [1e3, 5e2, 1e2, 5e1, 1e1, 5, 1, 5e-1, 1e-1, 5e-2, 1e-2]
DEPTHS = {8: "Eight", 16: "Sixteen", 32: "Thirtytwo", 64: "Sixtyfour", 128: "Onetwentyeight"}
IT900 = 8  # test every 100 iterations: index 8 is iteration 900


def acc900(r):
    a = r["test_acc"]
    if len(a) > IT900:
        return a[IT900]
    return a[-1] if a else None


def cell(d):
    recs = [json.loads(f.read_text()) for f in sorted(d.glob("seed*.json"))]
    if len(recs) < 3:
        return None
    mins = [min(r["train_loss"]) if r["train_loss"] else math.inf for r in recs]
    eligible = all(r["stop"] != "diverged" and math.isfinite(m) for r, m in zip(recs, mins))
    accs = [acc900(r) for r in recs]
    acc = float(np.mean(accs)) if all(x is not None for x in accs) else None
    return {"score": float(np.mean(mins)) if eligible else math.inf, "eligible": eligible,
            "acc": acc}


out, complete = {"rows": []}, True
deep_accs = []
for H, name in DEPTHS.items():
    cells = {(p, a): cell(G / f"depth_N512_H{H}" / f"plr{p:g}_alr{a:g}") for p in PLR for a in ALR}
    done = all(c is not None for c in cells.values())
    complete &= done
    elig = {k: c for k, c in cells.items() if c and c["eligible"]}
    best = min(elig, key=lambda k: elig[k]["score"]) if elig else None
    row = {"H": H, "complete": done, "eligible": len(elig), "best_lr": best,
           "best_acc": elig[best]["acc"] if best else None,
           "max_acc": max((c["acc"] for c in elig.values()), default=None)}
    out["rows"].append(row)
    if done:
        out[f"sp{name}Acc"] = f"{row['best_acc']:.2f}" if best else "no eligible cell"
    if H >= 64:
        deep_accs += [c["acc"] for c in elig.values() if c["acc"] is not None]
out["complete"] = complete
if complete:
    deep = [r for r in out["rows"] if r["H"] >= 64]
    out["spEligibleDeep"] = sum(r["eligible"] for r in deep)
    out["spMaxDeep"] = f"{max(deep_accs):.2f}" if deep_accs else "none"
    if any(r["best_acc"] is not None and r["best_acc"] > 15 for r in deep):
        out["spVerdict"] = "did not hold"
    elif any(r["best_acc"] is None for r in deep):
        out["spVerdict"] = "not evaluable"
    else:
        out["spVerdict"] = "held"
(HERE / "results" / "sp_grid.json").write_text(json.dumps(out, indent=1))
print({k: v for k, v in out.items() if k != "rows"})
