"""C1, standard parameterisation over the 4 x 11 grid (sp_grid.sh).

Reads runs/sp_grid/depth_N512_H<H>/plr*_alr*/seed*.json, writes
results/sp_grid.json. Cells are scored as in c3_summary.py: seed mean of the
minimum training loss, a cell with a diverged or non-finite run is
ineligible, the best cell is the eligible cell with the lowest score.
Accuracy of a cell is the seed mean of test accuracy after 900 iterations;
a run the training code stopped counts at its last test accuracy, or at 10%
if it had none (as drawn in Fig. 1).
C1 criterion for the standard parameterisation (fixed before any run): it
stays at or below 15% at H = 64 and 128. Here it is applied to the best cell
at each of those depths; a depth with no eligible cell counts as at or below
15%. spMaxDeep, the highest cell accuracy over all eligible cells at
H >= 64, is descriptive.
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
    return a[-1] if a else 10.0


def cell(d):
    recs = [json.loads(f.read_text()) for f in sorted(d.glob("seed*.json"))]
    if len(recs) < 3:
        return None
    mins = [min(r["train_loss"]) if r["train_loss"] else math.inf for r in recs]
    eligible = all(r["stop"] != "diverged" and math.isfinite(m) for r, m in zip(recs, mins))
    return {"score": float(np.mean(mins)) if eligible else math.inf, "eligible": eligible,
            "acc": float(np.mean([acc900(r) for r in recs]))}


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
        deep_accs += [c["acc"] for c in elig.values()]
out["complete"] = complete
if complete:
    deep = [r for r in out["rows"] if r["H"] >= 64]
    out["spEligibleDeep"] = sum(r["eligible"] for r in deep)
    out["spMaxDeep"] = f"{max(deep_accs):.2f}" if deep_accs else "none"
    held = all(r["best_acc"] is None or r["best_acc"] <= 15 for r in deep)
    out["spVerdict"] = "held" if held else "did not hold"
(HERE / "results" / "sp_grid.json").write_text(json.dumps(out, indent=1))
print({k: v for k, v in out.items() if k != "rows"})
