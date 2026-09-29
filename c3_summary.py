"""Claim C3 from transfer_relu.sh: does the best learning-rate cell transfer?

Reads runs/transfer/{width_N<N>_H8, depth_N512_H<H>}/plr*_alr*/seed*.json.
Writes results/c3.json.

Each cell is scored by the seed mean of the minimum training loss over the
epoch, as in the paper's Fig. A.32. Runs the training code stopped as
diverged, or with a non-finite minimum, make their cell ineligible. The best
cell is the eligible cell with the lowest score.
Criterion (main.tex, fixed before any run): the best cell equals the cell
found at H = 8, N = 512 in at least 4 of 5 depths and 4 of 5 widths, and lies
within one grid step (on both learning-rate axes) of it in all ten. The
reference is taken from the depth sweep at H = 8; that sweep point and the
width sweep point at N = 512 share settings apart from the test interval and
are listed separately.

Post hoc, descriptive, no threshold (defined after the width grids and part
of the H = 16 grid had been seen, before the other depth grids finished): at
each sweep point, with the reference cell c_ref and the best cell c_best,
  regret = log10 score(c_ref) - log10 score(c_best), in decades of minimum
           training loss; 0 when they match; None if c_ref is ineligible;
  rank   = rank of c_ref among eligible cells (1 = best);
  seed_regret = log10 m_k(c_ref) - log10 m_k(c_best) for each seed k.
The minimum training loss is the unsmoothed minimum over per-minibatch losses.
Taking the best cell of noisy seed means biases regret upwards.
Usage: python c3_summary.py
"""
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
T = HERE / "runs" / "transfer"
PLR = [5e-1, 1e-1, 5e-2, 1e-2]
ALR = [1e3, 5e2, 1e2, 5e1, 1e1, 5, 1, 5e-1, 1e-1, 5e-2, 1e-2]
SWEEPS = [("width", n, f"width_N{n}_H8") for n in (64, 128, 256, 512, 1024)] + \
         [("depth", h, f"depth_N512_H{h}") for h in (8, 16, 32, 64, 128)]


def seed_mins(d):
    recs = [json.loads(f.read_text()) for f in sorted(d.glob("seed*.json"))]
    return recs, [min(r["train_loss"]) if r["train_loss"] else math.inf for r in recs]


def cell_score(d):
    recs, mins = seed_mins(d)
    if len(recs) < 3:
        return None, len(recs)
    if any(r["stop"] == "diverged" or not math.isfinite(m) for r, m in zip(recs, mins)):
        return math.inf, len(recs)
    return float(np.mean(mins)), len(recs)


rows, complete = [], True
for axis, val, name in SWEEPS:
    grid = np.full((len(PLR), len(ALR)), np.nan)
    for i, p in enumerate(PLR):
        for j, a in enumerate(ALR):
            s, n = cell_score(T / name / f"plr{p:g}_alr{a:g}")
            if s is None:
                complete = False
            else:
                grid[i, j] = s
    finite = np.where(np.isfinite(grid), grid, np.inf)
    best = None
    if np.isfinite(finite).any():
        i, j = np.unravel_index(np.argmin(finite), finite.shape)
        best = [int(i), int(j)]
    rows.append({"axis": axis, "value": val, "best": best,
                 "best_lr": [PLR[best[0]], ALR[best[1]]] if best else None,
                 "eligible_cells": int(np.isfinite(grid).sum()),
                 "grid": [[None if not np.isfinite(x) else float(x) for x in r] for r in grid]})

out = {"param_lr": PLR, "activity_lr": ALR, "rows": rows, "complete": complete}
ref = next(r for r in rows if r["axis"] == "depth" and r["value"] == 8)["best"]
if complete and ref:
    for r in rows:
        b = r["best"]
        r["match"] = b == ref
        r["within_one"] = b is not None and max(abs(b[0] - ref[0]), abs(b[1] - ref[1])) <= 1
    n_d = sum(r["match"] for r in rows if r["axis"] == "depth")
    n_w = sum(r["match"] for r in rows if r["axis"] == "width")
    out.update(reference=ref, reference_lr=[PLR[ref[0]], ALR[ref[1]]],
               matches_depth=n_d, matches_width=n_w,
               all_within_one=all(r["within_one"] for r in rows))
    out["pass"] = n_d >= 4 and n_w >= 4 and out["all_within_one"]
if ref:
    for (axis, val, name), r in zip(SWEEPS, rows):
        g, b = np.array(r["grid"], dtype=float), r["best"]
        sc = g[ref[0], ref[1]]
        if b is None or not np.isfinite(sc):
            r.update(regret=None, rank=None, seed_regret=None)
            continue
        d = T / name
        _, mr = seed_mins(d / f"plr{PLR[ref[0]]:g}_alr{ALR[ref[1]]:g}")
        _, mb = seed_mins(d / f"plr{PLR[b[0]]:g}_alr{ALR[b[1]]:g}")
        r["regret"] = float(np.log10(sc) - np.log10(g[b[0], b[1]]))
        r["rank"] = int(np.sum(g[np.isfinite(g)] < sc)) + 1
        r["seed_regret"] = [float(np.log10(x) - np.log10(y)) for x, y in zip(mr, mb)]
(HERE / "results" / "c3.json").write_text(json.dumps(out, indent=1) + "\n")
for r in rows:
    print(f"{r['axis']:5s} {r['value']:5d} best {r['best_lr']} eligible {r['eligible_cells']}"
          f" regret {r.get('regret')} rank {r.get('rank')}")
print({k: v for k, v in out.items() if k not in ("rows",)})
