"""Figure 1: test accuracy against depth for muPC, SP PC and BP (claim C1).

Reads results/c1.json (c1_summary.py), which holds the final test accuracy
of every run of the official code and of lean_mupc.py.
Panel a: accuracy after one epoch against H for all methods, one point per
seed, a line through the reimplementation's seed means; runs the training
script stopped are drawn at their last accuracy (chance if none) with an x.
Panel b: the muPC and BP points on a narrow scale, with seed mean and s.d.
Usage: python fig1_plot.py [out.pdf]   (default ../paper/fig1.pdf)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
STYLE = HERE / "nature.mplstyle"
out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "paper" / "fig1.pdf"

plt.style.use(STYLE)
COLOR = {r"$\mu$PC": "#0072B2", "SP PC": "#D55E00", r"BP Depth-$\mu$P": "#000000"}
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(183 / 25.4, 2.4))

# Panel a reads results/c1.json, which also covers runs stopped before
# their first test point; those are drawn at chance with an x.
c1 = json.loads((HERE / "results" / "c1.json").read_text())
METHOD = {"mupc": r"$\mu$PC", "sp": "SP PC", "bp": r"BP Depth-$\mu$P"}
for src in ("lean", "official"):
    rows = sorted((r for r in c1["rows"] if r["source"] == src), key=lambda r: r["n_hidden"])
    for lab in COLOR:
        rs = [r for r in rows if METHOD[r["method"]] == lab]
        if not rs:
            continue
        for r in rs:
            for acc, stopped in zip(r["final_accs"], r["stopped"]):
                off = 1.06 if src == "official" else 1.0
                if stopped:
                    ax_a.scatter(r["n_hidden"] * off, acc, s=10, color=COLOR[lab], marker="x", linewidths=0.6)
                elif src == "lean":
                    ax_a.scatter(r["n_hidden"], acc, s=6, color=COLOR[lab], marker="s", alpha=0.5, linewidths=0)
                else:
                    ax_a.scatter(r["n_hidden"] * off, acc, s=10, facecolors="none", edgecolors=COLOR[lab], linewidths=0.6)
        if src == "lean":
            ax_a.plot([r["n_hidden"] for r in rs], [r["mean"] for r in rs], color=COLOR[lab], lw=0.8, label=lab)
ax_a.scatter([], [], s=6, color="grey", marker="s", label="reimplementation")
ax_a.scatter([], [], s=10, facecolors="none", edgecolors="grey", label="official code")
ax_a.scatter([], [], s=10, color="grey", marker="x", label="stopped by the script")

# Panel b: the muPC points of panel a on a narrow scale, with BP at H = 128.
for r in c1["rows"]:
    if r["method"] == "sp":
        continue
    col = COLOR[METHOD[r["method"]]]
    if r["source"] == "lean":
        ax_b.scatter([r["n_hidden"]] * len(r["final_accs"]), r["final_accs"], s=8, color=col,
                     marker="s", alpha=0.6, linewidths=0)
        ax_b.errorbar(r["n_hidden"] * 0.94, r["mean"], yerr=r["sd"] or 0, color=col, fmt="_", ms=5, lw=0.8)
    else:
        ax_b.scatter([r["n_hidden"] * 1.06] * len(r["final_accs"]), r["final_accs"], s=10,
                     facecolors="none", edgecolors=col, linewidths=0.6)
if any(r["method"] == "bp" for r in c1["rows"]):
    ax_b.scatter([], [], s=10, facecolors="none", edgecolors=COLOR[METHOD["bp"]], label=METHOD["bp"])
    ax_b.legend(frameon=False)

for ax in (ax_a, ax_b):
    ax.set_xscale("log", base=2)
    ax.set_xticks([8, 16, 32, 64, 128], ["8", "16", "32", "64", "128"])
    ax.minorticks_off()
    ax.set_xlabel("Hidden layers H")
ax_a.set_ylim(0, 100)
ax_a.set_xlabel("Hidden layers H")
ax_a.set_ylabel("Test accuracy after 1 epoch (%)")
ax_a.legend(frameon=False, loc="center right")
ax_b.set_ylabel("Test accuracy after 1 epoch (%)")
for ax, tag in ((ax_a, "a"), (ax_b, "b")):
    ax.text(-0.18, 1.02, tag, transform=ax.transAxes, fontweight="bold", va="bottom")
fig.tight_layout()
fig.savefig(out)
print(f"wrote {out}")
