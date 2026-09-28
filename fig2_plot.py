"""Figure 2: 128-layer muPC against BP, and how far each layer's weights move (C2).

Reads results/c2.json (c2_summary.py).
Panel a: seed-mean test accuracy at the last test point for muPC and BP on
each dataset, one point per seed.
Panel b: relative weight change ||W(t) - W(0)|| / ||W(0)|| at the end of
training against layer index (input layer 0, readout last), seed mean,
one line per dataset, log scale.
Usage: python fig2_plot.py [out.pdf]   (default ../paper/fig2.pdf)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "paper" / "fig2.pdf"
c2 = json.loads((HERE / "results" / "c2.json").read_text())
ds = [d for d, v in c2["datasets"].items() if v]

plt.style.use(HERE / "nature.mplstyle")
COLOR = {"mupc": "#0072B2", "bp": "#000000"}
LABEL = {"mupc": r"$\mu$PC", "bp": r"BP Depth-$\mu$P"}
DCOLOR = {"MNIST": "#0072B2", "Fashion-MNIST": "#009E73", "CIFAR10": "#D55E00"}
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(183 / 25.4, 2.4))

for k, d in enumerate(ds):
    for m, dx in (("mupc", -0.12), ("bp", 0.12)):
        v = c2["datasets"][d][m]
        ax_a.scatter(np.full(len(v["runs"]), k + dx), v["runs"], s=8, color=COLOR[m],
                     alpha=0.6, label=LABEL[m] if k == 0 else None)
        ax_a.hlines(v["mean"], k + dx - 0.08, k + dx + 0.08, color=COLOR[m])
ax_a.set_xticks(range(len(ds)), ds)
ax_a.set_ylabel("Test accuracy (%)")
ax_a.legend(frameon=False)
ax_a.set_title("a", loc="left", fontweight="bold")

for d in ds:
    wc = c2["datasets"][d]["weight_change"]["per_layer"]
    ax_b.plot(range(len(wc)), wc, color=DCOLOR[d], label=d)
ax_b.set_yscale("log")
ax_b.set_xlabel("Layer")
ax_b.set_ylabel(r"$\|W - W_0\| / \|W_0\|$")
ax_b.legend(frameon=False)
ax_b.set_title("b", loc="left", fontweight="bold")

fig.savefig(out)
print(out)
