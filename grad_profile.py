"""Per-layer gradient-norm profile for the muPC notebook-cell full arm.

Reads grad_norm_it1 (iteration 1) and the last entry of grad_norm (iteration
900, the last test point of a one-epoch run) for muPC at H = 128 on MNIST,
five seeds. Reports the median gradient Frobenius norm of the last TOP hidden
matrices against the rest, and their ratio. No backpropagation or CIFAR-10
gradient summary is produced. Written to results/grad_profile.json.
Usage: python grad_profile.py
"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "grad_profile.json")
TOP = 6


def split(g):
    g = np.asarray(g, dtype=float)
    hidden = g[1:-1]
    below, top = hidden[:-TOP], hidden[-TOP:]
    return {"below_median": float(np.median(below)), "top_median": float(np.median(top)),
            "ratio": float(np.median(top) / np.median(below))}


def mupc_full():
    seeds = {}
    for f in sorted(glob.glob(os.path.join(HERE, "runs", "freeze", "mupc_full", "seed*.json"))):
        d = json.load(open(f))
        seeds[d["seed"]] = {"it1": split(d["grad_norm_it1"]), "it900": split(d["grad_norm"][-1])}
    return seeds


def main():
    out = {"mupc_full": mupc_full()}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("mupc_full", {s: v["it900"]["ratio"] for s, v in out["mupc_full"].items()})


if __name__ == "__main__":
    main()
