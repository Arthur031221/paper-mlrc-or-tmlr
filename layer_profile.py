"""Per-layer relative weight change of muPC at H=128 on MNIST (runs/long/mupc_MNIST).

Reads the weight_change log of each finished seed and writes results/layer_profile.json:
for iteration 900 (the added-value arms stop there) and for the last test point,
the input layer, the hidden layers (median and max over all, and over all but the
top six) and the readout, plus the number of hidden layers whose relative change
is at least THRESH. Run with any python that has numpy.
"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs", "long", "mupc_MNIST")
OUT = os.path.join(HERE, "results", "layer_profile.json")
THRESH = 0.05
TOP = 6


def summarise(wc):
    wc = np.asarray(wc, dtype=float)
    hidden = wc[1:-1]
    return {
        "input": float(wc[0]),
        "readout": float(wc[-1]),
        "hidden_median": float(np.median(hidden)),
        "hidden_max": float(hidden.max()),
        "hidden_below_top_median": float(np.median(hidden[:-TOP])),
        "hidden_below_top_max": float(hidden[:-TOP].max()),
        "top_hidden": [float(x) for x in hidden[-TOP:]],
        "n_hidden_at_or_above_thresh": int((hidden >= THRESH).sum()),
        "first_hidden_at_or_above_thresh": (
            int(np.argmax(hidden >= THRESH)) + 1 if (hidden >= THRESH).any() else None),
    }


def main():
    seeds = {}
    for f in sorted(glob.glob(os.path.join(RUNS, "seed*.json"))):
        d = json.load(open(f))
        wc = d["weight_change"]
        every = d["config"]["test_every"]
        assert len(wc[0]) == d["config"]["n_hidden"] + 1, len(wc[0])
        i900 = 900 // every - 1
        seeds[d["seed"]] = {
            "n_hidden_matrices": len(wc[0]) - 2,
            "it900": summarise(wc[i900]),
            "last": summarise(wc[-1]),
            "last_iteration": every * len(wc),
            "profile_last": [float(x) for x in wc[-1]],
        }
    out = {"threshold": THRESH, "top": TOP, "n_seeds": len(seeds),
           "seeds": {str(k): v for k, v in sorted(seeds.items())}}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    for s, v in sorted(seeds.items()):
        a, b = v["it900"], v["last"]
        print(f"seed {s}: it900 in {a['input']:.2e} hid<top med {a['hidden_below_top_median']:.2e} "
              f"max {a['hidden_below_top_max']:.2e} n>=thr {a['n_hidden_at_or_above_thresh']} "
              f"(first {a['first_hidden_at_or_above_thresh']}) ro {a['readout']:.2f} | "
              f"it{v['last_iteration']} n>=thr {b['n_hidden_at_or_above_thresh']} "
              f"hid<top max {b['hidden_below_top_max']:.2e}")


if __name__ == "__main__":
    main()
