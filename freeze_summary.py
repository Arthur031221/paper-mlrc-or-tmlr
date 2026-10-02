"""Partial-freeze test (post hoc arm, designed after the weight-change profile).

Question (Goemaere et al., ePC App. D.1.1): does muPC at H = 128 train only the
top of the network? MNIST, one epoch, batch 64, notebook cell (weight lr 0.1,
activity lr 0.5), T = H, seeds 0 to 4. Arms, all in runs/freeze/:
  mupc_full         all layers train
  mupc_frz1-121     hidden W_1 .. W_121 frozen; input, W_122 .. W_127, readout train
  mupc_frz122-127   hidden W_122 .. W_127 frozen (descriptive only)
  bp_full, bp_frz1-121   BP Depth-muP (lr 5e-3), same arms
Decision rule, fixed before the runs: d = full - frz1-121, paired by seed,
at the last test point (iteration 900); 95% t-interval over the 5 seeds.
  upper end < 1 pp               -> lower-block freeze meets the 1 pp margin
  mean >= 1 pp and lower end > 0 -> mean loss meets 1 pp and interval excludes zero
  otherwise                      -> "inconclusive at 1 pp"
The same summary is written for BP and for the tuned-cell repeat
(runs/freeze_tuned/, same rule) and for CIFAR-10 (runs/freeze_cifar/, the C2
settings, seeds 0 to 2, descriptive: compared with runs/long).

Usage: python freeze_summary.py results/freeze.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
MARGIN = 1.0


def final_accs_lean(d):
    out = {}
    for f in sorted(Path(d).glob("seed*.json")):
        r = json.loads(f.read_text())
        if r["test_acc"]:
            out[r["seed"]] = r["test_acc"][-1]
    return out


def final_accs_bp(d):
    out = {}
    for f in sorted(Path(d).rglob("test_accs.npy")):
        # run_bpn.py trees end in .../<epochs>_epochs/<seed>/test_accs.npy
        out[int(f.parent.name)] = float(np.load(f)[-1])
    return out


def rule(full, frozen):
    seeds = sorted(set(full) & set(frozen))
    if len(seeds) < 2:
        return {"complete": False, "seeds": seeds}
    d = np.array([full[s] - frozen[s] for s in seeds])
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    t = stats.t.ppf(0.975, len(d) - 1)
    lo, hi = m - t * se, m + t * se
    if hi < MARGIN:
        verdict = "met the one-point margin"
    elif m >= MARGIN and lo > 0:
        verdict = "mean loss meets one point and interval excludes zero"
    else:
        verdict = "inconclusive at 1 pp"
    return {"complete": True, "seeds": seeds, "diff": d.tolist(), "mean": float(m),
            "ci95": [float(lo), float(hi)], "verdict": verdict}


def arm(accs):
    v = list(accs.values())
    return {"seeds": sorted(accs), "final_accs": [accs[s] for s in sorted(accs)],
            "mean": float(np.mean(v)) if v else None,
            "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None}


def main(out):
    R = HERE / "runs"
    res = {}
    for tag, root in (("notebook", R / "freeze"), ("tuned", R / "freeze_tuned")):
        if not root.exists():
            continue
        a = {k: final_accs_lean(root / k) for k in ("mupc_full", "mupc_frz1-121", "mupc_frz122-127")}
        res[tag] = {"arms": {k: arm(v) for k, v in a.items()},
                    "mupc_rule": rule(a["mupc_full"], a["mupc_frz1-121"])}
        if (root / "bp_full").exists():
            b = {k: final_accs_bp(root / k) for k in ("bp_full", "bp_frz1-121")}
            res[tag]["arms"].update({k: arm(v) for k, v in b.items()})
            res[tag]["bp_rule"] = rule(b["bp_full"], b["bp_frz1-121"])
    c = R / "freeze_cifar"
    if c.exists():
        res["cifar"] = {"mupc_frz1-121": arm(final_accs_lean(c / "mupc_frz1-121")),
                        "mupc_full": arm(final_accs_lean(R / "long" / "mupc_CIFAR10")),
                        "bp_frz1-121": arm(final_accs_bp(c / "bp_frz1-121"))}
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: {kk: vv.get("verdict") for kk, vv in v.items() if kk.endswith("rule")}
                      for k, v in res.items() if k != "cifar"}))


if __name__ == "__main__":
    main(sys.argv[1])
