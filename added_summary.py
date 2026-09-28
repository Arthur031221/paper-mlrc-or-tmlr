"""Added-value arms at H = 128 on MNIST: does muPC use its hidden layers?

Reads runs/lean_fig1/mupc_H128 (muPC, full training), runs/fig1/bp (official
BP Depth-muP, full training) and runs/added (added_value.sh). Writes
results/added.json.

Decision rule (IDEA.md, fixed before any of these runs), on test accuracy
after 900 iterations with differences paired by seed; "s.d." is the sample
standard deviation of the paired differences:
  decidable  if BP full - BP frozen >= 1 pp and >= 2 s.d.;
  supported  if muPC full - muPC frozen >= 1 pp and >= 2 s.d., and the muPC
             H = 128 mean is at least 1 pp above the muPC H = 1 mean;
  refuted    if muPC full - muPC frozen < 1 pp;
  otherwise  "hidden layers train, no gain over shallow".
Shallow controls (H = 1): the learning-rate cell with the lowest seed mean of
the minimum training loss; runs the training code stopped (diverged, or at
chance at the first test) are not eligible.
Usage: python added_summary.py
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
A = HERE / "runs" / "added"


def lean_runs(d):
    """seed -> record, for one lean_mupc.py output directory."""
    return {json.loads(f.read_text())["seed"]: json.loads(f.read_text()) for f in d.glob("seed*.json")}


def bp_runs(root):
    """(lr, H) -> seed -> {"test_acc", "train_loss"} for one run_bpn.py tree."""
    out = defaultdict(dict)
    for f in root.rglob("test_accs.npy"):
        parts = f.relative_to(root).parts
        lr = next(float(p[3:]) for p in parts if p.startswith("lr_"))
        h = next(int(p.split("_")[0]) for p in parts if p.endswith("_n_hidden"))
        out[(lr, h)][int(parts[-2])] = {
            "test_acc": np.load(f).tolist(),
            "train_loss": np.load(f.parent / "train_losses.npy").ravel().tolist()}
    return out


def final(rec, n=3):
    """Accuracy after 900 iterations, or None if the run was stopped before it."""
    acc = rec["test_acc"]
    return float(acc[n - 1]) if len(acc) >= n else None


def arm(runs):
    accs = {s: final(r) for s, r in runs.items()}
    return {"seeds": sorted(accs), "final_accs": [accs[s] for s in sorted(accs)]}


def paired(full, frozen):
    seeds = [s for s in full["seeds"] if s in frozen["seeds"]]
    f = dict(zip(full["seeds"], full["final_accs"]))
    z = dict(zip(frozen["seeds"], frozen["final_accs"]))
    d = [f[s] - z[s] for s in seeds if f[s] is not None and z[s] is not None]
    if len(d) < 2:
        return None
    return {"n": len(d), "mean": float(np.mean(d)), "sd": float(np.std(d, ddof=1)), "diffs": d}


def select(cells):
    """cells: label -> seed -> record. Lowest seed mean of min training loss."""
    scored = {}
    for lab, runs in cells.items():
        if not runs or any(final(r) is None for r in runs.values()):
            continue
        losses = [np.nanmin(r["train_loss"]) for r in runs.values()]
        scored[lab] = float(np.mean(losses))
    if not scored:
        return None, None
    best = min(scored, key=scored.get)
    return best, scored


arms = {
    "mupc_full": arm(lean_runs(HERE / "runs" / "lean_fig1" / "mupc_H128")),
    "mupc_frozen": arm(lean_runs(A / "mupc_H128_frozen")),
    "mupc_T4H": arm(lean_runs(A / "mupc_H128_T512")),
}
bp_full = bp_runs(HERE / "runs" / "fig1" / "bp")
bp_frozen = bp_runs(A / "bp_H128_frozen")
arms["bp_full"] = arm(bp_full.get((5e-3, 128), {}))
arms["bp_frozen"] = arm(bp_frozen.get((5e-3, 128), {}))

grid = {d.name: lean_runs(d) for d in (A / "mupc_H1_grid").glob("plr*_alr*")}
best, scores = select(grid)
arms["mupc_H1"] = dict(arm(grid[best]) if best else {"seeds": [], "final_accs": []}, cell=best)
bp1 = bp_runs(A / "bp_H1_grid")
best_bp, scores_bp = select({f"lr{lr:g}": r for (lr, h), r in bp1.items() if h == 1})
arms["bp_H1"] = dict(arm(next(r for (lr, h), r in bp1.items() if f"lr{lr:g}" == best_bp))
                     if best_bp else {"seeds": [], "final_accs": []}, cell=best_bp)

for a in arms.values():
    v = [x for x in a["final_accs"] if x is not None]
    a["mean"] = float(np.mean(v)) if v else None
    a["sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else None

out = {"arms": arms, "mupc_H1_scores": scores, "bp_H1_scores": scores_bp}
bp_gap = paired(arms["bp_full"], arms["bp_frozen"])
pc_gap = paired(arms["mupc_full"], arms["mupc_frozen"])
out.update(bp_gap=bp_gap, mupc_gap=pc_gap)
complete = bp_gap and pc_gap and arms["mupc_H1"]["mean"] is not None
if complete:
    if not (bp_gap["mean"] >= 1 and bp_gap["mean"] >= 2 * bp_gap["sd"]):
        verdict = "undecided"
    elif pc_gap["mean"] < 1:
        verdict = "refuted"
    elif pc_gap["mean"] >= 2 * pc_gap["sd"] and arms["mupc_full"]["mean"] - arms["mupc_H1"]["mean"] >= 1:
        verdict = "supported"
    else:
        verdict = "hidden layers train, no gain over shallow"
    out["verdict"] = verdict
out["complete"] = bool(complete)
(HERE / "results" / "added.json").write_text(json.dumps(out, indent=1) + "\n")
for k, a in arms.items():
    print(f"{k:12s} {a.get('cell') or '':16s} {a['final_accs']} mean {a['mean']}")
print("bp_gap", bp_gap and {k: bp_gap[k] for k in ("mean", "sd")},
      "mupc_gap", pc_gap and {k: pc_gap[k] for k in ("mean", "sd")}, "verdict", out.get("verdict"))
