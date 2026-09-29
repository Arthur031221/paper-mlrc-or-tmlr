"""Official code at H = 64 against the reimplementation (official_h64.sh).

Reads the official runs in runs/fig1/pc (H = 64, muPC) and the lean runs in
runs/lean_fig1 (H = 64, muPC), both seeds 0 to 2, and applies the agreement
rule of compare_lean.py: the lean seed mean at the last common test point lies
within the official seed range widened by 1 point on each side.
Writes results/official_h64.json.
Usage: python official_h64_summary.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
off = [np.load(f).tolist() for f in sorted((HERE / "runs" / "fig1" / "pc").rglob("test_accs.npy"))
       if "/64_n_hidden/" in str(f) and "/mupc_param/" in str(f)]
lean = [json.loads(f.read_text())["test_acc"]
        for f in sorted((HERE / "runs" / "lean_fig1").rglob("seed*.json"))
        if json.loads(f.read_text())["config"]["n_hidden"] == 64
        and json.loads(f.read_text())["config"]["param_type"] == "mupc"]
res = {"n_official": len(off), "n_lean": len(lean)}
if len(off) == 3 and len(lean) == 3:
    k = min(min(len(a) for a in off), min(len(a) for a in lean)) - 1
    o = [a[k] for a in off]
    m = float(np.mean([a[k] for a in lean]))
    ok = min(o) - 1 <= m <= max(o) + 1
    res.update(offSixtyfourMean=f"{np.mean(o):.2f}", offSixtyfourSd=f"{np.std(o, ddof=1):.2f}",
               offSixtyfourLean=f"{m:.2f}",
               offSixtyfourAgree="within the agreement rule" if ok else "outside the agreement rule",
               test_point_index=k)
(HERE / "results" / "official_h64.json").write_text(json.dumps(res, indent=1))
print(res)
