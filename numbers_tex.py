"""Generate paper/numbers.tex with the paper's experimental summary statistics.

Reads only files in results/. A macro whose source file is missing is
defined as \\pending so the draft compiles and the gap is visible.
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = HERE.parent / "paper" / "numbers.tex"


def sci(x):
    m, e = f"{x:.1e}".split("e")
    return f"{m}\\times 10^{{{int(e)}}}"


# Two-sided 97.5% quantile of Student's t by degrees of freedom.
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}


def paired_ci(x, y):
    """95% t interval of the per-seed difference x - y; both arms must share seeds."""
    assert x["seeds"] == y["seeds"], (x["seeds"], y["seeds"])
    d = [a - b for a, b in zip(x["final_accs"], y["final_accs"])]
    n = len(d)
    m = sum(d) / n
    sd = (sum((v - m) ** 2 for v in d) / (n - 1)) ** 0.5
    h = T975[n - 1] * sd / n ** 0.5
    return m - h, m + h


def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def main():
    macros = {}
    env = load("env.json")
    if env:
        for k in ("jax", "jaxlib", "equinox", "optax", "numpy"):
            macros["ver" + k.capitalize()] = env[k]
        macros["jpcCommit"] = env["jpc_commit"][:7]
        macros["gpuName"] = env["gpu"].split(",")[0].replace("NVIDIA GeForce ", "")

    eq = load("equiv.json")
    if eq:
        cases = {(c["param_type"], c["n_hidden"]): c for c in eq["cases"]}
        macros["eqIters"] = str(eq["cases"][0]["iterations"])
        sgd = max(max(c["sgd_lean_vs_official"]) for c in eq["cases"]
                  if "sgd_lean_vs_official" in c)
        macros["eqSgdMax"] = f"${sci(sgd)}$"
        c8 = cases[("mupc", 8)]
        macros["eqAdamLeanHeight"] = f"${sci(c8['adam_lean_vs_official'][-1])}$"
        macros["eqAdamUlpHeight"] = f"${sci(c8['adam_ulp_control_vs_official'][-1])}$"
        sp8 = cases[("sp", 8)]
        macros["eqSpAdam"] = f"{sp8['adam_lean_vs_official'][-1]:.2f}"
        macros["eqSpUlp"] = f"{sp8['adam_ulp_control_vs_official'][-1]:.2f}"

    prec = load("gpu_prec.json")
    if prec:
        dflt = [r for r in prec if r["precision"] == "default"]
        high = [r for r in prec if r["precision"] == "highest"]
        macros["precDefaultRuns"] = str(len(dflt))
        macros["precDefaultStopped"] = str(sum(r["stop"] == "no_learning" for r in dflt))
        macros["precHighestRuns"] = str(len(high))
        accs = [r["test_acc"] for r in high if r["test_acc"] is not None]
        macros["precHighestMin"] = f"{min(accs):.2f}" if accs else "\\pending"
        macros["precHighestMax"] = f"{max(accs):.2f}" if accs else "\\pending"
    else:
        for k in ("Runs", "Stopped"):
            macros["precDefault" + k] = "\\pending"
        for k in ("Runs", "Min", "Max"):
            macros["precHighest" + k] = "\\pending"

    c1 = load("c1.json")
    c1_keys = ("cOneMinMean", "cOneMaxMean", "cOneSpread", "cOneMinRun", "cOneSpDeepMax",
               "cOneVerdict", "cOneAgreeConfigs", "cOneAgreeRuns", "cOneSpStopped", "cOneSpRuns",
               "cOneDeepMean", "cOneDeepSd", "cOneBpMean", "cOneBpSd", "speedOfficial", "speedLean",
               "cOneAgreeTested", "cOneAgreeNoTest", "cOneSpDeepRuns", "cOneSpDeepNoTest")
    if c1 and c1["complete"] and load("lean_vs_official.json"):
        lvo = load("lean_vs_official.json")
        macros["cOneMinMean"] = f"{c1['mupc_min_mean']:.2f}"
        macros["cOneMaxMean"] = f"{c1['mupc_max_mean']:.2f}"
        macros["cOneSpread"] = f"{c1['mupc_spread']:.2f}"
        macros["cOneMinRun"] = f"{c1['mupc_min_run']:.2f}"
        macros["cOneSpDeepMax"] = f"{c1['sp_deep_max']:.0f}"
        macros["cOneVerdict"] = "passes" if c1["pass"] else "fails"
        macros["cOneAgreeConfigs"] = str(lvo["configs_compared"])
        macros["cOneAgreeRuns"] = str(sum(len(r["seeds"]) for r in lvo["rows"]))
        tested = [r for r in lvo["rows"] if r["lean_accs"]]
        macros["cOneAgreeTested"] = str(len(tested))
        macros["cOneAgreeNoTest"] = str(len(lvo["rows"]) - len(tested))
        sp = [r for r in c1["rows"] if r["method"] == "sp"]
        # c1_summary.py records a run with no test point as 10.0 with stopped=True.
        spd = [r for r in sp if r["source"] == "lean" and r["n_hidden"] >= 64]
        macros["cOneSpDeepRuns"] = str(sum(len(r["seeds"]) for r in spd))
        macros["cOneSpDeepNoTest"] = str(sum(1 for r in spd for a, st in zip(r["final_accs"], r["stopped"]) if st and a == 10.0))
        macros["cOneSpRuns"] = str(sum(len(r["seeds"]) for r in sp))
        macros["cOneSpStopped"] = str(sum(sum(r["stopped"]) for r in sp))
        bp = [r for r in c1["rows"] if r["method"] == "bp" and r["n_hidden"] == 128]
        deep = next(r for r in c1["rows"] if (r["source"], r["method"], r["n_hidden"]) == ("lean", "mupc", 128))
        macros["cOneDeepMean"] = f"{deep['mean']:.2f}"
        macros["cOneDeepSd"] = f"{deep['sd']:.2f}"
        macros["cOneBpMean"] = f"{bp[0]['mean']:.2f}" if bp else "\\pending"
        macros["cOneBpSd"] = f"{bp[0]['sd']:.2f}" if bp and bp[0]["sd"] is not None else "\\pending"
        h32 = next((r for r in lvo["rows"] if (r["param"], r["n_hidden"]) == ("mupc", 32)), None)
        if h32 and h32["official_seconds_per_run"]:
            macros["speedOfficial"] = f"{h32['official_seconds_per_run'] / 60:.0f}"
            macros["speedLean"] = f"{h32['lean_seconds_per_run']:.0f}"
        else:
            macros["speedOfficial"] = macros["speedLean"] = "\\pending"
    else:
        for k in c1_keys:
            macros[k] = "\\pending"

    add = load("added.json")
    add_arms = {"PcFull": "mupc_full", "PcFrozen": "mupc_frozen", "PcT": "mupc_T4H",
                "BpFull": "bp_full", "BpFrozen": "bp_frozen", "PcShallow": "mupc_H1",
                "BpShallow": "bp_H1"}
    add_keys = [f"add{k}{s}" for k in add_arms for s in ("", "Sd")] + [
        "addPcGap", "addPcGapSd", "addBpGap", "addBpGapSd", "addPcShallowPlr",
        "addPcShallowAlr", "addBpShallowLr", "addVerdict", "addShallowMinusDeep"] + [
        f"add{k}{e}" for k in ("PcGap", "BpGap", "ShallowMinusDeep") for e in ("Lo", "Hi")]
    if add and add["complete"]:
        for k, x, y in (("PcGap", "mupc_full", "mupc_frozen"), ("BpGap", "bp_full", "bp_frozen"),
                        ("ShallowMinusDeep", "mupc_H1", "mupc_full")):
            lo, hi = paired_ci(add["arms"][x], add["arms"][y])
            macros[f"add{k}Lo"], macros[f"add{k}Hi"] = f"{lo:.2f}", f"{hi:.2f}"
        a = add["arms"]
        for k, name in add_arms.items():
            macros[f"add{k}"] = f"{a[name]['mean']:.2f}" if a[name]["mean"] is not None else "\\pending"
            macros[f"add{k}Sd"] = f"{a[name]['sd']:.2f}" if a[name]["sd"] is not None else "\\pending"
        for k, g in (("Pc", add["mupc_gap"]), ("Bp", add["bp_gap"])):
            macros[f"add{k}Gap"] = f"{g['mean']:.2f}"
            macros[f"add{k}GapSd"] = f"{g['sd']:.2f}"
        plr, alr = a["mupc_H1"]["cell"][3:].split("_alr")
        macros["addPcShallowPlr"], macros["addPcShallowAlr"] = plr, alr
        macros["addBpShallowLr"] = a["bp_H1"]["cell"][2:]
        macros["addVerdict"] = add["verdict"]
        macros["addShallowMinusDeep"] = f"{a['mupc_H1']['mean'] - a['mupc_full']['mean']:.2f}"
    else:
        for k in add_keys:
            macros[k] = "\\pending"

    if add and add["complete"]:
        a = add["arms"]
        macros["addPcTMinusFull"] = f"{a['mupc_full']['mean'] - a['mupc_T4H']['mean']:.2f}"
    else:
        macros["addPcTMinusFull"] = "\\pending"

    gpu_shallow = load("gpu_shallow_check.json")
    for k in ("gpuShallowMean", "gpuShallowSd"):
        macros[k] = "\\pending"
    if gpu_shallow:
        macros["gpuShallowMean"] = f"{gpu_shallow['mean']:.2f}"
        macros["gpuShallowSd"] = f"{gpu_shallow['sd']:.2f}"

    # Partial-freeze test (freeze_summary.py). Nb: notebook cell, Tu: H = 128 grid-best cell.
    frz = load("freeze.json") or {}
    arm_names = {"PcFull": "mupc_full", "PcLow": "mupc_frz1-121", "PcTop": "mupc_frz122-127",
                 "BpFull": "bp_full", "BpLow": "bp_frz1-121"}
    for tag, key in (("Nb", "notebook"), ("Tu", "tuned")):
        r = frz.get(key, {})
        arms = r.get("arms", {})
        for k, name in arm_names.items():
            x = arms.get(name, {})
            ok = x.get("mean") is not None and len(x.get("seeds", [])) == 5
            macros[f"frz{tag}{k}"] = f"{x['mean']:.2f}" if ok else "\\pending"
            macros[f"frz{tag}{k}Sd"] = f"{x['sd']:.2f}" if ok else "\\pending"
        for k, rk in (("Pc", "mupc_rule"), ("Bp", "bp_rule")):
            ru = r.get(rk, {})
            done = ru.get("complete") and len(ru.get("seeds", [])) == 5
            macros[f"frz{tag}{k}Diff"] = f"{ru['mean']:.2f}" if done else "\\pending"
            macros[f"frz{tag}{k}Lo"] = f"{ru['ci95'][0]:.2f}" if done else "\\pending"
            macros[f"frz{tag}{k}Hi"] = f"{ru['ci95'][1]:.2f}" if done else "\\pending"
            macros[f"frz{tag}{k}Verdict"] = ru["verdict"] if done else "\\pending"
    cf = frz.get("cifar", {})
    for k, name in (("CfPcLow", "mupc_frz1-121"), ("CfBpLow", "bp_frz1-121")):
        x = cf.get(name, {})
        ok = x.get("mean") is not None and len(x.get("seeds", [])) == 3
        macros[f"frz{k}"] = f"{x['mean']:.2f}" if ok else "\\pending"
        macros[f"frz{k}Sd"] = f"{x['sd']:.2f}" if ok else "\\pending"

    # Standard parameterisation over the 4 x 11 grid (sp_summary.py).
    sp = load("sp_grid.json") or {}
    for k in ("spEightAcc", "spSixteenAcc", "spThirtytwoAcc", "spSixtyfourAcc",
              "spOnetwentyeightAcc", "spMaxDeep", "spVerdict", "spEligibleDeep"):
        macros[k] = str(sp[k]) if k in sp else "\\pending"

    # Official code at H = 64 against the reimplementation (official_h64.sh).
    o64 = load("official_h64.json") or {}
    for k in ("offSixtyfourMean", "offSixtyfourSd", "offSixtyfourLean", "offSixtyfourAgree"):
        macros[k] = str(o64[k]) if k in o64 else "\\pending"

    prof = load("layer_profile.json")
    prof_keys = ["profSeeds", "profHidden", "profThresh", "profNAbove", "profFirstAbove",
                 "profInMax", "profBelowMax", "profBelowMedMax", "profReadoutMin",
                 "profLastIt", "profBelowMaxLast", "profNAboveLast"]
    if prof and prof["n_seeds"]:
        s = list(prof["seeds"].values())
        a = [v["it900"] for v in s]
        b = [v["last"] for v in s]
        n_above = {x["n_hidden_at_or_above_thresh"] for x in a}
        first = {x["first_hidden_at_or_above_thresh"] for x in a}
        n_last = {x["n_hidden_at_or_above_thresh"] for x in b}
        assert len(n_above) == len(first) == len(n_last) == 1, (n_above, first, n_last)
        macros.update({
            "profSeeds": str(prof["n_seeds"]),
            "profHidden": str(s[0]["n_hidden_matrices"]),
            "profThresh": f"{prof['threshold']:g}",
            "profNAbove": str(n_above.pop()),
            "profFirstAbove": str(first.pop()),
            "profInMax": f"{max(x['input'] for x in a):.9g}",
            "profBelowMax": f"{max(x['hidden_below_top_max'] for x in a):.9g}",
            "profBelowMedMax": f"{max(x['hidden_below_top_median'] for x in a):.9g}",
            "profReadoutMin": f"{min(x['readout'] for x in a):.9g}",
            "profLastIt": str(s[0]["last_iteration"]),
            "profBelowMaxLast": f"{max(x['hidden_below_top_max'] for x in b):.9g}",
            "profNAboveLast": str(n_last.pop()),
        })
    else:
        for k in prof_keys:
            macros[k] = "\\pending"
    macros["profHidMedMax"] = (f"{max(x['it900']['hidden_median'] for x in prof['seeds'].values()):.9g}"
                               if prof and prof["n_seeds"] else "\\pending")

    # Same profile for the BP baseline with Depth-muP (bp_profile.py).
    bpp = load("layer_profile_bp.json")
    bpp_keys = ["bpProfSeeds", "bpProfAcc", "bpProfNAboveMin", "bpProfNAboveMax",
                "bpProfHidMedMin", "bpProfNAboveLastMin", "bpProfNAboveLastMax", "bpProfInMin"]
    if bpp and bpp["n_seeds"]:
        a = [v["it900"] for v in bpp["seeds"].values()]
        b = [v["it4500"] for v in bpp["seeds"].values()]
        macros.update({
            "bpProfSeeds": str(bpp["n_seeds"]),
            "bpProfAcc": f"{sum(x['test_acc'] for x in a) / len(a):.2f}",
            "bpProfNAboveMin": str(min(x["n_hidden_at_or_above_thresh"] for x in a)),
            "bpProfNAboveMax": str(max(x["n_hidden_at_or_above_thresh"] for x in a)),
            "bpProfHidMedMin": f"{min(x['hidden_median'] for x in a):.9g}",
            "bpProfNAboveLastMin": str(min(x["n_hidden_at_or_above_thresh"] for x in b)),
            "bpProfNAboveLastMax": str(max(x["n_hidden_at_or_above_thresh"] for x in b)),
            "bpProfInMin": f"{min(x['input'] for x in a):.9g}",
        })
    else:
        for k in bpp_keys:
            macros[k] = "\\pending"

    # Gradient-norm profile at the notebook cell (grad_profile.py), muPC only.
    gp = load("grad_profile.json")
    grad_keys = ["gradRatioItOneMin", "gradRatioItOneMax", "gradRatioLastMin", "gradRatioLastMax"]
    if gp and gp.get("mupc_full"):
        it1 = [v["it1"]["ratio"] for v in gp["mupc_full"].values()]
        it900 = [v["it900"]["ratio"] for v in gp["mupc_full"].values()]
        macros["gradRatioItOneMin"] = f"{min(it1):.0f}"
        macros["gradRatioItOneMax"] = f"{max(it1):.0f}"
        macros["gradRatioLastMin"] = f"{min(it900):.0f}"
        macros["gradRatioLastMax"] = f"{max(it900):.0f}"
    else:
        for k in grad_keys:
            macros[k] = "\\pending"

    c2 = load("c2.json")
    c2_ds = {"Mn": "MNIST", "Fm": "Fashion-MNIST", "Cf": "CIFAR10"}
    for k, ds in c2_ds.items():
        row = c2["datasets"].get(ds) if c2 else None
        keys = [f"ctwo{k}{m}" for m in ("Pc", "PcSd", "Bp", "BpSd", "Gap", "Pass", "HidMed", "In")]
        if not row:
            for key in keys:
                macros[key] = "\\pending"
            continue
        macros.update({
            f"ctwo{k}Pc": f"{row['mupc']['mean']:.2f}",
            f"ctwo{k}PcSd": f"{row['mupc']['sd']:.2f}",
            f"ctwo{k}Bp": f"{row['bp']['mean']:.2f}",
            f"ctwo{k}BpSd": f"{row['bp']['sd']:.2f}",
            f"ctwo{k}Gap": f"{row['gap']:.2f}",
            f"ctwo{k}Pass": "met" if row["pass"] else "not met",
            f"ctwo{k}HidMed": f"{row['weight_change']['hidden_median']:.4f}",
            f"ctwo{k}In": sci(row["weight_change"]["input"]),
        })

    # Which side of the CIFAR-10 band the muPC seeds fell on.
    cf = c2["datasets"].get("CIFAR10") if c2 else None
    if cf:
        from c2_summary import CRIT
        lo, hi = CRIT["CIFAR10"]
        runs = cf["mupc"]["runs"]
        macros["ctwoCfAbove"] = str(sum(r > hi for r in runs))
        macros["ctwoCfBelow"] = str(sum(r < lo for r in runs))
        macros["ctwoCfNSeeds"] = str(len(runs))
        macros["ctwoCfPcMin"] = f"{min(runs):.2f}"
    else:
        for key in ("ctwoCfAbove", "ctwoCfBelow", "ctwoCfNSeeds", "ctwoCfPcMin"):
            macros[key] = "\\pending"

    done = [r for r in (c2["datasets"].values() if c2 else []) if r]
    macros["ctwoNPass"] = str(sum(r["pass"] for r in done)) if c2 and c2["complete"] else "\\pending"

    c3 = load("c3.json")
    c3_keys = ["cthreeRefPlr", "cthreeRefAlr", "cthreeDepth", "cthreeWidth", "cthreeWithin",
               "cthreePass", "cthreeDeepPlr", "cthreeDeepAlr"]
    if c3 and c3["complete"] and "pass" in c3:
        deep = next(r for r in c3["rows"] if r["axis"] == "depth" and r["value"] == 128)
        macros.update({
            "cthreeRefPlr": f"{c3['reference_lr'][0]:g}",
            "cthreeRefAlr": f"{c3['reference_lr'][1]:g}",
            "cthreeDepth": str(c3["matches_depth"]),
            "cthreeWidth": str(c3["matches_width"]),
            "cthreeWithin": "every" if c3["all_within_one"] else "not every",
            "cthreePass": "passes" if c3["pass"] else "fails",
            "cthreeDeepPlr": f"{deep['best_lr'][0]:g}",
            "cthreeDeepAlr": f"{deep['best_lr'][1]:g}",
        })
    else:
        for k in c3_keys:
            macros[k] = "\\pending"

    # Per sweep point, filled as soon as that point's grid is complete.
    words = {8: "Eight", 16: "Sixteen", 32: "Thirtytwo", 64: "Sixtyfour", 128: "Onetwentyeight",
             256: "Twofiftysix", 512: "Fivetwelve", 1024: "Tentwentyfour"}
    for r in (c3["rows"] if c3 else []):
        k = "cthree" + ("W" if r["axis"] == "width" else "D") + words[r["value"]]
        if r.get("complete") and r.get("regret") is not None:
            sr = r["seed_regret"]
            macros.update({
                k + "Plr": f"{r['best_lr'][0]:g}", k + "Alr": f"{r['best_lr'][1]:g}",
                k + "Regret": f"{r['regret']:.2f}", k + "Rank": str(r["rank"]),
                k + "Eligible": str(r["eligible_cells"]),
                k + "SeedMin": f"{min(sr):.2f}", k + "SeedMax": f"{max(sr):.2f}",
                k + "AccRef": "--" if r.get("acc_ref") is None else f"{r['acc_ref']:.2f}",
                k + "AccBest": "--" if r.get("acc_best") is None else f"{r['acc_best']:.2f}",
                k + "AccBestMin": f"{min(r['acc_best_seeds']):.2f}" if r.get("acc_best_seeds") else "--",
            })
        else:
            for s_ in ("Plr", "Alr", "Regret", "Rank", "Eligible", "SeedMin", "SeedMax",
                       "AccRef", "AccBest", "AccBestMin"):
                macros[k + s_] = "\\pending"
    wrows = [r for r in (c3["rows"] if c3 else []) if r["axis"] == "width"]
    ref = next((r["best"] for r in (c3["rows"] if c3 else [])
                if r["axis"] == "depth" and r["value"] == 8 and r.get("complete")), None)
    if ref is not None and wrows and all(r.get("complete") for r in wrows):
        others = [r for r in wrows if r["value"] != 512]
        macros["cthreeWidthOther"] = str(sum(r["best"] == ref for r in others))
        macros["cthreeWidthOtherN"] = str(len(others))
    else:
        macros["cthreeWidthOther"] = macros["cthreeWidthOtherN"] = "\\pending"
    # Grid step sizes in decades, from the grids themselves.
    if c3:
        macros["cthreeStepPlr"] = f"{math.log10(2):.1f}"
        macros["cthreeStepAlrFive"] = f"{math.log10(5):.1f}"

    smoke = RES / "smoke_acc.csv"
    if smoke.exists():
        rows = [r.split(",") for r in smoke.read_text().split()[1:]]
        macros["smokeFinal"] = f"{float(rows[-1][-1]):.2f}"

    lines = ["% generated by code/numbers_tex.py; do not edit",
             "\\newcommand{\\pending}{\\textbf{[pending]}}"]
    for k, v in sorted(macros.items()):
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"{len(macros)} macros -> {OUT}")


if __name__ == "__main__":
    main()
