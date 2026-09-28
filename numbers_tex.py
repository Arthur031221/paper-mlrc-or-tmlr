"""Write paper/numbers.tex: every number the paper quotes, as LaTeX macros.

Reads only files in results/. A macro whose source file is missing is
defined as \\pending so the draft compiles and the gap is visible.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = HERE.parent / "paper" / "numbers.tex"


def sci(x):
    m, e = f"{x:.1e}".split("e")
    return f"{m}\\times 10^{{{int(e)}}}"


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
               "cOneDeepMean", "cOneDeepSd", "cOneBpMean", "cOneBpSd", "speedOfficial", "speedLean")
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
        sp = [r for r in c1["rows"] if r["method"] == "sp"]
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
        "addPcShallowAlr", "addBpShallowLr", "addVerdict"]
    if add and add["complete"]:
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
    else:
        for k in add_keys:
            macros[k] = "\\pending"

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
