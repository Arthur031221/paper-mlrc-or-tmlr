"""Summarize the post hoc partial-freeze test at the C3 tail-selected cell."""
import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE / "runs" / "tail_freeze"
ARMS = ("mupc_full", "mupc_frz1-121", "mupc_frz122-127")
SEEDS = (5, 6, 7, 8, 9)


def read_arm(name):
    records = {}
    for seed in SEEDS:
        path = ROOT / name / f"seed{seed}.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        config = record["config"]
        expected = {
            "dataset": "MNIST",
            "width": 512,
            "n_hidden": 128,
            "act_fn": "relu",
            "loss": "mse",
            "batch_size": 64,
            "max_epochs": 1,
            "max_infer_iters": 128,
            "test_every": 300,
            "param_lr": 0.5,
            "activity_lr": 0.5,
            "seeds": list(SEEDS),
        }
        if record["seed"] != seed or any(config.get(k) != v for k, v in expected.items()):
            raise ValueError(f"unexpected run configuration: {path}")
        if len(record["test_acc"]) >= 3:
            records[seed] = float(record["test_acc"][2])
    values = list(records.values())
    return {
        "seeds": sorted(records),
        "final_accs": [records[s] for s in sorted(records)],
        "mean": float(np.mean(values)) if values else None,
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
    }, records


def paired(full, frozen):
    seeds = sorted(set(full) & set(frozen))
    if len(seeds) < 2:
        return {"complete": False, "n": len(seeds)}
    diff = np.asarray([full[s] - frozen[s] for s in seeds], dtype=float)
    mean = float(diff.mean())
    se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
    critical = float(stats.t.ppf(0.975, len(diff) - 1))
    return {
        "complete": len(seeds) == len(SEEDS),
        "seeds": seeds,
        "n": len(seeds),
        "diff_full_minus_frozen_pp": diff.tolist(),
        "mean_pp": mean,
        "sd_pp": float(diff.std(ddof=1)),
        "ci95_pp": [mean - critical * se, mean + critical * se],
        "one_point_margin_met": bool(mean + critical * se < 1.0),
    }


def main():
    loaded = {name: read_arm(name) for name in ARMS}
    arms = {name: value[0] for name, value in loaded.items()}
    full = loaded["mupc_full"][1]
    contrasts = {
        name: paired(full, loaded[name][1])
        for name in ARMS[1:]
    }
    result = {
        "analysis": "post hoc partial-freeze comparison at the C3 tail-selected cell",
        "dataset": "MNIST",
        "architecture": {"width": 512, "hidden_layers": 128, "activation": "ReLU"},
        "selection": "mean of each seed's final 100 minibatch losses in the existing C3 grid",
        "learning_rates": {"parameter": 0.5, "activity": 0.5},
        "update": 900,
        "seeds": list(SEEDS),
        "arms": arms,
        "paired_full_minus_frozen": contrasts,
    }
    destination = HERE / "results" / "tail_freeze.json"
    destination.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
