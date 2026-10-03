"""Combine paired accuracy effects from the three H=128 freeze settings."""
import json
from pathlib import Path

from scipy import stats

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SELECTION_SEEDS = [0, 1, 2]


def records(arm):
    return dict(zip(arm["seeds"], arm["final_accs"]))


def interval(differences):
    values = [float(value) for value in differences]
    n = len(values)
    if n < 2:
        raise ValueError("at least two paired seeds are required")
    mean = sum(values) / n
    sd = stats.tstd(values)
    half_width = float(stats.t.ppf(0.975, n - 1) * sd / n**0.5)
    return {
        "n": n,
        "mean_pp": mean,
        "sd_pp": float(sd),
        "ci95_pp": [mean - half_width, mean + half_width],
        "differences_pp": values,
    }


def paired_difference(full, frozen):
    full_by_seed = records(full)
    frozen_by_seed = records(frozen)
    seeds = sorted(full_by_seed.keys() & frozen_by_seed.keys())
    values = [full_by_seed[seed] - frozen_by_seed[seed] for seed in seeds]
    result = interval(values)
    result["seeds"] = seeds
    return result


def row(condition, method, block, full, frozen, selection, selection_seeds):
    result = paired_difference(full, frozen)
    result.update({
        "condition": condition,
        "method": method,
        "frozen_hidden_matrices": block,
        "contrast": "full training minus frozen block",
        "selection": selection,
        "selection_seeds": selection_seeds,
        "evaluation_seed_overlap_with_selection": sorted(
            set(result["seeds"]) & set(selection_seeds)
        ),
    })
    return result


def main():
    freeze = json.loads((RESULTS / "freeze.json").read_text())
    c3 = json.loads((RESULTS / "c3.json").read_text())
    tail = json.loads((RESULTS / "tail_freeze.json").read_text())
    depth_best = next(
        item for item in c3["rows"]
        if item.get("axis") == "depth" and item.get("value") == 128
    )

    notebook = freeze["notebook"]["arms"]
    tuned = freeze["tuned"]["arms"]
    tail_arms = tail["arms"]
    notebook_seeds = sorted(freeze["notebook"]["mupc_rule"]["seeds"])
    tuned_seeds = sorted(freeze["tuned"]["mupc_rule"]["seeds"])
    tail_seeds = sorted(tail["seeds"])

    rows = [
        row("published notebook learning rates", "muPC", "W1-W121",
            notebook["mupc_full"], notebook["mupc_frz1-121"],
            "fixed published setting", []),
        row("published notebook learning rates", "muPC", "W122-W127",
            notebook["mupc_full"], notebook["mupc_frz122-127"],
            "fixed published setting", []),
        row("published notebook learning rates", "backpropagation",
            "W1-W121", notebook["bp_full"], notebook["bp_frz1-121"],
            "fixed published setting", []),
        row("C3 minimum mean-training-loss cell", "muPC", "W1-W121",
            tuned["mupc_full"], tuned["mupc_frz1-121"],
            "selected from C3 mean training loss", SELECTION_SEEDS),
        row("C3 minimum mean-training-loss cell", "muPC", "W122-W127",
            tuned["mupc_full"], tuned["mupc_frz122-127"],
            "selected from C3 mean training loss", SELECTION_SEEDS),
        row("C3 minimum final-100-loss cell", "muPC", "W1-W121",
            tail_arms["mupc_full"], tail_arms["mupc_frz1-121"],
            "selected from C3 final-100-minibatch mean loss", SELECTION_SEEDS),
        row("C3 minimum final-100-loss cell", "muPC", "W122-W127",
            tail_arms["mupc_full"], tail_arms["mupc_frz122-127"],
            "selected from C3 final-100-minibatch mean loss", SELECTION_SEEDS),
    ]

    notebook_mupc = paired_difference(
        notebook["mupc_full"], notebook["mupc_frz1-121"]
    )
    notebook_bp = paired_difference(
        notebook["bp_full"], notebook["bp_frz1-121"]
    )
    mupc_by_seed = dict(zip(notebook_mupc["seeds"], notebook_mupc["differences_pp"]))
    bp_by_seed = dict(zip(notebook_bp["seeds"], notebook_bp["differences_pp"]))
    common = sorted(mupc_by_seed.keys() & bp_by_seed.keys())
    contrast = interval([bp_by_seed[seed] - mupc_by_seed[seed] for seed in common])
    contrast.update({
        "condition": "published notebook learning rates",
        "methods": "backpropagation minus muPC freeze penalty",
        "frozen_hidden_matrices": "W1-W121",
        "seeds": common,
    })

    result = {
        "analysis": "paired H=128 accuracy contrasts for saved partial-freeze experiments",
        "dataset": "MNIST",
        "accuracy_unit": "percent",
        "difference_unit": "percentage points",
        "endpoint": "update 900",
        "interval": "paired two-sided 95% t interval over seeds",
        "selection_seed_ids_for_C3_cells": SELECTION_SEEDS,
        "minimum_mean_training_loss_cell": depth_best["best_lr"],
        "minimum_final_100_loss_cell": [
            tail["learning_rates"]["parameter"],
            tail["learning_rates"]["activity"],
        ],
        "rows": rows,
        "notebook_method_contrast": contrast,
        "notes": [
            "The notebook setting was fixed by the released configuration, not selected by C3.",
            "For the minimum mean-loss cell, evaluation seed IDs 0 to 2 overlap the C3 selection seed IDs.",
            "The final-100-loss cell was selected from the existing C3 grid and evaluated on seed IDs 5 to 9.",
            "Freezing effects describe accuracy changes while the other trainable weights adapt.",
        ],
        "sources": [
            "results/freeze.json",
            "results/c3.json",
            "results/tail_freeze.json",
        ],
    }
    destination = RESULTS / "freeze_contrast.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
