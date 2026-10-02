"""Check whether C3's selected learning-rate cells depend on the loss score.

For each existing C3 sweep point, rank the cells eligible under c3_summary.py
by the mean, across seeds, of the mean training loss in the final 100 logged
updates. This is a post hoc descriptive sensitivity analysis. It does not
replace the prespecified minimum-loss score or its verdict.
"""
import json
import math
from pathlib import Path
from statistics import fmean

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "transfer"
RESULTS = HERE / "results"
WINDOW = 100


def read_cell(directory):
    files = sorted(directory.glob("seed*.json"))
    records = [json.loads(path.read_text()) for path in files]
    if len(records) < 3:
        return None
    if any(record.get("stop") == "diverged" for record in records):
        return None
    losses = [record.get("train_loss", []) for record in records]
    if any(len(values) < WINDOW for values in losses):
        return None
    if any(not all(math.isfinite(value) for value in values) for values in losses):
        return None
    tail_means = [fmean(values[-WINDOW:]) for values in losses]
    endpoint_accuracy = [
        record["test_acc"][-1] if record.get("test_acc") else None
        for record in records
    ]
    return {
        "score": fmean(tail_means),
        "tail_means_by_seed": tail_means,
        "endpoint_accuracy_by_seed": endpoint_accuracy,
    }


def main():
    c3 = json.loads((RESULTS / "c3.json").read_text())
    param_lr = c3["param_lr"]
    activity_lr = c3["activity_lr"]
    reference = c3["reference"]
    sweeps = [
        ("width", width, f"width_N{width}_H8")
        for width in (64, 128, 256, 512, 1024)
    ] + [
        ("depth", depth, f"depth_N512_H{depth}")
        for depth in (8, 16, 32, 64, 128)
    ]

    rows = []
    for axis, value, name in sweeps:
        primary = next(
            row["best"] for row in c3["rows"]
            if row["axis"] == axis and row["value"] == value
        )
        scores = []
        for p_index, weight_lr in enumerate(param_lr):
            for a_index, activity in enumerate(activity_lr):
                details = read_cell(
                    RUNS / name / f"plr{weight_lr:g}_alr{activity:g}"
                )
                if details is not None:
                    scores.append((details["score"], p_index, a_index, details))

        if not scores:
            rows.append({"axis": axis, "value": value, "eligible_cells": 0})
            continue

        tail_score, p_index, a_index, tail = min(scores, key=lambda item: item[0])
        primary_index = tuple(primary)
        primary_result = next(
            details for score, i, j, details in scores
            if (i, j) == primary_index
        )
        reference_result = next(
            details for score, i, j, details in scores
            if (i, j) == tuple(reference)
        )
        rows.append({
            "axis": axis,
            "value": value,
            "eligible_cells": len(scores),
            "primary_best_index": primary,
            "primary_best_lr": [param_lr[primary[0]], activity_lr[primary[1]]],
            "tail_best_index": [p_index, a_index],
            "tail_best_lr": [param_lr[p_index], activity_lr[a_index]],
            "tail_best_is_reference": [p_index, a_index] == reference,
            "reference_tail_score": reference_result["score"],
            "primary_best_tail_score": primary_result["score"],
            "primary_best_to_reference_tail_ratio": (
                primary_result["score"] / reference_result["score"]
            ),
            "tail_best_score": tail_score,
            "tail_best_seed_means": tail["tail_means_by_seed"],
            "tail_best_endpoint_accuracy": tail["endpoint_accuracy_by_seed"],
        })

    result = {
        "analysis": "post hoc C3 score sensitivity",
        "alternative_score": "mean across seeds of each seed's final 100 logged minibatch losses",
        "window_updates": WINDOW,
        "reference_index": reference,
        "reference_lr": [param_lr[reference[0]], activity_lr[reference[1]]],
        "rows": rows,
    }
    destination = RESULTS / "c3_tail_sensitivity.json"
    destination.write_text(json.dumps(result, indent=1) + "\n")
    for row in rows:
        print(
            f"{row['axis']:5s} {row['value']:4d} "
            f"primary {row.get('primary_best_lr')} "
            f"tail {row.get('tail_best_lr')} "
            f"tail_best_is_reference {row.get('tail_best_is_reference')} "
            f"primary/reference_tail {row.get('primary_best_to_reference_tail_ratio')}"
        )
    print(f"wrote {destination.relative_to(HERE)}")


if __name__ == "__main__":
    main()
