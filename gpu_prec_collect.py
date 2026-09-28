"""Collect runs/gpu_prec into results/gpu_prec.json: final test accuracy and
stop reason per (precision, H, seed), GPU, muPC, Fig. 1 settings."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
rows = []
for f in sorted((HERE / "runs" / "gpu_prec").glob("*_H*/seed*.json")):
    r = json.loads(f.read_text())
    rows.append({"precision": r["config"].get("matmul_precision", f.parent.name.split("_")[0]),
                 "n_hidden": r["config"]["n_hidden"], "seed": r["seed"],
                 "test_acc": r["test_acc"][-1] if r["test_acc"] else None,
                 "stop": r["stop"], "device": r["device"]})
(HERE / "results" / "gpu_prec.json").write_text(json.dumps(rows, indent=1))
print(len(rows), "runs")
