"""Check that the shallow muPC control (H = 1, chosen on the CPU grid) reproduces on the GPU.

Reads runs/added/mupc_H1_gpu (freeze_test.sh, same cell as the CPU grid winner,
full float32 matmul precision) and results/added.json (the CPU arm). Writes
results/gpu_shallow_check.json.
Usage: python gpu_shallow_check.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def final(rec, n=3):
    acc = rec["test_acc"]
    return float(acc[n - 1]) if len(acc) >= n else None


def main():
    gpu_dir = HERE / "runs" / "added" / "mupc_H1_gpu"
    seeds = sorted(json.loads(f.read_text())["seed"] for f in gpu_dir.glob("seed*.json"))
    accs = [final(json.loads(f.read_text())) for f in sorted(gpu_dir.glob("seed*.json"))]
    mean = sum(accs) / len(accs)
    sd = (sum((a - mean) ** 2 for a in accs) / (len(accs) - 1)) ** 0.5
    added = json.loads((HERE / "results" / "added.json").read_text())
    cpu = added["arms"]["mupc_H1"]
    out = {"seeds": seeds, "final_accs": accs, "mean": mean, "sd": sd,
           "cpu_mean": cpu["mean"], "cpu_sd": cpu["sd"], "cell": cpu["cell"]}
    (HERE / "results" / "gpu_shallow_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
