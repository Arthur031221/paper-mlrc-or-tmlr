"""Record the software and hardware the reproduction runs on."""
import json
import subprocess
from importlib.metadata import version
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKGS = ["jax", "jaxlib", "equinox", "optax", "diffrax", "optimistix", "lineax",
        "jaxtyping", "numpy", "torch", "torchvision"]

env = {p: version(p) for p in PKGS}
env["jpc_commit"] = subprocess.check_output(
    ["git", "-C", str(HERE.parent / "vendor" / "upstream_jpc"), "rev-parse", "HEAD"], text=True).strip()
env["gpu"] = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
    text=True).strip()
(HERE / "results").mkdir(exist_ok=True)
(HERE / "results" / "env.json").write_text(json.dumps(env, indent=1) + "\n")
print(json.dumps(env))
