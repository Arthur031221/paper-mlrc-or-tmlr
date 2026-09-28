"""Per-layer relative weight change of the official BP baseline with Depth-muP.

The comparison for results/layer_profile.json: MNIST, H=128, width 512, Adam lr
5e-3, batch 64, MSE loss, as in long_runs.sh. Uses the official MLP, init_weights,
make_step, evaluate and data loaders (d_in set to 784, see run_bpn.py) and records
||W - W0|| / ||W0|| per layer, input layer first, at iterations 900 and 4500, with
the test accuracy at those points. Writes results/layer_profile_bp.json.
Run from the official repository root:  $PY $OLDPWD/bp_profile.py --seeds 0 1 2 3 4
"""
import argparse
import json
import os
import sys

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from experiments.datasets import get_dataloaders
from experiments.mupc_paper import train_bpn as m
from experiments.mupc_paper.utils import set_seed, init_weights

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from layer_profile import summarise, THRESH, TOP  # noqa: E402

OUT = os.path.join(HERE, "results", "layer_profile_bp.json")
CHECK = (900, 4500)


def weights(model):
    return [f.layers[1].weight for f in model.layers]


def change(model, w0):
    return [float(jnp.linalg.norm(w - v) / jnp.linalg.norm(v)) for w, v in zip(weights(model), w0)]


def run(seed, n_hidden=128, lr=5e-3, batch_size=64):
    set_seed(seed)
    model_key, init_key = jr.split(jr.PRNGKey(seed), 2)
    model = m.MLP(key=model_key, d_in=784, N=512, L=n_hidden + 1, d_out=10, act_fn="relu",
                  param_type="depth_mup", use_bias=False, use_skips=True)
    model = init_weights(model=model, init_fn_id="standard_gauss", key=init_key)
    w0 = weights(model)
    optim = optax.adam(lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    train_loader, test_loader = get_dataloaders("MNIST", batch_size)
    out, it = {}, 0
    while it < CHECK[-1]:
        for x, y in train_loader:
            model, opt_state, loss = m.make_step(model=model, optim=optim, opt_state=opt_state,
                                                 x=x.numpy(), y=y.numpy(), loss_id="mse")
            it += 1
            if it in CHECK:
                _, acc = m.evaluate(model, test_loader, "mse")
                out[it] = {"test_acc": float(acc), "change": change(model, w0)}
                print(f"seed {seed} it{it} acc {float(acc):.2f}", flush=True)
            if it >= CHECK[-1]:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    seeds = {}
    for s in a.seeds:
        r = run(s)
        seeds[str(s)] = {"n_hidden_matrices": len(r[900]["change"]) - 2,
                         **{f"it{k}": {"test_acc": v["test_acc"], **summarise(v["change"])}
                            for k, v in r.items()},
                         "profile_last": r[CHECK[-1]]["change"]}
    json.dump({"threshold": THRESH, "top": TOP, "n_seeds": len(seeds), "seeds": seeds},
              open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
