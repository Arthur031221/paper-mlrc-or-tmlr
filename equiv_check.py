"""Check lean_mupc.py against the official jpc functions, step by step.

Runs the official per-iteration calls (init_activities_with_ffwd,
update_activities T times, update_params) and the lean train step from the
same initial weights on the same MNIST batches, and records the largest
relative difference in the weights after each iteration. CPU, float32.

Adam divides each gradient entry by its own running scale, so rounding noise
in near-zero entries becomes a full-size step and two float32 runs of the same
maths drift apart. Two checks separate that from a real difference:
  sgd: both implementations with SGD on the weights, where rounding noise is
       not amplified; the lean run must stay within 1e-4 of the official one.
  adam: the lean drift is compared with a control, the official calls run
       again on inputs moved by one float32 ulp; the lean drift must be no
       larger than twice the control's.
Usage: JAX_PLATFORMS=cpu python equiv_check.py results/equiv.json
"""
import json
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch

import lean_mupc as lm
import jpc

ITERS = 20


def official_weights(model):
    Ws = [layer[1].weight for layer in model]
    return {"W0": Ws[0], "Wh": jnp.stack(Ws[1:-1]), "WL": Ws[-1]}


def rel(a, b):
    return max(float(jnp.linalg.norm(a[k] - b[k]) / jnp.linalg.norm(b[k])) for k in b)


def official_run(param_type, n_hidden, activity_lr, param_lr, data, seed=0, width=512,
                 optim="adam", x_ulp=False):
    """Weights after each of the first ITERS iterations of the official calls."""
    xtr, ytr = data["train"]
    d_in, L, T = xtr.shape[1], n_hidden + 1, n_hidden
    keys = jax.random.split(jax.random.PRNGKey(seed), 4)
    model = jpc.make_mlp(key=keys[0], input_dim=d_in, width=width, depth=L, output_dim=10,
                         act_fn="relu", use_bias=False, param_type=param_type)
    skip = jpc.make_skip_model(L)
    popt = getattr(optax, optim)(param_lr)
    pstate = popt.init((eqx.filter(model, eqx.is_array), skip))
    aopt = optax.sgd(activity_lr)

    torch.manual_seed(seed)
    loader = lm.index_loader(len(xtr), 64)
    traj = [official_weights(model)]
    for i, (idx,) in enumerate(loader):
        if i == ITERS:
            break
        x, y = xtr[idx.numpy()], ytr[idx.numpy()]
        if x_ulp:
            x = np.nextafter(x, np.float32(np.inf))
        acts = jpc.init_activities_with_ffwd(model=model, input=x, skip_model=skip,
                                             param_type=param_type)
        astate = aopt.init(acts)
        for _ in range(T):
            r = jpc.update_activities(params=(model, skip), activities=acts, optim=aopt,
                                      opt_state=astate, output=y, input=x, loss_id="mse",
                                      param_type=param_type)
            acts, astate = r["activities"], r["opt_state"]
        r = jpc.update_params(params=(model, skip), activities=acts, optim=popt,
                              opt_state=pstate, output=y, input=x, loss_id="mse",
                              param_type=param_type)
        model, pstate = r["model"], r["opt_state"]
        traj.append(official_weights(model))
    return traj


def lean_run(param_type, n_hidden, activity_lr, param_lr, data, p, seed=0, width=512,
             optim="adam"):
    xtr, ytr = data["train"]
    fns = lm.make_fns(param_type, xtr.shape[1], width, n_hidden, "relu")
    if optim == "adam":
        opt = lm.adam()
        state = lm.adam_init(opt, p, param_lr)
    else:
        opt = getattr(optax, optim)(param_lr)
        state = opt.init(p)
    train_step = lm.make_train_step(fns, n_hidden, opt)
    alr = jnp.asarray(activity_lr, jnp.float32)
    torch.manual_seed(seed)
    traj = [p]
    for i, (idx,) in enumerate(lm.index_loader(len(xtr), 64)):
        if i == ITERS:
            break
        p, state, _ = train_step(p, state, jnp.asarray(xtr[idx.numpy()]),
                                 jnp.asarray(ytr[idx.numpy()]), alr)
        traj.append(p)
    return traj


def run(param_type, n_hidden, activity_lr, param_lr, data):
    res = {"param_type": param_type, "n_hidden": n_hidden, "activity_lr": activity_lr,
           "param_lr": param_lr, "iterations": ITERS}
    off = official_run(param_type, n_hidden, activity_lr, param_lr, data)
    ctl = official_run(param_type, n_hidden, activity_lr, param_lr, data, x_ulp=True)
    lean = lean_run(param_type, n_hidden, activity_lr, param_lr, data, off[0])
    res["adam_lean_vs_official"] = [rel(a, b) for a, b in zip(lean[1:], off[1:])]
    res["adam_ulp_control_vs_official"] = [rel(a, b) for a, b in zip(ctl[1:], off[1:])]
    off = official_run(param_type, n_hidden, activity_lr, param_lr, data, optim="sgd")
    lean = lean_run(param_type, n_hidden, activity_lr, param_lr, data, off[0], optim="sgd")
    res["sgd_lean_vs_official"] = [rel(a, b) for a, b in zip(lean[1:], off[1:])]
    for k in ("adam_lean_vs_official", "adam_ulp_control_vs_official", "sgd_lean_vs_official"):
        assert all(np.isfinite(res[k])), f"non-finite weights in {k}"
    res["pass_sgd"] = bool(max(res["sgd_lean_vs_official"]) < 1e-4)
    res["pass_adam"] = bool(max(res["adam_lean_vs_official"])
                            <= 2 * max(max(res["adam_ulp_control_vs_official"]), 1e-7))
    return res


if __name__ == "__main__":
    data = lm.load_mnist()
    out = {"jax": jax.__version__, "device": str(jax.devices()[0]),
           "cases": [run("mupc", 8, 5e-1, 1e-1, data), run("sp", 8, 1e-2, 1e-1, data),
                     run("mupc", 32, 5e-1, 1e-1, data)]}
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    for c in out["cases"]:
        print(c["param_type"], c["n_hidden"],
              "adam lean", f"{max(c['adam_lean_vs_official']):.2e}",
              "control", f"{max(c['adam_ulp_control_vs_official']):.2e}",
              "sgd lean", f"{max(c['sgd_lean_vs_official']):.2e}",
              "pass", c["pass_sgd"], c["pass_adam"])
