"""Vectorised re-implementation of the official muPC training script.

Same model, energy, optimisers, initialisation and data order as
experiments/mupc_paper/train_pcn_no_metrics.py at the pinned jpc commit, but
the hidden layers are stacked into one array so that an inference step is two
batched matmuls, and a whole training iteration (T inference steps plus the
Adam update) is one compiled call. The official code loops over layers and
inference steps in Python and is host-bound on this machine.

Network, for L = n_hidden + 1 weight layers and activities z_0 .. z_{L-2}:
    e_0 = z_0 - a_1 W_0 x
    e_l = z_l - a_l W_l phi(z_{l-1}) - z_{l-1}      l = 1 .. L-2 (identity skip)
    e_L = y - a_L W_{L-1} phi(z_{L-2})
    F   = sum of 0.5 |e|^2 over layers and batch, divided by the batch size.
Scalings: SP all 1; muPC a_1 = 1/sqrt(D), a_l = 1/sqrt(N L), a_L = 1/N.

Data order: the official loaders are torch DataLoaders with shuffle=True and
drop_last=True, seeded by torch.manual_seed(seed). For MNIST we build loaders
of the same length over index tensors and iterate them in the same order as
the official loop, so each run sees the same batches as the official run.
Fashion-MNIST and CIFAR-10 (whose training set is augmented per sample) use
the official loaders themselves.

--freeze_hidden keeps the hidden weights W_1 .. W_{L-2} at initialisation and
trains only the input and output maps (an added-value arm, not in the paper).

Usage: python lean_mupc.py --out runs/lean/<name> --n_hidden 8 --param_type mupc
       --param_lr 1e-1 --activity_lr 5e-1 --seeds 0 1 2 [--log_weights]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
# The pinned jpc checkout (setup_upstream.sh); older layouts kept it in code/.
_VENDOR = HERE.parent / "vendor" / "upstream_jpc"
UPSTREAM = Path(os.environ.get("JPC_DIR", _VENDOR if _VENDOR.exists() else HERE / "upstream_jpc"))
DATA_DIR = Path(os.environ.get("MUPC_DATA", UPSTREAM / "datasets"))
sys.path.insert(0, str(UPSTREAM))

# Full float32 matmuls. With the GPU default, muPC at H = 8 and 16 stops at
# chance in all six runs; with "highest" all six reach the official accuracy
# (results/gpu_prec.json). MATMUL_PRECISION overrides it for that comparison.
PRECISION = os.environ.get("MATMUL_PRECISION", "highest")
jax.config.update("jax_default_matmul_precision", PRECISION)


def load_mnist():
    from torchvision.datasets import MNIST
    out = {}
    for split, train in (("train", True), ("test", False)):
        ds = MNIST(str(DATA_DIR), train=train, download=False)
        # ToTensor then Normalize(0.1307, 0.3081), in float32 as torchvision does.
        x = ds.data.reshape(len(ds.data), -1).to(torch.float32).div(255)
        x = x.sub(0.1307).div(0.3081)
        y = torch.nn.functional.one_hot(ds.targets, 10).to(torch.float32)
        out[split] = (x.numpy(), y.numpy())
    return out


def index_loader(n, batch_size):
    return DataLoader(TensorDataset(torch.arange(n)), batch_size=batch_size,
                      shuffle=True, drop_last=True)


def init_params(seed, d_in, width, n_hidden, param_type, act_fn):
    """Initial weights taken from jpc.make_mlp so they match the official run bit for bit."""
    import jpc
    L = n_hidden + 1
    keys = jax.random.split(jax.random.PRNGKey(seed), 4)
    model = jpc.make_mlp(key=keys[0], input_dim=d_in, width=width, depth=L,
                         output_dim=10, act_fn=act_fn, use_bias=False,
                         param_type=param_type)
    Ws = [layer[1].weight for layer in model]
    # n_hidden = 1 has no residual blocks: an empty stack, and scan does nothing.
    Wh = jnp.stack(Ws[1:-1]) if L > 2 else jnp.zeros((0, width, width), Ws[0].dtype)
    return {"W0": Ws[0], "Wh": Wh, "WL": Ws[-1]}


def scalings(param_type, d_in, width, n_hidden):
    L = n_hidden + 1
    if param_type == "sp":
        return 1.0, 1.0, 1.0
    a_L = 1.0 / width if param_type == "mupc" else 1.0 / np.sqrt(width)
    return 1.0 / np.sqrt(d_in), 1.0 / np.sqrt(width * L), a_L


ACTS = {"relu": jax.nn.relu, "tanh": jnp.tanh, "linear": lambda z: z}


def make_fns(param_type, d_in, width, n_hidden, act_fn, loss="mse"):
    """Feedforward pass, energy and test step. Learning rates are call arguments."""
    a1, al, aL = scalings(param_type, d_in, width, n_hidden)
    phi = ACTS[act_fn]

    def ffwd(p, x):
        z0 = a1 * x @ p["W0"].T

        def step(z, W):
            z = al * phi(z) @ W.T + z
            return z, z

        zlast, zh = jax.lax.scan(step, z0, p["Wh"])
        Z = jnp.concatenate([z0[None], zh], axis=0)
        return Z, aL * phi(zlast) @ p["WL"].T

    def energy(p, Z, x, y):
        B = x.shape[0]
        e0 = Z[0] - a1 * x @ p["W0"].T
        eh = Z[1:] - al * jnp.einsum("lbn,lmn->lbm", phi(Z[:-1]), p["Wh"]) - Z[:-1]
        out = aL * phi(Z[-1]) @ p["WL"].T
        if loss == "mse":
            eL = 0.5 * jnp.sum((y - out) ** 2)
        else:
            eL = -jnp.sum(y * jax.nn.log_softmax(out))
        return (0.5 * (jnp.sum(e0 ** 2) + jnp.sum(eh ** 2)) + eL) / B

    def out_loss(y, out):
        if loss == "mse":
            return 0.5 * jnp.mean((y - out) ** 2)
        return -jnp.mean(jnp.sum(y * jnp.log(jax.nn.softmax(out)), axis=-1))

    @jax.jit
    def test_batch(p, x, y):
        _, out = ffwd(p, x)
        acc = jnp.mean(jnp.argmax(y, 1) == jnp.argmax(out, 1)) * 100
        return out_loss(y, out), acc

    return ffwd, energy, out_loss, test_batch


def make_train_step(fns, T, opt):
    """One training iteration: feedforward init, T gradient steps on the
    activities, one optimiser step on the weights. The activity learning rate
    is an argument, so a learning-rate grid compiles once."""
    ffwd, energy, out_loss, _ = fns
    grad_z = jax.grad(energy, argnums=1)
    grad_p = jax.grad(energy, argnums=0)

    @jax.jit
    def train_step(p, opt_state, x, y, activity_lr):
        Z, out = ffwd(p, x)
        train_loss = out_loss(y, out)
        Z = jax.lax.fori_loop(0, T, lambda _, Z: Z - activity_lr * grad_z(p, Z, x, y), Z)
        g = grad_p(p, Z, x, y)
        upd, opt_state = opt.update(g, opt_state, p)
        return optax.apply_updates(p, upd), opt_state, train_loss

    return train_step


def adam(freeze_hidden=False):
    """Adam with its learning rate held in the optimiser state."""
    opt = optax.inject_hyperparams(optax.adam)(learning_rate=0.0)
    if freeze_hidden:
        opt = optax.multi_transform({"train": opt, "frozen": optax.set_to_zero()},
                                    {"W0": "train", "Wh": "frozen", "WL": "train"})
    return opt


def adam_init(opt, p, lr, freeze_hidden=False):
    state = opt.init(p)
    if freeze_hidden:
        inner = state.inner_states["train"].inner_state
        inner.hyperparams["learning_rate"] = jnp.asarray(lr, jnp.float32)
    else:
        state.hyperparams["learning_rate"] = jnp.asarray(lr, jnp.float32)
    return state


def weight_change(p, p0):
    """Relative change ||W - W0|| / ||W0|| per weight layer, input layer first."""
    Ws = [p["W0"], *p["Wh"], p["WL"]]
    W0s = [p0["W0"], *p0["Wh"], p0["WL"]]
    return [float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b)) for a, b in zip(Ws, W0s)]


class _Batches:
    """Yields (x, y) device batches in the official order."""

    def __init__(self, dataset, data, batch_size, train):
        if dataset == "MNIST":
            self.x, self.y = (jnp.asarray(v) for v in data["train" if train else "test"])
            self.loader = index_loader(len(self.x), batch_size)
        else:
            from experiments.datasets import get_dataset
            self.x = None
            self.loader = DataLoader(get_dataset(dataset, train=train, normalise=True),
                                     batch_size=batch_size, shuffle=True, drop_last=True)

    def __iter__(self):
        for batch in self.loader:
            if self.x is None:
                yield jnp.asarray(batch[0].numpy()), jnp.asarray(batch[1].numpy())
            else:
                idx = jnp.asarray(batch[0].numpy())
                yield self.x[idx], self.y[idx]


def train(seed, data, a, param_lr, activity_lr, train_step, test_batch, opt):
    np.random.seed(seed)
    torch.manual_seed(seed)
    p = init_params(seed, a.d_in, a.width, a.n_hidden, a.param_type, a.act_fn)
    p0 = p
    opt_state = adam_init(opt, p, param_lr, a.freeze_hidden)
    alr = jnp.asarray(activity_lr, jnp.float32)
    train_loader = _Batches(a.dataset, data, a.batch_size, True)
    test_loader = _Batches(a.dataset, data, a.batch_size, False)

    rec = {"train_loss": [], "test_loss": [], "test_acc": [], "weight_change": []}
    it, stop = 0, None
    for epoch in range(a.max_epochs):
        for x, y in train_loader:
            p, opt_state, loss = train_step(p, opt_state, x, y, alr)
            rec["train_loss"].append(loss)
            it += 1
            if it % a.test_every == 0:
                tl, ta = [], []
                for x, y in test_loader:
                    l_, a_ = test_batch(p, x, y)
                    tl.append(l_)
                    ta.append(a_)
                rec["test_loss"].append(float(jnp.mean(jnp.stack(tl))))
                rec["test_acc"].append(float(jnp.mean(jnp.stack(ta))))
                if a.log_weights:
                    rec["weight_change"].append(weight_change(p, p0))
            lf = float(loss)
            if np.isnan(lf) or np.isinf(lf):
                stop = "diverged"
                break
            # The official loop stops once the last test accuracy is below 15%.
            if rec["test_acc"] and rec["test_acc"][-1] < 15:
                stop = "no_learning"
                break
        if stop:
            break
    rec["train_loss"] = [float(v) for v in rec["train_loss"]]
    rec["iterations"] = it
    rec["stop"] = stop
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="MNIST", choices=["MNIST", "Fashion-MNIST", "CIFAR10"])
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--n_hidden", type=int, default=8)
    ap.add_argument("--act_fn", default="relu")
    ap.add_argument("--param_type", default="mupc")
    ap.add_argument("--loss", default="mse", choices=["mse", "ce"])
    ap.add_argument("--param_lr", type=float, nargs="+", default=[1e-1])
    ap.add_argument("--activity_lr", type=float, nargs="+", default=[5e-1])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_infer_iters", type=int, default=None, help="default: n_hidden")
    ap.add_argument("--max_epochs", type=int, default=1)
    ap.add_argument("--test_every", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--log_weights", action="store_true")
    ap.add_argument("--freeze_hidden", action="store_true")
    a = ap.parse_args()
    if a.max_infer_iters is None:
        a.max_infer_iters = a.n_hidden
    a.d_in = 32 * 32 * 3 if a.dataset == "CIFAR10" else 28 * 28
    root = Path(a.out).resolve()
    data = load_mnist() if a.dataset == "MNIST" else None
    if data is None:
        # The official dataset classes look for "datasets/" in the working directory.
        os.chdir(UPSTREAM)
    fns = make_fns(a.param_type, a.d_in, a.width, a.n_hidden, a.act_fn, a.loss)
    opt = adam(a.freeze_hidden)
    train_step = make_train_step(fns, a.max_infer_iters, opt)
    grid = len(a.param_lr) * len(a.activity_lr) > 1
    for plr in a.param_lr:
        for alr in a.activity_lr:
            out = root / f"plr{plr:g}_alr{alr:g}" if grid else root
            out.mkdir(parents=True, exist_ok=True)
            for seed in a.seeds:
                f = out / f"seed{seed}.json"
                if f.exists():
                    continue
                t0 = time.time()
                rec = train(seed, data, a, plr, alr, train_step, fns[3], opt)
                cfg = {**vars(a), "param_lr": plr, "activity_lr": alr}
                rec.update(config=cfg, seed=seed, seconds=round(time.time() - t0, 1),
                           jax=jax.__version__, device=str(jax.devices()[0]),
                           matmul_precision=PRECISION)
                f.write_text(json.dumps(rec))
                print(f"plr {plr:g} alr {alr:g} seed {seed}: acc {rec['test_acc']} "
                      f"stop {rec['stop']} {rec['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
