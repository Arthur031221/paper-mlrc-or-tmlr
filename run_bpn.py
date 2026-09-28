"""Run the official BP baseline (experiments/mupc_paper/train_bpn.py) on MNIST.

train_bpn.py builds its MLP with d_in=3072 written into the call (the CIFAR-10
input size; the comment beside it lists 784 as the other option), so it cannot
run on MNIST as released. This wrapper sets d_in from the dataset and otherwise
calls the official train_mlp with the official argument names and defaults.
--freeze_hidden (not an official option) zeroes the gradients of the hidden
layers, so Adam leaves them at initialisation and only the input map and the
readout train: the BP counterpart of lean_mupc.py --freeze_hidden.
Run from the official repository root, as the job scripts do.
"""
import sys

from experiments.mupc_paper import train_bpn as m

def frozen_hidden_step(m):
    """The official make_step with the hidden-layer gradients set to zero."""
    import equinox as eqx
    import jax.numpy as jnp

    def hidden(g):
        return [g.layers[i][1].weight for i in range(1, len(g.layers) - 1)]

    @eqx.filter_jit
    def make_step(model, optim, opt_state, x, y, loss_id="mse"):
        loss, grads = eqx.filter_value_and_grad(m.get_loss_fn(loss_id))(model, x, y)
        grads = eqx.tree_at(hidden, grads, [jnp.zeros_like(w) for w in hidden(grads)])
        updates, opt_state = optim.update(
            updates=grads, state=opt_state, params=eqx.filter(model, eqx.is_array))
        return eqx.apply_updates(model, updates), opt_state, loss
    return make_step


D_IN = {"MNIST": 784, "Fashion-MNIST": 784, "CIFAR10": 3072}

if __name__ == "__main__":
    # Parse with the official parser by running its argument section.
    argv = sys.argv[1:]
    dataset = argv[argv.index("--dataset") + 1] if "--dataset" in argv else "CIFAR10"
    if "--freeze_hidden" in sys.argv:
        sys.argv.remove("--freeze_hidden")
        m.make_step = frozen_hidden_step(m)
    orig = m.MLP
    m.MLP = lambda **kw: orig(**{**kw, "d_in": D_IN[dataset]})
    src = open(m.__file__).read()
    main = src[src.index('if __name__ == "__main__":'):].replace(
        'if __name__ == "__main__":', "if True:", 1)
    exec(compile(main, m.__file__, "exec"), vars(m))
