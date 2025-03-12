import toml
from flax import nnx

from layer import Layer
import jax.numpy as jnp

import matplotlib.pyplot as plt


class CEmodel(nnx.Module):
    def __init__(self, config):
        layers_config = config["layers"]
        self.layers = []
        for layer_config in layers_config:
            layer = Layer(
                config["grid"].copy(),
                layer_config["input"],
                layer_config["output"],
                config["basic"],
            )
            self.layers.append(layer)

    def __call__(self, masks: jnp.ndarray):
        tmp = masks
        for layer in self.layers:
            tmp = layer(tmp)
        return tmp

    def viz_abs(self, mask, verbose=False):
        num_layers = len(self.layers)
        fig, ax = plt.subplots(1, num_layers, figsize=(4 * num_layers, 6))

        funcs = []
        vmax_all = 0
        tmp = mask
        for (idx, layer) in enumerate(self.layers):
            if verbose:
                print(tmp)
            tmp, vmax, func = layer.viz_abs(tmp, ax=ax[idx])
            funcs.append(func)
            vmax_all = max(vmax_all, vmax)
        if verbose:
            print(tmp)

        for func in funcs:
            func(vmax_all)

        return fig, ax


if __name__ == "__main__":
    config_file = "./Configuration/2bit.toml"
    config = toml.load(config_file)
    model = CEmodel(config)

    print(model(
        jnp.array([
            [1, 0, 0, 1],
            [0, 1, 0, 1],
        ])
    ))

    model.viz_abs(jnp.array([1, 0, 0, 1]))
    plt.show()
