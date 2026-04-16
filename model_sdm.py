import matplotlib.pyplot as plt
import jax.numpy as jnp
from flax import nnx

from layer_sdm import LayerSDMDemux


class CEmodelSDMDemux(nnx.Module):
    def __init__(self, config):
        layers_config = config["layers"]
        simulation_config = config.get("simulation", {})
        self.layers = []
        for layer_config in layers_config:
            layer = LayerSDMDemux(
                config["grid"].copy(),
                layer_config,
                config["basic"],
                simulation_config,
            )
            self.layers.append(layer)

    def __call__(self, masks: jnp.ndarray):
        tmp = masks
        for layer in self.layers:
            tmp = layer(tmp)
        return tmp

    def viz_abs(self, mask, verbose=False):
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))

        funcs = []
        vmax_all = 0
        layer = self.layers[0]
        for mode_idx in range(3):
            if verbose:
                print(mask)
            _, vmax, func = layer.viz_abs(mask, mode_idx, ax=ax[mode_idx])
            funcs.append(func)
            vmax_all = max(vmax_all, vmax)

        for func in funcs:
            func(vmax_all)

        return fig, ax
