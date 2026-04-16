import matplotlib.pyplot as plt
import jax.numpy as jnp
from flax import nnx

from layer_wdm import LayerWDMDemux


class CEmodelWDMDemux(nnx.Module):
    def __init__(self, config):
        layers_config = config["layers"]
        simulation_config = config.get("simulation", {})
        self.layers = []
        for layer_config in layers_config:
            layer = LayerWDMDemux(
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
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        funcs = []
        vmax_all = 0
        layer = self.layers[0]
        for wavelength_idx in range(2):
            if verbose:
                print(mask)
            _, vmax, func = layer.viz_abs(mask, wavelength_idx, ax=ax[wavelength_idx])
            funcs.append(func)
            vmax_all = max(vmax_all, vmax)

        for func in funcs:
            func(vmax_all)

        return fig, ax
