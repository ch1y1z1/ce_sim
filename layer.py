import jax
import toml
import basis_generator_chi as bgc

import ceviche
from ceviche.fdfd import fdfd_ez
from ceviche.modes import insert_mode

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

import jax.numpy as jnp
import flax.nnx
from flax import nnx
import agjax

import functools

mpl.rcParams["font.family"] = "sans-serif"
jax.config.update("jax_enable_x64", True)


def sigmoid_b(E1, E0, alpha):
    x = -(E1 - E0) * alpha
    return 1 / (1 + jnp.exp(x))


def mode_overlap(E1, E2):
    """Defines an overlap integral between the simulated field and desired field"""
    return jnp.abs(jnp.sum(jnp.conj(E1) * E2))


def prepare_io(omega, resolution, input_list, output_list, epsr_total, m=1):
    ics = []
    for point in input_list:
        ic = insert_mode(omega, resolution, point["x"], point["y"], epsr_total, m=m)
        ics.append(ic)

    probes = []
    for point in output_list:
        probe = np.zeros(epsr_total.shape, dtype=complex)
        probe[point["x"], point["y"]] = 1
        probes.append(probe)

    return ics, probes


class Layer(flax.nnx.Module):
    def __init__(self, grid, input, output, basic):
        n_bits_i = input["n_bits"]
        grid["ny"] = int(256 / 5 * (2 * n_bits_i + 1))
        grid["nx"] = int(208)
        self.rho, self.bg_rho, self.opt_region, self.input_list, self.output_list = (
            bgc.init_layer(grid, input, output)
        )
        omega = 2 * jnp.pi * basic["freq"]

        self.grid = grid
        self.basic = basic

        self.epsr_total = bgc.epsr_parameterization(
            self.rho,
            self.bg_rho,
            self.opt_region,
            basic["epsilon_min"],
            basic["epsilon_max"],
        )

        self.ics, self.probes = prepare_io(
            omega,
            grid["resolution"],
            self.input_list,
            self.output_list,
            self.epsr_total,
        )

        self.simulation = fdfd_ez(
            omega, grid["resolution"], self.epsr_total, [grid["npml"], grid["npml"]]
        )

        self.E0s = []
        ic_total = 0
        for ic in self.ics:
            ic_total += ic
        _, _, Ez0 = self.simulation.solve(ic_total)
        for probe in self.probes:
            E0 = mode_overlap(Ez0, probe)
            self.E0s.append(E0 / 2 * basic["E0_scale"])

        self.rho_jax = flax.nnx.Param(self.rho)
        self.alpha = flax.nnx.Param(basic["alpha"])

    @functools.partial(
        agjax.wrap_for_jax, nondiff_argnums=(0,)
    )
    def solve(self, rho, source):
        rho = rho.reshape((self.grid["nx"], self.grid["ny"]))
        self.simulation.eps_r = bgc.epsr_parameterization(
            rho,
            self.bg_rho,
            self.opt_region,
            self.basic["epsilon_min"],
            self.basic["epsilon_max"],
        )
        _, _, Ez = self.simulation.solve(source)
        return Ez

    def __call__(self, masks: jnp.ndarray) -> jnp.ndarray:
        Ezs = []
        for mask in masks:
            source = jnp.sum(mask[:, None, None] * jnp.array(self.ics), axis=0)
            Ez = self.solve(self.rho_jax.value, source)
            Ezs.append(Ez)

        a_list = []
        for mdx in range(len(masks)):
            output_mask = []
            for i in range(len(self.probes)):
                a = sigmoid_b(
                    mode_overlap(Ezs[mdx], self.probes[i]), self.E0s[i], self.alpha.value
                )
                output_mask.append(a)
            a_list.append(jnp.array(output_mask))

        return jnp.array(a_list)

    def use_rho(self, rho):
        self.rho = rho.reshape((self.bg_rho.shape[0], self.bg_rho.shape[1]))

    def viz_abs(self, mask, ax):
        # reshape mask to match the dimensions of the input list for broadcasting
        source = jnp.sum(mask[:, None, None] * jnp.array(self.ics), axis=0)
        Ez = self.solve(self.rho_jax.value, source)

        output_mask = []
        for i in range(len(self.probes)):
            a = sigmoid_b(mode_overlap(Ez, self.probes[i]), self.E0s[i], self.alpha.value)
            output_mask.append(a)

        # ceviche.viz.abs(Ez, outline=self.epsr_total, ax=ax, cbar=False)
        for idx, point in enumerate(self.input_list):
            ax.plot(point["x"] * jnp.ones(len(point["y"])), point["y"], "w-", alpha=0.5)
            ax.text(
                point["x"] - 40,
                point["y"][len(point["y"]) // 2],
                f"{mask[idx]:.2f}",
                ha="center",
                va="center",
            )
        for idx, point in enumerate(self.output_list):
            ax.plot(point["x"] * jnp.ones(len(point["y"])), point["y"], "w-", alpha=0.5)
            ax.text(
                point["x"] + 40,
                point["y"][len(point["y"]) // 2],
                f"{output_mask[idx]:.2f}",
                ha="center",
                va="center",
            )

        return (
            jnp.array(output_mask),
            jnp.max(jnp.abs(Ez)),
            lambda vmax: (
                ceviche.viz.abs(
                    Ez, outline=self.epsr_total, ax=ax, cbar=False, vmax=vmax
                ),
                # 隐藏刻度
                ax.set_xticks([]),
                ax.set_yticks([]),
                # 隐藏标签
                ax.set_xlabel(""),
                ax.set_ylabel(""),
            ),
        )


if __name__ == "__main__":
    config = "./Configuration/2bit.toml"
    config = toml.load(config)
    train_config = config["train"]
    layer = config["layers"][0]
    la = Layer(config["grid"], layer["input"], layer["output"], config["basic"])

    # prepare dataset:
    masks = jnp.array([
        jnp.array([0, 1, 0, 1]),
        jnp.array([0, 1, 1, 0]),
        jnp.array([1, 0, 0, 1]),
        jnp.array([1, 0, 1, 0]),
    ])
    expected_output = [
        jnp.array([1, 1]),
        jnp.array([0, 0]),
        jnp.array([1, 0]),
        jnp.array([0, 1]),
    ]

    ret = la(masks)
    print(ret)

    import optax

    learning_rate = 0.03
    momentum = 0.9

    optimizer = flax.nnx.Optimizer(la, optax.adamw(learning_rate, momentum))
    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average('loss'),
    )


    def loss_fn(model: Layer, masks):
        out = model(masks)
        mse = jnp.sum(jnp.mean((jnp.array(out) - jnp.array(expected_output)) ** 2, axis=1))
        return mse


    # @flax.nnx.jit
    def train_step(model: Layer, optimizer: flax.nnx.Optimizer, metrics: flax.nnx.MultiMetric, batch):
        """Train for a single step."""
        grad_fn = nnx.value_and_grad(loss_fn)
        mse, grads = grad_fn(model, batch)
        metrics.update(loss=mse)  # In-place updates.
        optimizer.update(grads)  # In-place updates.


    # train
    epochs = 3
    for idx in range(epochs):
        train_step(la, optimizer, metrics, masks)
        print(f'loss: {metrics.compute()["loss"]}')
        metrics.reset()

    # viz
    fig, ax = plt.subplots(2, 2)
    _, vm1, fn1 = la.viz_abs(masks[0], ax=ax[0, 0])
    _, vm2, fn2 = la.viz_abs(masks[1], ax=ax[1, 0])
    _, vm3, fn3 = la.viz_abs(masks[2], ax=ax[0, 1])
    _, vm4, fn4 = la.viz_abs(masks[3], ax=ax[1, 1])
    vm = np.max(np.array([vm1, vm2, vm3, vm4]))
    fn1(vm), fn2(vm), fn3(vm), fn4(vm)
    plt.show()
