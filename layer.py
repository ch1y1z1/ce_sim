from typing import List
from matplotlib import pyplot as plt
import toml
import basis_generator_chi as bgc
import numpy as np

import ceviche
from ceviche.fdfd import fdfd_ez
from ceviche.jacobians import jacobian
from ceviche.modes import insert_mode

import autograd.numpy as npa
from ceviche.optimizers import adam_optimize

import matplotlib as mpl

mpl.rcParams["font.family"] = "sans-serif"


def sigmoid_b(E1, E0, basic):
    x = -(E1 - E0) * basic["alpha"]
    return 1 / (1 + npa.exp(x))


def mode_overlap(E1, E2):
    """Defines an overlap integral between the simulated field and desired field"""
    return npa.abs(npa.sum(npa.conj(E1) * E2))


def prepare_io(omega, resolution, input_list, output_list, epsr_total, m=1):
    ics = []
    for point in input_list:
        ic = insert_mode(omega, resolution, point["x"], point["y"], epsr_total, m=m)
        ics.append(ic)

    probes = []
    for point in output_list:
        probe = np.zeros(epsr_total.shape, dtype=np.complex128)
        probe[point["x"], point["y"]] = 1
        probes.append(probe)

    return ics, probes


class Layer:
    def __init__(self, grid, input, output, basic):
        n_bits_o = output["n_bits"]
        grid["ny"] = int(256 / 9 * (2 * n_bits_o + 1))
        grid["nx"] = int(208)
        self.rho, self.bg_rho, self.opt_region, self.input_list, self.output_list = (
            bgc.init_layer(grid, input, output)
        )
        omega = 2 * np.pi * basic["freq"]

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
        # TODO: this is not accurate

        ic_total = 0
        for ic in self.ics:
            ic_total += ic
        _, _, Ez0 = self.simulation.solve(ic_total)
        for probe in self.probes:
            E0 = mode_overlap(Ez0, probe)
            self.E0s.append(E0 / 2 * basic["E0_scale"])

    def objective(self, rho, masks: List[npa.ndarray]) -> List[npa.ndarray]:
        rho = rho.reshape((self.grid["nx"], self.grid["ny"]))
        self.simulation.eps_r = bgc.epsr_parameterization(
            rho,
            self.bg_rho,
            self.opt_region,
            self.basic["epsilon_min"],
            self.basic["epsilon_max"],
        )

        Ezs = []
        for mask in masks:
            source = npa.sum(mask[:, None, None] * npa.array(self.ics), axis=0)
            _, _, Ez = self.simulation.solve(source)
            Ezs.append(Ez)

        a_list = []
        for idx in range(len(masks)):
            output_mask = []
            for i in range(len(self.probes)):
                a = sigmoid_b(
                    mode_overlap(Ezs[idx], self.probes[i]), self.E0s[i], self.basic
                )
                output_mask.append(a)
            a_list.append(npa.array(output_mask))

        return a_list

    def use_rho(self, rho):
        self.rho = rho.reshape((self.bg_rho.shape[0], self.bg_rho.shape[1]))

    def viz_abs(self, mask, ax):
        self.simulation.eps_r = bgc.epsr_parameterization(
            self.rho,
            self.bg_rho,
            self.opt_region,
            self.basic["epsilon_min"],
            self.basic["epsilon_max"],
        )
        # reshape mask to match the dimensions of the input list for broadcasting
        source = npa.sum(mask[:, None, None] * npa.array(self.ics), axis=0)
        _, _, Ez = self.simulation.solve(source)

        output_mask = []
        for i in range(len(self.probes)):
            a = sigmoid_b(mode_overlap(Ez, self.probes[i]), self.E0s[i], self.basic)
            output_mask.append(a)

        # ceviche.viz.abs(Ez, outline=self.epsr_total, ax=ax, cbar=False)
        for idx, point in enumerate(self.input_list):
            ax.plot(point["x"] * np.ones(len(point["y"])), point["y"], "w-", alpha=0.5)
            ax.text(
                point["x"] - 40,
                point["y"][len(point["y"]) // 2],
                f"{mask[idx]:.2f}",
                ha="center",
                va="center",
            )
        for idx, point in enumerate(self.output_list):
            ax.plot(point["x"] * np.ones(len(point["y"])), point["y"], "w-", alpha=0.5)
            ax.text(
                point["x"] + 40,
                point["y"][len(point["y"]) // 2],
                f"{output_mask[idx]:.2f}",
                ha="center",
                va="center",
            )

        return (
            np.array(output_mask),
            npa.max(npa.abs(Ez)),
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
    config = "./Configuration/chi_config.toml"
    config = toml.load(config)
    train_config = config["train"]
    layer = config["layers"][0]
    la = Layer(config["grid"], layer["input"], layer["output"], config["basic"])

    # prepare dataset:
    masks = [
        npa.array([0, 1, 0, 1]),
        npa.array([0, 1, 1, 0]),
        npa.array([1, 0, 0, 1]),
        npa.array([1, 0, 1, 0]),
    ]
    expected_output = [
        npa.array([1, 1]),
        npa.array([0, 0]),
        npa.array([1, 0]),
        npa.array([0, 1]),
    ]

    def obj(rho):
        predict = la.objective(rho, masks)
        mse = npa.mean((npa.array(predict) - npa.array(expected_output)) ** 2, axis=1)
        return npa.sum(mse)

    obj_jac = jacobian(obj, mode="reverse")

    (rho_optimum, loss) = adam_optimize(
        obj,
        la.rho.flatten(),
        obj_jac,
        Nsteps=train_config["num_epochs"],
        direction="min",
        step_size=train_config["step_size"],
    )
    la.use_rho(rho_optimum)

    fig, ax = plt.subplots(2, 2)
    la.viz_abs(masks[0], ax=ax[0, 0])
    la.viz_abs(masks[1], ax=ax[1, 0])
    la.viz_abs(masks[2], ax=ax[0, 1])
    la.viz_abs(masks[3], ax=ax[1, 1])
    plt.show()
