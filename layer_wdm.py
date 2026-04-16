import functools

import agjax
import jax
import jax.numpy as jnp
import jax.scipy.signal as jsp_signal
import numpy as np
from flax import nnx

import basis_generator_chi_demux as bgc
import ceviche
from ceviche.modes import insert_mode
from simulation_backend import build_simulation_backend


jax.config.update("jax_enable_x64", True)


def mode_overlap(E1, E2):
    return jnp.abs(jnp.sum(jnp.conj(E1) * E1 * E2))


def prepare_io(omega, resolution, input_list, output_list, epsr_total, m=1):
    ics = [
        insert_mode(omega, resolution, point["x"], point["y"], epsr_total, m=m)
        for point in input_list
    ]

    probes = [
        insert_mode(omega, resolution, point["x"], point["y"], epsr_total, m=m)
        for point in output_list
    ]

    return ics, probes


def _create_blur_kernel(radius):
    coords = np.arange(-radius, radius + 1, dtype=np.float32)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    mask = xx * xx + yy * yy <= (radius + 0.5) * (radius + 0.5)
    kernel = np.where(mask, 1.0, 0.0).astype(np.float32)
    return kernel / kernel.sum()


def operator_blur(rho, radius=2, N=1):
    kernel = jnp.asarray(_create_blur_kernel(radius), dtype=rho.dtype)
    for _ in range(N):
        rho = jsp_signal.convolve2d(
            rho,
            kernel,
            mode="same",
            boundary="fill",
            fillvalue=0.0,
        )
    return rho


class LayerWDMDemux(nnx.Module):
    def __init__(self, grid, layer_config, basic, simulation_config=None):
        input_cfg = layer_config["input"]
        output_cfg = layer_config["output"]
        self.rho, self.bg_rho, self.opt_region, self.input_list, self.output_list = (
            bgc.init_layer(grid, input_cfg, output_cfg)
        )

        freq1 = 3e8 / basic["wavelength_1"]
        freq2 = 3e8 / basic["wavelength_2"]
        omega1 = 2 * jnp.pi * freq1
        omega2 = 2 * jnp.pi * freq2

        self.grid = grid
        self.basic = basic
        self.layer_config = layer_config
        self.simulation_config = simulation_config or {}
        self.backend_name = self.simulation_config.get("backend", "ceviche")
        self._is_fdfd_backend = self.backend_name in {
            "fdfd_solver",
            "fdfd_solver_scipy",
        }
        self.enable_batch = bool(
            self.simulation_config.get(
                "enable_batch",
                self._is_fdfd_backend,
            )
        )
        self.enable_batch = self.enable_batch and self._is_fdfd_backend

        self.epsr_total = bgc.epsr_parameterization(
            self.rho,
            self.bg_rho,
            self.opt_region,
            basic["epsilon_min"],
            basic["epsilon_max"],
        )

        self.ics1, self.probes1 = prepare_io(
            omega1,
            grid["resolution"],
            self.input_list,
            self.output_list,
            self.epsr_total,
        )
        self.ics2, self.probes2 = prepare_io(
            omega2,
            grid["resolution"],
            self.input_list,
            self.output_list,
            self.epsr_total,
        )

        self.simulation1 = build_simulation_backend(
            omega1,
            grid["resolution"],
            self.epsr_total,
            [grid["npml"], grid["npml"]],
            self.simulation_config,
        )
        self.simulation2 = build_simulation_backend(
            omega2,
            grid["resolution"],
            self.epsr_total,
            [grid["npml"], grid["npml"]],
            self.simulation_config,
        )

        _, _, Ez1 = self.simulation1.solve(self.ics1[0])
        _, _, Ez2 = self.simulation2.solve(self.ics2[0])
        self.E01 = mode_overlap(Ez1, self.probes1[0])
        self.E02 = mode_overlap(Ez2, self.probes2[1])

        self.rho_jax = nnx.Param(self.rho)

        self.epsr_min_jax = nnx.Param(jnp.ones(1) * self.basic["epsilon_min"])
        self.epsr_max_jax = nnx.Param(jnp.ones(1) * self.basic["epsilon_max"])

        self.probe_data = None

    def _resolve_eps_bounds(self):
        epsr_min = self.basic["epsilon_min"]
        epsr_max = self.basic["epsilon_max"]

        if self.basic.get("epsilon_min_learn_bool", False):
            epsr_min = self.epsr_min_jax.value[0]
        if self.basic.get("epsilon_max_learn_bool", False):
            epsr_max = self.epsr_max_jax.value[0]

        return epsr_min, epsr_max

    def _update_backend_eps(self, rho):
        epsr_min, epsr_max = self._resolve_eps_bounds()

        rho = rho.reshape((self.grid["nx"], self.grid["ny"]))
        eps_r = bgc.epsr_parameterization(
            rho,
            self.bg_rho,
            self.opt_region,
            epsr_min,
            epsr_max,
        )
        self.simulation1.eps_r = eps_r
        self.simulation2.eps_r = eps_r

    @functools.partial(agjax.wrap_for_jax, nondiff_argnums=(0,))
    def solve(self, rho, source1, source2):
        self._update_backend_eps(rho)
        _, _, Ez1 = self.simulation1.solve(source1)
        _, _, Ez2 = self.simulation2.solve(source2)
        return Ez1, Ez2

    def _solve_pair(self, source1, source2):
        _, _, Ez1 = self.simulation1.solve(source1)
        _, _, Ez2 = self.simulation2.solve(source2)
        return Ez1, Ez2

    def __call__(self, masks: jnp.ndarray) -> jnp.ndarray:
        self.rho_jax.value = self.pre_blur(self.rho_jax.value)

        sources1 = jnp.sum(masks[:, :, None, None] * jnp.array(self.ics1), axis=1)
        sources2 = jnp.sum(masks[:, :, None, None] * jnp.array(self.ics2), axis=1)

        if self._is_fdfd_backend:
            self._update_backend_eps(self.rho_jax.value)

            if (
                self.enable_batch
                and hasattr(self.simulation1, "solve_batch")
                and hasattr(self.simulation2, "solve_batch")
            ):
                _, _, Ezs1 = self.simulation1.solve_batch(sources1)
                _, _, Ezs2 = self.simulation2.solve_batch(sources2)
            else:
                Ezs1 = [self.simulation1.solve(source)[2] for source in sources1]
                Ezs2 = [self.simulation2.solve(source)[2] for source in sources2]
        else:
            Ezs1 = []
            Ezs2 = []
            for source1, source2 in zip(sources1, sources2):
                Ez1, Ez2 = self.solve(self.rho_jax.value, source1, source2)
                Ezs1.append(Ez1)
                Ezs2.append(Ez2)

        a_list_1, a_list_2 = self.overlap(Ezs1, Ezs2)
        self.probe_data = jnp.stack([a_list_1, a_list_2], axis=0)

        probe_data_11 = a_list_1[0, :] / self.E01
        probe_data_22 = a_list_2[1, :] / self.E02
        return jnp.squeeze(probe_data_11 * probe_data_22)

    def viz_abs(self, mask, wavelength_idx, ax):
        source1 = jnp.sum(mask[:, None, None] * jnp.array(self.ics1), axis=0)
        source2 = jnp.sum(mask[:, None, None] * jnp.array(self.ics2), axis=0)

        if self._is_fdfd_backend:
            self._update_backend_eps(self.rho_jax.value)
            Ez1, Ez2 = self._solve_pair(source1, source2)
        else:
            Ez1, Ez2 = self.solve(self.rho_jax.value, source1, source2)

        if wavelength_idx == 0:
            Ez = Ez1
        elif wavelength_idx == 1:
            Ez = Ez2
        else:
            raise ValueError("wavelength_idx must be 0 or 1")

        output_mask = (
            self.probe_data[wavelength_idx, :, 0]
            if self.probe_data is not None
            else jnp.zeros(len(self.output_list))
        )

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
                f"{output_mask[idx]:.3e}",
                ha="center",
                va="center",
            )

        ax.set_title(f"wavelength {wavelength_idx + 1}", fontsize=12)

        return (
            jnp.array(output_mask),
            jnp.max(jnp.abs(Ez)),
            lambda vmax: (
                ceviche.viz.abs(Ez, outline=self.epsr_total, ax=ax, cbar=False, vmax=vmax),
                ax.set_xticks([]),
                ax.set_yticks([]),
                ax.set_xlabel(""),
                ax.set_ylabel(""),
            ),
        )

    def overlap(self, Ezs1, Ezs2):
        overlaps_list1 = []
        overlaps_list2 = []
        for Ez1, Ez2 in zip(Ezs1, Ezs2):
            overlap_list1 = [mode_overlap(Ez1, self.probes1[i]) for i in range(len(self.probes1))]
            overlap_list2 = [mode_overlap(Ez2, self.probes2[i]) for i in range(len(self.probes2))]
            overlaps_list1.append(overlap_list1)
            overlaps_list2.append(overlap_list2)

        overlaps_jax1 = jnp.array(overlaps_list1)
        overlaps_jax2 = jnp.array(overlaps_list2)
        return overlaps_jax1.T, overlaps_jax2.T

    def operator_proj(self, rho, eta=0.5, beta=100, N=1):
        for _ in range(N):
            rho = jnp.divide(
                jnp.tanh(beta * eta) + jnp.tanh(beta * (rho - eta)),
                jnp.tanh(beta * eta) + jnp.tanh(beta * (1 - eta)),
            )
        return rho

    def pre_blur(self, rho):
        rho = jnp.asarray(bgc.mask_combine_rho(rho, self.bg_rho, self.opt_region))
        rho = operator_blur(
            rho,
            radius=int(self.basic.get("kernel_radius", 2)),
            N=int(self.basic.get("num_conv", 1)),
        )
        if self.basic.get("bin_bool", False):
            rho = self.operator_proj(
                rho,
                eta=float(self.basic.get("proj_eta", 0.5)),
                beta=float(self.basic.get("proj_beta", 100.0)),
                N=int(self.basic.get("num_proj", 1)),
            )
        rho = jnp.asarray(bgc.mask_combine_rho(rho, self.bg_rho, self.opt_region))
        return rho
