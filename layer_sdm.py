import functools

import agjax
import jax
import jax.numpy as jnp
from flax import nnx

import basis_generator_chi_demux as bgc
import ceviche
from ceviche.modes import insert_mode
from simulation_backend import build_simulation_backend


jax.config.update("jax_enable_x64", True)


def mode_overlap(E1, E2):
    return jnp.abs(jnp.sum(jnp.conj(E1) * E1 * E2))


def prepare_io(omega, resolution, input_list, output_list, epsr_total, m_in, m_out):
    point = input_list[0]
    ics = [
        insert_mode(omega, resolution, point["x"], point["y"], epsr_total, m=m)
        for m in m_in
    ]

    probes = [
        insert_mode(omega, resolution, point["x"], point["y"], epsr_total, m=m_out)
        for point in output_list
    ]

    return ics, probes


class LayerSDMDemux(nnx.Module):
    def __init__(self, grid, layer_config, basic, simulation_config=None):
        input_cfg = layer_config["input"]
        output_cfg = layer_config["output"]
        self.n_modes = int(output_cfg["n_bits"])
        self.rho, self.bg_rho, self.opt_region, self.input_list, self.output_list = (
            bgc.init_layer(grid, input_cfg, output_cfg)
        )

        freq = 3e8 / basic["wavelength"]
        omega = 2 * jnp.pi * freq

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

        mode_indices = layer_config.get("mode_indices")
        if mode_indices is None:
            mode_indices = list(range(1, self.n_modes + 1))
        output_mode_index = int(layer_config.get("output_mode_index", 1))

        self.ics, self.probes = prepare_io(
            omega,
            grid["resolution"],
            self.input_list,
            self.output_list,
            self.epsr_total,
            m_in=mode_indices,
            m_out=output_mode_index,
        )

        self.simulation = build_simulation_backend(
            omega,
            grid["resolution"],
            self.epsr_total,
            [grid["npml"], grid["npml"]],
            self.simulation_config,
        )

        self.E0 = []
        for i in range(self.n_modes):
            _, _, Ez0 = self.simulation.solve(self.ics[i])
            self.E0.append(mode_overlap(Ez0, self.probes[i]))

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
        self.simulation.eps_r = eps_r

    @functools.partial(agjax.wrap_for_jax, nondiff_argnums=(0,))
    def solve(self, rho, source):
        self._update_backend_eps(rho)
        _, _, Ez = self.simulation.solve(source)
        return Ez

    def __call__(self, masks: jnp.ndarray) -> jnp.ndarray:
        if self.basic.get("bin_bool", False):
            self.rho_jax.value = self.operator_proj(
                self.rho_jax.value,
                eta=float(self.basic.get("proj_eta", 0.5)),
                beta=float(self.basic.get("proj_beta", 100.0)),
                N=int(self.basic.get("num_proj", 1)),
            )

        mask = masks[0]
        sources = [
            jnp.sum(mask[:, None, None] * jnp.array(self.ics[i]), axis=0)
            for i in range(self.n_modes)
        ]

        if self._is_fdfd_backend:
            self._update_backend_eps(self.rho_jax.value)

            if self.enable_batch and hasattr(self.simulation, "solve_batch"):
                _, _, Ezs = self.simulation.solve_batch(jnp.stack(sources, axis=0))
            else:
                Ezs = [self.simulation.solve(source)[2] for source in sources]
        else:
            Ezs = [self.solve(self.rho_jax.value, source) for source in sources]

        overlaps = jnp.array([mode_overlap(Ezs[i], self.probes[i]) for i in range(self.n_modes)])
        self.probe_data = overlaps

        return jnp.prod(self.probe_data / jnp.array(self.E0))

    def viz_abs(self, mask, mode_idx, ax):
        source = jnp.sum(mask[:, None, None] * jnp.array(self.ics[mode_idx]), axis=0)

        if self._is_fdfd_backend:
            self._update_backend_eps(self.rho_jax.value)
            Ez = self.simulation.solve(source)[2]
        else:
            Ez = self.solve(self.rho_jax.value, source)

        output_mask = self.probe_data if self.probe_data is not None else jnp.zeros(self.n_modes)

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
                f"{output_mask[mode_idx]:.3e}",
                ha="center",
                va="center",
            )

        ax.set_title(f"mode {mode_idx + 1}", fontsize=12)

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

    def operator_proj(self, rho, eta=0.5, beta=100, N=1):
        for _ in range(N):
            rho = jnp.divide(
                jnp.tanh(beta * eta) + jnp.tanh(beta * (rho - eta)),
                jnp.tanh(beta * eta) + jnp.tanh(beta * (1 - eta)),
            )

        return rho
