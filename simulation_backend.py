from typing import Any, Dict, Optional, Protocol, Tuple

import jax.numpy as jnp

from ceviche.fdfd import fdfd_ez
from spsolver.bindings_superlu_dist import DistSolveConfig
from fdfd_solver.dxy_matrix import DxyMatrix
from fdfd_solver.make_A_b import make_A
from fdfd_solver.solver import make_solver_pair


class EzSimulationBackend(Protocol):
    @property
    def eps_r(self) -> jnp.ndarray:
        ...

    @eps_r.setter
    def eps_r(self, new_eps: jnp.ndarray) -> None:
        ...

    def solve(self, source: jnp.ndarray):
        ...

    def solve_batch(self, sources: jnp.ndarray):
        ...


def _normalize_npml(npml) -> Tuple[int, int]:
    if isinstance(npml, int):
        return (npml, npml)
    if isinstance(npml, (list, tuple)) and len(npml) == 2:
        return (int(npml[0]), int(npml[1]))
    raise ValueError(f"Invalid npml: {npml}")


class CevicheEzBackend:
    def __init__(self, omega: float, dL: float, eps_r: jnp.ndarray, npml):
        npml_pair = _normalize_npml(npml)
        self._sim = fdfd_ez(omega, dL, eps_r, [npml_pair[0], npml_pair[1]])

    @property
    def eps_r(self) -> jnp.ndarray:
        return self._sim.eps_r

    @eps_r.setter
    def eps_r(self, new_eps: jnp.ndarray) -> None:
        self._sim.eps_r = new_eps

    def solve(self, source: jnp.ndarray):
        return self._sim.solve(source)

    def solve_batch(self, sources: jnp.ndarray):
        ezs = [self._sim.solve(src)[2] for src in sources]
        Ez = jnp.stack(ezs, axis=0)
        zeros = jnp.zeros_like(Ez)
        return zeros, zeros, Ez


class FdfdSolverEzBackend:
    def __init__(
        self,
        omega: float,
        dL: float,
        eps_r: jnp.ndarray,
        npml,
        enable_symbolic_reuse: bool = False,
        enable_superlu_dist: bool = False,
        superlu_dist_config: Optional[DistSolveConfig] = None,
    ):
        npml_pair = _normalize_npml(npml)
        nx, ny = eps_r.shape

        self.omega = omega
        self.shape = (nx, ny)
        self._eps_r = eps_r
        self._dxy = DxyMatrix(omega, dL, self.shape, npml_pair)

        self._solve_single, self._solve_batch = make_solver_pair(
            enable_symbolic_reuse=enable_symbolic_reuse,
            use_superlu_dist=enable_superlu_dist,
            dist_config=superlu_dist_config,
        )

    @property
    def eps_r(self) -> jnp.ndarray:
        return self._eps_r

    @eps_r.setter
    def eps_r(self, new_eps: jnp.ndarray) -> None:
        if tuple(new_eps.shape) != self.shape:
            raise ValueError(
                f"eps_r shape changed from {self.shape} to {tuple(new_eps.shape)}"
            )
        self._eps_r = new_eps

    def _make_A(self):
        nx, ny = self.shape
        return make_A(self._dxy, self._eps_r, nx, ny, self.omega)

    def solve(self, source: jnp.ndarray):
        entries_a, indices_a = self._make_A()
        b = (1j * self.omega) * source.flatten()
        ez_vec = self._solve_single(entries_a, indices_a, b)

        Ez = ez_vec.reshape(self.shape)
        zeros = jnp.zeros_like(Ez)
        return zeros, zeros, Ez

    def solve_batch(self, sources: jnp.ndarray):
        entries_a, indices_a = self._make_A()
        batch_size = sources.shape[0]
        b = (1j * self.omega) * sources.reshape((batch_size, -1))
        ez_vec = self._solve_batch(entries_a, indices_a, b)

        Ez = ez_vec.reshape((batch_size,) + self.shape)
        zeros = jnp.zeros_like(Ez)
        return zeros, zeros, Ez


def build_simulation_backend(
    omega: float,
    dL: float,
    eps_r: jnp.ndarray,
    npml,
    simulation_config: Optional[Dict[str, Any]] = None,
) -> EzSimulationBackend:
    simulation_config = simulation_config or {}
    backend = simulation_config.get("backend", "ceviche")

    if backend == "ceviche":
        return CevicheEzBackend(omega, dL, eps_r, npml)

    if backend == "fdfd_solver":
        enable_symbolic_reuse = bool(
            simulation_config.get("enable_symbolic_reuse", True)
        )
        enable_superlu_dist = bool(
            simulation_config.get("enable_superlu_dist", False)
        )
        dist_cfg_raw = simulation_config.get("superlu_dist", {}) or {}
        dist_cfg = DistSolveConfig(
            nrow=int(dist_cfg_raw.get("nrow", 2)),
            ncol=int(dist_cfg_raw.get("ncol", 1)),
            rowperm=int(dist_cfg_raw.get("rowperm", 0)),
            colperm=int(dist_cfg_raw.get("colperm", 0)),
            int64=int(dist_cfg_raw.get("int64", 1)),
            algo3d=int(dist_cfg_raw.get("algo3d", 0)),
            verbosity=bool(dist_cfg_raw.get("verbosity", False)),
            library_path=dist_cfg_raw.get("library_path"),
        )
        return FdfdSolverEzBackend(
            omega,
            dL,
            eps_r,
            npml,
            enable_symbolic_reuse=enable_symbolic_reuse,
            enable_superlu_dist=enable_superlu_dist,
            superlu_dist_config=dist_cfg,
        )

    raise ValueError(
        f"Unknown simulation backend: {backend}. Use 'ceviche' or 'fdfd_solver'."
    )
