from typing import Any, Dict, Optional, Protocol, Tuple

import jax.numpy as jnp

from ceviche.fdfd import fdfd_ez
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
        solver_config: Optional[Dict[str, Any]] = None,
    ):
        npml_pair = _normalize_npml(npml)
        nx, ny = eps_r.shape

        self.omega = omega
        self.shape = (nx, ny)
        self._eps_r = eps_r
        self._dxy = DxyMatrix(omega, dL, self.shape, npml_pair)

        solver_config = solver_config or {}
        self._solve_single, self._solve_batch = make_solver_pair(
            single_solver=solver_config.get("single_solver"),
            batch_factorize=solver_config.get("batch_factorize"),
            batch_solver=solver_config.get("batch_solver"),
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
        return FdfdSolverEzBackend(
            omega,
            dL,
            eps_r,
            npml,
            solver_config=simulation_config.get("fdfd_solver", {}),
        )

    raise ValueError(
        f"Unknown simulation backend: {backend}. Use 'ceviche' or 'fdfd_solver'."
    )
