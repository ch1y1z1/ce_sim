from __future__ import annotations

import atexit
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from spsolver import is_available, spanalyze, spfactorize, sprefactorize, spsolve
from spsolver.bindings_superlu_dist import (
    DistSolveConfig,
    is_available as is_superlu_dist_available,
    spfactorize as spfactorize_dist,
    spsolve as spsolve_dist,
)


def _require_superlu_backend(use_superlu_dist: bool) -> None:
    if use_superlu_dist:
        if is_superlu_dist_available():
            return
        raise RuntimeError(
            "fdfd_solver superlu_dist backend is unavailable. "
            "Ensure mpirun exists and build superlu_dist Python bridge "
            "(libsuperlu_dist_python) first."
        )

    if is_available():
        return
    raise RuntimeError(
        "fdfd_solver backend requires custom spsolver superlu binding. "
        "Please build spsolver/nanobind and ensure _superlu_nb is importable."
    )


class _SymbolicReuseCache:
    def __init__(self):
        self.factor = None
        self._indptr: Optional[np.ndarray] = None
        self._indices: Optional[np.ndarray] = None
        self._n: Optional[int] = None

    def factorize(
        self,
        A_csc: sp.csc_matrix,
        *,
        use_superlu_dist: bool,
        dist_config: DistSolveConfig | None,
    ):
        n = int(A_csc.shape[0])
        indptr = np.asarray(A_csc.indptr)
        indices = np.asarray(A_csc.indices)

        if self.factor is None:
            if use_superlu_dist:
                self.factor = spfactorize_dist(A_csc, config=dist_config)
            else:
                self.factor = spanalyze(A_csc)
            self._indptr = indptr.copy()
            self._indices = indices.copy()
            self._n = n
        else:
            if n != self._n:
                raise ValueError(
                    f"Symbolic reuse expected n={self._n}, got n={n}."
                )
            if not np.array_equal(indptr, self._indptr) or not np.array_equal(
                indices, self._indices
            ):
                raise ValueError(
                    "enable_symbolic_reuse=True but sparse pattern changed."
                )

        if use_superlu_dist:
            self.factor.refactorize(A_csc)
        else:
            sprefactorize(self.factor, A_csc.data)
        return self.factor

    def clear(self) -> None:
        self.factor = None
        self._indptr = None
        self._indices = None
        self._n = None


_ACTIVE_REUSE_CACHES: list[_SymbolicReuseCache] = []


def _register_reuse_cache(cache: Optional[_SymbolicReuseCache]) -> None:
    if cache is not None:
        _ACTIVE_REUSE_CACHES.append(cache)


def _clear_registered_reuse_caches() -> None:
    for cache in _ACTIVE_REUSE_CACHES:
        cache.clear()
    _ACTIVE_REUSE_CACHES.clear()


atexit.register(_clear_registered_reuse_caches)


def _factorize(
    A_csc: sp.csc_matrix,
    cache: Optional[_SymbolicReuseCache],
    *,
    use_superlu_dist: bool,
    dist_config: DistSolveConfig | None,
):
    _require_superlu_backend(use_superlu_dist=use_superlu_dist)

    if cache is None:
        if use_superlu_dist:
            return spfactorize_dist(A_csc, config=dist_config)
        return spfactorize(A_csc)

    return cache.factorize(
        A_csc,
        use_superlu_dist=use_superlu_dist,
        dist_config=dist_config,
    )


def make_solver(
    enable_symbolic_reuse: bool = False,
    use_superlu_dist: bool = False,
    dist_config: DistSolveConfig | None = None,
):
    _require_superlu_backend(use_superlu_dist=use_superlu_dist)
    cache = _SymbolicReuseCache() if enable_symbolic_reuse else None
    _register_reuse_cache(cache)
    cache_t = (
        _SymbolicReuseCache()
        if enable_symbolic_reuse and use_superlu_dist
        else cache
    )
    if cache_t is not cache:
        _register_reuse_cache(cache_t)

    def _solve_impl(entries_a: jnp.ndarray, indices_a: jnp.ndarray, b: jnp.ndarray):
        entries_a = np.asarray(entries_a)
        b = np.asarray(b)
        N = b.shape[0]
        A = sp.coo_matrix((entries_a, indices_a), shape=(N, N)).tocsc()
        F = _factorize(
            A,
            cache,
            use_superlu_dist=use_superlu_dist,
            dist_config=dist_config,
        )
        if use_superlu_dist:
            x = spsolve_dist(F, b, overwrite_b=False, transpose=False)
        else:
            x = spsolve(F, b, overwrite_b=False, transpose=False)
        return np.asarray(x, dtype=b.dtype)

    def _adj_impl(entries_a: jnp.ndarray, indices_a: jnp.ndarray, g: jnp.ndarray):
        entries_a = np.asarray(entries_a)
        g = np.asarray(g)
        N = g.shape[0]
        A = sp.coo_matrix((entries_a, indices_a), shape=(N, N)).tocsc()
        if use_superlu_dist:
            F = _factorize(
                A.T.tocsc(),
                cache_t,
                use_superlu_dist=True,
                dist_config=dist_config,
            )
            adj = spsolve_dist(F, -g, overwrite_b=False, transpose=False)
        else:
            F = _factorize(
                A,
                cache,
                use_superlu_dist=False,
                dist_config=dist_config,
            )
            adj = spsolve(F, -g, overwrite_b=False, transpose=True)
        return np.asarray(adj, dtype=g.dtype)

    @jax.custom_vjp
    def solve(entries_a, indices_a, b):
        return jax.pure_callback(
            _solve_impl, jax.ShapeDtypeStruct(b.shape, b.dtype), entries_a, indices_a, b
        )

    def solve_fwd(entries_a, indices_a, b):
        x = solve(entries_a, indices_a, b)
        return x, (entries_a, x, indices_a)

    def solve_bwd(res, g):
        entries_a, x, indices_a = res

        adj = jax.pure_callback(
            _adj_impl, jax.ShapeDtypeStruct(g.shape, g.dtype), entries_a, indices_a, g
        )

        i, j = indices_a
        d_entries = adj[i] * x[j]
        d_b = -adj
        return d_entries, None, d_b

    solve.defvjp(solve_fwd, solve_bwd)
    return solve


def make_batch_solver(
    enable_symbolic_reuse: bool = False,
    use_superlu_dist: bool = False,
    dist_config: DistSolveConfig | None = None,
):
    _require_superlu_backend(use_superlu_dist=use_superlu_dist)
    cache = _SymbolicReuseCache() if enable_symbolic_reuse else None
    _register_reuse_cache(cache)
    cache_t = (
        _SymbolicReuseCache()
        if enable_symbolic_reuse and use_superlu_dist
        else cache
    )
    if cache_t is not cache:
        _register_reuse_cache(cache_t)

    def _solve_impl(entries_a, indices_a, b):
        entries_a = np.asarray(entries_a)
        b = np.asarray(b)
        N = b.shape[1]
        A = sp.coo_matrix((entries_a, indices_a), shape=(N, N)).tocsc()
        F = _factorize(
            A,
            cache,
            use_superlu_dist=use_superlu_dist,
            dist_config=dist_config,
        )
        if use_superlu_dist:
            x = spsolve_dist(F, b, overwrite_b=False, transpose=False)
        else:
            x = spsolve(F, b, overwrite_b=False, transpose=False)
        return np.asarray(x, dtype=b.dtype)

    def _adj_impl(entries_a, indices_a, g):
        entries_a = np.asarray(entries_a)
        g = np.asarray(g)
        N = g.shape[1]
        A = sp.coo_matrix((entries_a, indices_a), shape=(N, N)).tocsc()
        if use_superlu_dist:
            F = _factorize(
                A.T.tocsc(),
                cache_t,
                use_superlu_dist=True,
                dist_config=dist_config,
            )
            adj = spsolve_dist(F, -g, overwrite_b=False, transpose=False)
        else:
            F = _factorize(
                A,
                cache,
                use_superlu_dist=False,
                dist_config=dist_config,
            )
            adj = spsolve(F, -g, overwrite_b=False, transpose=True)
        return np.asarray(adj, dtype=g.dtype)

    @jax.custom_vjp
    def solve(entries_a, indices_a, b):
        return jax.pure_callback(
            _solve_impl, jax.ShapeDtypeStruct(b.shape, b.dtype), entries_a, indices_a, b
        )

    def solve_fwd(entries_a, indices_a, b):
        x = solve(entries_a, indices_a, b)
        return x, (entries_a, x, indices_a)

    def solve_bwd(res, g):
        entries_a, x, indices_a = res

        adj = jax.pure_callback(
            _adj_impl, jax.ShapeDtypeStruct(g.shape, g.dtype), entries_a, indices_a, g
        )

        i, j = indices_a
        d_entries = jnp.sum(adj[:, i] * x[:, j], axis=0)
        d_b = -adj
        return d_entries, None, d_b

    solve.defvjp(solve_fwd, solve_bwd)
    return solve


def make_solver_pair(
    enable_symbolic_reuse: bool = False,
    use_superlu_dist: bool = False,
    dist_config: DistSolveConfig | None = None,
):
    solve_single = make_solver(
        enable_symbolic_reuse=enable_symbolic_reuse,
        use_superlu_dist=use_superlu_dist,
        dist_config=dist_config,
    )
    solve_batch = make_batch_solver(
        enable_symbolic_reuse=enable_symbolic_reuse,
        use_superlu_dist=use_superlu_dist,
        dist_config=dist_config,
    )
    return solve_single, solve_batch


solve, solve_batch = make_solver_pair(enable_symbolic_reuse=False)
