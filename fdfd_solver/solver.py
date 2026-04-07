import numpy as np
import scipy.sparse as sp
import jax.numpy as jnp
import jax
from threadpoolctl import threadpool_limits
from typing import Any, Callable, Optional, Union


def spsolve(A: sp.coo_matrix, b: jnp.ndarray) -> jnp.ndarray:
    with threadpool_limits(limits=1, user_api="blas"):
        return sp.linalg.spsolve(A, b)


def splu_factorize(A: sp.spmatrix):
    with threadpool_limits(limits=1, user_api="blas"):
        if not sp.isspmatrix_csc(A):
            A = A.tocsc()
        return sp.linalg.splu(A)


def splu_solve(F, b: jnp.ndarray) -> jnp.ndarray:
    with threadpool_limits(limits=1, user_api="blas"):
        return F.solve(b)


def make_solver(spsolve_func: Callable[[sp.spmatrix, jnp.ndarray], jnp.ndarray]):
    def _solve_impl(entries_a: jnp.ndarray, indices_a: jnp.ndarray, b: jnp.ndarray):
        N = b.shape[0]
        A = sp.coo_matrix((entries_a, indices_a), shape=(N, N))
        A = A.tocsc()
        x = spsolve_func(A, b)
        return np.asarray(x, dtype=b.dtype)

    def _adj_impl(entries_a: jnp.ndarray, indices_a: jnp.ndarray, g: jnp.ndarray):
        N = g.shape[0]
        A = sp.coo_matrix((entries_a, indices_a), shape=(N, N))
        A = A.tocsc()
        adj = spsolve_func(A.T, -g)
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


def make_batch_solver(spfactorize, spsolve):
    """创建支持批量右端项的求解器。

    Args:
        spfactorize: 稀疏矩阵预分解函数，如 scipy.sparse.linalg.splu
        spsolve: 使用分解结果求解的函数，签名 (Factor, b) -> x

    输入 b 的 shape 为 (batch, N)，输出 x 的 shape 为 (batch, N)。
    """

    def _solve_impl(entries_a, indices_a, b):
        # b: (batch, N)
        entries_a, b = np.asarray(entries_a), np.asarray(b)
        N = b.shape[1]
        A = sp.coo_matrix((entries_a, indices_a), shape=(N, N)).tocsc()
        F = spfactorize(A)
        x = np.stack([spsolve(F, bi) for bi in b])
        return np.asarray(x, dtype=b.dtype)

    def _adj_impl(entries_a, indices_a, g):
        # g: (batch, N)
        entries_a, g = np.asarray(entries_a), np.asarray(g)
        N = g.shape[1]
        A = sp.coo_matrix((entries_a, indices_a), shape=(N, N)).tocsc()
        Ft = spfactorize(A.T)
        adj = np.stack([spsolve(Ft, -gi) for gi in g])
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
        # g: (batch, N), x: (batch, N)
        entries_a, x, indices_a = res

        adj = jax.pure_callback(
            _adj_impl, jax.ShapeDtypeStruct(g.shape, g.dtype), entries_a, indices_a, g
        )

        # entries 的梯度：对每个 batch 求 adj[i] * x[j]，然后沿 batch 维求和
        i, j = indices_a
        d_entries = jnp.sum(adj[:, i] * x[:, j], axis=0)
        d_b = -adj
        return d_entries, None, d_b

    solve.defvjp(solve_fwd, solve_bwd)
    return solve


def _resolve_single_solver(
    solver: Optional[Union[str, Callable[[sp.spmatrix, jnp.ndarray], jnp.ndarray]]]
) -> Callable[[sp.spmatrix, jnp.ndarray], jnp.ndarray]:
    if solver is None:
        return spsolve
    if callable(solver):
        return solver

    registry = {
        "spsolve": spsolve,
    }
    if solver not in registry:
        raise ValueError(f"Unknown single solver: {solver}")
    return registry[solver]


def _resolve_factorize(
    factorize: Optional[Union[str, Callable[[sp.spmatrix], Any]]]
) -> Callable[[sp.spmatrix], Any]:
    if factorize is None:
        return splu_factorize
    if callable(factorize):
        return factorize

    registry = {
        "splu": splu_factorize,
    }
    if factorize not in registry:
        raise ValueError(f"Unknown factorize backend: {factorize}")
    return registry[factorize]


def _resolve_factor_solve(
    solve_fn: Optional[Union[str, Callable[[Any, jnp.ndarray], jnp.ndarray]]]
) -> Callable[[Any, jnp.ndarray], jnp.ndarray]:
    if solve_fn is None:
        return splu_solve
    if callable(solve_fn):
        return solve_fn

    registry = {
        "solve": splu_solve,
    }
    if solve_fn not in registry:
        raise ValueError(f"Unknown factor-solve backend: {solve_fn}")
    return registry[solve_fn]


def make_solver_pair(
    single_solver: Optional[
        Union[str, Callable[[sp.spmatrix, jnp.ndarray], jnp.ndarray]]
    ] = None,
    batch_factorize: Optional[Union[str, Callable[[sp.spmatrix], Any]]] = None,
    batch_solver: Optional[Union[str, Callable[[Any, jnp.ndarray], jnp.ndarray]]] = None,
):
    """Build single-RHS and batch-RHS linear solvers with replaceable backends."""

    solve_single = make_solver(_resolve_single_solver(single_solver))
    solve_batch = make_batch_solver(
        _resolve_factorize(batch_factorize),
        _resolve_factor_solve(batch_solver),
    )
    return solve_single, solve_batch


# Default solvers: SciPy spsolve for single RHS and splu+solve for batched RHS.
solve, solve_batch = make_solver_pair()
