import numpy as np
import scipy.sparse as sp
import jax.numpy as jnp
import jax
from threadpoolctl import threadpool_limits
from typing import Callable


def spsolve(A: sp.coo_matrix, b: jnp.ndarray) -> jnp.ndarray:
    with threadpool_limits(limits=1, user_api="blas"):
        return sp.linalg.spsolve(A, b)


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
        return adj[i] * x[j], None, None

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
        return d_entries, None, None

    solve.defvjp(solve_fwd, solve_bwd)
    return solve
