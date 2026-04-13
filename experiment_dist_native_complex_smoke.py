from __future__ import annotations

from dataclasses import asdict

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import typer
from loguru import logger

from spsolver.bindings_superlu_dist import DistSolveConfig, spfactorize, spsolve

app = typer.Typer(add_completion=False)


def _make_matrix(n: int, density: float, seed: int, diag_shift: float) -> sp.csc_matrix:
    rng = np.random.default_rng(seed)

    nnz_target = max(1, int(n * n * density))
    rows = rng.integers(0, n, size=nnz_target)
    cols = rng.integers(0, n, size=nnz_target)
    vals = rng.standard_normal(nnz_target) + 1j * rng.standard_normal(nnz_target)

    A = sp.coo_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.complex128).tocsc()
    A = A + sp.eye(n, format="csc", dtype=np.complex128) * diag_shift
    return A


def _max_residual(A: sp.csc_matrix, x: np.ndarray, b: np.ndarray) -> float:
    r = A @ x - b
    return float(np.max(np.abs(r)))


def _check_close(label: str, got: np.ndarray, ref: np.ndarray, atol: float, rtol: float) -> None:
    err = np.max(np.abs(got - ref))
    tol = atol + rtol * np.max(np.abs(ref))
    logger.info(f"[{label}] max(|x_dist - x_ref|) = {err:.3e}, tol = {tol:.3e}")
    if not np.allclose(got, ref, atol=atol, rtol=rtol):
        raise RuntimeError(f"{label} failed allclose check")


@app.command()
def main(
    n: int = typer.Option(200, "--n", min=8, help="Matrix dimension."),
    density: float = typer.Option(0.05, "--density", min=1e-6, max=1.0),
    seed: int = typer.Option(0, "--seed"),
    diag_shift: float = typer.Option(10.0, "--diag-shift", min=1e-12),
    repeats: int = typer.Option(5, "--repeats", min=1),
    warmup: int = typer.Option(1, "--warmup", min=0),
    batch: int = typer.Option(4, "--batch", min=1),
    nrow: int = typer.Option(1, "--nrow", min=1),
    ncol: int = typer.Option(1, "--ncol", min=1),
    rowperm: int = typer.Option(1, "--rowperm"),
    colperm: int = typer.Option(2, "--colperm"),
    int64: int = typer.Option(1, "--int64"),
    algo3d: int = typer.Option(0, "--algo3d"),
    verbosity: bool = typer.Option(False, "--verbosity"),
    launcher: str = typer.Option("", "--launcher", help="Override launcher, e.g. mpirun or srun."),
    launcher_extra_args: str = typer.Option("", "--launcher-extra-args", help="Extra launcher args split by spaces."),
    wait_timeout_sec: float = typer.Option(600.0, "--wait-timeout-sec", min=1.0),
    library_path: str = typer.Option("", "--library-path", help="Directory containing libsuperlu_dist_python."),
    legacy_real_block: bool = typer.Option(False, "--legacy-real-block", help="Use legacy real 2N block fallback instead of native complex path."),
    atol: float = typer.Option(1e-8, "--atol", min=0.0),
    rtol: float = typer.Option(1e-7, "--rtol", min=0.0),
) -> None:
    cfg = DistSolveConfig(
        nrow=nrow,
        ncol=ncol,
        rowperm=rowperm,
        colperm=colperm,
        int64=int64,
        algo3d=algo3d,
        verbosity=verbosity,
        library_path=library_path.strip() or None,
        launcher=launcher.strip() or None,
        launcher_extra_args=tuple(x for x in launcher_extra_args.split() if x),
        wait_timeout_sec=wait_timeout_sec,
        native_complex=not legacy_real_block,
    )

    logger.info("DistSolveConfig:")
    for k, v in asdict(cfg).items():
        logger.info(f"  - {k}: {v}")

    A = _make_matrix(n, density, seed, diag_shift)
    A_lu = spla.splu(A)

    rng = np.random.default_rng(seed + 1)

    factor = spfactorize(A, cfg)

    for _ in range(warmup):
        b_warm = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        _ = spsolve(factor, b_warm, overwrite_b=False)

    # 1) Single RHS repeated solve
    for i in range(repeats):
        b = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        x_dist = spsolve(factor, b, overwrite_b=False)
        x_ref = A_lu.solve(b)

        res = _max_residual(A, x_dist, b)
        logger.info(f"[single #{i}] residual_inf = {res:.3e}")
        _check_close(f"single #{i}", x_dist, x_ref, atol, rtol)

    # 2) Batch RHS solve (wrapper expects shape [batch, n])
    B = rng.standard_normal((batch, n)) + 1j * rng.standard_normal((batch, n))
    X_dist = spsolve(factor, B, overwrite_b=False)
    for i in range(batch):
        x_ref = A_lu.solve(B[i])
        res = _max_residual(A, X_dist[i], B[i])
        logger.info(f"[batch #{i}] residual_inf = {res:.3e}")
        _check_close(f"batch #{i}", X_dist[i], x_ref, atol, rtol)

    # 3) Same-pattern refactorize check
    scale = 1.0 + 0.05 * (rng.standard_normal(A.nnz) + 1j * rng.standard_normal(A.nnz))
    A2 = sp.csc_matrix((A.data * scale, A.indices, A.indptr), shape=A.shape)
    A2_lu = spla.splu(A2)
    factor.refactorize(A2)

    b2 = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    x2_dist = spsolve(factor, b2, overwrite_b=False)
    x2_ref = A2_lu.solve(b2)

    res2 = _max_residual(A2, x2_dist, b2)
    logger.info(f"[refactorize] residual_inf = {res2:.3e}")
    _check_close("refactorize", x2_dist, x2_ref, atol, rtol)

    logger.info("PASS: native complex superlu_dist smoke test completed.")


if __name__ == "__main__":
    app()
