from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import toml
import typer
from loguru import logger

from fdfd_solver.constants import EPSILON_0, MU_0
from fdfd_solver.dxy_matrix import DxyMatrix

matplotlib.use("Agg")
import matplotlib.pyplot as plt


C0 = 1.0 / math.sqrt(EPSILON_0 * MU_0)

app = typer.Typer(add_completion=False)


@dataclass
class MatrixShapeMetrics:
    n: int
    seed: int
    dof: int
    nnz: int
    density: float
    bandwidth_max: int
    bandwidth_q50: float
    bandwidth_q90: float
    bandwidth_q99: float
    bandwidth_max_rcm: int
    bandwidth_reduction_rcm: float
    nnz_diag_ratio: float
    l1_diag_ratio: float
    near_diag_k1_ratio: float
    near_diag_k2_ratio: float
    near_diag_kn_ratio: float
    near_diag_k2n_ratio: float
    row_nnz_min: int
    row_nnz_max: int
    row_nnz_mean: float
    col_nnz_min: int
    col_nnz_max: int
    col_nnz_mean: float
    diag_abs_min: float
    diag_abs_median: float
    diag_abs_max: float
    diag_near_zero_ratio: float
    diag_dominant_row_ratio: float
    struct_symmetric_match_ratio: float


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _parse_n_values(n_values: str) -> list[int]:
    vals: list[int] = []
    for token in n_values.split(","):
        t = token.strip()
        if not t:
            continue
        vals.append(int(t))
    if not vals:
        raise ValueError("empty --n-values")
    return vals


def _build_eps_pair(
    n: int,
    eps_hi: float,
    eps_lo: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(n)[:, None]
    y = np.arange(n)[None, :]
    block = max(1, n // 16)
    mask = ((x // block + y // block) % 2).astype(np.float64)

    eps_base = eps_lo + (eps_hi - eps_lo) * mask
    noise = rng.normal(loc=0.0, scale=0.08, size=(n, n))
    eps_update = np.clip(eps_base * (1.0 + noise), eps_lo, eps_hi)
    return eps_base, eps_update


def _build_fdfd_matrix(
    eps_r: np.ndarray,
    wavelength: float,
    points_per_wavelength: int,
    npml: int,
) -> sp.csc_matrix:
    if eps_r.ndim != 2 or eps_r.shape[0] != eps_r.shape[1]:
        raise ValueError(f"eps_r must be square 2D array, got {eps_r.shape}")

    n = int(eps_r.shape[0])
    dL = wavelength / float(points_per_wavelength)
    omega = 2.0 * math.pi * C0 / wavelength

    dxy = DxyMatrix(omega, dL, (n, n), (npml, npml))
    lap = (dxy.dxf * dxy.dxb + dxy.dyf * dxy.dyb) * (-1.0 / MU_0)
    diag = sp.diags(
        (eps_r.reshape(-1) * (omega**2) * (-EPSILON_0)),
        offsets=0,
        shape=(n * n, n * n),
        dtype=np.complex128,
        format="csc",
    )
    return (lap + diag).tocsc()


def _bandwidth_max(A: sp.spmatrix) -> int:
    rr, cc = A.nonzero()
    if rr.size == 0:
        return 0
    return int(np.abs(rr - cc).max())


def _analyze_matrix(A: sp.csc_matrix, n: int, seed: int) -> MatrixShapeMetrics:
    dof = int(A.shape[0])
    nnz = int(A.nnz)
    density = float(nnz / (dof * dof))

    rr, cc = A.nonzero()
    offsets = np.abs(rr - cc).astype(np.int64)
    bw_max = int(offsets.max()) if offsets.size else 0
    bw_q50 = float(np.quantile(offsets, 0.50)) if offsets.size else 0.0
    bw_q90 = float(np.quantile(offsets, 0.90)) if offsets.size else 0.0
    bw_q99 = float(np.quantile(offsets, 0.99)) if offsets.size else 0.0

    near_k1 = float(np.mean(offsets <= 1)) if offsets.size else 0.0
    near_k2 = float(np.mean(offsets <= 2)) if offsets.size else 0.0
    near_kn = float(np.mean(offsets <= n)) if offsets.size else 0.0
    near_k2n = float(np.mean(offsets <= 2 * n)) if offsets.size else 0.0

    diag = A.diagonal()
    diag_abs = np.abs(diag)
    diag_abs_min = float(diag_abs.min()) if diag_abs.size else float("nan")
    diag_abs_median = float(np.median(diag_abs)) if diag_abs.size else float("nan")
    diag_abs_max = float(diag_abs.max()) if diag_abs.size else float("nan")
    diag_scale = max(diag_abs_max, 1.0)
    diag_near_zero_ratio = float(np.mean(diag_abs <= (1e-14 * diag_scale))) if diag_abs.size else float("nan")

    absA = np.abs(A).tocsr()
    row_abs_sum = np.asarray(absA.sum(axis=1)).ravel()
    offdiag_abs_sum = row_abs_sum - diag_abs
    dominance = diag_abs / np.maximum(offdiag_abs_sum, 1e-300)
    diag_dominant_row_ratio = float(np.mean(dominance >= 1.0))

    l1_diag = float(diag_abs.sum())
    l1_total = float(np.abs(A.data).sum())
    l1_diag_ratio = l1_diag / l1_total if l1_total > 0 else float("nan")

    nnz_diag = int(np.count_nonzero(rr == cc)) if rr.size else 0
    nnz_diag_ratio = float(nnz_diag / nnz) if nnz > 0 else float("nan")

    row_nnz = np.diff(A.tocsr().indptr)
    col_nnz = np.diff(A.indptr)

    A_pattern = A.copy()
    A_pattern.data = np.ones_like(A_pattern.data)
    asym = (A_pattern != A_pattern.T).astype(np.int8)
    unmatched = int(asym.nnz)
    matched = max(nnz - unmatched, 0)
    struct_symmetric_match_ratio = float(matched / nnz) if nnz > 0 else float("nan")

    perm = csgraph.reverse_cuthill_mckee(A_pattern, symmetric_mode=False)
    A_rcm = A[perm, :][:, perm]
    bw_rcm = _bandwidth_max(A_rcm)
    bw_reduction = float(bw_max / bw_rcm) if bw_rcm > 0 else float("nan")

    return MatrixShapeMetrics(
        n=n,
        seed=seed,
        dof=dof,
        nnz=nnz,
        density=density,
        bandwidth_max=bw_max,
        bandwidth_q50=bw_q50,
        bandwidth_q90=bw_q90,
        bandwidth_q99=bw_q99,
        bandwidth_max_rcm=bw_rcm,
        bandwidth_reduction_rcm=bw_reduction,
        nnz_diag_ratio=nnz_diag_ratio,
        l1_diag_ratio=l1_diag_ratio,
        near_diag_k1_ratio=near_k1,
        near_diag_k2_ratio=near_k2,
        near_diag_kn_ratio=near_kn,
        near_diag_k2n_ratio=near_k2n,
        row_nnz_min=int(row_nnz.min()),
        row_nnz_max=int(row_nnz.max()),
        row_nnz_mean=float(row_nnz.mean()),
        col_nnz_min=int(col_nnz.min()),
        col_nnz_max=int(col_nnz.max()),
        col_nnz_mean=float(col_nnz.mean()),
        diag_abs_min=diag_abs_min,
        diag_abs_median=diag_abs_median,
        diag_abs_max=diag_abs_max,
        diag_near_zero_ratio=diag_near_zero_ratio,
        diag_dominant_row_ratio=diag_dominant_row_ratio,
        struct_symmetric_match_ratio=struct_symmetric_match_ratio,
    )


def _plot_matrix_shape(A: sp.csc_matrix, n: int, seed: int, path: Path) -> None:
    _ensure_parent(path)
    rr, cc = A.nonzero()
    offsets = np.abs(rr - cc)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].scatter(cc, rr, s=0.2, alpha=0.45, linewidths=0)
    axes[0].invert_yaxis()
    axes[0].set_title(f"Sparsity Pattern (n={n}, seed={seed})")
    axes[0].set_xlabel("column")
    axes[0].set_ylabel("row")

    bins = np.linspace(0, max(int(offsets.max()), 1), 80)
    axes[1].hist(offsets, bins=bins, color="#2667a5", alpha=0.85)
    axes[1].set_yscale("log")
    axes[1].set_title("|row - col| distribution")
    axes[1].set_xlabel("offset")
    axes[1].set_ylabel("count (log scale)")

    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_csv(rows: list[MatrixShapeMetrics], path: Path) -> None:
    _ensure_parent(path)
    if not rows:
        raise ValueError("no rows to write")

    keys = list(asdict(rows[0]).keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))


def _write_meta(rows: list[MatrixShapeMetrics], path: Path, params: dict[str, object]) -> None:
    _ensure_parent(path)
    payload = {
        "params": params,
        "rows": [asdict(r) for r in rows],
    }
    path.write_text(toml.dumps(payload), encoding="utf-8")


def _summarize(rows: list[MatrixShapeMetrics]) -> dict[str, float]:
    def mean_of(name: str) -> float:
        vals = [float(getattr(r, name)) for r in rows]
        return float(np.mean(vals))

    return {
        "cases": float(len(rows)),
        "density_mean": mean_of("density"),
        "bandwidth_max_mean": mean_of("bandwidth_max"),
        "bandwidth_reduction_rcm_mean": mean_of("bandwidth_reduction_rcm"),
        "near_diag_k2_mean": mean_of("near_diag_k2_ratio"),
        "near_diag_kn_mean": mean_of("near_diag_kn_ratio"),
        "l1_diag_ratio_mean": mean_of("l1_diag_ratio"),
        "diag_dominant_row_ratio_mean": mean_of("diag_dominant_row_ratio"),
        "struct_symmetric_match_ratio_mean": mean_of("struct_symmetric_match_ratio"),
    }


@app.command()
def main(
    n_values: str = typer.Option("200", "--n-values", help="Comma-separated n values, e.g. 80,120,200"),
    samples_per_n: int = typer.Option(3, "--samples-per-n", min=1),
    wavelength: float = typer.Option(1.55e-6, "--wavelength"),
    points_per_wavelength: int = typer.Option(20, "--ppw", min=4),
    eps_si: float = typer.Option(12.25, "--eps-si"),
    eps_sio2: float = typer.Option(2.085, "--eps-sio2"),
    npml: int = typer.Option(20, "--npml", min=0),
    seed: int = typer.Option(0, "--seed"),
    out_csv: str = typer.Option("experiments/matrix_shape/matrix_shape_metrics.csv", "--out-csv"),
    out_meta: str = typer.Option("experiments/matrix_shape/matrix_shape_metrics.toml", "--out-meta"),
    out_plot_dir: str = typer.Option("experiments/matrix_shape/plots", "--out-plot-dir"),
) -> None:
    ns = _parse_n_values(n_values)
    plot_dir = Path(out_plot_dir)
    csv_path = Path(out_csv)
    meta_path = Path(out_meta)
    rows: list[MatrixShapeMetrics] = []

    logger.info(
        f"Start matrix-shape analysis: n_values={ns}, samples_per_n={samples_per_n}, npml={npml}"
    )

    case_id = 0
    for n in ns:
        for local_idx in range(samples_per_n):
            case_seed = seed + case_id
            case_id += 1

            rng = np.random.default_rng(case_seed)
            _, eps_update = _build_eps_pair(n, eps_si, eps_sio2, rng)
            A = _build_fdfd_matrix(eps_update, wavelength, points_per_wavelength, npml)

            metrics = _analyze_matrix(A, n=n, seed=case_seed)
            rows.append(metrics)

            plot_path = plot_dir / f"A_shape_n{n}_seed{case_seed}.png"
            _plot_matrix_shape(A, n=n, seed=case_seed, path=plot_path)

            logger.info(
                (
                    f"[n={n}, seed={case_seed}] dof={metrics.dof}, nnz={metrics.nnz}, "
                    f"bw_max={metrics.bandwidth_max}, near(k<=2)={metrics.near_diag_k2_ratio:.3f}, "
                    f"near(k<=n)={metrics.near_diag_kn_ratio:.3f}, "
                    f"diag_dom_rows={metrics.diag_dominant_row_ratio:.3f}, "
                    f"rcm_reduction={metrics.bandwidth_reduction_rcm:.2f}x"
                )
            )

    _write_csv(rows, csv_path)
    summary = _summarize(rows)
    _write_meta(
        rows,
        meta_path,
        params={
            "n_values": ns,
            "samples_per_n": samples_per_n,
            "wavelength": wavelength,
            "points_per_wavelength": points_per_wavelength,
            "eps_si": eps_si,
            "eps_sio2": eps_sio2,
            "npml": npml,
            "seed": seed,
            "summary": summary,
        },
    )

    logger.info(f"CSV saved: {csv_path}")
    logger.info(f"Meta saved: {meta_path}")
    logger.info(f"Plots saved under: {plot_dir}")
    logger.info(f"Summary: {summary}")


if __name__ == "__main__":
    app()
