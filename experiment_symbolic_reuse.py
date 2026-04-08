from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import toml
import typer
from loguru import logger

from fdfd_solver.constants import EPSILON_0, MU_0
from fdfd_solver.dxy_matrix import DxyMatrix
from spsolver import profile_phases, spanalyze, sprefactorize, spsolve

C0 = 1.0 / math.sqrt(EPSILON_0 * MU_0)

app = typer.Typer(add_completion=False)


@dataclass
class ReuseBenchRow:
    n: int
    N: int
    nnz: int
    sparsity_percent: float
    T_perm_ms: float
    T_symb_ms: float
    T_num_ms: float
    T_solve_ms: float
    T_total_ms: float
    T_reuse_ms: float
    speedup: float


@dataclass
class FullTiming:
    t_perm_ms: float
    t_symb_ms: float
    t_num_ms: float
    t_solve_ms: float
    t_total_ms: float


@dataclass
class ReuseTiming:
    t_num_ms: float
    t_solve_ms: float
    t_reuse_ms: float


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _build_eps_pair(
    n: int,
    eps_hi: float,
    eps_lo: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    # Construct a deterministic base pattern and a perturbed update with unchanged sparsity pattern.
    x = np.arange(n)[:, None]
    y = np.arange(n)[None, :]
    block = max(1, n // 16)
    mask = ((x // block + y // block) % 2).astype(np.float64)

    eps_base = eps_lo + (eps_hi - eps_lo) * mask

    noise = rng.normal(loc=0.0, scale=0.08, size=(n, n))
    eps_update = eps_base * (1.0 + noise)
    eps_update = np.clip(eps_update, eps_lo, eps_hi)
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


def _rhs(n_dof: int, rng: np.random.Generator) -> np.ndarray:
    real = rng.standard_normal(n_dof)
    imag = rng.standard_normal(n_dof)
    return (real + 1j * imag).astype(np.complex128)


def _run_full_once(
    A: sp.csc_matrix,
    colperm: int,
) -> FullTiming:
    prof = profile_phases(A, colperm=colperm)
    t_perm = float(prof["t_perm_ms"])
    t_symb = float(prof["t_symb_ms"])
    t_num_ms = float(prof["t_num_ms"])
    t_solve_ms = float(prof["t_solve_ms"])
    t_total_ms = float(prof["t_total_ms"])

    return FullTiming(
        t_perm_ms=t_perm,
        t_symb_ms=t_symb,
        t_num_ms=t_num_ms,
        t_solve_ms=t_solve_ms,
        t_total_ms=t_total_ms,
    )


def _run_reuse_once(
    A_symbolic: sp.csc_matrix,
    A_update: sp.csc_matrix,
    b: np.ndarray,
) -> ReuseTiming:
    factor = spanalyze(A_symbolic)

    t0 = time.perf_counter()
    sprefactorize(factor, np.asarray(A_update.data, dtype=np.complex128, order="C"))
    t1 = time.perf_counter()
    _ = spsolve(factor, b, overwrite_b=False)
    t2 = time.perf_counter()

    t_num_ms = (t1 - t0) * 1e3
    t_solve_ms = (t2 - t1) * 1e3
    t_reuse_ms = (t2 - t0) * 1e3
    return ReuseTiming(t_num_ms=t_num_ms, t_solve_ms=t_solve_ms, t_reuse_ms=t_reuse_ms)


def _mean(vals: Iterable[float]) -> float:
    arr = np.asarray(list(vals), dtype=np.float64)
    return float(arr.mean()) if arr.size else float("nan")


def _write_csv(rows: list[ReuseBenchRow], path: Path) -> None:
    _ensure_parent(path)
    header = (
        "n,N,nnz,sparsity_percent,T_perm_ms,T_symb_ms,T_num_ms,"
        "T_solve_ms,T_total_ms,T_reuse_ms,speedup"
    )
    lines = [header]
    for row in rows:
        lines.append(
            ",".join(
                [
                    str(row.n),
                    str(row.N),
                    str(row.nnz),
                    f"{row.sparsity_percent:.8f}",
                    f"{row.T_perm_ms:.6f}",
                    f"{row.T_symb_ms:.6f}",
                    f"{row.T_num_ms:.6f}",
                    f"{row.T_solve_ms:.6f}",
                    f"{row.T_total_ms:.6f}",
                    f"{row.T_reuse_ms:.6f}",
                    f"{row.speedup:.6f}",
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_latex_rows(rows: list[ReuseBenchRow], path: Path) -> None:
    _ensure_parent(path)
    lines = []
    for row in rows:
        lines.append(
            "{n} & {tp:.2f} & {ts:.2f} & {tn:.2f} & {tv:.2f} & {tt:.2f} & {tr:.2f} \\\\".format(
                n=row.n,
                tp=row.T_perm_ms,
                ts=row.T_symb_ms,
                tn=row.T_num_ms,
                tv=row.T_solve_ms,
                tt=row.T_total_ms,
                tr=row.T_reuse_ms,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plot(rows: list[ReuseBenchRow], path: Path) -> None:
    _ensure_parent(path)
    ns = [r.n for r in rows]
    spd = [r.speedup for r in rows]

    with plt.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    ):
        fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=220)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        ax.plot(
            ns,
            spd,
            color="black",
            marker="o",
            markersize=5.6,
            linewidth=1.8,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.2,
            zorder=3,
        )

        # Keep geometric spacing while forcing plain-number tick labels (8, 16, ...).
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns])
        ax.minorticks_off()

        y_max = max(spd) if spd else 1.0
        y_min = min(spd) if spd else 0.0
        y_span = max(y_max - y_min, 1e-6)
        y_bottom = y_min - 0.18 * y_span
        y_top = y_max + 0.20 * y_span
        ax.set_ylim(y_bottom, y_top)

        # Make the left bound slightly smaller than the first x tick so the y-axis is visually separated.
        x_left = ns[0] / 1.3
        x_right = ns[-1] * 1.2
        ax.set_xlim(x_left, x_right)

        for i, (x, y) in enumerate(zip(ns, spd)):
            # Project each point to x-axis (y=0) and y-axis (left spine) with dashed helper lines.
            ax.annotate(
                "",
                xy=(x, y),
                xycoords="data",
                xytext=(x, 0.0),
                textcoords=("data", "axes fraction"),
                arrowprops={
                    "arrowstyle": "-",
                    "linestyle": "--",
                    "color": "0.58",
                    "linewidth": 0.9,
                    "alpha": 0.9,
                },
                zorder=1,
            )
            ax.annotate(
                "",
                xy=(x, y),
                xycoords="data",
                xytext=(0.0, y),
                textcoords=("axes fraction", "data"),
                arrowprops={
                    "arrowstyle": "-",
                    "linestyle": "--",
                    "color": "0.58",
                    "linewidth": 0.9,
                    "alpha": 0.9,
                },
                zorder=1,
            )

            y_offset = 8 + (i % 2) * 4
            ax.annotate(
                f"{y:.2f}",
                xy=(x, y),
                xytext=(0, y_offset),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9.5,
                zorder=4,
            )

        ax.set_xlabel("Grid size $n$ ($N = n^2$)", fontsize=11.5)
        ax.set_ylabel("Speedup $ S = T_{{total}} / T_{{reuse}}$", fontsize=11.5)
        ax.set_title("Symbolic Reuse Speedup", fontsize=12.5)

        ax.grid(axis="y", linestyle="--", linewidth=0.75, alpha=0.35)
        ax.tick_params(direction="in", top=True, right=True, labelsize=10.5)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)


def _write_meta(rows: list[ReuseBenchRow], path: Path) -> None:
    _ensure_parent(path)
    payload = {
        "rows": [asdict(r) for r in rows],
    }
    path.write_text(toml.dumps(payload), encoding="utf-8")


@app.command()
def main(
    sizes: str = typer.Option(
        "10,20,40,80,160,320",
        "--sizes",
        help="Comma separated grid sizes n.",
    ),
    repeats: int = typer.Option(10, "--repeats", min=1),
    wavelength: float = typer.Option(1.55e-6, "--wavelength"),
    points_per_wavelength: int = typer.Option(20, "--ppw", min=4),
    eps_si: float = typer.Option(12.25, "--eps-si"),
    eps_sio2: float = typer.Option(2.085, "--eps-sio2"),
    npml: int = typer.Option(0, "--npml", min=0),
    colperm: int = typer.Option(3, "--colperm", help="SuperLU ColPerm enum, default 3=COLAMD."),
    seed: int = typer.Option(0, "--seed"),
    out_csv: str = typer.Option(
        "experiments/symbolic_reuse/reuse_detail.csv",
        "--out-csv",
    ),
    out_plot: str = typer.Option("figures/reuse_accu.png", "--out-plot"),
    out_latex: str = typer.Option(
        "experiments/symbolic_reuse/reuse_detail_rows.tex",
        "--out-latex",
    ),
    out_meta: str = typer.Option(
        "experiments/symbolic_reuse/reuse_detail.toml",
        "--out-meta",
    ),
) -> None:
    ns = [int(x.strip()) for x in sizes.split(",") if x.strip()]
    rng = np.random.default_rng(seed)

    logger.info("Start symbolic reuse benchmark")
    logger.info(f"sizes={ns}, repeats={repeats}, wavelength={wavelength}, ppw={points_per_wavelength}")

    rows: list[ReuseBenchRow] = []
    for n in ns:
        logger.info(f"Benchmark n={n}")
        eps_base, eps_update = _build_eps_pair(n, eps_si, eps_sio2, rng)
        A_base = _build_fdfd_matrix(eps_base, wavelength, points_per_wavelength, npml)
        A_update = _build_fdfd_matrix(eps_update, wavelength, points_per_wavelength, npml)

        N = n * n
        nnz = int(A_update.nnz)
        sparsity_percent = 100.0 * nnz / float(N * N)
        b = _rhs(N, rng)

        full_runs: list[FullTiming] = []
        reuse_runs: list[ReuseTiming] = []
        for _ in range(repeats):
            full_runs.append(_run_full_once(A_update, colperm=colperm))
            reuse_runs.append(_run_reuse_once(A_base, A_update, b))

        t_perm = _mean(x.t_perm_ms for x in full_runs)
        t_symb = _mean(x.t_symb_ms for x in full_runs)
        t_num = _mean(x.t_num_ms for x in full_runs)
        t_solve = _mean(x.t_solve_ms for x in full_runs)
        t_total = _mean(x.t_total_ms for x in full_runs)
        t_reuse = _mean(x.t_reuse_ms for x in reuse_runs)
        speedup = t_total / t_reuse if t_reuse > 0 else float("nan")

        row = ReuseBenchRow(
            n=n,
            N=N,
            nnz=nnz,
            sparsity_percent=sparsity_percent,
            T_perm_ms=t_perm,
            T_symb_ms=t_symb,
            T_num_ms=t_num,
            T_solve_ms=t_solve,
            T_total_ms=t_total,
            T_reuse_ms=t_reuse,
            speedup=speedup,
        )
        rows.append(row)
        logger.info(
            f"n={n}, N={N}, nnz={nnz}, total={t_total:.3f} ms, reuse={t_reuse:.3f} ms, speedup={speedup:.3f}x"
        )

    csv_path = Path(out_csv)
    plot_path = Path(out_plot)
    latex_path = Path(out_latex)
    meta_path = Path(out_meta)

    _write_csv(rows, csv_path)
    _write_latex_rows(rows, latex_path)
    _write_plot(rows, plot_path)
    _write_meta(rows, meta_path)

    logger.info(f"CSV saved: {csv_path}")
    logger.info(f"LaTeX rows saved: {latex_path}")
    logger.info(f"Plot saved: {plot_path}")
    logger.info(f"Meta saved: {meta_path}")


if __name__ == "__main__":
    app()
