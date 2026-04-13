from __future__ import annotations

import csv
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
from spsolver import profile_phases, spfactorize, spsolve

C0 = 1.0 / math.sqrt(EPSILON_0 * MU_0)

app = typer.Typer(add_completion=False)


@dataclass
class BatchDetailRow:
    n: int
    N: int
    nnz: int
    sparsity_percent: float
    T_factor_ms: float
    T_solve_ms: float
    factor_solve_ratio: float


@dataclass
class BatchSpeedupRow:
    n: int
    M: int
    T_naive_ms: float
    T_batch_ms: float
    speedup: float


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _mean(vals: Iterable[float]) -> float:
    arr = np.asarray(list(vals), dtype=np.float64)
    return float(arr.mean()) if arr.size else float("nan")


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


def _sample_rhs_batch(n_dof: int, batch_size: int, rng: np.random.Generator) -> np.ndarray:
    real = rng.standard_normal((batch_size, n_dof))
    imag = rng.standard_normal((batch_size, n_dof))
    return (real + 1j * imag).astype(np.complex128)


def _measure_factor_solve_profile(
    A: sp.csc_matrix,
    colperm: int,
    repeats: int,
) -> tuple[float, float]:
    t_factor_samples: list[float] = []
    t_solve_samples: list[float] = []
    for _ in range(repeats):
        prof = profile_phases(A, colperm=colperm)
        t_factor_samples.append(float(prof["t_num_ms"]))
        t_solve_samples.append(float(prof["t_solve_ms"]))
    return _mean(t_factor_samples), _mean(t_solve_samples)


def _time_naive_once(A: sp.csc_matrix, rhs_batch: np.ndarray) -> float:
    t0 = time.perf_counter()
    for b in rhs_batch:
        factor = spfactorize(A)
        _ = spsolve(factor, b, overwrite_b=False)
    t1 = time.perf_counter()
    return (t1 - t0) * 1e3


def _time_batch_once(A: sp.csc_matrix, rhs_batch: np.ndarray) -> float:
    t0 = time.perf_counter()
    factor = spfactorize(A)
    for b in rhs_batch:
        _ = spsolve(factor, b, overwrite_b=False)
    t1 = time.perf_counter()
    return (t1 - t0) * 1e3


def _write_detail_csv(rows: list[BatchDetailRow], path: Path) -> None:
    _ensure_parent(path)
    header = "n,N,nnz,sparsity_percent,T_factor_ms,T_solve_ms,factor_solve_ratio"
    lines = [header]
    for row in rows:
        lines.append(
            ",".join(
                [
                    str(row.n),
                    str(row.N),
                    str(row.nnz),
                    f"{row.sparsity_percent:.8f}",
                    f"{row.T_factor_ms:.6f}",
                    f"{row.T_solve_ms:.6f}",
                    f"{row.factor_solve_ratio:.6f}",
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_speedup_csv(rows: list[BatchSpeedupRow], path: Path) -> None:
    _ensure_parent(path)
    header = "n,M,T_naive_ms,T_batch_ms,speedup"
    lines = [header]
    for row in rows:
        lines.append(
            ",".join(
                [
                    str(row.n),
                    str(row.M),
                    f"{row.T_naive_ms:.6f}",
                    f"{row.T_batch_ms:.6f}",
                    f"{row.speedup:.6f}",
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_speedup_csv(path: Path) -> list[BatchSpeedupRow]:
    if not path.exists():
        raise FileNotFoundError(f"speedup csv not found: {path}")

    rows: list[BatchSpeedupRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"n", "M", "T_naive_ms", "T_batch_ms", "speedup"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"invalid speedup csv header in {path}, required: {sorted(required)}"
            )

        for item in reader:
            rows.append(
                BatchSpeedupRow(
                    n=int(item["n"]),
                    M=int(item["M"]),
                    T_naive_ms=float(item["T_naive_ms"]),
                    T_batch_ms=float(item["T_batch_ms"]),
                    speedup=float(item["speedup"]),
                )
            )

    if not rows:
        raise ValueError(f"speedup csv has no data rows: {path}")
    return rows


def _write_latex_detail_rows(rows: list[BatchDetailRow], path: Path) -> None:
    _ensure_parent(path)
    lines: list[str] = []
    for row in rows:
        lines.append(
            "{n} & {tf:.2f} & {ts:.2f} & {rt:.2f} \\\\".format(
                n=row.n,
                tf=row.T_factor_ms,
                ts=row.T_solve_ms,
                rt=row.factor_solve_ratio,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plot(rows: list[BatchSpeedupRow], path: Path) -> None:
    _ensure_parent(path)

    by_n: dict[int, list[BatchSpeedupRow]] = {}
    for row in rows:
        by_n.setdefault(row.n, []).append(row)
    for n in by_n:
        by_n[n].sort(key=lambda x: x.M)

    with plt.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    ):
        fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=220)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        marker_cycle = ["o", "s", "^", "D", "v", "P"]
        for i, n in enumerate(sorted(by_n)):
            r = by_n[n]
            xs = [x.M for x in r]
            ys = [x.speedup for x in r]
            ax.plot(
                xs,
                ys,
                color="black",
                linewidth=1.5,
                marker=marker_cycle[i % len(marker_cycle)],
                markersize=5.2,
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=1.0,
                label=f"n={n}",
            )

        m_values = sorted({x.M for x in rows})
        ax.set_xscale("log", base=2)
        ax.set_xticks(m_values)
        ax.set_xticklabels([str(m) for m in m_values])
        ax.minorticks_off()

        y_vals = [x.speedup for x in rows]
        y_max = max(y_vals) if y_vals else 1.0
        y_min = min(y_vals) if y_vals else 0.0
        y_span = max(y_max - y_min, 1e-6)
        ax.set_ylim(y_min - 0.12 * y_span, y_max + 0.18 * y_span)

        ax.set_xlabel("Batch size $M$", fontsize=11.5)
        ax.set_ylabel("Speedup $S = T_{naive} / T_{batch}$", fontsize=11.5)
        ax.set_title("Batch RHS Solve Speedup", fontsize=12.5)

        ax.grid(axis="y", linestyle="--", linewidth=0.75, alpha=0.35)
        ax.tick_params(direction="in", top=True, right=True, labelsize=10.5)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

        ax.legend(frameon=False, ncols=2, fontsize=9.5, loc="best")
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)


def _write_meta(
    detail_rows: list[BatchDetailRow],
    speedup_rows: list[BatchSpeedupRow],
    path: Path,
) -> None:
    _ensure_parent(path)
    payload = {
        "detail_rows": [asdict(r) for r in detail_rows],
        "speedup_rows": [asdict(r) for r in speedup_rows],
    }
    path.write_text(toml.dumps(payload), encoding="utf-8")


@app.command()
def main(
    sizes: str = typer.Option(
        "10,20,40,80,160,320",
        "--sizes",
        help="Comma separated grid sizes n.",
    ),
    batch_sizes: str = typer.Option("4,8,16,32,64", "--batch-sizes"),
    repeats: int = typer.Option(10, "--repeats", min=1),
    wavelength: float = typer.Option(1.55e-6, "--wavelength"),
    points_per_wavelength: int = typer.Option(20, "--ppw", min=4),
    eps_si: float = typer.Option(12.25, "--eps-si"),
    eps_sio2: float = typer.Option(2.085, "--eps-sio2"),
    npml: int = typer.Option(0, "--npml", min=0),
    colperm: int = typer.Option(3, "--colperm", help="SuperLU ColPerm enum, default 3=COLAMD."),
    seed: int = typer.Option(0, "--seed"),
    out_detail_csv: str = typer.Option(
        "experiments/batch_rhs/batch_detail.csv",
        "--out-detail-csv",
    ),
    out_speedup_csv: str = typer.Option(
        "experiments/batch_rhs/batch_speedup.csv",
        "--out-speedup-csv",
    ),
    out_plot: str = typer.Option("figures/batch_speedup.png", "--out-plot"),
    redraw_only: bool = typer.Option(
        False,
        "--redraw-only",
        help="Only redraw plot from existing speedup CSV without rerunning experiments.",
    ),
    redraw_from_speedup_csv: str = typer.Option(
        "",
        "--redraw-from-speedup-csv",
        help="Existing speedup CSV path used when --redraw-only is enabled. Default uses --out-speedup-csv.",
    ),
    out_latex: str = typer.Option(
        "experiments/batch_rhs/batch_detail_rows.tex",
        "--out-latex",
    ),
    out_meta: str = typer.Option(
        "experiments/batch_rhs/batch_detail.toml",
        "--out-meta",
    ),
) -> None:
    plot_path = Path(out_plot)
    default_speedup_csv_path = Path(out_speedup_csv)

    if redraw_only:
        source_path = (
            Path(redraw_from_speedup_csv)
            if redraw_from_speedup_csv.strip()
            else default_speedup_csv_path
        )
        logger.info(f"Redraw-only mode: source={source_path}, out_plot={plot_path}")
        speedup_rows = _read_speedup_csv(source_path)
        _write_plot(speedup_rows, plot_path)
        logger.info(f"Plot saved: {plot_path}")
        return

    ns = [int(x.strip()) for x in sizes.split(",") if x.strip()]
    ms = [int(x.strip()) for x in batch_sizes.split(",") if x.strip()]
    rng = np.random.default_rng(seed)

    logger.info("Start batch RHS benchmark")
    logger.info(f"sizes={ns}, batch_sizes={ms}, repeats={repeats}, ppw={points_per_wavelength}")

    detail_rows: list[BatchDetailRow] = []
    speedup_rows: list[BatchSpeedupRow] = []

    for n in ns:
        logger.info(f"Benchmark n={n}")
        _, eps_update = _build_eps_pair(n, eps_si, eps_sio2, rng)
        A = _build_fdfd_matrix(eps_update, wavelength, points_per_wavelength, npml)

        N = n * n
        nnz = int(A.nnz)
        sparsity_percent = 100.0 * nnz / float(N * N)

        t_factor, t_solve = _measure_factor_solve_profile(A, colperm=colperm, repeats=repeats)
        ratio = t_factor / t_solve if t_solve > 0 else float("nan")

        detail_rows.append(
            BatchDetailRow(
                n=n,
                N=N,
                nnz=nnz,
                sparsity_percent=sparsity_percent,
                T_factor_ms=t_factor,
                T_solve_ms=t_solve,
                factor_solve_ratio=ratio,
            )
        )

        logger.info(
            f"n={n}, factor={t_factor:.3f} ms, solve={t_solve:.3f} ms, ratio={ratio:.2f}"
        )

        for m in ms:
            naive_samples: list[float] = []
            batch_samples: list[float] = []

            for _ in range(repeats):
                rhs_batch = _sample_rhs_batch(N, m, rng)
                naive_samples.append(_time_naive_once(A, rhs_batch))
                batch_samples.append(_time_batch_once(A, rhs_batch))

            t_naive = _mean(naive_samples)
            t_batch = _mean(batch_samples)
            speedup = t_naive / t_batch if t_batch > 0 else float("nan")

            speedup_rows.append(
                BatchSpeedupRow(
                    n=n,
                    M=m,
                    T_naive_ms=t_naive,
                    T_batch_ms=t_batch,
                    speedup=speedup,
                )
            )

            logger.info(
                f"n={n}, M={m}, naive={t_naive:.3f} ms, batch={t_batch:.3f} ms, speedup={speedup:.3f}x"
            )

    detail_csv_path = Path(out_detail_csv)
    speedup_csv_path = default_speedup_csv_path
    latex_path = Path(out_latex)
    meta_path = Path(out_meta)

    _write_detail_csv(detail_rows, detail_csv_path)
    _write_speedup_csv(speedup_rows, speedup_csv_path)
    _write_latex_detail_rows(detail_rows, latex_path)
    _write_plot(speedup_rows, plot_path)
    _write_meta(detail_rows, speedup_rows, meta_path)

    logger.info(f"Detail CSV saved: {detail_csv_path}")
    logger.info(f"Speedup CSV saved: {speedup_csv_path}")
    logger.info(f"LaTeX rows saved: {latex_path}")
    logger.info(f"Plot saved: {plot_path}")
    logger.info(f"Meta saved: {meta_path}")


if __name__ == "__main__":
    app()
