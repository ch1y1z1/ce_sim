from __future__ import annotations

import csv
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import toml
import typer
from loguru import logger

import spsolver.bindings_superlu_dist as dist_mod
from fdfd_solver.constants import EPSILON_0, MU_0
from fdfd_solver.dxy_matrix import DxyMatrix
from spsolver import profile_phases as profile_phases_serial
from spsolver import sprefactorize as sprefactorize_serial
from spsolver import spanalyze as spanalyze_serial
from spsolver.bindings_superlu_dist import DistSolveConfig, spfactorize as spfactorize_dist

C0 = 1.0 / math.sqrt(EPSILON_0 * MU_0)

app = typer.Typer(add_completion=False)


@dataclass(frozen=True)
class ParallelSpec:
    ntasks: int
    cpus_per_task: int
    nrow: int
    ncol: int


@dataclass
class ParallelRow:
    ntasks: int
    cpus_per_task: int
    nrow: int
    ncol: int
    grid: str
    factor_time_s: float
    speedup: float
    status: str
    error: str
    cold_full_factor_s: float = float("nan")
    warm_refactorize_s: float = float("nan")
    superlu_profile_colperm: int = -1
    superlu_phase_colperm_ms: float = float("nan")
    superlu_phase_etree_ms: float = float("nan")
    superlu_phase_fact_ms: float = float("nan")
    superlu_profile_status: str = "na"
    superlu_profile_error: str = ""
    serial_baseline_refactorize_s: float = float("nan")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _mean(vals: Iterable[float]) -> float:
    arr = np.asarray(list(vals), dtype=np.float64)
    return float(arr.mean()) if arr.size else float("nan")


def _parse_int_list(text: str) -> list[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("list option is empty")
    return values


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


def _all_grids(ntasks: int) -> list[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for d in range(1, int(math.sqrt(ntasks)) + 1):
        if ntasks % d != 0:
            continue
        e = ntasks // d
        pairs.add((d, e))
        pairs.add((e, d))
    return sorted(pairs)


def _build_specs(ntasks_values: list[int], cpus_values: list[int]) -> list[ParallelSpec]:
    specs: list[ParallelSpec] = []
    seen: set[tuple[int, int, int, int]] = set()

    baseline = (1, 1, 1, 1)
    if baseline not in seen:
        seen.add(baseline)
        specs.append(ParallelSpec(*baseline))

    for ntasks in ntasks_values:
        for cpus in cpus_values:
            for nrow, ncol in _all_grids(ntasks):
                key = (ntasks, cpus, nrow, ncol)
                if key in seen:
                    continue
                seen.add(key)
                specs.append(ParallelSpec(ntasks=ntasks, cpus_per_task=cpus, nrow=nrow, ncol=ncol))

    return specs


def _set_thread_env(cpus_per_task: int) -> dict[str, Optional[str]]:
    keys = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]
    old = {k: os.environ.get(k) for k in keys}
    value = str(cpus_per_task)
    for k in keys:
        os.environ[k] = value
    return old


def _restore_thread_env(old_env: dict[str, Optional[str]]) -> None:
    for key, value in old_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _cleanup_dist_sessions() -> None:
    cleanup_fn = getattr(dist_mod, "_cleanup_sessions", None)
    if callable(cleanup_fn):
        cleanup_fn()


def _profile_superlu_factor_breakdown(A: sp.csc_matrix, *, colperm: int) -> dict[str, Any]:
    try:
        prof = profile_phases_serial(A, colperm=colperm)
        phase_ms = prof.get("phase_ms", {})
        return {
            "colperm": int(colperm),
            "phase_colperm_ms": float(phase_ms.get("COLPERM", float("nan"))),
            "phase_etree_ms": float(phase_ms.get("ETREE", float("nan"))),
            "phase_fact_ms": float(phase_ms.get("FACT", float("nan"))),
            "status": "ok",
            "error": "",
        }
    except Exception as exc:
        return {
            "colperm": int(colperm),
            "phase_colperm_ms": float("nan"),
            "phase_etree_ms": float("nan"),
            "phase_fact_ms": float("nan"),
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _measure_serial_factor_time(
    A: sp.csc_matrix,
    *,
    repeats: int,
    warmup: int,
) -> tuple[float, float, str]:
    old_env = _set_thread_env(1)
    try:
        nzval = np.asarray(A.data, dtype=np.complex128, order="C")

        t0 = time.perf_counter()
        factor = spanalyze_serial(A)
        sprefactorize_serial(factor, nzval)
        t1 = time.perf_counter()
        cold_full_factor = t1 - t0

        for _ in range(max(0, warmup)):
            sprefactorize_serial(factor, nzval)

        samples: list[float] = []
        for _ in range(repeats):
            t2 = time.perf_counter()
            sprefactorize_serial(factor, nzval)
            t3 = time.perf_counter()
            samples.append(t3 - t2)

        return cold_full_factor, _mean(samples), ""
    except Exception as exc:
        return float("nan"), float("nan"), f"{type(exc).__name__}: {exc}"
    finally:
        _restore_thread_env(old_env)


def _measure_factor_time(
    A: sp.csc_matrix,
    spec: ParallelSpec,
    *,
    repeats: int,
    warmup: int,
    rowperm: int,
    colperm: int,
    int64: int,
    algo3d: int,
    verbosity: bool,
    launcher: Optional[str],
    launcher_extra_args: tuple[str, ...],
    wait_timeout_sec: float,
    library_path: Optional[str],
) -> tuple[float, float, str]:
    old_env = _set_thread_env(spec.cpus_per_task)
    try:
        _cleanup_dist_sessions()
        cfg = DistSolveConfig(
            nrow=spec.nrow,
            ncol=spec.ncol,
            rowperm=rowperm,
            colperm=colperm,
            int64=int64,
            algo3d=algo3d,
            verbosity=verbosity,
            library_path=library_path,
            launcher=launcher,
            launcher_extra_args=launcher_extra_args,
            wait_timeout_sec=wait_timeout_sec,
        )

        t_cold0 = time.perf_counter()
        factor = spfactorize_dist(A, config=cfg)
        t_cold1 = time.perf_counter()
        cold_full_factor = t_cold1 - t_cold0

        for _ in range(max(0, warmup)):
            factor.refactorize(A)

        times: list[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            factor.refactorize(A)
            t1 = time.perf_counter()
            times.append(t1 - t0)

        _cleanup_dist_sessions()
        return cold_full_factor, _mean(times), ""
    except Exception as exc:
        _cleanup_dist_sessions()
        return float("nan"), float("nan"), f"{type(exc).__name__}: {exc}"
    finally:
        _restore_thread_env(old_env)


def _attach_superlu_profile(rows: list[ParallelRow], profile: dict[str, Any]) -> None:
    for r in rows:
        r.superlu_profile_colperm = int(profile.get("colperm", -1))
        r.superlu_phase_colperm_ms = float(profile.get("phase_colperm_ms", float("nan")))
        r.superlu_phase_etree_ms = float(profile.get("phase_etree_ms", float("nan")))
        r.superlu_phase_fact_ms = float(profile.get("phase_fact_ms", float("nan")))
        r.superlu_profile_status = str(profile.get("status", "na"))
        r.superlu_profile_error = str(profile.get("error", ""))


def _write_csv(rows: list[ParallelRow], path: Path) -> None:
    _ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "ntasks",
                "cpus_per_task",
                "nrow",
                "ncol",
                "grid",
                "factor_time_s",
                "cold_full_factor_s",
                "warm_refactorize_s",
                "speedup",
                "superlu_profile_colperm",
                "superlu_phase_colperm_ms",
                "superlu_phase_etree_ms",
                "superlu_phase_fact_ms",
                "superlu_profile_status",
                "superlu_profile_error",
                "serial_baseline_refactorize_s",
                "status",
                "error",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.ntasks,
                    row.cpus_per_task,
                    row.nrow,
                    row.ncol,
                    row.grid,
                    f"{row.factor_time_s:.9e}" if math.isfinite(row.factor_time_s) else "nan",
                    f"{row.cold_full_factor_s:.9e}" if math.isfinite(row.cold_full_factor_s) else "nan",
                    f"{row.warm_refactorize_s:.9e}" if math.isfinite(row.warm_refactorize_s) else "nan",
                    f"{row.speedup:.6f}" if math.isfinite(row.speedup) else "nan",
                    str(row.superlu_profile_colperm) if row.superlu_profile_colperm >= 0 else "na",
                    f"{row.superlu_phase_colperm_ms:.9e}" if math.isfinite(row.superlu_phase_colperm_ms) else "nan",
                    f"{row.superlu_phase_etree_ms:.9e}" if math.isfinite(row.superlu_phase_etree_ms) else "nan",
                    f"{row.superlu_phase_fact_ms:.9e}" if math.isfinite(row.superlu_phase_fact_ms) else "nan",
                    row.superlu_profile_status,
                    row.superlu_profile_error,
                    f"{row.serial_baseline_refactorize_s:.9e}" if math.isfinite(row.serial_baseline_refactorize_s) else "nan",
                    row.status,
                    row.error,
                ]
            )


def _read_csv(path: Path) -> list[ParallelRow]:
    if not path.exists():
        raise FileNotFoundError(f"parallel csv not found: {path}")

    rows: list[ParallelRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "ntasks",
            "cpus_per_task",
            "nrow",
            "ncol",
            "grid",
            "factor_time_s",
            "speedup",
            "status",
            "error",
        }
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"invalid csv header in {path}")

        for item in reader:
            ft = float(item["factor_time_s"]) if item["factor_time_s"] else float("nan")
            spd = float(item["speedup"]) if item["speedup"] else float("nan")
            cold = float(item.get("cold_full_factor_s", "nan") or "nan")
            warm = float(item.get("warm_refactorize_s", item.get("factor_time_s", "nan")) or "nan")
            prof_colperm_text = item.get("superlu_profile_colperm", "na")
            rows.append(
                ParallelRow(
                    ntasks=int(item["ntasks"]),
                    cpus_per_task=int(item["cpus_per_task"]),
                    nrow=int(item["nrow"]),
                    ncol=int(item["ncol"]),
                    grid=item["grid"],
                    factor_time_s=ft,
                    speedup=spd,
                    status=item["status"],
                    error=item["error"],
                    cold_full_factor_s=cold,
                    warm_refactorize_s=warm,
                    superlu_profile_colperm=int(prof_colperm_text) if prof_colperm_text not in ("", "na") else -1,
                    superlu_phase_colperm_ms=float(item.get("superlu_phase_colperm_ms", "nan") or "nan"),
                    superlu_phase_etree_ms=float(item.get("superlu_phase_etree_ms", "nan") or "nan"),
                    superlu_phase_fact_ms=float(item.get("superlu_phase_fact_ms", "nan") or "nan"),
                    superlu_profile_status=item.get("superlu_profile_status", "na"),
                    superlu_profile_error=item.get("superlu_profile_error", ""),
                    serial_baseline_refactorize_s=float(
                        item.get("serial_baseline_refactorize_s", "nan") or "nan"
                    ),
                )
            )

    if not rows:
        raise ValueError(f"csv has no data rows: {path}")
    return rows


def _write_latex_rows(rows: list[ParallelRow], path: Path) -> None:
    _ensure_parent(path)
    lines: list[str] = []
    for row in rows:
        t = f"{row.factor_time_s:.3e}" if math.isfinite(row.factor_time_s) else "---"
        s = f"{row.speedup:.2f}\\times" if math.isfinite(row.speedup) else "---"
        lines.append(
            "{nt} & {cp} & ${nr}\\times{nc}$ & {t} & {s} \\\\".format(
                nt=row.ntasks,
                cp=row.cpus_per_task,
                nr=row.nrow,
                nc=row.ncol,
                t=t,
                s=s,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plot(rows: list[ParallelRow], path: Path) -> None:
    _ensure_parent(path)
    ok_rows = [r for r in rows if r.status == "ok" and math.isfinite(r.speedup)]
    if not ok_rows:
        raise ValueError("no successful rows to plot")

    labels = [f"{r.ntasks}/{r.cpus_per_task}\n{r.nrow}x{r.ncol}" for r in ok_rows]
    values = [r.speedup for r in ok_rows]
    x = np.arange(len(ok_rows))

    with plt.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    ):
        fig, ax = plt.subplots(figsize=(max(7.6, 0.52 * len(ok_rows)), 4.9), dpi=220)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        bars = ax.bar(
            x,
            values,
            color="white",
            edgecolor="black",
            linewidth=1.0,
            width=0.72,
            zorder=3,
        )

        ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1.0, zorder=2)

        for rect, v in zip(bars, values):
            ax.annotate(
                f"{v:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2.0, v),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Speedup vs serial warm refactorize", fontsize=11)
        ax.set_xlabel("ntasks / cpus-per-task and process grid", fontsize=11)
        ax.set_title("SuperLU_DIST Parallel Configuration Study", fontsize=12)

        y_max = max(values)
        ax.set_ylim(0.0, y_max * 1.25 if y_max > 0 else 1.0)
        ax.grid(axis="y", linestyle="--", linewidth=0.75, alpha=0.35, zorder=1)
        ax.tick_params(direction="in", top=True, right=True)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)


def _write_meta(
    rows: list[ParallelRow],
    path: Path,
    n: int,
    repeats: int,
    warmup: int,
    superlu_profile: dict[str, Any],
    serial_baseline: dict[str, Any],
) -> None:
    _ensure_parent(path)
    payload = {
        "matrix": {"n": n, "N": n * n},
        "measurement": {"repeats": repeats, "warmup": warmup},
        "serial_baseline": serial_baseline,
        "superlu_profile": superlu_profile,
        "rows": [asdict(r) for r in rows],
    }
    path.write_text(toml.dumps(payload), encoding="utf-8")


def _compute_speedups(rows: list[ParallelRow], *, baseline: float) -> list[ParallelRow]:
    out: list[ParallelRow] = []
    for r in rows:
        spd = float("nan")
        if baseline is not None and r.status == "ok" and math.isfinite(r.factor_time_s) and r.factor_time_s > 0:
            spd = baseline / r.factor_time_s
        out.append(
            ParallelRow(
                ntasks=r.ntasks,
                cpus_per_task=r.cpus_per_task,
                nrow=r.nrow,
                ncol=r.ncol,
                grid=r.grid,
                factor_time_s=r.factor_time_s,
                speedup=spd,
                status=r.status,
                error=r.error,
                cold_full_factor_s=r.cold_full_factor_s,
                warm_refactorize_s=r.warm_refactorize_s,
                superlu_profile_colperm=r.superlu_profile_colperm,
                superlu_phase_colperm_ms=r.superlu_phase_colperm_ms,
                superlu_phase_etree_ms=r.superlu_phase_etree_ms,
                superlu_phase_fact_ms=r.superlu_phase_fact_ms,
                superlu_profile_status=r.superlu_profile_status,
                superlu_profile_error=r.superlu_profile_error,
                serial_baseline_refactorize_s=baseline,
            )
        )
    return out


@app.command()
def main(
    n: int = typer.Option(200, "--n", min=8, help="Grid size n; matrix dimension is N=n^2."),
    ntasks_list: str = typer.Option("1,2,4", "--ntasks-list"),
    cpus_per_task_list: str = typer.Option("1,2,4,8", "--cpus-per-task-list"),
    repeats: int = typer.Option(5, "--repeats", min=1),
    warmup: int = typer.Option(1, "--warmup", min=1),
    wavelength: float = typer.Option(1.55e-6, "--wavelength"),
    points_per_wavelength: int = typer.Option(20, "--ppw", min=4),
    eps_si: float = typer.Option(12.25, "--eps-si"),
    eps_sio2: float = typer.Option(2.085, "--eps-sio2"),
    npml: int = typer.Option(20, "--npml", min=0),
    rowperm: int = typer.Option(1, "--rowperm"),
    colperm: int = typer.Option(2, "--colperm"),
    int64: int = typer.Option(1, "--int64"),
    algo3d: int = typer.Option(0, "--algo3d"),
    verbosity: bool = typer.Option(False, "--verbosity"),
    launcher: str = typer.Option("", "--launcher", help="Override launcher, e.g. mpirun or srun."),
    launcher_extra_args: str = typer.Option("", "--launcher-extra-args", help="Extra launcher args split by spaces."),
    wait_timeout_sec: float = typer.Option(600.0, "--wait-timeout-sec", min=1.0),
    library_path: str = typer.Option("", "--library-path", help="Directory containing libsuperlu_dist_python."),
    seed: int = typer.Option(0, "--seed"),
    out_csv: str = typer.Option("experiments/dist_parallel/parallel_config.csv", "--out-csv"),
    out_latex: str = typer.Option(
        "experiments/dist_parallel/parallel_config_rows.tex",
        "--out-latex",
    ),
    out_plot: str = typer.Option("figures/parallel_config_speedup.png", "--out-plot"),
    out_meta: str = typer.Option("experiments/dist_parallel/parallel_config.toml", "--out-meta"),
    redraw_only: bool = typer.Option(
        False,
        "--redraw-only",
        help="Only redraw plot from existing csv without rerunning experiments.",
    ),
    redraw_from_csv: str = typer.Option(
        "",
        "--redraw-from-csv",
        help="Existing csv path used when --redraw-only is enabled. Default uses --out-csv.",
    ),
) -> None:
    csv_path = Path(out_csv)
    latex_path = Path(out_latex)
    plot_path = Path(out_plot)
    meta_path = Path(out_meta)

    if redraw_only:
        source_csv = Path(redraw_from_csv) if redraw_from_csv.strip() else csv_path
        logger.info(f"Redraw-only mode: source={source_csv}, out_plot={plot_path}")
        rows = _read_csv(source_csv)
        _write_plot(rows, plot_path)
        logger.info(f"Plot saved: {plot_path}")
        return

    ntasks_values = _parse_int_list(ntasks_list)
    cpus_values = _parse_int_list(cpus_per_task_list)
    specs = _build_specs(ntasks_values, cpus_values)

    logger.info("Start SuperLU_DIST parallel configuration benchmark")
    logger.info(
        f"n={n}, repeats={repeats}, warmup={warmup}, configs={len(specs)}, "
        f"ntasks={ntasks_values}, cpus={cpus_values}"
    )

    rng = np.random.default_rng(seed)
    _, eps_update = _build_eps_pair(n, eps_si, eps_sio2, rng)
    A = _build_fdfd_matrix(eps_update, wavelength, points_per_wavelength, npml)

    parsed_launcher = launcher.strip() or None
    parsed_library_path = library_path.strip() or None
    parsed_launcher_extra_args = tuple(x for x in launcher_extra_args.split() if x)
    serial_cold_t, serial_warm_t, serial_err = _measure_serial_factor_time(
        A,
        repeats=repeats,
        warmup=warmup,
    )
    if serial_err:
        logger.warning(f"[serial_baseline] failed: {serial_err}")
    else:
        logger.info(
            f"[serial_baseline] cold_full_factor={serial_cold_t:.3e} s, warm_refactorize={serial_warm_t:.3e} s"
        )

    superlu_profile = _profile_superlu_factor_breakdown(A, colperm=colperm)

    if superlu_profile.get("status") == "ok":
        logger.info(
            (
                f"[superlu_profile] colperm={superlu_profile['colperm']}, "
                f"perm_ms={superlu_profile['phase_colperm_ms']:.3e}, "
                f"etree_ms={superlu_profile['phase_etree_ms']:.3e}, "
                f"numeric_ms={superlu_profile['phase_fact_ms']:.3e}"
            )
        )
    else:
        logger.warning(f"[superlu_profile] failed: {superlu_profile.get('error', '')}")

    rows_raw: list[ParallelRow] = []
    for spec in specs:
        logger.info(
            "Run config: ntasks={nt}, cpus-per-task={cp}, grid={nr}x{nc}".format(
                nt=spec.ntasks,
                cp=spec.cpus_per_task,
                nr=spec.nrow,
                nc=spec.ncol,
            )
        )
        cold_factor_t, warm_refactor_t, err = _measure_factor_time(
            A,
            spec,
            repeats=repeats,
            warmup=warmup,
            rowperm=rowperm,
            colperm=colperm,
            int64=int64,
            algo3d=algo3d,
            verbosity=verbosity,
            launcher=parsed_launcher,
            launcher_extra_args=parsed_launcher_extra_args,
            wait_timeout_sec=wait_timeout_sec,
            library_path=parsed_library_path,
        )

        status = "ok" if not err else "failed"
        row = ParallelRow(
            ntasks=spec.ntasks,
            cpus_per_task=spec.cpus_per_task,
            nrow=spec.nrow,
            ncol=spec.ncol,
            grid=f"{spec.nrow}x{spec.ncol}",
            factor_time_s=warm_refactor_t,
            speedup=float("nan"),
            status=status,
            error=err,
            cold_full_factor_s=cold_factor_t,
            warm_refactorize_s=warm_refactor_t,
        )
        rows_raw.append(row)

        if status == "ok":
            logger.info(
                f"cold_full_factor={cold_factor_t:.6e} s, warm_refactorize={warm_refactor_t:.6e} s"
            )
        else:
            logger.warning(f"failed: {err}")

    rows = _compute_speedups(rows_raw, baseline=serial_warm_t)
    _attach_superlu_profile(rows, superlu_profile)
    _write_csv(rows, csv_path)
    _write_latex_rows(rows, latex_path)
    _write_meta(
        rows,
        meta_path,
        n=n,
        repeats=repeats,
        warmup=warmup,
        superlu_profile=superlu_profile,
        serial_baseline={
            "cold_full_factor_s": serial_cold_t,
            "warm_refactorize_s": serial_warm_t,
            "status": "ok" if not serial_err else "failed",
            "error": serial_err,
        },
    )

    try:
        _write_plot(rows, plot_path)
        logger.info(f"Plot saved: {plot_path}")
    except Exception as exc:
        logger.warning(f"plot skipped: {type(exc).__name__}: {exc}")

    logger.info(f"CSV saved: {csv_path}")
    logger.info(f"LaTeX rows saved: {latex_path}")
    logger.info(f"Meta saved: {meta_path}")


if __name__ == "__main__":
    app()
