from __future__ import annotations

import csv
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

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
from spsolver import spsolve as spsolve_serial
from spsolver.bindings_superlu_dist import (
    DistSolveConfig,
    spfactorize as spfactorize_dist,
    spsolve as spsolve_dist,
)

C0 = 1.0 / math.sqrt(EPSILON_0 * MU_0)

app = typer.Typer(add_completion=False)


@dataclass
class BackendResult:
    backend: str
    factor_cold_s: float
    solve_cold_s: float
    factor_warm_s: float
    solve_warm_s: float
    startup_overhead_est_s: float
    solve_overhead_est_s: float
    total_overhead_est_s: float
    status: str
    error: str
    cold_full_factor_s: float = float("nan")
    warm_refactorize_s: float = float("nan")
    superlu_profile_colperm: int = -1
    superlu_t_perm_ms: float = float("nan")
    superlu_t_symb_ms: float = float("nan")
    superlu_t_num_ms: float = float("nan")
    superlu_phase_colperm_ms: float = float("nan")
    superlu_phase_etree_ms: float = float("nan")
    superlu_phase_fact_ms: float = float("nan")
    superlu_profile_status: str = "na"
    superlu_profile_error: str = ""


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _mean(vals: Iterable[float]) -> float:
    arr = np.asarray(list(vals), dtype=np.float64)
    return float(arr.mean()) if arr.size else float("nan")


def _set_single_thread_env() -> dict[str, Optional[str]]:
    keys = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]
    old = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ[k] = "1"
    return old


def _restore_env(old_env: dict[str, Optional[str]]) -> None:
    for k, v in old_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _cleanup_dist_sessions() -> None:
    cleanup_fn = getattr(dist_mod, "_cleanup_sessions", None)
    if callable(cleanup_fn):
        cleanup_fn()


def _profile_superlu_factor_breakdown(A: sp.csc_matrix, *, colperm: int) -> dict[str, Any]:
    try:
        prof = profile_phases_serial(A, colperm=colperm)
        phase_ms = prof.get("phase_ms", {})
        colperm_phase = float(phase_ms.get("COLPERM", float("nan")))
        etree_phase = float(phase_ms.get("ETREE", float("nan")))
        fact_phase = float(phase_ms.get("FACT", float("nan")))
        return {
            "colperm": int(colperm),
            "t_perm_ms": float(prof.get("t_perm_ms", float("nan"))),
            "t_symb_ms": float(prof.get("t_symb_ms", float("nan"))),
            "t_num_ms": float(prof.get("t_num_ms", float("nan"))),
            "phase_colperm_ms": colperm_phase,
            "phase_etree_ms": etree_phase,
            "phase_fact_ms": fact_phase,
            "status": "ok",
            "error": "",
        }
    except Exception as exc:
        return {
            "colperm": int(colperm),
            "t_perm_ms": float("nan"),
            "t_symb_ms": float("nan"),
            "t_num_ms": float("nan"),
            "phase_colperm_ms": float("nan"),
            "phase_etree_ms": float("nan"),
            "phase_fact_ms": float("nan"),
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _attach_superlu_profile(rows: list[BackendResult], profile: dict[str, Any]) -> None:
    for r in rows:
        r.superlu_profile_colperm = int(profile.get("colperm", -1))
        r.superlu_t_perm_ms = float(profile.get("t_perm_ms", float("nan")))
        r.superlu_t_symb_ms = float(profile.get("t_symb_ms", float("nan")))
        r.superlu_t_num_ms = float(profile.get("t_num_ms", float("nan")))
        r.superlu_phase_colperm_ms = float(profile.get("phase_colperm_ms", float("nan")))
        r.superlu_phase_etree_ms = float(profile.get("phase_etree_ms", float("nan")))
        r.superlu_phase_fact_ms = float(profile.get("phase_fact_ms", float("nan")))
        r.superlu_profile_status = str(profile.get("status", "na"))
        r.superlu_profile_error = str(profile.get("error", ""))


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


def _sample_rhs(n_dof: int, rng: np.random.Generator) -> np.ndarray:
    real = rng.standard_normal(n_dof)
    imag = rng.standard_normal(n_dof)
    return (real + 1j * imag).astype(np.complex128)


def _measure_serial(
    A: sp.csc_matrix,
    b: np.ndarray,
    *,
    repeats: int,
    warmup: int,
) -> BackendResult:
    refactor_warm_samples: list[float] = []
    solve_warm_samples: list[float] = []

    try:
        nzval = np.asarray(A.data, dtype=np.complex128, order="C")

        # Cold run
        t0 = time.perf_counter()
        factor = spanalyze_serial(A)
        sprefactorize_serial(factor, nzval)
        t1 = time.perf_counter()
        _ = spsolve_serial(factor, b, overwrite_b=False)
        t2 = time.perf_counter()

        factor_cold = t1 - t0
        solve_cold = t2 - t1

        # Warmup + steady-state using in-place numeric refactorization
        for i in range(warmup + repeats):
            t3 = time.perf_counter()
            sprefactorize_serial(factor, nzval)
            t4 = time.perf_counter()
            _ = spsolve_serial(factor, b, overwrite_b=False)
            t5 = time.perf_counter()

            if i >= warmup:
                refactor_warm_samples.append(t4 - t3)
                solve_warm_samples.append(t5 - t4)

        factor_warm = _mean(refactor_warm_samples)
        solve_warm = _mean(solve_warm_samples)

        return BackendResult(
            backend="superlu",
            factor_cold_s=factor_cold,
            solve_cold_s=solve_cold,
            factor_warm_s=factor_warm,
            solve_warm_s=solve_warm,
            cold_full_factor_s=factor_cold,
            warm_refactorize_s=factor_warm,
            startup_overhead_est_s=float("nan"),
            solve_overhead_est_s=float("nan"),
            total_overhead_est_s=float("nan"),
            status="ok",
            error="",
        )
    except Exception as exc:
        return BackendResult(
            backend="superlu",
            factor_cold_s=float("nan"),
            solve_cold_s=float("nan"),
            factor_warm_s=float("nan"),
            solve_warm_s=float("nan"),
            cold_full_factor_s=float("nan"),
            warm_refactorize_s=float("nan"),
            startup_overhead_est_s=float("nan"),
            solve_overhead_est_s=float("nan"),
            total_overhead_est_s=float("nan"),
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )


def _measure_dist(
    A: sp.csc_matrix,
    b: np.ndarray,
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
    native_complex: bool,
) -> BackendResult:
    factor_warm_samples: list[float] = []
    solve_warm_samples: list[float] = []

    try:
        _cleanup_dist_sessions()
        cfg = DistSolveConfig(
            nrow=1,
            ncol=1,
            rowperm=rowperm,
            colperm=colperm,
            int64=int64,
            algo3d=algo3d,
            verbosity=verbosity,
            library_path=library_path,
            launcher=launcher,
            launcher_extra_args=launcher_extra_args,
            wait_timeout_sec=wait_timeout_sec,
            native_complex=native_complex,
        )

        # Cold run includes worker startup + first factorization
        t0 = time.perf_counter()
        factor = spfactorize_dist(A, config=cfg)
        t1 = time.perf_counter()
        _ = spsolve_dist(factor, b, overwrite_b=False)
        t2 = time.perf_counter()

        factor_cold = t1 - t0
        solve_cold = t2 - t1

        # Warmup + steady-state using refactorize on same worker session
        for i in range(warmup + repeats):
            t3 = time.perf_counter()
            factor.refactorize(A)
            t4 = time.perf_counter()
            _ = spsolve_dist(factor, b, overwrite_b=False)
            t5 = time.perf_counter()

            if i >= warmup:
                factor_warm_samples.append(t4 - t3)
                solve_warm_samples.append(t5 - t4)

        factor_warm = _mean(factor_warm_samples)
        solve_warm = _mean(solve_warm_samples)

        startup_est = factor_cold - factor_warm if math.isfinite(factor_warm) else float("nan")
        solve_overhead_est = solve_cold - solve_warm if math.isfinite(solve_warm) else float("nan")
        total_overhead_est = (
            (factor_cold + solve_cold) - (factor_warm + solve_warm)
            if math.isfinite(factor_warm) and math.isfinite(solve_warm)
            else float("nan")
        )

        _cleanup_dist_sessions()
        return BackendResult(
            backend="superlu_dist_1x1",
            factor_cold_s=factor_cold,
            solve_cold_s=solve_cold,
            factor_warm_s=factor_warm,
            solve_warm_s=solve_warm,
            cold_full_factor_s=factor_cold,
            warm_refactorize_s=factor_warm,
            startup_overhead_est_s=startup_est,
            solve_overhead_est_s=solve_overhead_est,
            total_overhead_est_s=total_overhead_est,
            status="ok",
            error="",
        )
    except Exception as exc:
        _cleanup_dist_sessions()
        return BackendResult(
            backend="superlu_dist_1x1",
            factor_cold_s=float("nan"),
            solve_cold_s=float("nan"),
            factor_warm_s=float("nan"),
            solve_warm_s=float("nan"),
            cold_full_factor_s=float("nan"),
            warm_refactorize_s=float("nan"),
            startup_overhead_est_s=float("nan"),
            solve_overhead_est_s=float("nan"),
            total_overhead_est_s=float("nan"),
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )


def _write_csv(rows: list[BackendResult], path: Path) -> None:
    _ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "backend",
                "factor_cold_s",
                "solve_cold_s",
                "factor_warm_s",
                "solve_warm_s",
                "cold_full_factor_s",
                "warm_refactorize_s",
                "startup_overhead_est_s",
                "solve_overhead_est_s",
                "total_overhead_est_s",
                "superlu_profile_colperm",
                "superlu_t_perm_ms",
                "superlu_t_symb_ms",
                "superlu_t_num_ms",
                "superlu_phase_colperm_ms",
                "superlu_phase_etree_ms",
                "superlu_phase_fact_ms",
                "superlu_profile_status",
                "superlu_profile_error",
                "status",
                "error",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.backend,
                    f"{r.factor_cold_s:.9e}" if math.isfinite(r.factor_cold_s) else "nan",
                    f"{r.solve_cold_s:.9e}" if math.isfinite(r.solve_cold_s) else "nan",
                    f"{r.factor_warm_s:.9e}" if math.isfinite(r.factor_warm_s) else "nan",
                    f"{r.solve_warm_s:.9e}" if math.isfinite(r.solve_warm_s) else "nan",
                    f"{r.cold_full_factor_s:.9e}" if math.isfinite(r.cold_full_factor_s) else "nan",
                    f"{r.warm_refactorize_s:.9e}" if math.isfinite(r.warm_refactorize_s) else "nan",
                    f"{r.startup_overhead_est_s:.9e}" if math.isfinite(r.startup_overhead_est_s) else "nan",
                    f"{r.solve_overhead_est_s:.9e}" if math.isfinite(r.solve_overhead_est_s) else "nan",
                    f"{r.total_overhead_est_s:.9e}" if math.isfinite(r.total_overhead_est_s) else "nan",
                    str(r.superlu_profile_colperm) if r.superlu_profile_colperm >= 0 else "na",
                    f"{r.superlu_t_perm_ms:.9e}" if math.isfinite(r.superlu_t_perm_ms) else "nan",
                    f"{r.superlu_t_symb_ms:.9e}" if math.isfinite(r.superlu_t_symb_ms) else "nan",
                    f"{r.superlu_t_num_ms:.9e}" if math.isfinite(r.superlu_t_num_ms) else "nan",
                    f"{r.superlu_phase_colperm_ms:.9e}" if math.isfinite(r.superlu_phase_colperm_ms) else "nan",
                    f"{r.superlu_phase_etree_ms:.9e}" if math.isfinite(r.superlu_phase_etree_ms) else "nan",
                    f"{r.superlu_phase_fact_ms:.9e}" if math.isfinite(r.superlu_phase_fact_ms) else "nan",
                    r.superlu_profile_status,
                    r.superlu_profile_error,
                    r.status,
                    r.error,
                ]
            )


def _write_latex_rows(rows: list[BackendResult], path: Path) -> None:
    _ensure_parent(path)
    lines: list[str] = []
    for r in rows:
        if r.status != "ok":
            lines.append("{} & --- & --- & --- \\\\".format(r.backend))
            continue

        lines.append(
            "{name} & {fc:.3e} & {sc:.3e} & {ov:.3e} \\\\".format(
                name=r.backend,
                fc=r.factor_warm_s,
                sc=r.solve_warm_s,
                ov=r.total_overhead_est_s if math.isfinite(r.total_overhead_est_s) else 0.0,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_meta(
    rows: list[BackendResult],
    path: Path,
    *,
    n: int,
    repeats: int,
    warmup: int,
    native_complex: bool,
    superlu_profile: dict[str, Any],
) -> None:
    _ensure_parent(path)
    payload = {
        "matrix": {"n": n, "N": n * n},
        "measurement": {"repeats": repeats, "warmup": warmup},
        "dist_runtime": {"native_complex": bool(native_complex)},
        "superlu_profile": superlu_profile,
        "rows": [asdict(r) for r in rows],
    }
    path.write_text(toml.dumps(payload), encoding="utf-8")


@app.command()
def main(
    n: int = typer.Option(200, "--n", min=8, help="Grid size n; matrix dimension is N=n^2."),
    repeats: int = typer.Option(5, "--repeats", min=1),
    warmup: int = typer.Option(2, "--warmup", min=0),
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
    legacy_real_block: bool = typer.Option(
        False,
        "--legacy-real-block",
        help="Use legacy real 2N block fallback instead of native complex path.",
    ),
    seed: int = typer.Option(0, "--seed"),
    out_csv: str = typer.Option(
        "experiments/dist_parallel/serial_vs_dist_baseline.csv",
        "--out-csv",
    ),
    out_latex: str = typer.Option(
        "experiments/dist_parallel/serial_vs_dist_baseline_rows.tex",
        "--out-latex",
    ),
    out_meta: str = typer.Option(
        "experiments/dist_parallel/serial_vs_dist_baseline.toml",
        "--out-meta",
    ),
) -> None:
    csv_path = Path(out_csv)
    latex_path = Path(out_latex)
    meta_path = Path(out_meta)

    logger.info(
        f"Start serial vs dist(1x1) baseline benchmark: n={n}, repeats={repeats}, warmup={warmup}"
    )

    rng = np.random.default_rng(seed)
    _, eps_update = _build_eps_pair(n, eps_si, eps_sio2, rng)
    A = _build_fdfd_matrix(eps_update, wavelength, points_per_wavelength, npml)
    b = _sample_rhs(n * n, rng)

    parsed_launcher = launcher.strip() or None
    parsed_library_path = library_path.strip() or None
    parsed_launcher_extra_args = tuple(x for x in launcher_extra_args.split() if x)
    native_complex = not legacy_real_block
    logger.info(f"dist native_complex={native_complex}")

    old_env = _set_single_thread_env()
    try:
        superlu_profile = _profile_superlu_factor_breakdown(A, colperm=colperm)
        serial_result = _measure_serial(A, b, repeats=repeats, warmup=warmup)
        dist_result = _measure_dist(
            A,
            b,
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
            native_complex=native_complex,
        )
    finally:
        _restore_env(old_env)

    rows = [serial_result, dist_result]
    _attach_superlu_profile(rows, superlu_profile)
    _write_csv(rows, csv_path)
    _write_latex_rows(rows, latex_path)
    _write_meta(
        rows,
        meta_path,
        n=n,
        repeats=repeats,
        warmup=warmup,
        native_complex=native_complex,
        superlu_profile=superlu_profile,
    )

    if superlu_profile.get("status") == "ok":
        logger.info(
            (
                f"[superlu_profile] colperm={superlu_profile['colperm']}, "
                f"t_perm_ms={superlu_profile['t_perm_ms']:.3e}, "
                f"t_symb_ms={superlu_profile['t_symb_ms']:.3e}, "
                f"t_num_ms={superlu_profile['t_num_ms']:.3e}, "
                f"COLPERM_phase_ms={superlu_profile['phase_colperm_ms']:.3e}, "
                f"ETREE_phase_ms={superlu_profile['phase_etree_ms']:.3e}, "
                f"FACT_phase_ms={superlu_profile['phase_fact_ms']:.3e}"
            )
        )
    else:
        logger.warning(f"[superlu_profile] failed: {superlu_profile.get('error', '')}")

    for r in rows:
        if r.status == "ok":
            logger.info(
                (
                    f"[{r.backend}] cold(factor/solve)=({r.factor_cold_s:.3e}/{r.solve_cold_s:.3e}) s, "
                    f"warm(refactorize/solve)=({r.factor_warm_s:.3e}/{r.solve_warm_s:.3e}) s, "
                    f"overhead(total)={r.total_overhead_est_s:.3e} s"
                )
            )
        else:
            logger.warning(f"[{r.backend}] failed: {r.error}")

    logger.info(f"CSV saved: {csv_path}")
    logger.info(f"LaTeX rows saved: {latex_path}")
    logger.info(f"Meta saved: {meta_path}")


if __name__ == "__main__":
    app()
