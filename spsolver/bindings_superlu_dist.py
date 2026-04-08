from __future__ import annotations

import atexit
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class DistSolveConfig:
    nrow: int = 2
    ncol: int = 1
    rowperm: int = 0
    colperm: int = 0
    int64: int = 1
    algo3d: int = 0
    verbosity: bool = False
    library_path: Optional[str] = None
    launcher: Optional[str] = None
    launcher_extra_args: tuple[str, ...] = ()
    wait_timeout_sec: float = 600.0


def _has_superlu_python_lib(path: Path) -> bool:
    if not path.is_dir():
        return False
    patterns = (
        "libsuperlu_dist_python.so*",
        "libsuperlu_dist_python.dylib*",
    )
    for pattern in patterns:
        if any(path.glob(pattern)):
            return True
    return False


def _discover_superlu_python_lib_path(config: DistSolveConfig) -> Optional[str]:
    if config.library_path:
        return str(config.library_path)

    env_path = os.getenv("SUPERLU_PYTHON_LIB_PATH")
    if env_path:
        return env_path

    spsolver_root = Path(__file__).resolve().parent
    superlu_dist_root = spsolver_root / "superlu_dist"

    candidates: list[Path] = [
        superlu_dist_root / "build" / "PYTHON",
        superlu_dist_root / "build",
    ]
    for build_dir in sorted(superlu_dist_root.glob("build*")):
        candidates.append(build_dir / "PYTHON")
        candidates.append(build_dir)

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if _has_superlu_python_lib(candidate):
            return str(candidate)

    return None


def _worker_script_path() -> Path:
    return Path(__file__).resolve().parent / "superlu_dist" / "PYTHON" / "pddrive_worker.py"


def _resolve_launcher(config: DistSolveConfig) -> str:
    if config.launcher:
        return config.launcher
    # Prefer mpirun by default: many Slurm clusters disable/limit srun MPI plugins
    # for nested steps, while mpirun remains available inside allocations.
    if shutil.which("mpirun") is not None:
        return "mpirun"
    if os.getenv("SLURM_JOB_ID") and shutil.which("srun") is not None:
        return "srun"
    return "mpirun"


def _build_launcher_prefix(launcher: str, nproc: int, extra_args: tuple[str, ...]) -> list[str]:
    if launcher == "srun":
        slurm_job_id = os.getenv("SLURM_JOB_ID")
        if not slurm_job_id:
            raise RuntimeError(
                "launcher='srun' requires a Slurm allocation (SLURM_JOB_ID not found). "
                "Run via sbatch/salloc, or set simulation.superlu_dist.launcher='mpirun'."
            )

        slurm_ntasks = os.getenv("SLURM_NTASKS")
        if slurm_ntasks is not None:
            try:
                if int(slurm_ntasks) < nproc:
                    raise RuntimeError(
                        "requested superlu_dist processes exceed Slurm allocation: "
                        f"nproc={nproc}, SLURM_NTASKS={slurm_ntasks}"
                    )
            except ValueError:
                pass
        return ["srun", "-n", str(nproc), *extra_args]
    return [launcher, "-n", str(nproc), *extra_args]


def is_available(config: Optional[DistSolveConfig] = None) -> bool:
    if not _worker_script_path().exists():
        return False
    cfg = config or DistSolveConfig()
    launcher = _resolve_launcher(cfg)
    return shutil.which(launcher) is not None


class _DistWorkerSession:
    def __init__(self, config: DistSolveConfig):
        launcher = _resolve_launcher(config)
        if not is_available(config):
            raise RuntimeError(
                "superlu_dist worker is unavailable. Ensure launcher exists "
                f"(launcher={launcher}) and "
                "spsolver/superlu_dist/PYTHON/pddrive_worker.py is present."
            )

        nproc = int(config.nrow) * int(config.ncol)
        if nproc <= 0:
            raise ValueError(f"invalid process grid: nrow={config.nrow}, ncol={config.ncol}")

        self._config = config
        self._launcher = launcher
        self._tmpdir = Path(tempfile.mkdtemp(prefix="superlu_dist_"))
        self._control_file = self._tmpdir / "control.txt"
        self._data_file = self._tmpdir / "data.bin"
        self._result_file = self._tmpdir / "result.bin"
        self._stdout_log = self._tmpdir / "worker.stdout.log"
        self._stderr_log = self._tmpdir / "worker.stderr.log"

        env = os.environ.copy()
        env["CONTROL_FILE"] = str(self._control_file)
        env["DATA_FILE"] = str(self._data_file)
        env["RESULT_FILE"] = str(self._result_file)
        env["PYTHONUNBUFFERED"] = "1"
        self._library_path = _discover_superlu_python_lib_path(config)
        if self._library_path:
            env["SUPERLU_PYTHON_LIB_PATH"] = self._library_path

        cmd = _build_launcher_prefix(
            self._launcher,
            nproc,
            tuple(config.launcher_extra_args),
        ) + [
            sys.executable,
            "-u",
            str(_worker_script_path()),
            "-r",
            str(config.nrow),
            "-c",
            str(config.ncol),
            "-p",
            str(config.rowperm),
            "-q",
            str(config.colperm),
            "-s",
            "0",
            "-i",
            "0",
            "-m",
            "0",
            "-n",
            "1",
            "-t",
            "0",
        ]

        self._cmd = cmd
        self._stdout_fp = self._stdout_log.open("w", encoding="utf-8")
        self._stderr_fp = self._stderr_log.open("w", encoding="utf-8")

        self._proc = subprocess.Popen(
            cmd,
            cwd=str(self._tmpdir),
            env=env,
            stdout=self._stdout_fp,
            stderr=self._stderr_fp,
            text=True,
        )
        time.sleep(0.05)
        if self._proc.poll() is not None:
            raise RuntimeError(
                "superlu_dist worker exited during startup. "
                f"details: {self._diagnostic_snapshot()}"
            )

    def _read_log_tail(self, path: Path, max_bytes: int = 16384) -> str:
        if not path.exists():
            return ""
        try:
            data = path.read_bytes()
            if len(data) > max_bytes:
                data = data[-max_bytes:]
            return data.decode("utf-8", errors="replace").strip()
        except Exception:
            return ""

    def _diagnostic_snapshot(self) -> str:
        control_flag = ""
        if self._control_file.exists():
            try:
                control_flag = self._control_file.read_text(encoding="utf-8").strip()
            except Exception:
                control_flag = "<unreadable>"

        stderr_tail = self._read_log_tail(self._stderr_log)
        stdout_tail = self._read_log_tail(self._stdout_log)
        return (
            f"launcher={self._launcher}, cmd={' '.join(self._cmd)}, tmpdir={self._tmpdir}, "
            f"library_path={self._library_path!r}, "
            f"control_flag={control_flag!r}, stderr_tail={stderr_tail!r}, stdout_tail={stdout_tail!r}"
        )

    def _timeout_hint(self, expected: str) -> str:
        try:
            control_flag = self._control_file.read_text(encoding="utf-8").strip()
        except Exception:
            control_flag = ""

        if expected == "done" and control_flag == "init":
            if self._launcher == "srun":
                return (
                    " hint: worker appears stuck during MPI/pdbridge init. "
                    "If not in a Slurm allocation, do not use launcher='srun'. "
                    "If in Slurm, try setting simulation.superlu_dist.launcher_extra_args=['--mpi=pmix'] "
                    "or switch launcher='mpirun'."
                )
            return (
                " hint: worker appears stuck during MPI/pdbridge init. "
                "Verify mpi4py was built against the same MPI as launcher and "
                "SUPERLU_PYTHON_LIB_PATH points to directory containing libsuperlu_dist_python.so."
            )

        if expected == "done" and control_flag == "factor":
            return (
                " hint: worker reached factor stage; factorization may be very slow for this matrix/pivot settings. "
                "Try increasing simulation.superlu_dist.wait_timeout_sec (e.g. 300-1800). "
                "If using rowperm=0 and colperm=0, try rowperm=1 and colperm=2 to reduce fill-in as a diagnostic."
            )
        return ""

    def _wait_flag(self, expected: str, timeout_sec: float = 600.0) -> None:
        start = time.time()
        while True:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    "superlu_dist worker exited unexpectedly. "
                    f"details: {self._diagnostic_snapshot()}"
                )
            if self._control_file.exists():
                try:
                    flag = self._control_file.read_text(encoding="utf-8").strip()
                    if flag == expected:
                        return
                except Exception:
                    pass

            if timeout_sec > 0 and (time.time() - start) > timeout_sec:
                raise TimeoutError(
                    f"timed out waiting for superlu_dist worker flag '{expected}'. "
                    f"details: {self._diagnostic_snapshot()}"
                    f"{self._timeout_hint(expected)}"
                )
            time.sleep(0.002)

    def _set_flag(self, flag: str) -> None:
        self._control_file.write_text(flag, encoding="utf-8")

    def factorize(self, A: sp.csc_matrix) -> None:
        payload = (A, int(self._config.int64), int(self._config.algo3d))
        with self._data_file.open("wb") as f:
            pickle.dump(payload, f)

        self._set_flag("init")
        self._wait_flag("done", timeout_sec=float(self._config.wait_timeout_sec))

        self._set_flag("factor")
        self._wait_flag("done", timeout_sec=float(self._config.wait_timeout_sec))

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        nrhs = int(rhs.shape[-1])
        with self._data_file.open("wb") as f:
            pickle.dump((rhs, nrhs), f)

        self._set_flag("solve")
        self._wait_flag("done", timeout_sec=float(self._config.wait_timeout_sec))

        with self._result_file.open("rb") as f:
            out = pickle.load(f)
        return np.asarray(out, dtype=np.float64)

    def free_lu(self) -> None:
        self._set_flag("free")
        self._wait_flag("clean", timeout_sec=float(self._config.wait_timeout_sec))

    def close(self) -> None:
        if self._proc.poll() is None:
            try:
                self._set_flag("terminate")
            except Exception:
                pass
            try:
                self._proc.wait(timeout=10)
            except Exception:
                self._proc.kill()

        try:
            self._stdout_fp.close()
        except Exception:
            pass
        try:
            self._stderr_fp.close()
        except Exception:
            pass

        try:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        except Exception:
            pass


_SESSIONS: dict[DistSolveConfig, _DistWorkerSession] = {}


def _get_session(config: DistSolveConfig) -> _DistWorkerSession:
    session = _SESSIONS.get(config)
    if session is None:
        session = _DistWorkerSession(config)
        _SESSIONS[config] = session
    return session


def _cleanup_sessions() -> None:
    for session in list(_SESSIONS.values()):
        session.close()
    _SESSIONS.clear()


atexit.register(_cleanup_sessions)


def _complex_to_real_block_csc(A: sp.csc_matrix) -> sp.csc_matrix:
    Ar = A.real.tocsc()
    Ai = A.imag.tocsc()
    top = sp.hstack([Ar, -Ai], format="csc")
    bottom = sp.hstack([Ai, Ar], format="csc")
    return sp.vstack([top, bottom], format="csc")


class DistFactor:
    def __init__(self, A: sp.csc_matrix, config: DistSolveConfig):
        self._config = config
        self._session = _get_session(config)
        self._n = int(A.shape[0])
        self._indptr = np.asarray(A.indptr).copy()
        self._indices = np.asarray(A.indices).copy()
        self.refactorize(A)

    def refactorize(self, A: sp.csc_matrix) -> None:
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"matrix must be square, got {A.shape}")
        if int(A.shape[0]) != self._n:
            raise ValueError(f"shape changed: expected {self._n}, got {A.shape[0]}")

        indptr = np.asarray(A.indptr)
        indices = np.asarray(A.indices)
        if not np.array_equal(indptr, self._indptr) or not np.array_equal(indices, self._indices):
            raise ValueError("superlu_dist factor pattern changed unexpectedly")

        A_real = _complex_to_real_block_csc(A)
        self._session.factorize(A_real)

    def solve_inplace(self, rhs: np.ndarray) -> None:
        if rhs.ndim == 1:
            if rhs.shape[0] != self._n:
                raise ValueError(f"rhs length mismatch: expected {self._n}, got {rhs.shape[0]}")
            real_rhs = np.empty((2 * self._n, 1), dtype=np.float64)
            real_rhs[: self._n, 0] = rhs.real
            real_rhs[self._n :, 0] = rhs.imag
            real_out = self._session.solve(real_rhs)
            rhs[:] = real_out[: self._n, 0] + 1j * real_out[self._n :, 0]
            return

        if rhs.ndim != 2:
            raise ValueError(f"rhs must be 1D or 2D, got ndim={rhs.ndim}")
        if rhs.shape[1] != self._n:
            raise ValueError(f"rhs second dimension mismatch: expected {self._n}, got {rhs.shape[1]}")

        batch = rhs.shape[0]
        for i in range(batch):
            self.solve_inplace(rhs[i])


def spfactorize(A: sp.spmatrix, config: DistSolveConfig | None = None) -> DistFactor:
    if not sp.isspmatrix(A):
        raise TypeError("spfactorize (dist) expects a scipy.sparse matrix")

    csc = A if sp.isspmatrix_csc(A) else A.tocsc()
    cfg = config or DistSolveConfig()
    return DistFactor(csc, cfg)


def spsolve(
    factor: DistFactor,
    b: np.ndarray,
    overwrite_b: bool = False,
    transpose: bool = False,
) -> np.ndarray:
    if transpose:
        raise ValueError(
            "superlu_dist wrapper does not support transpose solve directly; "
            "factorize A.T separately and solve with transpose=False."
        )

    rhs = np.asarray(b, dtype=np.complex128, order="C")
    if not rhs.flags.writeable:
        rhs = np.array(rhs, dtype=np.complex128, order="C", copy=True)

    out = rhs if overwrite_b else np.array(rhs, dtype=np.complex128, order="C", copy=True)
    factor.solve_inplace(out)
    return out
