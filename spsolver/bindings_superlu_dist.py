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
    library_path: str | None = None


def _worker_script_path() -> Path:
    return Path(__file__).resolve().parent / "superlu_dist" / "PYTHON" / "pddrive_worker.py"


def is_available() -> bool:
    return (
        shutil.which("mpirun") is not None
        and _worker_script_path().exists()
    )


class _DistWorkerSession:
    def __init__(self, config: DistSolveConfig):
        if not is_available():
            raise RuntimeError(
                "superlu_dist worker is unavailable. Ensure mpirun exists and "
                "spsolver/superlu_dist/PYTHON/pddrive_worker.py is present."
            )

        nproc = int(config.nrow) * int(config.ncol)
        if nproc <= 0:
            raise ValueError(f"invalid process grid: nrow={config.nrow}, ncol={config.ncol}")

        self._config = config
        self._tmpdir = Path(tempfile.mkdtemp(prefix="superlu_dist_"))
        self._control_file = self._tmpdir / "control.txt"
        self._data_file = self._tmpdir / "data.bin"
        self._result_file = self._tmpdir / "result.bin"

        env = os.environ.copy()
        env["CONTROL_FILE"] = str(self._control_file)
        env["DATA_FILE"] = str(self._data_file)
        env["RESULT_FILE"] = str(self._result_file)
        if config.library_path:
            env["SUPERLU_PYTHON_LIB_PATH"] = str(config.library_path)

        cmd = [
            "mpirun",
            "-n",
            str(nproc),
            sys.executable,
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

        self._proc = subprocess.Popen(
            cmd,
            cwd=str(self._tmpdir),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

    def _read_stderr(self) -> str:
        if self._proc.stderr is None:
            return ""
        try:
            return self._proc.stderr.read()
        except Exception:
            return ""

    def _wait_flag(self, expected: str, timeout_sec: float = 600.0) -> None:
        start = time.time()
        while True:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    "superlu_dist worker exited unexpectedly. "
                    f"stderr: {self._read_stderr().strip()}"
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
                    f"timed out waiting for superlu_dist worker flag '{expected}'"
                )
            time.sleep(0.001)

    def _set_flag(self, flag: str) -> None:
        self._control_file.write_text(flag, encoding="utf-8")

    def factorize(self, A: sp.csc_matrix) -> None:
        payload = (A, int(self._config.int64), int(self._config.algo3d))
        with self._data_file.open("wb") as f:
            pickle.dump(payload, f)

        self._set_flag("init")
        self._wait_flag("done")

        self._set_flag("factor")
        self._wait_flag("done")

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        nrhs = int(rhs.shape[-1])
        with self._data_file.open("wb") as f:
            pickle.dump((rhs, nrhs), f)

        self._set_flag("solve")
        self._wait_flag("done")

        with self._result_file.open("rb") as f:
            out = pickle.load(f)
        return np.asarray(out, dtype=np.float64)

    def free_lu(self) -> None:
        self._set_flag("free")
        self._wait_flag("clean")

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
