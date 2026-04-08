from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import scipy.sparse as sp

_HAS_NANOBIND_SUPERLU = False
_IMPORT_ERROR: Exception | None = None

try:
    _nb = importlib.import_module("._superlu_nb", __package__)
    _INDEX_SIZE_BYTES = _nb.index_size_bytes
    _spfactorize_nb = _nb.spfactorize
    _spanalyze_nb = _nb.spanalyze
    _sprefactorize_nb = _nb.sprefactorize
    _spsolve_nb = _nb.spsolve
    _profile_symbolic_nb = getattr(_nb, "profile_symbolic", None)
    _profile_phases_nb = getattr(_nb, "profile_phases", None)

    _HAS_NANOBIND_SUPERLU = True
except Exception as exc:  # pragma: no cover - import behavior depends on local build
    _IMPORT_ERROR = exc
    _INDEX_SIZE_BYTES = 4


def is_available() -> bool:
    return _HAS_NANOBIND_SUPERLU


def _require_backend() -> None:
    if _HAS_NANOBIND_SUPERLU:
        return

    detail = f"{type(_IMPORT_ERROR).__name__}: {_IMPORT_ERROR}" if _IMPORT_ERROR else "unknown import error"
    raise RuntimeError(
        "spsolver nanobind backend is unavailable. "
        "Build extension at spsolver/nanobind first. "
        f"Detail: {detail}"
    )


def _index_dtype() -> Any:
    return np.int64 if int(_INDEX_SIZE_BYTES) == 8 else np.int32


def spfactorize(A: sp.spmatrix):
    _require_backend()

    if not sp.isspmatrix(A):
        raise TypeError("spfactorize expects a scipy.sparse matrix")

    if A.shape[0] != A.shape[1]:
        raise ValueError(f"matrix must be square, got {A.shape}")

    csc = A if sp.isspmatrix_csc(A) else A.tocsc()
    indptr = np.asarray(csc.indptr, dtype=_index_dtype(), order="C")
    indices = np.asarray(csc.indices, dtype=_index_dtype(), order="C")
    data = np.asarray(csc.data, dtype=np.complex128, order="C")

    return _spfactorize_nb(indptr, indices, data, int(csc.shape[0]))


def spanalyze(A: sp.spmatrix):
    _require_backend()

    if not sp.isspmatrix(A):
        raise TypeError("spanalyze expects a scipy.sparse matrix")
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"matrix must be square, got {A.shape}")

    csc = A if sp.isspmatrix_csc(A) else A.tocsc()
    indptr = np.asarray(csc.indptr, dtype=_index_dtype(), order="C")
    indices = np.asarray(csc.indices, dtype=_index_dtype(), order="C")
    return _spanalyze_nb(indptr, indices, int(csc.shape[0]))


def sprefactorize(factor, nzval: np.ndarray) -> None:
    _require_backend()
    data = np.asarray(nzval, dtype=np.complex128, order="C")
    if data.ndim != 1:
        raise ValueError(f"nzval must be 1D, got ndim={data.ndim}")
    _sprefactorize_nb(factor, data)


def spsolve(
    factor,
    b: np.ndarray,
    overwrite_b: bool = False,
    transpose: bool = False,
) -> np.ndarray:
    _require_backend()

    rhs = np.asarray(b, dtype=np.complex128, order="C")
    # nanobind overloads currently require writable arrays for rhs.
    if not rhs.flags.writeable:
        rhs = np.array(rhs, dtype=np.complex128, order="C", copy=True)

    if rhs.ndim not in (1, 2):
        raise ValueError(f"rhs must be 1D or 2D, got ndim={rhs.ndim}")

    return np.asarray(
        _spsolve_nb(
            factor,
            rhs,
            overwrite_b=overwrite_b,
            transpose=transpose,
        )
    )


def profile_symbolic(A: sp.spmatrix, colperm: int = 3) -> dict[str, float]:
    _require_backend()
    if _profile_symbolic_nb is None:
        raise RuntimeError(
            "profile_symbolic is unavailable in current _superlu_nb build. "
            "Please rebuild spsolver/nanobind."
        )

    if not sp.isspmatrix(A):
        raise TypeError("profile_symbolic expects a scipy.sparse matrix")
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"matrix must be square, got {A.shape}")

    csc = A if sp.isspmatrix_csc(A) else A.tocsc()
    indptr = np.asarray(csc.indptr, dtype=_index_dtype(), order="C")
    indices = np.asarray(csc.indices, dtype=_index_dtype(), order="C")

    raw = _profile_symbolic_nb(indptr, indices, int(csc.shape[0]), int(colperm))
    return {
        "t_perm_ms": float(raw["t_perm_ms"]),
        "t_symb_ms": float(raw["t_symb_ms"]),
        "nnz": float(raw["nnz"]),
    }


def profile_phases(A: sp.spmatrix, colperm: int = 3) -> dict[str, Any]:
    _require_backend()
    if _profile_phases_nb is None:
        raise RuntimeError(
            "profile_phases is unavailable in current _superlu_nb build. "
            "Please rebuild spsolver/nanobind."
        )

    if not sp.isspmatrix(A):
        raise TypeError("profile_phases expects a scipy.sparse matrix")
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"matrix must be square, got {A.shape}")

    csc = A if sp.isspmatrix_csc(A) else A.tocsc()
    indptr = np.asarray(csc.indptr, dtype=_index_dtype(), order="C")
    indices = np.asarray(csc.indices, dtype=_index_dtype(), order="C")
    data = np.asarray(csc.data, dtype=np.complex128, order="C")

    raw = _profile_phases_nb(indptr, indices, data, int(csc.shape[0]), int(colperm))
    phase_ms_raw = raw["phase_ms"]
    phase_ops_raw = raw["phase_ops"]

    phase_ms = {str(k): float(phase_ms_raw[k]) for k in phase_ms_raw}
    phase_ops = {str(k): float(phase_ops_raw[k]) for k in phase_ops_raw}

    return {
        "phase_ms": phase_ms,
        "phase_ops": phase_ops,
        "t_perm_ms": float(raw["t_perm_ms"]),
        "t_symb_ms": float(raw["t_symb_ms"]),
        "t_num_ms": float(raw["t_num_ms"]),
        "t_solve_ms": float(raw["t_solve_ms"]),
        "t_total_ms": float(raw["t_total_ms"]),
        "nnz": float(raw["nnz"]),
    }
