import numpy as np
import scipy.sparse as sp
import fdfd_solver.s_matrix as sm


# 导数矩阵
class DxyMatrix:
    dxf: sp.spmatrix
    dxb: sp.spmatrix
    dyf: sp.spmatrix
    dyb: sp.spmatrix

    def __init__(
        self, omega: float, dL: float, nxy: tuple[int, int], npml: tuple[int, int]
    ):
        self.dxf, self.dxb, self.dyf, self.dyb = build_derivative_matrix(
            omega, dL, nxy, npml
        )


def build_derivative_matrix(
    omega: float, dL: float, nxy: tuple[int, int], npml: tuple[int, int]
) -> tuple[sp.spmatrix, sp.spmatrix, sp.spmatrix, sp.spmatrix]:
    dxf, dxb, dyf, dyb = build_raw_derivative_matrix(dL, nxy)
    sxf, sxb, syf, syb = build_pml_S_matrix(omega, dL, nxy, npml)

    dxf = sxf * dxf
    dxb = sxb * dxb
    dyf = syf * dyf
    dyb = syb * dyb

    return dxf, dxb, dyf, dyb


def build_raw_derivative_matrix(
    dL: float, nxy: tuple[int, int]
) -> tuple[sp.spmatrix, sp.spmatrix, sp.spmatrix, sp.spmatrix]:
    nx, ny = nxy

    dxf = sp.diags([-1, 1, 1], [0, 1, -nx + 1], shape=(nx, nx), dtype=np.complex128)
    dxf = 1 / dL * sp.kron(dxf, sp.eye(ny))

    dxb = sp.diags([1, -1, -1], [0, -1, nx - 1], shape=(nx, nx), dtype=np.complex128)
    dxb = 1 / dL * sp.kron(dxb, sp.eye(ny))

    dyf = sp.diags([-1, 1, 1], [0, 1, -ny + 1], shape=(ny, ny), dtype=np.complex128)
    dyf = 1 / dL * sp.kron(sp.eye(nx), dyf)

    dyb = sp.diags([1, -1, -1], [0, -1, ny - 1], shape=(ny, ny), dtype=np.complex128)
    dyb = 1 / dL * sp.kron(sp.eye(nx), dyb)

    return dxf, dxb, dyf, dyb


def build_pml_S_matrix(
    omega: float, dL: float, nxy: tuple[int, int], npml: tuple[int, int]
) -> tuple[sp.spmatrix, sp.spmatrix, sp.spmatrix, sp.spmatrix]:
    sxf, sxb, syf, syb = sm.create_S_matrices(omega, nxy, npml, dL)
    return sxf, sxb, syf, syb


if __name__ == "__main__":
    omega = 2 * np.pi * 200e12
    dL = 25e-9
    nxy = (200, 80)
    npml = (20, 20)
    dxf, dxb, dyf, dyb = build_derivative_matrix(omega, dL, nxy, npml)
    print(dxf)
    print(dxb)
    print(dyf)
    print(dyb)
