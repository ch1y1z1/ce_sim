from fdfd_solver.dxy_matrix import DxyMatrix
from fdfd_solver.make_A_b import make_A, make_b
from fdfd_solver.solver import solve, solve_batch


def fdfd_solve(omega, dL, nx, ny, npml, epsr, src):
    dxy = DxyMatrix(omega, dL, (nx, ny), npml)
    A = make_A(dxy, epsr, nx, ny, omega)
    b = make_b(src, omega)
    Ez = solve(A, b)
    return Ez


def fdfd_solve_batch(omega, dL, nx, ny, npml, epsr, srcs):
    dxy = DxyMatrix(omega, dL, (nx, ny), npml)
    A = make_A(dxy, epsr, nx, ny, omega)
    bs = [make_b(src, omega) for src in srcs]
    Ezs = solve_batch(A, bs)
    return Ezs
