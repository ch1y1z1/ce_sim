import numpy as np
import jax.numpy as jnp
from fdfd_solver.dxy_matrix import DxyMatrix
from fdfd_solver.constants import MU_0, EPSILON_0


# 这个函数会生成 A 矩阵：
# A = (\nabla^2 + k^2)
# 其中 \nabla^2 是导数矩阵，k^2 是介电常数矩阵
# \nabla^2 = \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2}
# k^2 = k0^2 * eps_r
def make_A(
    dxy: DxyMatrix, eps_r: jnp.ndarray, nx: int, ny: int, omega: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    nabla2 = (dxy.dxf * dxy.dxb + dxy.dyf * dxy.dyb) * (-1 / MU_0)
    # k2 = sp.spdiags(eps_r.flatten(), 0, nx * ny, nx * ny) * (omega**2) * (-EPSILON_0)
    k2 = eps_r * (omega**2) * (-EPSILON_0)
    
    # 为了使自动微分工作，我们需要使用 indices 和 data 来构建稀疏矩阵
    # indices 是非零元素的索引，data 是非零元素的值
    # 这样 jax 可以追踪这些值的变化
    # 这里使用 COO 格式
    N = nx * ny
    k2_indices = np.vstack((np.arange(N), np.arange(N)))
    k2_data = k2.flatten()
    
    nabla2_coo = nabla2.tocoo()
    nabla2_indices = np.vstack((nabla2_coo.row, nabla2_coo.col))
    nabla2_data = nabla2_coo.data
    
    return jnp.hstack((nabla2_data, k2_data)), np.hstack((nabla2_indices, k2_indices))


# b = f(x, y)
# source: Jz(x, y)
def make_b(src: jnp.ndarray, omega: float) -> jnp.ndarray:
    return 1j * omega * src.flatten()
