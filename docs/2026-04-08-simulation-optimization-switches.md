# Simulation 优化开关需求留痕（2026-04-08）

## 需求约束

根据当前项目决策，以下三种优化策略仅在 `fdfd_solver` 后端实现并生效：

- 符号分析复用（symbolic reuse）
- 批量 RHS 求解（batch solve）
- SuperLU_DIST 分布式求解（distributed solve）

当 `simulation.backend != "fdfd_solver"` 时，即使配置中显式开启优化开关，也应自动忽略并回退到后端默认路径。

## 本次落地范围（第一步）

本次先完成 `enable_batch` 开关：

- 配置键：`simulation.enable_batch`
- 生效条件：`simulation.backend == "fdfd_solver"`
- 关闭行为：逐 RHS 求解（不走 `solve_batch`）

## 本次落地范围（第二步）

完成 `enable_symbolic_reuse` 开关：

- 配置键：`simulation.enable_symbolic_reuse`
- 生效条件：`simulation.backend == "fdfd_solver"`
- 开启行为：使用 SuperLU `SamePattern_SameRowPerm` 复用
  - 列置换/消元树等符号分析数据
  - L/U 矩阵数据结构及内存空间

同时移除旧配置段 `simulation.fdfd_solver.single_solver/batch_factorize/batch_solver`，
`fdfd_solver` 后端统一使用自定义绑定的 SuperLU，不再使用 SciPy 求解实现。

已完成实现：

- `simulation_backend.py` 将 `simulation.enable_symbolic_reuse` 下发到 `FdfdSolverEzBackend`。
- `fdfd_solver/solver.py` 在 `make_solver_pair(enable_symbolic_reuse=...)` 中切换
  - 复用模式：`spanalyze + sprefactorize`
  - 非复用模式：`spfactorize`
- 复用模式增加稀疏模式一致性检查（`indptr/indices` 不一致时报错）。

## 问题修复留痕

在命令 `uv run train.py -o simulation.enable_batch=false` 的回归测试中，曾出现
`Autograd ArrayBox is not a valid JAX type`。

根因是：`fdfd_solver` 在关闭 batch 后走到了 `agjax` 包装路径，导致 `ArrayBox` 与 `jax.numpy` 的拼接流程不兼容。

修复方式：

- 当 `backend == "fdfd_solver"` 时，无论 batch 开关状态都走 `simulation` 原生求解路径。
- 开启 batch：调用 `solve_batch`。
- 关闭 batch：改为逐个 source 调 `simulation.solve(source)[2]`，不再经过 `agjax` 包装函数。

修复后，命令 `uv run train.py -o simulation.enable_batch=false -o train.num_epochs=1` 可正常完成至少 1 个 epoch。

在 symbolic reuse 实装过程中还修复了两类问题：

- `_superlu_nb` 导入失败：`undefined symbol: dmach`
  - 通过开启 SuperLU double 路径并重编解决。
- `spsolve` 调用签名不匹配（JAX 回调 RHS 为只读数组）
  - Python 适配层在 RHS 不可写时自动转为可写数组。

此外，为避免解释器退出阶段的 nanobind 泄漏告警，已在 `fdfd_solver/solver.py`
增加 `atexit` 缓存清理。

## 代码落点

- `layer.py`
  - 读取 `simulation.enable_batch`
  - 将其与后端类型联合判断，确保只在 `fdfd_solver` 后端启用

- `Configuration/*.toml`
  - 在注释中明确 `enable_batch` 仅对 `fdfd_solver` 生效

## 训练 JIT 尝试开关

为方便性能实验，新增训练开关：

- 配置键：`train.enable_jit`
- 行为：
  - 开启时优先使用 `nnx.jit(train_step)`
  - 若执行失败，自动记录日志并回退到非 JIT 路径继续训练

说明：该开关用于“尝试启用 JIT”，避免因局部算子不兼容导致训练直接中断。

验证命令（已通过）：

```bash
uv run train.py -o train.num_epochs=1 -o train.enable_jit=true
```

## 后续计划

- 第三步：增加 `enable_superlu_dist` 开关与分布式后端路由。

## 本次落地范围（第三步）

完成 `enable_superlu_dist` 开关：

- 配置键：`simulation.enable_superlu_dist`
- 生效条件：`simulation.backend == "fdfd_solver"`
- 兼容关系：
  - 与 `enable_batch` 兼容：batch 模式仍由 `solve_batch` 路径驱动
  - 与 `enable_symbolic_reuse` 兼容：复用开关仍可用（dist 模式下按稀疏模式复用会话）

新增 `simulation.superlu_dist` 配置段：

- `nrow = 2`
- `ncol = 1`
- `rowperm = 0`（NOROWPERM，单位置换）
- `colperm = 0`（NATURAL，单位置换）

实现要点：

- 在 `simulation_backend.py` 中读取并下发 `enable_superlu_dist` 与 `simulation.superlu_dist.*`。
- 在 `fdfd_solver/solver.py` 中扩展 `make_solver_pair(..., use_superlu_dist, dist_config)`。
- 新增 `spsolver/bindings_superlu_dist.py`：
  - 通过 SuperLU_DIST 自带 worker 协议（`control/data/result`）驱动分布式求解
  - 进程网格按 `nrow*ncol` 启动（本次固定默认 2x1）
  - 置换使用单位矩阵（`rowperm=0`, `colperm=0`）
  - 对复数系统使用实数块矩阵等价变换以兼容 `pdbridge` 的 double 接口

## 当前回归结果（2026-04-08）

以下最小回归命令均已通过（1 epoch）：

```bash
uv run train.py -f Configuration/2bit.toml \
  -o train.num_epochs=1 \
  -o train.enable_jit=false \
  -o save_load.save=false \
  -o save_load.viz_interval=1000 \
  -o simulation.enable_symbolic_reuse=true

uv run train.py -f Configuration/2bit.toml \
  -o train.num_epochs=1 \
  -o train.enable_jit=false \
  -o save_load.save=false \
  -o save_load.viz_interval=1000 \
  -o simulation.enable_symbolic_reuse=false
```
