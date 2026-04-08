# SuperLU nanobind 绑定留痕（2026-04-08）

## 目标

将本地仓库中的 SuperLU 通过 nanobind 绑定为 Python 可调用接口，优先支持训练流程中的：

- `spfactorize`
- `spsolve`

并接入现有 `fdfd_solver` 后端工厂，作为可选高性能求解后端。

## 本次改动范围

### 1) C++ 绑定实现

- 新增/完善：`spsolver/nanobind/src/superlu_nb.cpp`
- 模块名：`_superlu_nb`
- 导出接口：
  - `spfactorize(colptr, rowind, nzval, n) -> ZFactor`
  - `spsolve(factor, rhs, overwrite_b=True)`（支持 1D/2D RHS）

关键实现点：

- 因子对象 `ZFactor` 生命周期内持有 `L/U` 因子。
- `solve_inplace` 路径优先原地写回，减少临时分配。
- 因子化前增加 CSC 输入合法性检查：
  - `colptr[0] == 0`
  - `colptr` 单调不减
  - `colptr[n] == nnz`
  - `rowind` 在合法范围
- 释放逻辑统一化，避免失败路径资源泄漏。
- 暴露索引位宽信息给 Python：
  - `index_size_bytes`
  - `index_is_64bit`

### 2) Python 适配层

- 新增：`spsolver/bindings_superlu.py`
- 更新：`spsolver/__init__.py`

关键实现点：

- 对外统一导出：
  - `spfactorize(A)`
  - `spsolve(factor, b, overwrite_b=False)`
  - `is_available()`
- 自动根据 `_superlu_nb.index_size_bytes` 选择索引 dtype（`int32/int64`）。
- 在扩展未编译时，抛出明确错误并提示构建路径。

### 3) 训练求解后端接入

- 更新：`fdfd_solver/solver.py`

当前策略：

- `fdfd_solver` 后端统一只使用自定义 SuperLU 绑定。
- 不再使用 SciPy 求解路径。
- `enable_symbolic_reuse=true` 时走 `spanalyze + sprefactorize + spsolve`。
- `enable_symbolic_reuse=false` 时走 `spfactorize + spsolve`。

## 构建方法

在项目根目录执行（使用当前 Python 环境）：

```bash
cmake -S spsolver/nanobind -B spsolver/nanobind/build \
  -DPython_EXECUTABLE=/home/chi/code/bysj/ce_sim/.venv/bin/python
cmake --build spsolver/nanobind/build -j
```

构建成功后会生成：

- `spsolver/_superlu_nb*.so`

如果环境中没有 `nanobind`，先安装：

```bash
uv pip install --python /home/chi/code/bysj/ce_sim/.venv/bin/python "nanobind>=2.2.0"
```

## 配置示例

在配置文件 `simulation` 顶层可指定：

```toml
[simulation]
backend = "fdfd_solver"
enable_batch = true
enable_symbolic_reuse = true
```

其中 `enable_batch` 与 `enable_symbolic_reuse` 仅在 `fdfd_solver` 后端生效。

## 零分配说明

- `spsolve(..., overwrite_b=True)` 走原地写回路径，可避免额外 RHS 拷贝。
- `spfactorize` 阶段对 CSC 索引与值数组优先使用连续内存视图；若上游 dtype 不匹配会发生必要转换。
- 对 2D RHS 的 C-contiguous `(batch, N)` 输入，当前实现在 C++ 内逐行求解，避免 Python 循环，但仍会为每个 RHS 调用底层求解例程。

## 已知限制

- 当前绑定针对复数双精度（`complex128`）路径。
- 默认接入的是本地 SuperLU（串行）；已增加可选 SuperLU_DIST 路由（通过 `simulation.enable_superlu_dist` 启用）。
- 若本地未安装 `nanobind` 或编译链缺失，扩展会导入失败，需先完成构建。

### SuperLU_DIST 选项补充（2026-04-08）

- 新增配置：`simulation.enable_superlu_dist`
- 新增参数段：
  - `simulation.superlu_dist.nrow = 2`
  - `simulation.superlu_dist.ncol = 1`
  - `simulation.superlu_dist.rowperm = 0`（单位行置换）
  - `simulation.superlu_dist.colperm = 0`（单位列置换）
- 兼容性：可与 `enable_batch`、`enable_symbolic_reuse` 同时使用（仅 `fdfd_solver` 后端生效）。

## 2026-04-08 追加修复

- 修复 `_superlu_nb` 导入失败：`undefined symbol: dmach`
  - 原因：SuperLU 构建时关闭了 double 路径。
  - 处理：在 `spsolver/nanobind/CMakeLists.txt` 中启用 `enable_double ON` 并重编。
- 修复 JAX 回调下 `spsolve` 参数不匹配
  - 现象：只读 RHS 传入 nanobind 可写 ndarray 重载时报错。
  - 处理：在 `spsolver/bindings_superlu.py` 中，当 RHS 不可写时自动拷贝为可写 C 连续数组。
- 修复 symbolic reuse 模式退出时 nanobind 泄漏告警
  - 处理：在 `fdfd_solver/solver.py` 注册 `atexit` 清理符号复用缓存。

## 后续建议

- 增加基准脚本，对比 `splu` 与 `superlu_*` 在训练 batch 大小下的实际速度。
- 下一步扩展 SuperLU_DIST 绑定，并在多进程/MPI 训练场景中接入。

## 本地验证结果

已在当前工作区完成以下验证：

- `cmake` 配置成功。
- `_superlu_nb` 成功编译并输出到 `spsolver/`。
- Python 快速检查通过：
  - `spanalyze/sprefactorize/spsolve` 单 RHS 正确。
  - `spsolve(..., transpose=True)` 正确。
- 训练最小回归通过（`num_epochs=1`）：
  - `simulation.enable_symbolic_reuse=true`
  - `simulation.enable_symbolic_reuse=false`