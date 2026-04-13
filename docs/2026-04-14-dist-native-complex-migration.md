# 2026-04-14 superlu_dist native complex 迁移留痕

本文档记录本轮针对 superlu_dist Python 绑定与实验脚本的迁移/对齐工作，确保后续可追溯。

## 1. 目标

- 在 complex128 场景下，dist 路径默认使用 native complex 求解链路。
- 保留 legacy real 2N block 回退能力，用于兼容和 A/B 对照。
- 将 dist 相关实验脚本统一到同一参数风格和记录方式。

## 2. 关键改动

### 2.1 superlu_dist Python bridge（仓库内）

- 新增 `pzbridge`（complex16）桥接实现并接入构建：
  - [spsolver/superlu_dist/PYTHON/pzbridge.c](spsolver/superlu_dist/PYTHON/pzbridge.c)
  - [spsolver/superlu_dist/PYTHON/pzbridge.h](spsolver/superlu_dist/PYTHON/pzbridge.h)
  - [spsolver/superlu_dist/PYTHON/CMakeLists.txt](spsolver/superlu_dist/PYTHON/CMakeLists.txt)
- ctypes 层新增 `pzbridge_*` 符号装载：
  - [spsolver/superlu_dist/PYTHON/pdbridge.py](spsolver/superlu_dist/PYTHON/pdbridge.py)
- worker 支持按 `use_native_complex` 分支调用 native complex 或 legacy double：
  - [spsolver/superlu_dist/PYTHON/pddrive_worker.py](spsolver/superlu_dist/PYTHON/pddrive_worker.py)

### 2.2 Python 上层 bindings

- DistSolveConfig 新增 `native_complex`（默认 True）并透传到 dist worker：
  - [spsolver/bindings_superlu_dist.py](spsolver/bindings_superlu_dist.py)
- 保留 legacy real block 路径作为回退模式。

### 2.3 实验脚本统一

- native complex smoke：
  - [experiment_dist_native_complex_smoke.py](experiment_dist_native_complex_smoke.py)
  - 统一为 Typer 风格，支持 `--legacy-real-block`。
- 并行实验：
  - [experiment_dist_parallel.py](experiment_dist_parallel.py)
  - 增加 `--legacy-real-block`，写入 `dist_runtime.native_complex`。
- 串行对比分布式基线：
  - [experiment_dist_vs_serial.py](experiment_dist_vs_serial.py)
  - 增加 `--legacy-real-block`，写入 `dist_runtime.native_complex`。

## 3. 运行模式约定

- 默认：native complex（推荐）
- 回退：传入 `--legacy-real-block`

示例：

```bash
python experiment_dist_vs_serial.py
python experiment_dist_parallel.py
python experiment_dist_parallel.py --legacy-real-block
```

## 4. 验证与限制

- 已完成脚本级语法/静态检查。
- 当前设备不具备 MPI 可运行环境，端到端分布式运行需在目标设备验证。

## 5. 产出与追溯建议

- 每次实验执行保留：CSV、LaTeX rows、TOML（含 runtime 模式字段）。
- 若调整求解路径或默认参数，需同步更新 docs 并新增日期留痕文档。

## 6. 训练流程同步

- 训练调用链已显式接入 `simulation.superlu_dist.native_complex` 配置项，默认 `true`：
  - [simulation_backend.py](simulation_backend.py)
  - [train.py](train.py)
- 当 `simulation.backend="fdfd_solver"` 且 `enable_superlu_dist=true` 时，训练日志会打印 `superlu_dist.native_complex`。
- 后端一次性诊断日志会输出 dist 网格与 `native_complex` 状态：
  - [fdfd_solver/solver.py](fdfd_solver/solver.py)
- 配置模板已补齐字段示例：
  - [Configuration/2bit.toml](Configuration/2bit.toml)
  - [Configuration/5bit.toml](Configuration/5bit.toml)
  - [Configuration/6bit.toml](Configuration/6bit.toml)

## 7. 并行实验扩展（factor + solve）

- [experiment_dist_parallel.py](experiment_dist_parallel.py) 已扩展为同时统计 warm factor 与 warm solve。
- CSV/LaTeX/TOML 均新增 solve 相关字段（例如 `solve_time_s`、`solve_speedup`）。
- 图像产出由 1 张扩展为 2 张：
  - [figures/parallel_config_factor_speedup.png](figures/parallel_config_factor_speedup.png)
  - [figures/parallel_config_solve_speedup.png](figures/parallel_config_solve_speedup.png)

## 8. 训练总耗时日志

- [train.py](train.py) 在完成初始化后、进入首个 epoch 前开始计时。
- 在 epoch 循环结束后输出总用时日志：`训练总用时（epoch阶段）: ...s`。
