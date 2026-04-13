# 分布式并行求解实验（实验三）

本文档对应 [experiment_dist_parallel.py](experiment_dist_parallel.py)，用于生成：

- 并行配置结果表数据（CSV）
- LaTeX 表格行（可直接填入论文）
- 速度提升图 [figures/parallel_config_speedup.png](figures/parallel_config_speedup.png)
- 元数据（TOML）

## 1. 运行实验

默认参数：矩阵规模 n=200，重复 10 次，配置集合来自 ntasks={1,2,4} 与 cpus-per-task={1,2,4,8}，并自动枚举满足 nrow*ncol=ntasks 的网格形状。

```bash
python experiment_dist_parallel.py
```

显式参数示例：

```bash
python experiment_dist_parallel.py \
  --n 200 \
  --ntasks-list 1,2,4 \
  --cpus-per-task-list 1,2,4,8 \
  --repeats 10 \
  --warmup 1 \
  --launcher mpirun
```

若在 Slurm 作业中希望改用 srun，可指定：

```bash
python experiment_dist_parallel.py \
  --launcher srun \
  --launcher-extra-args "--mpi=pmix"
```

## 2. 仅重绘（不重跑实验）

使用已有 CSV 直接重绘：

```bash
python experiment_dist_parallel.py --redraw-only
```

指定数据源并输出到目标图片：

```bash
python experiment_dist_parallel.py --redraw-only \
  --redraw-from-csv experiments/dist_parallel/parallel_config.csv \
  --out-plot figures/parallel_config_speedup.png
```

## 3. 输出文件

- [experiments/dist_parallel/parallel_config.csv](experiments/dist_parallel/parallel_config.csv)
- [experiments/dist_parallel/parallel_config_rows.tex](experiments/dist_parallel/parallel_config_rows.tex)
- [experiments/dist_parallel/parallel_config.toml](experiments/dist_parallel/parallel_config.toml)
- [figures/parallel_config_speedup.png](figures/parallel_config_speedup.png)

## 4. 字段说明

- ntasks: MPI 进程数
- cpus_per_task: 每进程线程数（通过 OMP_NUM_THREADS 等环境变量设置）
- nrow, ncol: SuperLU_DIST 二维进程网格
- factor_time_s: 数值分解耗时（秒）
- speedup: 相对于基准配置 (1 task, 1 thread, 1x1) 的加速比
- status: ok 或 failed
- error: 失败时的错误信息
