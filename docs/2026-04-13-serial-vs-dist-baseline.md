# serial superlu vs superlu_dist(1x1) 基线测试

本文档对应 [experiment_dist_vs_serial.py](experiment_dist_vs_serial.py)，用于比较：

- superlu（串行）
- superlu_dist（1 task / 1 thread / 1x1）

在相同 200x200 FDFD 问题上的 factor 与 solve 耗时，并量化 dist 启动常数项。

## 1. 运行

```bash
python experiment_dist_vs_serial.py
```

显式参数示例：

```bash
python experiment_dist_vs_serial.py \
  --n 200 \
  --repeats 10 \
  --warmup 2 \
  --rowperm 1 \
  --colperm 2 \
  --launcher mpirun
```

## 2. 输出

- [experiments/dist_parallel/serial_vs_dist_baseline.csv](experiments/dist_parallel/serial_vs_dist_baseline.csv)
- [experiments/dist_parallel/serial_vs_dist_baseline_rows.tex](experiments/dist_parallel/serial_vs_dist_baseline_rows.tex)
- [experiments/dist_parallel/serial_vs_dist_baseline.toml](experiments/dist_parallel/serial_vs_dist_baseline.toml)

## 3. 指标说明

- factor_cold_s: 冷启动 factor 时间
- solve_cold_s: 冷启动 solve 时间
- factor_warm_s: 稳态 factor 平均时间
- solve_warm_s: 稳态 solve 平均时间
- startup_overhead_est_s (dist): `factor_cold_s - factor_warm_s`
- solve_overhead_est_s (dist): `solve_cold_s - solve_warm_s`
- total_overhead_est_s (dist): `(factor_cold_s + solve_cold_s) - (factor_warm_s + solve_warm_s)`

说明：overhead 为经验估计量，用于衡量冷启动额外常数项，不是严格分解出的唯一启动时间。
