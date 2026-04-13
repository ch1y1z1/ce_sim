# 多右端向量批量求解实验（实验二）

本文档对应 [experiment_batch_rhs.py](experiment_batch_rhs.py)，用于生成：

- 表格数据（数值分解/三角求解耗时与比值）
- 批量加速比数据（朴素方法 vs 批量方法）
- 图 [figures/batch_speedup.png](figures/batch_speedup.png)
- LaTeX 表格行（可直接填入论文表格）

## 1. 先重编 nanobind 扩展

若近期修改过 spsolver 绑定，先重编：

```bash
cmake -S spsolver/nanobind -B spsolver/nanobind/build \
  -DPython_EXECUTABLE=$(which python) \
  -DCE_SUPERLU_USE_INTERNAL_BLAS=OFF \
  -DCE_SUPERLU_BLA_VENDOR=OpenBLAS
cmake --build spsolver/nanobind/build -j
```

## 2. 运行实验

默认参数（6 个规模、5 个批量、每组重复 10 次）：

```bash
python experiment_batch_rhs.py
```

显式参数示例：

```bash
python experiment_batch_rhs.py \
  --sizes 10,20,40,80,160,320 \
  --batch-sizes 4,8,16,32,64 \
  --repeats 10 \
  --wavelength 1.55e-6 \
  --ppw 20 \
  --eps-si 12.25 \
  --eps-sio2 2.085
```

仅使用已有数据重绘（不重跑实验）：

```bash
python experiment_batch_rhs.py --redraw-only
```

若 speedup CSV 不在默认位置：

```bash
python experiment_batch_rhs.py --redraw-only \
  --redraw-from-speedup-csv experiments/batch_rhs/batch_speedup.csv \
  --out-plot figures/batch_speedup.png
```

## 3. 输出文件

- [experiments/batch_rhs/batch_detail.csv](experiments/batch_rhs/batch_detail.csv)
- [experiments/batch_rhs/batch_speedup.csv](experiments/batch_rhs/batch_speedup.csv)
- [experiments/batch_rhs/batch_detail_rows.tex](experiments/batch_rhs/batch_detail_rows.tex)
- [experiments/batch_rhs/batch_detail.toml](experiments/batch_rhs/batch_detail.toml)
- [figures/batch_speedup.png](figures/batch_speedup.png)

## 4. 口径说明

- `T_factor_ms`: 单次数值分解耗时（来自 SuperLU 内部阶段统计）
- `T_solve_ms`: 单次三角求解耗时（来自 SuperLU 内部阶段统计）
- `factor_solve_ratio`: `T_factor_ms / T_solve_ms`
- `T_naive_ms`: 朴素方法总耗时（每个右端向量都重新分解并求解）
- `T_batch_ms`: 批量方法总耗时（单次分解 + M 次求解）
- `speedup`: `T_naive_ms / T_batch_ms`
