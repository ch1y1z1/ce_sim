# 符号分析复用实验（实验一）

本文档对应 `experiment_symbolic_reuse.py`，用于生成：

- 论文表格所需数据（CSV）
- 加速比图（`figures/reuse_accu.png`）
- LaTeX 表格行片段（可直接粘贴到表格）

## 1. 先重编 nanobind 扩展

新增了 `profile_phases`（基于 SuperLU `SuperLUStat_t` 内置统计）绑定函数，需重编 `_superlu_nb`：

```bash
cmake -S spsolver/nanobind -B spsolver/nanobind/build \
  -DPython_EXECUTABLE=$(which python) \
  -DCE_SUPERLU_USE_INTERNAL_BLAS=OFF \
  -DCE_SUPERLU_BLA_VENDOR=OpenBLAS
cmake --build spsolver/nanobind/build -j
```

## 2. 运行实验

默认参数已经与你论文实验描述一致（6 个规模、每组重复 10 次）：

```bash
python experiment_symbolic_reuse.py
```

若要显式指定：

```bash
python experiment_symbolic_reuse.py \
  --sizes 10,20,40,80,160,320 \
  --repeats 10 \
  --wavelength 1.55e-6 \
  --ppw 20 \
  --eps-si 12.25 \
  --eps-sio2 2.085 \
  --out-plot figures/reuse_accu.png
```

## 3. 输出文件

- `experiments/symbolic_reuse/reuse_detail.csv`
- `experiments/symbolic_reuse/reuse_detail_rows.tex`
- `experiments/symbolic_reuse/reuse_detail.toml`
- `figures/reuse_accu.png`

## 4. 列定义

CSV 列与论文表字段对应关系：

- `T_perm_ms` -> $T_{\mathrm{perm}}$
- `T_symb_ms` -> $T_{\mathrm{symb}}$
- `T_num_ms` -> $T_{\mathrm{num}}$
- `T_solve_ms` -> $T_{\mathrm{solve}}$
- `T_total_ms` -> $T_{\mathrm{total}}$
- `T_reuse_ms` -> $T_{\mathrm{reuse}}$
- `speedup` -> $S = T_{\mathrm{total}}/T_{\mathrm{reuse}}$
