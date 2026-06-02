# ml_benchmark — 四种回归方法的合成数据基准与可视化

对比课堂上学的四种方法，在**刻意设计的多种合成数据集**上的表现，并把
**超参数的影响**可视化出来，回答这几个问题：

- 每种方法分别**擅长什么样的数据**？
- 在哪些**超参数**下表现最好（learning rate / 树深 / 树的数量 / 何时停）？
- **特征工程**到底能帮多大忙？

## 四个模型

| key | 模型 | 机制 | 直觉 |
|-----|------|------|------|
| `linear` | Linear Regression | 最小二乘 | 高偏差、低方差，只能拟合线性关系 |
| `tree` | Decision Tree | 单棵 CART | 低偏差、高方差，极易过拟合 |
| `random_forest` | Random Forest | **bagging**（并行、降方差） | 多棵深树取平均 |
| `gbt` | Gradient Boosted Trees | **boosting**（串行、降偏差） | 浅树逐步纠错，可子采样 |

## 数据集（`datasets.py`）

| 名称 | 信号 | 考察点 |
|------|------|--------|
| `linear_additive` | Σ wᵢxᵢ + ε | 纯线性（叠加后加噪声）→ 线性回归主场 |
| `linear_noise_before` | Σ wᵢ(xᵢ+εᵢ) | 噪声在**叠加前**注入特征，信噪结构不同 |
| `interaction` | x₀·x₁ | 二元交互，线性回归看不见 |
| `quadratic` | x₀² | 单变量非线性 |
| `logarithmic` | log(x₀) | 单调凹 |
| `high_noise` | 线性，SNR≈1 | 极大噪声，考验抗过拟合 |
| `mixed_interaction` | x₀·log(x₁) | 交互 × 非线性 |
| `periodic` | sin(x₀) | 低频周期 |
| `high_freq_periodic` | sin(5x₀) | 强波动，局部斜率大 |
| `step` | 阶跃 | 分段常数，树的天然主场 |

每个数据集都带一个**特征工程器**（多项式 / 对数 / 傅里叶项），
用于对比线性回归在「原始 vs 工程化特征」下的差距。

## 用法

```bash
# 完整跑（含超参网格搜索，较慢）
python -m ml_benchmark.run

# 缩小网格，快速预览
python -m ml_benchmark.run --quick

# 只看默认超参
python -m ml_benchmark.run --no-tune
```

产出：

- `ml_benchmark/report.html` — **自包含、可交互**的 Plotly 报告（离线打开，图例可点选）
- `ml_benchmark/overview.png` — 静态总览热力图（模型 × 数据集 的 test R²）

## 报告四块内容

1. **总览热力图** — 模型 × 数据集 的 test R²，一眼看出谁擅长什么
2. **1D 拟合曲线** — 散点 + 真相曲线 + 各模型拟合，直观看欠/过拟合
3. **超参扫描** — train/test R² 随超参变化（偏差/方差权衡、收益递减、早停）；
   含 GBT 的 `staged_predict` 逐棵树曲线，直观展示**何时早停**
4. **特征工程增益** — 线性回归在原始 vs 工程化特征上的对比

## 作为库调用

```python
from ml_benchmark import datasets, benchmark, tuning, visualize

d = datasets.build("interaction")
results, extras = benchmark.run_dataset(d, quick=True)

# 单轴超参扫描
sw = tuning.sweep(d, "gbt", "learning_rate")

# GBT 早停曲线
st = tuning.gbt_staged(d)
```
