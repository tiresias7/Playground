"""一键入口: 跑全部基准并生成交互式 HTML 报告 (+ 一张总览 PNG)。

用法:
    python -m ml_benchmark.run                 # 完整跑 (含网格搜索, 较慢)
    python -m ml_benchmark.run --quick         # 缩小网格, 快速预览
    python -m ml_benchmark.run --no-tune       # 只看默认超参表现
    python -m ml_benchmark.run --out report.html
"""
from __future__ import annotations

import argparse
import os

from . import benchmark as bm
from . import datasets as ds
from . import tuning as tu
from . import visualize as viz


def _overview_png(results, path):
    """用 matplotlib 另存一张静态总览热力图, 方便快速预览/分享。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from . import models as mdl

    rows = viz._best_per_pair(results, use_tuned=True)
    models = [s.name for s in mdl.MODELS.values()]
    datasets = sorted({r["dataset"] for r in rows})
    Z = np.full((len(models), len(datasets)), np.nan)
    for i, m in enumerate(models):
        for j, d in enumerate(datasets):
            v = next((r["test_r2"] for r in rows
                      if r["model"] == m and r["dataset"] == d), np.nan)
            Z[i, j] = v

    fig, ax = plt.subplots(figsize=(12, 4.2))
    im = ax.imshow(Z, vmin=-0.2, vmax=1.0, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9)
    for i in range(len(models)):
        for j in range(len(datasets)):
            if not np.isnan(Z[i, j]):
                ax.text(j, i, f"{Z[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color="black")
    # 标题用英文: matplotlib 默认字体 (DejaVu Sans) 不含中文字形
    ax.set_title("test R²  (tuned; greener = better)")
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ml_benchmark/report.html")
    ap.add_argument("--png", default="ml_benchmark/overview.png")
    ap.add_argument("--quick", action="store_true", help="缩小网格, 快速预览")
    ap.add_argument("--no-tune", dest="tune", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    datasets = ds.build_all()
    print("跑基准 (模型 × 数据集) ...")
    results, all_extras = bm.run_all(datasets, tune=args.tune,
                                     quick=args.quick, seed=args.seed)

    figs = [viz.fig_overview_heatmap(results)]

    # 1D 拟合曲线 (每个可画的数据集一张)
    for dname, payload in all_extras.items():
        f = viz.fig_fit_curves(dname, payload)
        if f is not None:
            figs.append(f)

    # 精选超参扫描 (覆盖三类模型的代表性现象)
    dmap = {d.name: d for d in datasets}
    print("\n超参扫描 ...")
    sweeps = [
        ("quadratic", "tree", "max_depth"),            # 偏差/方差权衡
        ("interaction", "random_forest", "n_estimators"),  # 收益递减
        ("high_freq_periodic", "gbt", "learning_rate"),    # 学习率影响
        ("high_noise", "gbt", "max_depth"),                # 高噪声下浅树更稳
    ]
    for dname, mkey, axis in sweeps:
        if dname in dmap:
            figs.append(viz.fig_sweep(tu.sweep(dmap[dname], mkey, axis,
                                               seed=args.seed)))

    # GBT 早停曲线 (staged_predict)
    print("GBT 逐棵树早停曲线 ...")
    for dname in ("high_freq_periodic", "interaction"):
        if dname in dmap:
            figs.append(viz.fig_gbt_staged(
                tu.gbt_staged(dmap[dname], seed=args.seed)))

    # 特征工程增益
    figs.append(viz.fig_feature_engineering(all_extras))

    # 学习曲线 (需要多少数据?) — 选简单→复杂三个代表性数据集
    print("\n学习曲线 ...")
    for dname in ("linear_additive", "interaction", "high_freq_periodic"):
        if dname in dmap:
            figs.append(viz.fig_learning_curve(
                tu.learning_curve(dmap[dname], seed=args.seed)))

    # 训练时间对比
    figs.append(viz.fig_fit_time(results))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    viz.build_report(figs, args.out)
    _overview_png(results, args.png)

    print(f"\n✅ 报告: {args.out}")
    print(f"✅ 总览图: {args.png}")
    _print_summary(results)


def _print_summary(results):
    """文字版结论: 每个数据集 test R² 最高的模型。"""
    rows = viz._best_per_pair(results, use_tuned=True)
    by_ds: dict[str, list] = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], []).append(r)
    print("\n各数据集最佳模型 (调优后 test R²):")
    for d in sorted(by_ds):
        best = max(by_ds[d], key=lambda r: r["test_r2"])
        print(f"  {d:>20}: {best['model']:<24} R²={best['test_r2']:.3f}")


if __name__ == "__main__":
    main()
