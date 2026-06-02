"""超参数单轴扫描: 固定其他超参, 扫描一个关键超参, 记录 train/test R²。

这正是课堂上关心的"什么时候停 / 学习率多大 / 树多深"等问题的可视化:
  - Decision Tree 的 max_depth -> 经典偏差/方差权衡 (深 -> train↑ test↓)
  - Random Forest 的 n_estimators -> 收益递减, test 趋于平台
  - GBT 的 n_estimators -> 早停信号 (test 先降后回升 = 过拟合, 该停了)
  - GBT 的 learning_rate -> 大步快但易过冲, 小步稳但需更多树
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from . import datasets as ds
from . import models as mdl


def sweep(dataset: ds.Dataset, model_key: str, axis: str, values=None,
          fixed: dict | None = None, test_size=0.25, seed=0):
    """沿单个超参 axis 扫描, 返回每个取值下的 train/test R²。"""
    spec = mdl.MODELS[model_key]
    values = values or spec.sweep_axes[axis]
    fixed = dict(fixed or {})

    Xtr, Xte, ytr, yte = train_test_split(
        dataset.X, dataset.y, test_size=test_size, random_state=seed)

    rows = []
    for v in values:
        est = spec.factory(**{**fixed, axis: v})
        est.fit(Xtr, ytr)
        rows.append(dict(
            value=v,
            train_r2=r2_score(ytr, est.predict(Xtr)),
            test_r2=r2_score(yte, est.predict(Xte)),
        ))
    return dict(model=spec.name, dataset=dataset.name, axis=axis, rows=rows)


def sweep_all_axes(dataset: ds.Dataset, model_key: str, **kw):
    """扫描某模型在该数据集上的所有 sweep_axes。"""
    spec = mdl.MODELS[model_key]
    return {axis: sweep(dataset, model_key, axis, **kw)
            for axis in spec.sweep_axes}


def gbt_staged(dataset: ds.Dataset, learning_rate=0.1, max_depth=3,
               n_estimators=800, test_size=0.25, seed=0):
    """利用 GBT 的 staged_predict 一次拟合得到逐棵树的 test 曲线。

    这是观察"早停点"的最直接方式: test R² 见顶后再加树只会过拟合。
    """
    from sklearn.ensemble import GradientBoostingRegressor

    Xtr, Xte, ytr, yte = train_test_split(
        dataset.X, dataset.y, test_size=test_size, random_state=seed)
    gb = GradientBoostingRegressor(
        n_estimators=n_estimators, learning_rate=learning_rate,
        max_depth=max_depth, random_state=0)
    gb.fit(Xtr, ytr)

    stages = np.arange(1, n_estimators + 1)
    test_r2 = np.array([r2_score(yte, p) for p in gb.staged_predict(Xte)])
    train_r2 = np.array([r2_score(ytr, p) for p in gb.staged_predict(Xtr)])
    best_n = int(stages[np.argmax(test_r2)])
    return dict(stages=stages, train_r2=train_r2, test_r2=test_r2,
                best_n_estimators=best_n, best_test_r2=float(test_r2.max()),
                learning_rate=learning_rate, max_depth=max_depth,
                dataset=dataset.name)


def learning_curve(dataset: ds.Dataset, model_keys=None, fractions=None,
                   test_size=0.3, seed=0):
    """学习曲线: 固定 test 集, 改变训练样本数, 看 test R² 如何随数据量增长。

    回答课堂提到的"需要多少数据"——
      - 线性回归在简单线性数据上很快触顶, 再多数据帮助有限;
      - 树类/集成在复杂数据上对数据量更"饥渴", 数据越多越占优。
    用同一份随机排列取前 k 个, 保证小训练集是大训练集的子集 (嵌套)。
    """
    model_keys = model_keys or list(mdl.MODELS)
    fractions = fractions or [0.05, 0.1, 0.2, 0.35, 0.55, 0.8, 1.0]

    Xtr_full, Xte, ytr_full, yte = train_test_split(
        dataset.X, dataset.y, test_size=test_size, random_state=seed)
    n = len(ytr_full)
    order = np.random.default_rng(seed).permutation(n)

    curves = {}
    for key in model_keys:
        spec = mdl.MODELS[key]
        sizes, scores = [], []
        for frac in fractions:
            k = max(15, int(frac * n))
            idx = order[:k]
            est = spec.factory()
            est.fit(Xtr_full[idx], ytr_full[idx])
            sizes.append(k)
            scores.append(r2_score(yte, est.predict(Xte)))
        curves[spec.name] = dict(sizes=sizes, test_r2=scores, color=spec.color)
    return dict(dataset=dataset.name, curves=curves)
