"""四个回归模型的统一封装与超参数搜索空间。

对应课堂上比较的四种方法:
  1. Linear Regression  —— 线性模型, 高偏差/低方差, 只能拟合线性关系
  2. Decision Tree      —— 单棵树, 低偏差/高方差, 容易过拟合
  3. Random Forest      —— bagging (并行, 降方差)
  4. Gradient Boosted   —— boosting (串行, 降偏差), 也可对样本/特征做子采样

每个模型给出:
  - factory: 接受超参数返回 sklearn estimator (已套上标准化, 对线性模型尤其重要)
  - param_grid: 课堂提到的可调超参 (learning_rate / depth / n_estimators / ...)
  - sweep_axes: 用于"逐一扫描某个超参看 train/test 曲线"的关键轴
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


@dataclass
class ModelSpec:
    name: str
    factory: Callable[..., Any]
    # 网格搜索空间: {超参名: [候选值, ...]}
    param_grid: dict[str, list]
    # 单轴扫描: {超参名: [由小到大的取值]} —— 用来画偏差/方差权衡曲线
    sweep_axes: dict[str, list] = field(default_factory=dict)
    color: str = "#1f77b4"
    blurb: str = ""


def _linear_factory(**kw):
    # 线性模型必须标准化, 否则不同尺度特征权重不可比
    return Pipeline([("scale", StandardScaler()), ("lr", LinearRegression(**kw))])


def _tree_factory(max_depth=None, min_samples_leaf=1, **kw):
    return DecisionTreeRegressor(
        max_depth=max_depth, min_samples_leaf=min_samples_leaf,
        random_state=0, **kw,
    )


def _rf_factory(n_estimators=200, max_depth=None, max_features=1.0,
                min_samples_leaf=1, **kw):
    return RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        max_features=max_features, min_samples_leaf=min_samples_leaf,
        random_state=0, n_jobs=-1, **kw,
    )


def _gbt_factory(n_estimators=300, learning_rate=0.1, max_depth=3,
                 subsample=1.0, min_samples_leaf=1, **kw):
    return GradientBoostingRegressor(
        n_estimators=n_estimators, learning_rate=learning_rate,
        max_depth=max_depth, subsample=subsample,
        min_samples_leaf=min_samples_leaf, random_state=0, **kw,
    )


MODELS: dict[str, ModelSpec] = {
    "linear": ModelSpec(
        name="Linear Regression",
        factory=_linear_factory,
        param_grid={},  # 普通最小二乘无超参 (可换 Ridge 加 alpha)
        sweep_axes={},
        color="#1f77b4",
        blurb="高偏差; 只能拟合线性关系, 但样本少/线性强时最稳。",
    ),
    "tree": ModelSpec(
        name="Decision Tree",
        factory=_tree_factory,
        param_grid={
            "max_depth": [2, 3, 5, 8, 12, None],
            "min_samples_leaf": [1, 5, 20, 50],
        },
        sweep_axes={"max_depth": [1, 2, 3, 4, 6, 8, 10, 14, 20]},
        color="#ff7f0e",
        blurb="单棵树, 深度↑ -> 偏差↓方差↑, 极易过拟合。",
    ),
    "random_forest": ModelSpec(
        name="Random Forest",
        factory=_rf_factory,
        param_grid={
            "n_estimators": [100, 300],
            "max_depth": [5, 10, None],
            "max_features": [0.3, 0.6, 1.0],
            "min_samples_leaf": [1, 5, 20],
        },
        sweep_axes={
            "n_estimators": [5, 10, 25, 50, 100, 200, 400],
            "max_features": [0.2, 0.4, 0.6, 0.8, 1.0],
        },
        color="#2ca02c",
        blurb="bagging 多棵深树, 靠平均降方差; 树越多越稳但收益递减。",
    ),
    "gbt": ModelSpec(
        name="Gradient Boosted Trees",
        factory=_gbt_factory,
        param_grid={
            "n_estimators": [100, 300, 600],
            "learning_rate": [0.3, 0.1, 0.05, 0.02],
            "max_depth": [2, 3, 4],
            "subsample": [1.0, 0.7],
        },
        sweep_axes={
            "n_estimators": [10, 25, 50, 100, 200, 400, 800],
            "learning_rate": [0.5, 0.3, 0.1, 0.05, 0.02, 0.01],
            "max_depth": [1, 2, 3, 4, 5, 6],
        },
        color="#d62728",
        blurb="boosting 浅树串行纠错; learning_rate 与 n_estimators 此消彼长。",
    ),
}


def build(model_key: str, **params):
    return MODELS[model_key].factory(**params)
