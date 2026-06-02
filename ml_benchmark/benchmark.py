"""跑「模型 × 数据集」对比, 并为每个模型搜索最佳超参。

产出结构化结果, 供 visualize.py 绘图:
  - 每个 (dataset, model) 的默认表现与调优后表现 (train/test R², RMSE)
  - 最佳超参组合
  - 1D 数据集上的拟合曲线 (默认 vs 调优 vs 真相)
  - 特征工程对线性回归的增益
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import datasets as ds
from . import models as mdl


@dataclass
class FitResult:
    dataset: str
    model: str
    train_r2: float
    test_r2: float
    train_rmse: float
    test_rmse: float
    fit_time: float
    best_params: dict = field(default_factory=dict)
    tuned: bool = False


def _evaluate(est, Xtr, ytr, Xte, yte):
    t0 = time.perf_counter()
    est.fit(Xtr, ytr)
    fit_time = time.perf_counter() - t0
    ptr, pte = est.predict(Xtr), est.predict(Xte)
    return est, dict(
        train_r2=r2_score(ytr, ptr),
        test_r2=r2_score(yte, pte),
        train_rmse=float(np.sqrt(mean_squared_error(ytr, ptr))),
        test_rmse=float(np.sqrt(mean_squared_error(yte, pte))),
        fit_time=fit_time,
    )


def _curve_1d(est, dataset: ds.Dataset, n=400):
    """在单维数据集上, 沿 x0 的密集网格生成预测曲线。"""
    x = dataset.X[:, 0]
    grid = np.linspace(x.min(), x.max(), n).reshape(-1, 1)
    return grid.ravel(), est.predict(grid)


def run_dataset(dataset: ds.Dataset, model_keys=None, test_size=0.25,
                tune=True, quick=False, cv=3, seed=0, verbose=True):
    """在单个数据集上跑全部模型。返回 (results, extras)。"""
    model_keys = model_keys or list(mdl.MODELS)
    Xtr, Xte, ytr, yte = train_test_split(
        dataset.X, dataset.y, test_size=test_size, random_state=seed)

    results: list[FitResult] = []
    curves: dict[str, dict] = {}     # model -> {x, y_default, y_tuned}

    for key in model_keys:
        spec = mdl.MODELS[key]

        # --- 默认超参 ---
        est_def = spec.factory()
        est_def, m_def = _evaluate(est_def, Xtr, ytr, Xte, yte)
        results.append(FitResult(dataset.name, spec.name, **m_def,
                                 best_params={}, tuned=False))

        best_est = est_def
        if tune and spec.param_grid:
            grid = _maybe_shrink(spec.param_grid) if quick else spec.param_grid
            gs = GridSearchCV(spec.factory(), grid, cv=cv,
                              scoring="r2", n_jobs=-1)
            gs.fit(Xtr, ytr)
            best_est, m_tuned = _evaluate(
                spec.factory(**gs.best_params_), Xtr, ytr, Xte, yte)
            results.append(FitResult(dataset.name, spec.name, **m_tuned,
                                     best_params=gs.best_params_, tuned=True))

        if dataset.is_1d:
            gx, gy_def = _curve_1d(est_def, dataset)
            _, gy_best = _curve_1d(best_est, dataset)
            curves[spec.name] = dict(x=gx, y_default=gy_def, y_tuned=gy_best)

        if verbose:
            tag = results[-1]
            print(f"  [{dataset.name:>18}] {spec.name:<24} "
                  f"test R²={tag.test_r2:6.3f}  (train {tag.train_r2:6.3f})"
                  f"{'  *tuned' if tag.tuned else ''}")

    extras = {
        "curves": curves,
        "feature_engineering": _feature_engineering_gain(dataset, seed, test_size),
        "true_curve": _true_curve_1d(dataset) if dataset.is_1d else None,
        "scatter": _scatter_1d(dataset) if dataset.is_1d else None,
    }
    return results, extras


def _feature_engineering_gain(dataset: ds.Dataset, seed, test_size):
    """对比线性回归在 原始特征 vs 工程化特征 上的 test R²。"""
    if dataset.engineer is None:
        return None
    Xtr, Xte, ytr, yte = train_test_split(
        dataset.X, dataset.y, test_size=test_size, random_state=seed)

    raw = Pipeline([("s", StandardScaler()), ("lr", LinearRegression())])
    raw.fit(Xtr, ytr)
    raw_r2 = r2_score(yte, raw.predict(Xte))

    Xtr_e, names = dataset.engineer(Xtr)
    Xte_e, _ = dataset.engineer(Xte)
    eng = Pipeline([("s", StandardScaler()), ("lr", LinearRegression())])
    eng.fit(Xtr_e, ytr)
    eng_r2 = r2_score(yte, eng.predict(Xte_e))
    return dict(raw_r2=raw_r2, eng_r2=eng_r2, n_eng_features=len(names),
                feature_names=names[:12])


def _true_curve_1d(dataset: ds.Dataset, n=400):
    if dataset.true_func is None:
        return None
    x = dataset.X[:, 0]
    grid = np.linspace(x.min(), x.max(), n).reshape(-1, 1)
    return grid.ravel(), dataset.true_func(grid)


def _scatter_1d(dataset: ds.Dataset, n=600):
    idx = np.random.default_rng(0).choice(
        len(dataset.y), size=min(n, len(dataset.y)), replace=False)
    return dataset.X[idx, 0], dataset.y[idx]


def _maybe_shrink(grid: dict) -> dict:
    """quick 模式: 每个超参只留首/中/尾, 控制网格规模。"""
    out = {}
    for k, v in grid.items():
        if len(v) <= 3:
            out[k] = v
        else:
            out[k] = [v[0], v[len(v) // 2], v[-1]]
    return out


def run_all(datasets=None, model_keys=None, tune=True, quick=False,
            seed=0, verbose=True):
    datasets = datasets or ds.build_all()
    all_results, all_extras = [], {}
    for d in datasets:
        if verbose:
            print(f"\n=== {d.name}: {d.description} ===")
        res, extras = run_dataset(d, model_keys=model_keys, tune=tune,
                                  quick=quick, seed=seed, verbose=verbose)
        all_results.extend(res)
        all_extras[d.name] = {"dataset": d, "extras": extras}
    return all_results, all_extras


def results_to_frame(results: list[FitResult]):
    import pandas as pd
    return pd.DataFrame([r.__dict__ for r in results])
