"""合成数据集生成器。

每个生成器返回一个 Dataset 对象，封装了:
  - 特征矩阵 X 与目标 y
  - 真实函数 true_func (用于在 1D 数据集上画出"真相曲线", 直观对比欠/过拟合)
  - feature engineering 函数 engineer (用于回答"特征工程能帮多大忙")

设计原则: 每种数据集都刻意凸显某一类模型的长处或短处。
  - 纯线性   -> 线性回归占优, 树类要很多分裂才能逼近
  - 交互/非线性 -> 树类占优, 线性回归无能为力(除非做特征工程)
  - 周期/波动  -> 都很难; 特征工程(加 sin 项)能救线性回归
  - 高噪声   -> 考验抗方差能力, bagging / 正则起作用
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass
class Dataset:
    name: str
    description: str
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    # 真实信号函数 f(X) (不含噪声). 仅在能解析表达时提供, 用于可视化。
    true_func: Optional[Callable[[np.ndarray], np.ndarray]] = None
    # 特征工程: 把原始 X 映射到扩展特征空间, 返回 (X_eng, eng_feature_names)
    engineer: Optional[Callable[[np.ndarray], tuple[np.ndarray, list[str]]]] = None
    # 噪声相对信号的标准差比例, 仅作记录
    noise_level: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def is_1d(self) -> bool:
        """只有一个"有效"输入维度时, 可以画拟合曲线。"""
        return self.meta.get("plot_1d", False)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _signal_noise_scale(signal: np.ndarray, snr_ratio: float) -> float:
    """根据信号标准差与目标噪声比, 算出噪声的绝对标准差。"""
    sig_std = float(np.std(signal))
    sig_std = sig_std if sig_std > 1e-9 else 1.0
    return snr_ratio * sig_std


# --------------------------------------------------------------------------- #
# 1. 线性叠加, 叠加后加噪声 (线性回归主场)
# --------------------------------------------------------------------------- #
def make_linear_additive(n=1500, d=5, noise=0.2, seed=0) -> Dataset:
    rng = _rng(seed)
    X = rng.uniform(-3, 3, size=(n, d))
    w = rng.uniform(-2, 2, size=d)
    b = rng.uniform(-1, 1)
    signal = X @ w + b
    sigma = _signal_noise_scale(signal, noise)
    y = signal + rng.normal(0, sigma, size=n)

    def true_func(Xq, w=w, b=b):
        return Xq @ w + b

    return Dataset(
        name="linear_additive",
        description=f"y = Σ wᵢxᵢ + b + ε  (d={d}, 噪声叠加在最后)",
        X=X, y=y, feature_names=[f"x{i}" for i in range(d)],
        true_func=true_func, engineer=_poly_engineer(degree=2),
        noise_level=noise, meta={"plot_1d": False},
    )


# --------------------------------------------------------------------------- #
# 2. 线性叠加, 但噪声先加到每个 feature 再叠加
#    y = Σ wᵢ(xᵢ + εᵢ).  数学上仍线性, 但噪声经权重放大且相关, 信噪结构不同。
# --------------------------------------------------------------------------- #
def make_linear_noise_before(n=1500, d=5, noise=0.2, seed=1) -> Dataset:
    rng = _rng(seed)
    X = rng.uniform(-3, 3, size=(n, d))
    w = rng.uniform(-2, 2, size=d)
    b = rng.uniform(-1, 1)
    clean = X @ w + b
    sigma = _signal_noise_scale(X, noise)            # 噪声以特征尺度衡量
    X_noisy = X + rng.normal(0, sigma, size=(n, d))  # 先污染特征
    y = X_noisy @ w + b                              # 再叠加 (无额外输出噪声)

    def true_func(Xq, w=w, b=b):
        return Xq @ w + b

    return Dataset(
        name="linear_noise_before",
        description=f"y = Σ wᵢ(xᵢ+εᵢ) + b  (d={d}, 噪声在叠加前注入特征)",
        X=X, y=y, feature_names=[f"x{i}" for i in range(d)],
        true_func=true_func, engineer=_poly_engineer(degree=2),
        noise_level=noise, meta={"plot_1d": False,
                                 "note": "观测到的 X 是干净的, 但 y 由被污染的 X 生成 -> 不可约误差"},
    )


# --------------------------------------------------------------------------- #
# 3. 二元交互 y = x1*x2 (+ 弱线性项)
# --------------------------------------------------------------------------- #
def make_interaction(n=1500, d=4, noise=0.15, seed=2) -> Dataset:
    rng = _rng(seed)
    X = rng.uniform(-3, 3, size=(n, d))
    signal = X[:, 0] * X[:, 1] + 0.5 * X[:, 2]
    sigma = _signal_noise_scale(signal, noise)
    y = signal + rng.normal(0, sigma, size=n)

    def true_func(Xq):
        return Xq[:, 0] * Xq[:, 1] + 0.5 * Xq[:, 2]

    return Dataset(
        name="interaction",
        description="y = x0·x1 + 0.5·x2 + ε  (纯交互, 线性回归看不见)",
        X=X, y=y, feature_names=[f"x{i}" for i in range(d)],
        true_func=true_func, engineer=_poly_engineer(degree=2, interaction_only=False),
        noise_level=noise, meta={"plot_1d": False},
    )


# --------------------------------------------------------------------------- #
# 4. 二次项 y = x1^2
# --------------------------------------------------------------------------- #
def make_quadratic(n=1200, noise=0.15, seed=3) -> Dataset:
    rng = _rng(seed)
    x = rng.uniform(-3, 3, size=(n, 1))
    signal = (x[:, 0] ** 2) - 2.0
    sigma = _signal_noise_scale(signal, noise)
    y = signal + rng.normal(0, sigma, size=n)

    return Dataset(
        name="quadratic",
        description="y = x0² − 2 + ε  (单变量非线性)",
        X=x, y=y, feature_names=["x0"],
        true_func=lambda Xq: Xq[:, 0] ** 2 - 2.0,
        engineer=_poly_engineer(degree=3),
        noise_level=noise, meta={"plot_1d": True},
    )


# --------------------------------------------------------------------------- #
# 5. 对数项 y = log|x1|
# --------------------------------------------------------------------------- #
def make_logarithmic(n=1200, noise=0.15, seed=4) -> Dataset:
    rng = _rng(seed)
    x = rng.uniform(0.1, 6, size=(n, 1))
    signal = np.log(x[:, 0])
    sigma = _signal_noise_scale(signal, noise)
    y = signal + rng.normal(0, sigma, size=n)

    return Dataset(
        name="logarithmic",
        description="y = log(x0) + ε  (x0∈[0.1,6], 单调凹)",
        X=x, y=y, feature_names=["x0"],
        true_func=lambda Xq: np.log(np.clip(Xq[:, 0], 1e-6, None)),
        engineer=_log_engineer(),
        noise_level=noise, meta={"plot_1d": True},
    )


# --------------------------------------------------------------------------- #
# 6. 高噪声线性 (低信噪比)
# --------------------------------------------------------------------------- #
def make_high_noise(n=1500, d=5, noise=1.0, seed=5) -> Dataset:
    rng = _rng(seed)
    X = rng.uniform(-3, 3, size=(n, d))
    w = rng.uniform(-2, 2, size=d)
    signal = X @ w
    sigma = _signal_noise_scale(signal, noise)   # 噪声 ≈ 信号同量级
    y = signal + rng.normal(0, sigma, size=n)

    return Dataset(
        name="high_noise",
        description=f"y = Σ wᵢxᵢ + ε  (SNR≈1, 噪声极大, 考验抗过拟合)",
        X=X, y=y, feature_names=[f"x{i}" for i in range(d)],
        true_func=lambda Xq, w=w: Xq @ w,
        engineer=_poly_engineer(degree=2),
        noise_level=noise, meta={"plot_1d": False},
    )


# --------------------------------------------------------------------------- #
# 7. 混合交互 y = x1 * log|x2|
# --------------------------------------------------------------------------- #
def make_mixed_interaction(n=1500, noise=0.15, seed=6) -> Dataset:
    rng = _rng(seed)
    x0 = rng.uniform(-3, 3, size=n)
    x1 = rng.uniform(0.1, 6, size=n)
    X = np.column_stack([x0, x1])
    signal = x0 * np.log(x1)
    sigma = _signal_noise_scale(signal, noise)
    y = signal + rng.normal(0, sigma, size=n)

    return Dataset(
        name="mixed_interaction",
        description="y = x0·log(x1) + ε  (交互 × 非线性)",
        X=X, y=y, feature_names=["x0", "x1"],
        true_func=lambda Xq: Xq[:, 0] * np.log(np.clip(Xq[:, 1], 1e-6, None)),
        engineer=_mixed_engineer(),
        noise_level=noise, meta={"plot_1d": False},
    )


# --------------------------------------------------------------------------- #
# 8. 周期 y = sin(x)
# --------------------------------------------------------------------------- #
def make_periodic(n=1200, noise=0.1, seed=7) -> Dataset:
    rng = _rng(seed)
    x = rng.uniform(-2 * np.pi, 2 * np.pi, size=(n, 1))
    signal = np.sin(x[:, 0])
    sigma = _signal_noise_scale(signal, noise)
    y = signal + rng.normal(0, sigma, size=n)

    return Dataset(
        name="periodic",
        description="y = sin(x0) + ε  (低频周期)",
        X=x, y=y, feature_names=["x0"],
        true_func=lambda Xq: np.sin(Xq[:, 0]),
        engineer=_fourier_engineer(freqs=(1, 2, 3)),
        noise_level=noise, meta={"plot_1d": True},
    )


# --------------------------------------------------------------------------- #
# 9. 高频周期 y = sin(5x)
# --------------------------------------------------------------------------- #
def make_high_freq_periodic(n=2000, noise=0.1, seed=8) -> Dataset:
    rng = _rng(seed)
    x = rng.uniform(-np.pi, np.pi, size=(n, 1))
    signal = np.sin(5 * x[:, 0])
    sigma = _signal_noise_scale(signal, noise)
    y = signal + rng.normal(0, sigma, size=n)

    return Dataset(
        name="high_freq_periodic",
        description="y = sin(5·x0) + ε  (强波动, 局部斜率大)",
        X=x, y=y, feature_names=["x0"],
        true_func=lambda Xq: np.sin(5 * Xq[:, 0]),
        engineer=_fourier_engineer(freqs=(1, 3, 5, 7)),
        noise_level=noise, meta={"plot_1d": True},
    )


# --------------------------------------------------------------------------- #
# 10. 阶跃 (树的主场, 线性回归与之对照)
# --------------------------------------------------------------------------- #
def make_step(n=1200, noise=0.1, seed=9) -> Dataset:
    rng = _rng(seed)
    x = rng.uniform(-3, 3, size=(n, 1))
    signal = np.where(x[:, 0] > 0, 1.0, -1.0) + np.where(x[:, 0] > 1.5, 1.0, 0.0)
    sigma = _signal_noise_scale(signal, noise)
    y = signal + rng.normal(0, sigma, size=n)

    return Dataset(
        name="step",
        description="y = 阶跃函数 + ε  (分段常数, 树的天然主场)",
        X=x, y=y, feature_names=["x0"],
        true_func=lambda Xq: np.where(Xq[:, 0] > 0, 1.0, -1.0)
        + np.where(Xq[:, 0] > 1.5, 1.0, 0.0),
        engineer=_poly_engineer(degree=3),
        noise_level=noise, meta={"plot_1d": True},
    )


# --------------------------------------------------------------------------- #
# 特征工程器 (回答: 特征工程能帮多大忙?)
# --------------------------------------------------------------------------- #
def _poly_engineer(degree=2, interaction_only=False):
    from sklearn.preprocessing import PolynomialFeatures

    def fn(X):
        pf = PolynomialFeatures(degree=degree, interaction_only=interaction_only,
                                include_bias=False)
        Xe = pf.fit_transform(X)
        return Xe, list(pf.get_feature_names_out())

    return fn


def _log_engineer():
    def fn(X):
        Xe = np.column_stack([X, np.log(np.clip(np.abs(X) + 1e-6, 1e-6, None))])
        names = [f"x{i}" for i in range(X.shape[1])] + [f"log|x{i}|" for i in range(X.shape[1])]
        return Xe, names

    return fn


def _mixed_engineer():
    def fn(X):
        logx1 = np.log(np.clip(X[:, 1], 1e-6, None))
        Xe = np.column_stack([X, logx1, X[:, 0] * logx1])
        return Xe, ["x0", "x1", "log(x1)", "x0·log(x1)"]

    return fn


def _fourier_engineer(freqs=(1, 2, 3)):
    def fn(X):
        cols, names = [X], [f"x{i}" for i in range(X.shape[1])]
        for f in freqs:
            cols.append(np.sin(f * X))
            cols.append(np.cos(f * X))
            names += [f"sin({f}x{i})" for i in range(X.shape[1])]
            names += [f"cos({f}x{i})" for i in range(X.shape[1])]
        return np.column_stack(cols), names

    return fn


# --------------------------------------------------------------------------- #
# 注册表
# --------------------------------------------------------------------------- #
ALL_DATASETS: dict[str, Callable[..., Dataset]] = {
    "linear_additive": make_linear_additive,
    "linear_noise_before": make_linear_noise_before,
    "interaction": make_interaction,
    "quadratic": make_quadratic,
    "logarithmic": make_logarithmic,
    "high_noise": make_high_noise,
    "mixed_interaction": make_mixed_interaction,
    "periodic": make_periodic,
    "high_freq_periodic": make_high_freq_periodic,
    "step": make_step,
}


def build_all(seed_offset: int = 0) -> list[Dataset]:
    return [fn(seed=i + seed_offset) for i, fn in enumerate(ALL_DATASETS.values())]


def build(name: str, **kwargs) -> Dataset:
    if name not in ALL_DATASETS:
        raise KeyError(f"未知数据集 {name!r}; 可选: {list(ALL_DATASETS)}")
    return ALL_DATASETS[name](**kwargs)
