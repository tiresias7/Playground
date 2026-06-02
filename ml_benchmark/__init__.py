"""ml_benchmark: 对比四种回归方法的合成数据基准与可视化工具。

模块:
  datasets   合成数据集生成器 (线性/交互/非线性/周期/高噪声/...)
  models     四个模型封装与超参搜索空间
  benchmark  模型 × 数据集 对比 + 超参网格搜索
  tuning     超参数单轴扫描 / GBT 逐棵树早停曲线
  visualize  生成交互式 Plotly HTML 报告
  run        一键生成完整报告 (python -m ml_benchmark.run)
"""
from . import benchmark, datasets, models, tuning, visualize  # noqa: F401

__all__ = ["datasets", "models", "benchmark", "tuning", "visualize"]
