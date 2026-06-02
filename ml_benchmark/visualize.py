"""可视化: 生成自包含、可交互的 Plotly HTML 报告 (离线即可打开)。

报告包含四块:
  1. 总览热力图      —— 模型 × 数据集 的 test R², 一眼看出谁擅长什么
  2. 1D 拟合曲线     —— 散点 + 真相 + 各模型拟合, 直观看欠/过拟合 (图例可点选)
  3. 超参扫描曲线    —— train/test R² 随超参变化, 看偏差/方差权衡与早停
  4. 特征工程增益    —— 线性回归在 原始 vs 工程化特征 上的对比
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from . import models as mdl

_MODEL_COLOR = {spec.name: spec.color for spec in mdl.MODELS.values()}


# --------------------------------------------------------------------------- #
# 1. 总览热力图
# --------------------------------------------------------------------------- #
def fig_overview_heatmap(results, use_tuned=True):
    rows = _best_per_pair(results, use_tuned)
    models = [s.name for s in mdl.MODELS.values()]
    datasets = sorted({r["dataset"] for r in rows})
    z, text = [], []
    for m in models:
        zr, tr = [], []
        for d in datasets:
            v = next((r["test_r2"] for r in rows
                      if r["model"] == m and r["dataset"] == d), np.nan)
            zr.append(v)
            tr.append("" if np.isnan(v) else f"{v:.2f}")
        z.append(zr)
        text.append(tr)

    fig = go.Figure(go.Heatmap(
        z=z, x=datasets, y=models, text=text, texttemplate="%{text}",
        zmin=-0.2, zmax=1.0, colorscale="RdYlGn",
        colorbar=dict(title="test R²"),
        hovertemplate="模型=%{y}<br>数据集=%{x}<br>test R²=%{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title="① 总览: 各模型在各数据集上的 test R² (越绿越好)",
        xaxis=dict(tickangle=-35), height=380,
        margin=dict(l=170, t=60, b=120))
    return fig


# --------------------------------------------------------------------------- #
# 2. 1D 拟合曲线
# --------------------------------------------------------------------------- #
def fig_fit_curves(dataset_name, extras):
    e = extras["extras"]
    if not e.get("curves"):
        return None
    fig = go.Figure()

    sx, sy = e["scatter"]
    fig.add_trace(go.Scattergl(
        x=sx, y=sy, mode="markers", name="样本",
        marker=dict(size=4, color="lightgray", opacity=0.6)))

    if e.get("true_curve"):
        tx, ty = e["true_curve"]
        fig.add_trace(go.Scatter(
            x=tx, y=ty, mode="lines", name="真相 f(x)",
            line=dict(color="black", width=3, dash="dot")))

    for model_name, c in e["curves"].items():
        color = _MODEL_COLOR.get(model_name, "#555")
        fig.add_trace(go.Scatter(
            x=c["x"], y=c["y_tuned"], mode="lines",
            name=f"{model_name} (调优)",
            line=dict(color=color, width=2)))
        fig.add_trace(go.Scatter(
            x=c["x"], y=c["y_default"], mode="lines",
            name=f"{model_name} (默认)", visible="legendonly",
            line=dict(color=color, width=1, dash="dash")))

    fig.update_layout(
        title=f"② 拟合曲线 · {dataset_name} — {extras['dataset'].description}",
        xaxis_title="x0", yaxis_title="y", height=460,
        legend=dict(orientation="h", y=-0.18))
    return fig


# --------------------------------------------------------------------------- #
# 3. 超参扫描
# --------------------------------------------------------------------------- #
def fig_sweep(sweep_result):
    rows = sweep_result["rows"]
    xs = [str(r["value"]) for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["train_r2"] for r in rows],
                             mode="lines+markers", name="train R²",
                             line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=xs, y=[r["test_r2"] for r in rows],
                             mode="lines+markers", name="test R²",
                             line=dict(color="#d62728")))
    fig.update_layout(
        title=f"③ 超参扫描 · {sweep_result['model']} · {sweep_result['axis']} "
              f"(数据集 {sweep_result['dataset']})",
        xaxis_title=sweep_result["axis"], yaxis_title="R²", height=380,
        legend=dict(orientation="h", y=-0.2))
    return fig


def fig_gbt_staged(staged):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=staged["stages"], y=staged["train_r2"],
                             mode="lines", name="train R²",
                             line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=staged["stages"], y=staged["test_r2"],
                             mode="lines", name="test R²",
                             line=dict(color="#d62728")))
    fig.add_vline(x=staged["best_n_estimators"], line_dash="dot",
                  line_color="green",
                  annotation_text=f"早停≈{staged['best_n_estimators']}棵")
    fig.update_layout(
        title=f"③' GBT 逐棵树曲线 · {staged['dataset']} "
              f"(lr={staged['learning_rate']}, depth={staged['max_depth']}) "
              f"— test 见顶后再加树即过拟合",
        xaxis_title="树的数量 (n_estimators)", yaxis_title="R²", height=380,
        legend=dict(orientation="h", y=-0.2))
    return fig


# --------------------------------------------------------------------------- #
# 4. 特征工程增益
# --------------------------------------------------------------------------- #
def fig_feature_engineering(all_extras):
    names, raw, eng = [], [], []
    for dname, payload in all_extras.items():
        fe = payload["extras"].get("feature_engineering")
        if not fe:
            continue
        names.append(dname)
        raw.append(fe["raw_r2"])
        eng.append(fe["eng_r2"])
    if not names:
        return None
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=raw, name="线性回归 · 原始特征",
                         marker_color="#9ecae1"))
    fig.add_trace(go.Bar(x=names, y=eng, name="线性回归 · 工程化特征",
                         marker_color="#08519c"))
    fig.update_layout(
        title="④ 特征工程能帮多大忙? (线性回归 test R²)",
        barmode="group", xaxis=dict(tickangle=-35),
        yaxis_title="test R²", height=420,
        margin=dict(b=130), legend=dict(orientation="h", y=-0.35))
    return fig


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #
def _best_per_pair(results, use_tuned):
    """每个 (dataset, model) 取调优结果(若有), 否则默认结果。"""
    by_pair: dict[tuple, dict] = {}
    for r in results:
        d = r.__dict__ if hasattr(r, "__dict__") else r
        key = (d["dataset"], d["model"])
        cur = by_pair.get(key)
        if cur is None:
            by_pair[key] = d
        elif use_tuned and d["tuned"] and not cur["tuned"]:
            by_pair[key] = d
    return list(by_pair.values())


# --------------------------------------------------------------------------- #
# 报告组装: 多个 figure -> 单个自包含 HTML
# --------------------------------------------------------------------------- #
def build_report(figs_with_titles, out_path, page_title="ML 模型基准报告"):
    import plotly.io as pio

    sections = []
    for fig in figs_with_titles:
        if fig is None:
            continue
        sections.append(pio.to_html(fig, full_html=False,
                                    include_plotlyjs=False))
    body = "\n<hr style='margin:32px 0;border:none;border-top:1px solid #eee'>\n".join(sections)
    html = f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>{page_title}</title>
<script type="text/javascript">{get_plotlyjs()}</script>
<style>
 body {{ font-family: -apple-system, "Segoe UI", "PingFang SC", sans-serif;
        max-width: 1100px; margin: 24px auto; padding: 0 16px; color:#222; }}
 h1 {{ font-size: 22px; }} .intro {{ color:#555; line-height:1.6; }}
</style></head><body>
<h1>{page_title}</h1>
<p class="intro">对比 Linear Regression / Decision Tree / Random Forest /
Gradient Boosted Trees 四种回归方法, 在多种刻意设计的合成数据集上的表现,
以及关键超参数的影响。图例可点击切换曲线显隐。</p>
{body}
</body></html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path
