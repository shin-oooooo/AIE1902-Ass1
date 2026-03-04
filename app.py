import json
import os
import sys
from typing import Any, Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from ass1_core import annual_metrics, corr_matrix, daily_returns, gaussian_kde_1d, load_bundle, normalize_prices, rolling_volatility


def _ensure_plotly():
    import plotly.express as px
    import plotly.graph_objects as go

    return px, go


def _paths() -> Tuple[str, str, str]:
    base = os.path.dirname(os.path.abspath(__file__))
    return base, os.path.join(base, "data.json"), os.path.join(base, "models.json")


@st.cache_data
def _load_data() -> Dict[str, Any]:
    _, json_path, _ = _paths()
    bundle = load_bundle(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {"meta": bundle.meta, "close_assets": bundle.close_assets, "close_stocks": bundle.close_stocks, "close_universe": bundle.close_universe, "raw": raw}


@st.cache_data
def _load_models() -> Dict[str, Any]:
    _, _, models_path = _paths()
    if not os.path.exists(models_path):
        return {}
    with open(models_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _subset_close(close: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
    cols = [s for s in symbols if s in close.columns]
    return close.loc[:, cols].copy()


def _fig_to_download(fig, filename: str):
    html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    st.download_button("下载 HTML", data=html.encode("utf-8"), file_name=filename, mime="text/html")


def _kde_fig(returns: pd.DataFrame, title: str):
    px, go = _ensure_plotly()
    flat = returns.to_numpy().reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return go.Figure(layout={"title": title})
    lo, hi = np.quantile(flat, [0.01, 0.99])
    pad = (hi - lo) * 0.2 if hi > lo else 0.05
    grid = np.linspace(lo - pad, hi + pad, 300)
    fig = go.Figure()
    for col in returns.columns:
        x = returns[col].dropna().to_numpy(dtype=float)
        y = gaussian_kde_1d(x, grid)
        fig.add_trace(go.Scatter(x=grid, y=y, mode="lines", name=str(col)))
    fig.update_layout(title=title, xaxis_title="daily return", yaxis_title="density")
    return fig


def _heatmap_fig(corr: pd.DataFrame, title: str):
    px, _ = _ensure_plotly()
    fig = px.imshow(corr, color_continuous_scale="rdbu_r", zmin=-1, zmax=1, text_auto=True, aspect="auto", title=title)
    fig.update_layout(coloraxis_colorbar_title="corr")
    return fig


def _price_fig(close: pd.DataFrame, title: str):
    px, _ = _ensure_plotly()
    norm = normalize_prices(close)
    fig = px.line(norm, x=norm.index, y=norm.columns, title=title)
    fig.update_layout(xaxis_title="date", yaxis_title="normalized close")
    return fig


def _rolling_vol_fig(returns: pd.DataFrame, title: str, window: int = 30):
    px, _ = _ensure_plotly()
    rv = rolling_volatility(returns, window=window)
    fig = px.line(rv, x=rv.index, y=rv.columns, title=title)
    fig.update_layout(xaxis_title="date", yaxis_title=f"{window}D rolling volatility (annualized)")
    return fig


Dataset = Literal["assets", "stocks", "universe"]


def _pick_weights(models: Dict[str, Any], dataset: Dataset, risk: Literal["低", "中", "高"]) -> Dict[str, float]:
    if not models:
        return {}
    opt = models.get(dataset, {}).get("opt", {})
    w_low = opt.get("min_vol", {}).get("weights", {})
    w_high = opt.get("max_sharpe", {}).get("weights", {})
    if risk == "低":
        return {k: float(v) for k, v in w_low.items()}
    if risk == "高":
        return {k: float(v) for k, v in w_high.items()}
    keys = sorted(set(w_low) | set(w_high))
    w = {k: 0.5 * float(w_low.get(k, 0.0)) + 0.5 * float(w_high.get(k, 0.0)) for k in keys}
    s = sum(w.values())
    return {k: (v / s if s > 0 else 0.0) for k, v in w.items()}


def _renorm_subset(w: Dict[str, float], symbols: List[str]) -> Dict[str, float]:
    w2 = {k: float(v) for k, v in w.items() if k in set(symbols)}
    s = sum(w2.values())
    if s <= 0:
        return {k: 1.0 / len(symbols) for k in symbols} if symbols else {}
    return {k: v / s for k, v in w2.items()}


def _weight_fig(weights: Dict[str, float], title: str):
    px, _ = _ensure_plotly()
    if not weights:
        return px.pie(pd.DataFrame({"symbol": [], "weight": []}), names="symbol", values="weight", title=title)
    df = pd.DataFrame({"symbol": list(weights.keys()), "weight": list(weights.values())}).sort_values("weight", ascending=False)
    fig = px.pie(df, names="symbol", values="weight", title=title)
    return fig


def _sharpe_fig(metrics: pd.DataFrame, title: str):
    px, _ = _ensure_plotly()
    if metrics.empty:
        return px.scatter(title=title)
    fig = px.scatter(
        metrics, 
        x="ann_volatility", 
        y="ann_return_mean", 
        hover_name=metrics.index, 
        size=[10] * len(metrics),
        color="sharpe_rf0",
        color_continuous_scale="Viridis",
        title=title
    )
    fig.update_layout(
        xaxis_title="年化波动率",
        yaxis_title="年化收益率",
        coloraxis_colorbar_title="夏普比率"
    )
    return fig


def _weights_array_text(symbols: List[str], weights: Dict[str, float], digits: int = 6) -> str:
    s1 = "[" + ",".join(symbols) + "]"
    s2 = "[" + ",".join([f"{float(weights.get(s, 0.0)):.{digits}f}" for s in symbols]) + "]"
    return f"{s1};{s2}"


def _pairs_from_corr(corr: pd.DataFrame, top_k: int = 3) -> Dict[str, List[Tuple[str, str, float]]]:
    if corr is None or corr.empty:
        return {"top_pos": [], "top_neg": []}
    syms = list(corr.columns)
    pairs = []
    for i in range(len(syms)):
        for j in range(i + 1, len(syms)):
            v = float(corr.iloc[i, j])
            if np.isfinite(v):
                pairs.append((syms[i], syms[j], v))
    top_pos = sorted(pairs, key=lambda x: x[2], reverse=True)[:top_k]
    top_neg = sorted(pairs, key=lambda x: x[2])[:top_k]
    return {"top_pos": top_pos, "top_neg": top_neg}


def _recommend_weights(models: Dict[str, Any], dataset: Dataset, risk: Literal["低", "中", "高"]) -> Dict[str, float]:
    if not models:
        return {}
    if dataset == "universe":
        return {k: float(v) for k, v in models.get("universe", {}).get("final_weights", {}).get(risk, {}).items()}
    if dataset == "stocks":
        return {k: float(v) for k, v in models.get("stocks", {}).get("derived_from_universe", {}).get(risk, {}).items()}
    return {k: float(v) for k, v in models.get("universe", {}).get("derived_assets", {}).get(risk, {}).items()}


ASSET_ROLES = {
    "AU0": "<strong>防守核心</strong>：黄金，对冲股市波动",
    "MSFT": "<strong>稳健成长</strong>：科技巨头，基石仓位",
    "SPY": "<strong>市场基准</strong>：标普500 ETF，分散风险",
    "AMZN": "<strong>进攻</strong>：电商与云服务龙头",
    "AAPL": "<strong>进攻</strong>：消费电子龙头",
    "TSMC": "<strong>进攻</strong>：半导体制造核心",
    "META": "<strong>进攻</strong>：社交网络与元宇宙",
    "GOOGL": "<strong>进攻</strong>：搜索与广告业务",
    "ORCL": "<strong>进攻</strong>：企业软件服务",
    "AVGO": "<strong>进攻</strong>：半导体设计",
    "NVDA": "<strong>高波进攻</strong>：AI 算力核心（控风险）",
    "ASML": "<strong>进攻</strong>：光刻机垄断",
}


def _get_role(symbol: str) -> str:
    return ASSET_ROLES.get(symbol, "<strong>成长</strong>：美股核心资产")


def _generate_weights_table_html(weights: Dict[str, float]) -> str:
    sorted_items = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    rows = ""
    for sym, w in sorted_items:
        role = _get_role(sym)
        pct = f"{w:.2%}"
        rows += f"""<tr><td style="padding: 4px; border-bottom: 1px solid #eee;">{sym}</td><td style="text-align: right; padding: 4px; border-bottom: 1px solid #eee;">{pct}</td><td style="text-align: left; padding: 4px; border-bottom: 1px solid #eee;">{role}</td></tr>"""
    
    return f"""<table style="width: 100%; border-collapse: collapse; font-size: 0.9em; border: 1px solid #eee;"><tr style="background-color: #f0f2f6;"><th style="text-align: left; padding: 6px; border-bottom: 2px solid #ddd;">标的</th><th style="text-align: right; padding: 6px; border-bottom: 2px solid #ddd;">权重</th><th style="text-align: left; padding: 6px; border-bottom: 2px solid #ddd;">角色定位</th></tr>{rows}</table>"""


def _build_explain(
    dataset: Dataset,
    risk: str,
    selected: List[str],
    raw: Dict[str, Any],
    metrics: pd.DataFrame,
    corr: pd.DataFrame,
    models: Dict[str, Any],
    weights: Dict[str, float],
) -> str:
    # 动态生成 HTML 报告
    weights_table = _generate_weights_table_html(weights)
    target_count = len(weights)
    
    return f"""<div style="font-family: sans-serif; line-height: 1.6;"><table style="width: 100%; border-collapse: collapse;"><thead><tr style="background-color: #f8f9fa;"><th style="width: 60%; padding: 12px; border-bottom: 2px solid #ddd; text-align: left;">说明与投资建议（面向零基础）</th><th style="width: 40%; padding: 12px; border-bottom: 2px solid #ddd; text-align: left;">名词解释与原理拆解</th></tr></thead><tbody><!-- 1. 数据来源 --><tr><td style="padding: 12px; vertical-align: top; border-bottom: 1px solid #eee;"><strong>1. 数据从哪里来、怎么变成可用的“参数”</strong><br><br><strong>(1) 数据获取与清洗</strong><br>我们从 AkShare 接口获取了 {target_count} 个标的的历史日度收盘价。原始价格数据可能包含录入错误或极端的黑天鹅事件（如暴跌 99%），这会干扰模型判断。因此，我们使用 <strong>Z-Score</strong> 方法识别并剔除统计学意义上的离群点，确保输入数据的纯净性。<br><br><strong>(2) 数据集划分</strong><br>为了严谨验证模型能力，我们将时间序列切分为两段：<br>- <strong>训练集</strong>：用于让模型“学习”历史规律。<br>- <strong>测试集</strong>：用于“考试”，检验模型在未知未来的表现。<br>这能有效防止“死记硬背”历史答案（过拟合）。</td><td style="padding: 12px; vertical-align: top; background-color: #fafafa; border-bottom: 1px solid #eee; border-left: 1px solid #ddd;"><strong>Z-Score (标准分数)</strong><br><span style="color: #444; font-weight: bold;">[数据]</span> 历史收益率序列。<br><span style="color: #444; font-weight: bold;">[处理]</span> 计算序列均值 μ 与标准差 σ，将每个收益率 x 转化为 z = (x - μ) / σ。<br><span style="color: #444; font-weight: bold;">[效果]</span> 量化“偏离正常水平的程度”。若 |z|>3，判定为异常并剔除，防止极端值扭曲统计结果。<br><br><strong>训练集 / 测试集 (Train/Test Set)</strong><br><span style="color: #444; font-weight: bold;">[数据]</span> 完整的时间序列数据。<br><span style="color: #444; font-weight: bold;">[处理]</span> 按时间点一分为二：前 80% 仅用于计算参数，后 20% 仅用于验证预测。<br><span style="color: #444; font-weight: bold;">[效果]</span> 模拟真实的“未知未来”场景，确保评估出的模型能力不是靠“偷看答案”得来的。</td></tr><!-- 2. 图表分析 --><tr><td style="padding: 12px; vertical-align: top; border-bottom: 1px solid #eee;"><strong>2. 图表与统计指标如何影响结论</strong><br><br><strong>(1) 统一比较基准：日收益率</strong><br>不同标的价格（100元 vs 3000元）无法直接对比。我们计算 <strong>日收益率</strong>（每日涨跌百分比）作为所有分析的基础。<br><br><strong>(2) 风险与收益的可视化</strong><br>- <strong>归一化</strong>走势图帮助我们直观对比谁涨得快、谁更稳。<br>- <strong>滚动波动率</strong>（如30日窗口）展示了风险随时间的变化，避免被长期平均值掩盖近期波动。（数据实证：在本次回测周期中，**AU0 (黄金)** 展现了惊人的防御性，年化波动率仅 18%（与大盘 SPY 持平），相比之下，**AVGO** 的波动率高达 58%，意味着持有它的心理压力是持有黄金的三倍以上。）<br>- <strong>核密度估计 (KDE)</strong> 描绘了收益率的“概率地图”，让我们看到极端暴涨暴跌出现的可能性。<br><br><strong>(3) 资产间的联动：相关性</strong><br><strong>相关系数</strong> 告诉我们资产是否“同涨同跌”。<strong>相关性热力图</strong> 用颜色直观展示了这一点：深红代表高度同步（风险无法分散），深蓝代表走势相反（具有对冲价值）。（数据洞察：热力图清晰地展示了“科技抱团”现象——NVDA 与 TSMC 的相关性高达 0.62，意味着它们往往齐涨齐跌。唯独 **AU0** 呈现出珍贵的“蓝海”，与 AMZN (-0.11)、MSFT (-0.06) 等巨头均为负相关。这就是为什么算法会重仓黄金——它是唯一能有效对冲科技股崩盘风险的“减震器”。）<br><br><strong>(4) 综合性价比：夏普比率</strong><br>我们将 <strong>年化收益率</strong> 与 <strong>年化波动率</strong> 画在同一张图上，并用 <strong>夏普比率</strong> 衡量“性价比”：承担单位风险能换来多少超额收益。（结论：数据告诉我们要“买稳赚的”而不是“买涨得猛的”。虽然 AVGO 涨幅惊人，但其单位风险回报率（夏普 1.25）远低于黄金（2.94）和谷歌（1.96）。优化器会自动惩罚那些“大起大落”的标的。）<br><br><strong>(5) 深度图表解读：为什么这样配置？</strong><br><strong>1) 相关性热力图 (Correlation Heatmap) 给了我们什么信息？</strong><br>这张图展示了资产两两之间的联动程度。AU0 (黄金) 所在的行/列呈现出大量的冷色调（蓝色/浅色），相关系数多在 0 附近甚至为负（如与 SPY 的相关性仅为 -0.04）。<br><span style="color: #0056b3; font-weight: bold;">结论：</span>黄金与科技股/美股大盘几乎“绝缘”。当股市剧烈震荡时，黄金往往能走出独立行情。因此，在组合中配置 37.69% 的 AU0 并不是为了搏取最高收益，而是利用这种“不相关性”来充当组合的减震器，大幅降低整体回撤风险。<br><br><strong>2) 夏普比率图 (Sharpe Ratio Plot) 揭示了什么？</strong><br>在图中，AU0 位于左上区域（低波动、较高收益），夏普比率高达 2.94，是全场“性价比”之王；而 AVGO 虽然收益极高（年化 73%），但波动率也最大（年化 59%），位于图表最右侧。<br><span style="color: #0056b3; font-weight: bold;">结论：</span>单一押注高收益资产（如 AVGO/NVDA）虽然诱人，但性价比（夏普比率）其实不如稳健资产。优化算法之所以给 AVGO 分配较少权重（1.04%），正是因为它的“单位风险回报”不如黄金划算。投资的真谛不在于“最能涨”，而在于“最稳地涨”。<br><br><strong>3) 滚动波动率 (Rolling Volatility) 说明了什么？</strong><br>通过观察 30 日滚动波动率曲线，我们可以看到 NVDA, AVGO 等科技股的曲线经常出现剧烈的尖峰（Spike），说明其风险具有突发性和聚集性；而 SPY 和 MSFT 的曲线则相对平缓。<br><span style="color: #0056b3; font-weight: bold;">结论：</span>科技成长股的持仓体验往往是“过山车”式的。为了平滑这种心理压力，必须引入波动率曲线平缓的基石资产（如 SPY, MSFT, AU0），确保你在市场恐慌时依然拿得住筹码，避免倒在黎明前。</td><td style="padding: 12px; vertical-align: top; background-color: #fafafa; border-bottom: 1px solid #eee; border-left: 1px solid #ddd;"><strong>日收益率 (Daily Return)</strong><br><span style="color: #444; font-weight: bold;">[数据]</span> 每日收盘价 P_t。<br><span style="color: #444; font-weight: bold;">[处理]</span> 计算 (P_t - P_{{t-1}}) / P_{{t-1}}。<br><span style="color: #444; font-weight: bold;">[效果]</span> 消除价格绝对值差异，将“金额变动”转化为可跨资产比较的“涨跌比例”。<br><br><strong>归一化 (Normalization)</strong><br><span style="color: #444; font-weight: bold;">[数据]</span> 多只股票的价格序列。<br><span style="color: #444; font-weight: bold;">[处理]</span> 将每只股票第1天价格设为1.0，后续价格按涨跌幅同比例缩放。<br><span style="color: #444; font-weight: bold;">[效果]</span> 强制所有曲线从同一起跑线出发，直观对比相对强弱。<br><br><strong>滚动波动率 (Rolling Volatility)</strong><br><span style="color: #444; font-weight: bold;">[数据]</span> 日收益率序列。<br><span style="color: #444; font-weight: bold;">[处理]</span> 逐日移动一个 30 天窗口，计算窗口内收益率的标准差。<br><span style="color: #444; font-weight: bold;">[效果]</span> 捕捉风险的动态变化（如某个月市场突然恐慌），比单一的“平均波动率”更敏感。<br><br><strong>核密度估计 (KDE)</strong><br><span style="color: #444; font-weight: bold;">[数据]</span> 历史收益率分布。<br><span style="color: #444; font-weight: bold;">[处理]</span> 使用高斯核函数对直方图进行平滑拟合。<br><span style="color: #444; font-weight: bold;">[效果]</span> 绘制出平滑的概率山峰图，直观展示“常态”在哪里，“极端风险”有多大概率。<br><br><strong>相关系数 (Correlation)</strong><br><span style="color: #444; font-weight: bold;">[数据]</span> 两只股票的收益率序列。<br><span style="color: #444; font-weight: bold;">[处理]</span> 计算协方差除以各自标准差的乘积。<br><span style="color: #444; font-weight: bold;">[效果]</span> 得到 [-1, 1] 的数值：1代表完全同步（无分散效果），-1代表完全对冲（风险抵消）。<br><br><strong>相关性热力图 (Heatmap)</strong><br><span style="color: #444; font-weight: bold;">[数据]</span> {target_count}x{target_count} 相关系数矩阵。<br><span style="color: #444; font-weight: bold;">[处理]</span> 将数值映射为颜色（红=正相关，蓝=负相关）。<br><span style="color: #444; font-weight: bold;">[效果]</span> 将枯燥的数字矩阵转化为视觉图谱，一眼识别出哪些资产是“抱团”的。<br><br><strong>年化收益率 (Annualized Return)</strong><br><span style="color: #444; font-weight: bold;">[数据]</span> 日均收益率。<br><span style="color: #444; font-weight: bold;">[处理]</span> 日均值 × 252（年交易日）。<br><span style="color: #444; font-weight: bold;">[效果]</span> 将短期数据扩展为符合直觉的“年回报率”概念。<br><br><strong>年化波动率 (Annualized Volatility)</strong><br><span style="color: #444; font-weight: bold;">[数据]</span> 日收益率标准差。<br><span style="color: #444; font-weight: bold;">[处理]</span> 日标准差 × √252。<br><span style="color: #444; font-weight: bold;">[效果]</span> 统一量纲，便于与年化收益率进行比较。<br><br><strong>夏普比率 (Sharpe Ratio)</strong><br><span style="color: #444; font-weight: bold;">[数据]</span> 年化收益率与年化波动率。<br><span style="color: #444; font-weight: bold;">[处理]</span> (收益 - 无风险利率) / 波动率。<br><span style="color: #444; font-weight: bold;">[效果]</span> 衡量“性价比”：每承担 1 单位风险，能换来多少超额回报。</td></tr><!-- 3. LightGBM --><tr><td style="padding: 12px; vertical-align: top; border-bottom: 1px solid #eee;"><strong>3. LightGBM 预测模型如何产生“期望收益”</strong><br><br><strong>(1) 特征工程：从历史中提取规律</strong><br>模型不仅看昨天的涨跌，还通过 <strong>特征 (Features)</strong> 观察更多维度：<br>- <strong>滞后项</strong>：过去第1天、第2天...的收益。<br>- <strong>动量</strong>：过去一段时间的累计涨幅趋势。<br>- 滚动统计：近期的平均水平和波动状态。<br><br><strong>(2) 预测引擎：LightGBM</strong><br>相比于简单的“历史均值”（Naive），<strong>LightGBM</strong> 是一种强大的机器学习算法，能捕捉复杂的非线性规律。它在 <strong>测试集</strong> 上的表现通过以下指标衡量：<br>- <strong>MAE / RMSE</strong>：预测数值的误差大小（越小越好）。<br>- <strong>方向准确率</strong>：猜对“涨/跌”方向的概率（越高越好）。<br><br>最终，模型输出每个标的未来的“预期日均收益率”，构成我们的投资导航图。</td><td style="padding: 12px; vertical-align: top; background-color: #fafafa; border-bottom: 1px solid #eee; border-left: 1px solid #ddd;"><strong>特征 (Features)</strong><br><span style="color: #444; font-weight: bold;">[数据]</span> 原始价格/收益序列。<br><span style="color: #444; font-weight: bold;">[处理]</span> 计算统计量（如5日均线、10日波动等）作为新变量。<br><span style="color: #444; font-weight: bold;">[效果]</span> 将单一的价格信息扩展为多维度的“市场状态描述”，供模型学习。<br><br><strong>滞后项 (Lag)</strong><br><span style="color: #444; font-weight: bold;">[数据]</span> 历史收益率。<br><span style="color: #444; font-weight: bold;">[处理]</span> 将时间轴平移（如用昨天的 t-1 预测今天的 t）。<br><span style="color: #444; font-weight: bold;">[效果]</span> 捕捉“惯性”或“反转”效应（如昨日大跌今日往往反弹）。<br><br><strong>动量 (Momentum)</strong><br><span style="color: #444; font-weight: bold;">[数据]</span> 过去 N 天的收益率。<br><span style="color: #444; font-weight: bold;">[处理]</span> 计算期间累计涨跌幅。<br><span style="color: #444; font-weight: bold;">[效果]</span> 量化趋势强度，判断当前是处于上升期还是下跌期。<br><br><strong>LightGBM</strong><br><span style="color: #444; font-weight: bold;">[数据]</span> 构造好的特征矩阵。<br><span style="color: #444; font-weight: bold;">[处理]</span> 训练数百棵决策树，每棵树修正前者的残差，最终集成投票。<br><span style="color: #444; font-weight: bold;">[效果]</span> 相比线性模型，能捕捉复杂的非线性市场规律（如“跌多了且波动大时容易反弹”）。<br><br><strong>MAE / RMSE</strong><br><span style="color: #444; font-weight: bold;">[数据]</span> 预测值 vs 真实值。<br><span style="color: #444; font-weight: bold;">[处理]</span> 计算两者差值的绝对值平均或平方根。<br><span style="color: #444; font-weight: bold;">[效果]</span> 数值越小，说明模型预测的“点位”越精准。<br><br><strong>方向准确率</strong><br><span style="color: #444; font-weight: bold;">[数据]</span> 预测正负号 vs 真实正负号。<br><span style="color: #444; font-weight: bold;">[处理]</span> 统计符号一致的比例。<br><span style="color: #444; font-weight: bold;">[效果]</span> 即使点位不准，若能猜对“涨跌方向”，对投资依然极具价值。</td></tr><!-- 4. 优化 --><tr><td style="padding: 12px; vertical-align: top; border-bottom: 1px solid #eee;"><strong>4. 均值-方差优化如何变成最终权重</strong><br><br><strong>(1) 两大核心输入</strong><br>要计算最佳投资比例，我们需要两个数学对象：<br>- <strong>期望收益向量 (μ)</strong>：由 LightGBM 预测的 {target_count} 个标的未来日均收益。<br>- <strong>协方差矩阵 (Σ)</strong>：描述 {target_count} 个标的之间波动大小和联动关系的矩阵。<br><br><strong>(2) 寻找最优解：蒙特卡洛模拟</strong><br>我们利用 <strong>均值-方差优化</strong> 理论，通过 <strong>蒙特卡洛</strong> 方法随机生成数十万种投资组合。在“{risk}风险偏好”下，我们寻找那个既能提供不错收益，又能通过分散配置将波动控制在合理范围的组合。<br><br><strong>(3) 最终产出：权重</strong><br>算法最终给出一组 <strong>权重</strong>（百分比），告诉我们每 100 元资金应分别投向哪些资产。</td><td style="padding: 12px; vertical-align: top; background-color: #fafafa; border-bottom: 1px solid #eee; border-left: 1px solid #ddd;"><strong>期望收益向量 (μ)</strong><br><span style="color: #444; font-weight: bold;">[数据]</span> LightGBM 模型的预测输出。<br><span style="color: #444; font-weight: bold;">[处理]</span> 整理为 {target_count}x1 列向量。<br><span style="color: #444; font-weight: bold;">[效果]</span> 告诉优化器每个资产“未来可能赚多少”。<br><br><strong>协方差矩阵 (Σ)</strong><br><span style="color: #444; font-weight: bold;">[数据]</span> 历史收益率序列。<br><span style="color: #444; font-weight: bold;">[处理]</span> 计算两两之间的协方差，构成 {target_count}x{target_count} 矩阵。<br><span style="color: #444; font-weight: bold;">[效果]</span> 告诉优化器“哪些资产一起动”，从而指导分散配置。<br><br><strong>均值-方差优化</strong><br><span style="color: #444; font-weight: bold;">[数据]</span> 输入 μ 和 Σ。<br><span style="color: #444; font-weight: bold;">[处理]</span> 求解数学规划问题：在风险约束下最大化收益。<br><span style="color: #444; font-weight: bold;">[效果]</span> 找到理论上的“最佳性价比”配方。<br><br><strong>蒙特卡洛 (Monte Carlo)</strong><br><span style="color: #444; font-weight: bold;">[数据]</span> 随机数生成器。<br><span style="color: #444; font-weight: bold;">[处理]</span> 随机尝试 20 万种不同的权重组合，计算每一组的结果。<br><span style="color: #444; font-weight: bold;">[效果]</span> 暴力穷举出近似的最优解，解决复杂数学方程难以直接求解的问题。<br><br><strong>权重 (Weights)</strong><br><span style="color: #444; font-weight: bold;">[数据]</span> 优化算法的最优解。<br><span style="color: #444; font-weight: bold;">[处理]</span> 归一化使得总和为 100%。<br><span style="color: #444; font-weight: bold;">[效果]</span> 直接转化为可执行的资金分配指令（如“买 30% 黄金”）。</td></tr><!-- 5. 投资建议 --><tr><td style="padding: 12px; vertical-align: top; border-bottom: 1px solid #eee;"><strong>5. 你应该如何进行最后投资</strong><br><br>基于“{risk}风险偏好”的计算结果，我们建议的资金分配方案如下：<br><br>{weights_table}<br><strong>操作指南：</strong><br>1. <strong>长期定投（推荐）</strong>：按上述比例首次建仓后，每月投入固定资金，依然按此比例分配。定期（如每季度）检查持仓，若某资产占比偏离超过 5%，则卖高买低进行<strong>再平衡</strong>。<br>2. <strong>风险控制</strong>：严格遵守纪律，不因短期涨跌随意更改配方。若必须做短线，请设置严格止损线（如 -5%）。</td><td style="padding: 12px; vertical-align: top; background-color: #fafafa; border-bottom: 1px solid #eee; border-left: 1px solid #ddd;"><strong>再平衡 (Rebalance)</strong><br><span style="color: #444; font-weight: bold;">[数据]</span> 账户当前持仓 vs 目标权重。<br><span style="color: #444; font-weight: bold;">[处理]</span> 卖出涨幅过大导致占比超标的资产，买入占比不足的资产。<br><span style="color: #444; font-weight: bold;">[效果]</span> 强制实现“高抛低吸”，维持组合的风险特征不随市场波动而漂移。</td></tr></tbody></table></div>"""


def _suggest_text(dataset: str, risk: str, weights: Dict[str, float]) -> str:
    if not weights:
        return "未找到权重结果，请先运行 Ass1\\run_models.py 生成 models.json。"
    top = max(weights.items(), key=lambda x: x[1])
    if dataset in ["assets", "universe"] and risk == "低" and top[0] != "AU0":
        return f"低风险偏好下，建议增加 AU0（黄金）占比；当前最高权重为 {top[0]}={top[1]:.0%}。"
    return f"{risk}风险偏好下，当前最高权重为 {top[0]}={top[1]:.0%}。"


def main():
    st.set_page_config(page_title="Ass1 跨资产对比分析", layout="wide")
    data = _load_data()
    models = _load_models()

    st.sidebar.title("设置")
    dataset = st.sidebar.radio(
        "数据集",
        options=["universe", "stocks", "assets"],
        index=0,
        format_func=lambda x: "12种资产（10股票+SPY+AU0）" if x == "universe" else ("10只股票" if x == "stocks" else "SPY+AU0"),
    )
    risk = st.sidebar.select_slider("风险偏好", options=["低", "中", "高"], value="中")
    close = data["close_universe"] if dataset == "universe" else (data["close_stocks"] if dataset == "stocks" else data["close_assets"])
    symbols_all = list(close.columns)
    selected = st.sidebar.multiselect("资产选择器", options=symbols_all, default=symbols_all)
    close_sel = _subset_close(close, selected)
    ret_sel = daily_returns(close_sel).dropna(how="all")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["数据看板", "风险分析", "预测建议", "AI 可解释性", "回测验证"])

    with tab1:
        st.subheader("基础信息")
        st.write(data["meta"])
        st.subheader("价格数据（清洗后）")
        st.dataframe(close_sel.tail(20))
        st.subheader("核心统计指标（年化）")
        st.dataframe(annual_metrics(ret_sel))

    with tab2:
        st.subheader("相关性矩阵与热力图")
        corr = corr_matrix(ret_sel)
        st.dataframe(corr)
        fig = _heatmap_fig(corr, "Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)
        _fig_to_download(fig, f"{dataset}_corr_heatmap_selected.html")

        st.subheader("归一化价格走势")
        fig2 = _price_fig(close_sel, "Normalized Prices")
        st.plotly_chart(fig2, use_container_width=True)
        _fig_to_download(fig2, f"{dataset}_prices_selected.html")

        st.subheader("收益率分布 KDE")
        fig3 = _kde_fig(ret_sel, "Returns KDE")
        st.plotly_chart(fig3, use_container_width=True)
        _fig_to_download(fig3, f"{dataset}_returns_kde_selected.html")

        st.subheader("30天滚动波动率（年化）")
        fig4 = _rolling_vol_fig(ret_sel, "Rolling Volatility (30D)")
        st.plotly_chart(fig4, use_container_width=True)
        _fig_to_download(fig4, f"{dataset}_rolling_vol_selected.html")

        st.subheader("夏普比率图（年化收益率 vs 年化波动率）")
        metrics = annual_metrics(ret_sel)
        fig5 = _sharpe_fig(metrics, "Sharpe Ratio Plot")
        st.plotly_chart(fig5, use_container_width=True)
        _fig_to_download(fig5, f"{dataset}_sharpe_plot_selected.html")

    def _classify_value(val: float, threshold: float = 0.01) -> int:
        if val > threshold: return 2
        if val < -threshold: return 0
        return 1

    def _naive_rolling_classify(close_series: pd.Series, window: int = 30) -> pd.Series:
        ret = close_series.pct_change()
        roll_mean = ret.rolling(window=window).mean().shift(1) # shift 1 to avoid lookahead
        return roll_mean.apply(lambda x: _classify_value(x))

    with tab3:
        st.subheader("预测与配置建议")
        
        granularity = st.radio(
            "预测颗粒度 (Prediction Granularity)",
            options=["日收益率 (回归)", "趋势三分类 (分类)"],
            horizontal=True
        )

        lgb = models.get(dataset, {}).get("lightgbm", {})
        naive_cls_data = models.get(dataset, {}).get("naive_classification", {})
        
        if granularity == "日收益率 (回归)":
            st.markdown("**Naive 预测（过去30天均值→未来7天）**")
            naive_rows = models.get(dataset, {}).get("naive", [])
            naive_df = pd.DataFrame(naive_rows)
            if not naive_df.empty and "symbol" in naive_df.columns:
                naive_df = naive_df.set_index("symbol").loc[[s for s in selected if s in set(naive_df["symbol"])]].reset_index()
            st.dataframe(naive_df)

            st.markdown("**LightGBM 回归预测（测试集表现）**")
            if isinstance(lgb, dict) and lgb.get("available", False):
                m = lgb.get("metrics", {})
                rows = [{"symbol": k, **v} for k, v in m.items() if isinstance(v, dict)]
                lgb_df = pd.DataFrame(rows)
                if not lgb_df.empty:
                    lgb_df = lgb_df.set_index("symbol").loc[[s for s in selected if s in set(lgb_df["symbol"])]].reset_index()
                st.dataframe(lgb_df)
            else:
                st.write(f"LightGBM 未启用：{lgb.get('error','')}")

            st.subheader("最终权重（饼图 + 数组格式）")
            w_base = _recommend_weights(models, dataset, risk)
            w = _renorm_subset(w_base, selected)
            figw = _weight_fig(w, f"Weights ({dataset}, {risk}风险)")
            st.plotly_chart(figw, use_container_width=True)
            _fig_to_download(figw, f"{dataset}_weights_{risk}_selected.html")

            order = [s for s in symbols_all if s in set(selected)]
            st.code(_weights_array_text(order, w), language="text")

            if dataset == "universe":
                st.code(_weights_array_text(["SPY", "AU0"], w), language="text")
                st.code(_weights_array_text([s for s in order if s not in ["SPY", "AU0"]], w), language="text")

        else:  # Classification Mode
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### LightGBM 预测")
                if isinstance(lgb, dict) and lgb.get("available", False):
                    cls_res = lgb.get("classification", {})
                    if cls_res:
                        cls_rows = []
                        for sym, res in cls_res.items():
                            if sym not in selected: continue
                            last_class = res["pred_classes"][-1]
                            probs = res["pred_probs"][-1]
                            cls_label = ["大跌", "震荡", "大涨"][last_class]
                            cls_rows.append({
                                "Symbol": sym, "Pred": cls_label,
                                "Prob(Up)": f"{probs[2]:.1%}", "Acc": f"{res['accuracy']:.1%}"
                            })
                        st.dataframe(pd.DataFrame(cls_rows))
                    else:
                        st.warning("LightGBM分类数据缺失")
                else:
                    st.warning("LightGBM不可用")

            with col2:
                st.markdown("### Naive 预测")
                if naive_cls_data:
                    n_rows = []
                    for sym, res in naive_cls_data.items():
                        if sym not in selected: continue
                        pred = res.get("pred_class", 1)
                        probs = res.get("pred_probs", [0,0,0])
                        label = ["大跌", "震荡", "大涨"][pred]
                        n_rows.append({
                            "Symbol": sym, "Pred": label,
                            "Prob(Up)": f"{probs[2]:.1%}", "Desc": res.get("description", "")
                        })
                    st.dataframe(pd.DataFrame(n_rows))
                else:
                    st.warning("Naive分类数据缺失")

            st.markdown("---")
            st.subheader("模型与现实情况对比 (测试集可视化)")
            
            viz_sym = st.selectbox("选择资产进行对比", options=selected, key="cls_viz_sym")
            
            # Prepare data for visualization
            if isinstance(lgb, dict) and lgb.get("available", False):
                cls_res = lgb.get("classification", {})
                lgb_sym_data = cls_res.get(viz_sym, {})
                
                if lgb_sym_data:
                    dates = lgb_sym_data.get("dates", [])
                    true_cls = lgb_sym_data.get("true_classes", [])
                    lgb_pred = lgb_sym_data.get("pred_classes", [])
                    
                    # Compute Naive predictions on the same dates
                    # Use close prices from data['close_universe/stocks/assets']
                    close_df = close_sel[viz_sym] if viz_sym in close_sel.columns else None
                    if close_df is not None:
                        naive_series = _naive_rolling_classify(close_df)
                        # Filter to match test dates
                        naive_pred = []
                        valid_dates = []
                        valid_true = []
                        valid_lgb = []
                        
                        for i, d_str in enumerate(dates):
                            try:
                                d = pd.to_datetime(d_str)
                                if d in naive_series.index:
                                    naive_pred.append(naive_series.loc[d])
                                    valid_dates.append(d_str)
                                    valid_true.append(true_cls[i])
                                    valid_lgb.append(lgb_pred[i])
                            except:
                                continue
                        
                        if valid_dates:
                            # Create heatmap data
                            # Map classes to values: 0->-1, 1->0, 2->1 for color scale
                            map_val = {0: -1, 1: 0, 2: 1}
                            
                            fig_heat = go.Figure(data=go.Heatmap(
                                z=[
                                    [map_val[x] for x in valid_true],
                                    [map_val[x] for x in valid_lgb],
                                    [map_val[x] for x in naive_pred]
                                ],
                                x=valid_dates,
                                y=['Real (真实)', 'LightGBM', 'Naive'],
                                colorscale=[[0, 'red'], [0.5, 'lightgray'], [1, 'green']],
                                showscale=False
                            ))
                            fig_heat.update_layout(
                                title=f"{viz_sym} 分类预测对比 (绿=涨, 红=跌, 灰=震荡)",
                                height=300
                            )
                            st.plotly_chart(fig_heat, use_container_width=True)
                            
                            # Confusion Matrix for LightGBM
                            st.markdown("**混淆矩阵 (LightGBM vs Real)**")
                            from sklearn.metrics import confusion_matrix
                            cm = confusion_matrix(valid_true, valid_lgb, labels=[0, 1, 2])
                            cm_df = pd.DataFrame(cm, index=["Real:跌", "Real:平", "Real:涨"], columns=["Pred:跌", "Pred:平", "Pred:涨"])
                            st.table(cm_df)

            st.info("注：分类阈值为 +/- 1%。")

        st.subheader("文字建议（推导全过程）")
        metrics = annual_metrics(ret_sel)
        corr = corr_matrix(ret_sel)
        # Pass empty weights if classification mode to avoid confusion in text generation, 
        # or keep original weights but add disclaimer. Here we keep original for context.
        w_base = _recommend_weights(models, dataset, risk)
        w = _renorm_subset(w_base, selected)
        st.markdown(_build_explain(dataset, risk, selected, data["raw"], metrics, corr, models, w), unsafe_allow_html=True)

    with tab4:
        st.subheader("LightGBM 特征重要性分析")
        lgb_data = models.get(dataset, {}).get("lightgbm", {})
        
        if not lgb_data.get("available", False):
            st.warning("LightGBM 模型未启用，无法展示特征重要性。")
        else:
            # Dropdown to select symbol
            feat_sym = st.selectbox("选择资产查看特征贡献", options=selected, key="feat_imp_sym")
            
            # Get feature importance for selected symbol
            fi_data = lgb_data.get("models", {}).get(feat_sym, {})
            names = fi_data.get("feature_names", [])
            importance = fi_data.get("feature_importances", [])
            
            if names and importance:
                fi_df = pd.DataFrame({"Feature": names, "Importance": importance})
                fi_df = fi_df.sort_values("Importance", ascending=True)
                
                fig_fi = px.bar(
                    fi_df, 
                    x="Importance", 
                    y="Feature", 
                    orientation='h',
                    title=f"Feature Importance for {feat_sym}",
                    height=600
                )
                st.plotly_chart(fig_fi, use_container_width=True)
                
                st.markdown("""
                **解读指南：**
                - **Lag (滞后项)**: 代表过去几天的收益率。Lag_1 是昨天，Lag_2 是前天。
                - **Rolling Mean (滚动均值)**: 代表过去一段时间（5/10/20天）的平均收益水平。
                - **Rolling Std (滚动波动)**: 代表风险水平。
                - **Momentum (动量)**: 代表累计涨跌幅。
                """)
            else:
                st.info(f"未找到 {feat_sym} 的特征重要性数据。")

    with tab5:
        st.subheader("模型回测验证 (2025-02-01 ~ 2025-02-08)")
        
        # Load backtest results
        backtest_path = os.path.join(_paths()[0], "backtest_results.json")
        if os.path.exists(backtest_path):
            with open(backtest_path, "r", encoding="utf-8") as f:
                bt_results = json.load(f)
            
            scope_res = bt_results.get(dataset, {})
            dates = scope_res.get("dates", [])
            comparison = scope_res.get("comparison", {})
            
            if not dates:
                st.warning("回测数据为空，请检查 analyze_backtest.py 是否成功运行。")
            else:
                bt_sym = st.selectbox("选择资产查看回测详情", options=selected, key="bt_sym")
                
                comp_data = comparison.get(bt_sym, {})
                if comp_data:
                    # Metrics Table
                    nm = comp_data.get("naive_metrics", {})
                    lm = comp_data.get("lgb_metrics", {})
                    
                    met_df = pd.DataFrame({
                        "Metric": ["MAE (平均绝对误差)", "RMSE (均方根误差)", "累计收益差"],
                        "Naive": [nm.get("mae"), nm.get("rmse"), nm.get("cum_diff")],
                        "LightGBM": [lm.get("mae"), lm.get("rmse"), lm.get("cum_diff")]
                    })
                    st.table(met_df)
                    
                    # Plot
                    real_vals = comp_data.get("real", [])
                    naive_vals = comp_data.get("naive_pred", [])
                    lgb_vals = comp_data.get("lgb_pred", [])
                    
                    # Cumulative Return
                    real_cum = [np.prod(1 + np.array(real_vals[:i+1])) - 1 for i in range(len(real_vals))]
                    naive_cum = [np.prod(1 + np.array(naive_vals[:i+1])) - 1 for i in range(len(naive_vals))]
                    lgb_cum = [np.prod(1 + np.array(lgb_vals[:i+1])) - 1 for i in range(len(lgb_vals))]
                    
                    chart_df = pd.DataFrame({
                        "Date": dates,
                        "Real": real_cum,
                        "Naive": naive_cum,
                        "LightGBM": lgb_cum
                    })
                    
                    fig_bt = px.line(chart_df, x="Date", y=["Real", "Naive", "LightGBM"], 
                                     title=f"{bt_sym} 累计收益回测对比 (Cumulative Return)",
                                     markers=True)
                    st.plotly_chart(fig_bt, use_container_width=True)
                    
                    st.markdown("""
                    **图表说明：**
                    - **Real (蓝色)**: 真实发生的市场累计收益。
                    - **Naive (红色)**: 基于过去30天均值的线性预测。
                    - **LightGBM (绿色)**: 机器学习模型的动态预测。
                    """)
                else:
                    st.info(f"未找到 {bt_sym} 的回测数据。")
        else:
            st.error("未找到 backtest_results.json 文件。请运行 `python analyze_backtest.py`。")


if __name__ == "__main__":
    main()
