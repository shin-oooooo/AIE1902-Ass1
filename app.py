# -*- coding: utf-8 -*-
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

from ass1_core import annual_metrics, corr_matrix, daily_returns, gaussian_kde_1d, get_prob_summary, load_bundle, normalize_prices, rolling_volatility

# Kronos integration
try:
    from kronos_predictor import kronos_forecast, KRONOS_AVAILABLE, run_kronos_optimization
except Exception as e:
    st.warning(f"Kronos 导入失败: {e}")
    kronos_forecast = None
    KRONOS_AVAILABLE = False
    run_kronos_optimization = None


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
    
    return f"""<div style="font-family: sans-serif; line-height: 1.6;"><table style="width: 100%; border-collapse: collapse;"><thead><tr style="background-color: #f8f9fa;"><th style="width: 60%; padding: 12px; border-bottom: 2px solid #ddd; text-align: left;">说明与投资建议（面向零基础）</th><th style="width: 40%; padding: 12px; border-bottom: 2px solid #ddd; text-align: left;">名词解释与原理拆解</th></tr></thead><tbody><!-- 1. 数据来源 --><tr><td style="padding: 12px; vertical-align: top; border-bottom: 1px solid #eee;"><strong>1. 数据从哪里来、怎么变成可用的“参数”</strong><br><br><strong>(1) 数据获取与清洗</strong><br>我们从 AkShare 接口获取了 {target_count} 个标的的历史日度收盘价。原始价格数据可能包含录入错误或极端的黑天鹅事件（如暴跌 99%），这会干扰模型判断。因此，我们使用 <strong>Z-Score</strong> 方法识别并剔除统计学意义上的离群点，确保输入数据的纯净性。<br><br><strong>(2) 数据集划分</strong><br>为了严谨验证模型能力，我们将时间序列切分为两段：<br>- <strong>训练集</strong>：用于让模型“学习”历史规律。<br>- <strong>测试集</strong>：用于“考试”，检验模型在未知未来的表现。<br>这能有效防止“死记硬背”历史答案（过拟合）。</td><td style="padding: 12px; vertical-align: top; background-color: #fafafa; border-bottom: 1px solid #eee; border-left: 1px solid #ddd;"><strong>Z-Score (标准分数)</strong><br><span style="color: #444; font-weight: bold;">[数据]</span> 历史收益率序列。<br><span style="color: #444; font-weight: bold;">[处理]</span> 计算序列均值 μ 与标准差 σ，将每个收益率 x 转化为 z = (x - μ) / σ。<br><span style="color: #444; font-weight: bold;">[效果]</span> 量化“偏离正常水平的程度”。若 |z|>3，判定为异常并剔除，防止极端值扭曲统计结果。<br><br><strong>训练集 / 测试集 (Train/Test Set)</strong><br><span style="color: #444; font-weight: bold;">[数据]</span> 完整的时间序列数据。<br><span style="color: #444; font-weight: bold;">[处理]</span> 按时间点一分为二：前 80% 仅用于计算参数，后 20% 仅用于验证预测。<br><span style="color: #444; font-weight: bold;">[效果]</span> 模拟真实的“未知未来”场景，确保评估出的模型能力不是靠“偷看答案”得来的。</td></tr><!-- 2. 图表分析 --><tr><td style="padding: 12px; vertical-align: top; border-bottom: 1px solid #eee;"><strong>2. 图表与统计指标如何影响结论</strong><br><br><strong>(1) 统一比较基准：日收益率</strong><br>不同标的价格（100元 vs 3000元）无法直接对比。我们计算 <strong>日收益率</strong>（每日涨跌百分比）作为所有分析的基础。<br><br><strong>(2) 风险与收益的可视化</strong><br>- <strong>归一化</strong>对比图：让所有资产从 1.0 开始起跑，直观看出谁跑得快（收益）、谁波动大（风险）。<br>- <strong>波动分布 (KDE)</strong>：展示收益率的分布情况。越集中的曲线代表越稳健。<br><br><strong>(3) 核心指标：年化夏普比率 (Sharpe Ratio)</strong><br>衡量“每一份风险换回了多少超额收益”。夏普比率越高，代表该资产的性价比越高。</td><td style="padding: 12px; vertical-align: top; background-color: #fafafa; border-bottom: 1px solid #eee; border-left: 1px solid #ddd;"><strong>归一化 (Normalization)</strong><br><span style="color: #444; font-weight: bold;">[原理]</span> P_norm = P_t / P_initial。<br><span style="color: #444; font-weight: bold;">[意义]</span> 消除价格绝对规模的影响，只看百分比变化。<br><br><strong>夏普比率 (Sharpe Ratio)</strong><br><span style="color: #444; font-weight: bold;">[公式]</span> (R_p - R_f) / σ_p。<br><span style="color: #444; font-weight: bold;">[意义]</span> 收益减去无风险收益（如国债），再除以波动的标准差。这是投资界的“性价比”金标准。</td></tr><!-- 3. 组合优化 --><tr><td style="padding: 12px; vertical-align: top; border-bottom: 1px solid #eee;"><strong>3. 怎么算出“最优”配置比例？</strong><br><br><strong>(1) 现代组合理论 (MPT)</strong><br>鸡蛋不要放在一个篮子里，但怎么放才最科学？我们通过计算资产间的 <strong>相关性</strong>（谁和谁同步涨跌），利用数学方法寻找在给定风险水平下收益最高的点。<br><br><strong>(2) 本次推荐的配置方案</strong><br>{weights_table}<br><br><strong>(3) 专家建议</strong><br>{_suggest_text(dataset, risk, weights)}</td><td style="padding: 12px; vertical-align: top; background-color: #fafafa; border-bottom: 1px solid #eee; border-left: 1px solid #ddd;"><strong>资产相关性 (Correlation)</strong><br><span style="color: #444; font-weight: bold;">[取值]</span> -1 到 +1。+1 代表完全同步，-1 代表完全反向。<br><span style="color: #444; font-weight: bold;">[意义]</span> 找到相关性低的资产组合，可以抵消部分波动，实现“1+1>2”的控风险效果。<br><br><strong>有效边界 (Efficient Frontier)</strong><br><span style="color: #444; font-weight: bold;">[原理]</span> 通过数万次模拟（蒙特卡洛法），在“风险-收益”坐标系中连成的一条曲线。<br><span style="color: #444; font-weight: bold;">[意义]</span> 曲线上的点代表了当前资产组合在数学上的极限最优解。</td></tr></tbody></table></div>"""


def _suggest_text(dataset: str, risk: str, weights: Dict[str, float]) -> str:
    if not weights:
        return "未找到权重结果，请先运行 Ass1\\run_models.py 生成 models.json。"
    top = max(weights.items(), key=lambda x: x[1])
    if dataset in ["assets", "universe"] and risk == "低" and top[0] != "AU0":
        return f"低风险偏好下，建议增加 AU0（黄金）占比；当前最高权重为 {top[0]}={top[1]:.0%}。"
    return f"{risk}风险偏好下，当前最高权重为 {top[0]}={top[1]:.0%}。"


def _plot_efficient_frontier(ret_sel: pd.DataFrame, sim_count: int = 5000):
    """
    Step 4: 影子博弈图表 - 开发中栏 Plotly 有效边界图
    """
    from portfolio import optimize_monte_carlo
    
    # 运行蒙特卡洛模拟获取散点数据
    if ret_sel.empty or len(ret_sel.columns) < 2:
        st.warning("⚠️ 资产数据不完整，无法计算有效边界。")
        return None

    try:
        # 统一使用 pandas 的 mean 和 cov，它们默认处理 NaN (pairwise deletion)
        # 这确保了模拟的散点簇与“当前组合”点位在同一个坐标系内
        mu_d = ret_sel.mean().to_numpy()
        cov_d = ret_sel.cov().to_numpy()
        
        # 填充 NaN 为 0 防止矩阵运算崩溃
        mu_d = np.nan_to_num(mu_d)
        cov_d = np.nan_to_num(cov_d)

        trading_days = 252
        
        rng = np.random.default_rng(42)
        w_all = rng.dirichlet(alpha=np.ones(len(ret_sel.columns)), size=sim_count)
        
        # 计算年化收益 and 波动
        mu_a = (w_all @ mu_d) * trading_days
        var_a = np.einsum("ij,jk,ik->i", w_all, cov_d, w_all) * trading_days
        vol_a = np.sqrt(np.maximum(var_a, 0.0))
        
        # 避免除以 0
        sharpe = np.zeros_like(mu_a)
        mask = vol_a > 1e-6
        sharpe[mask] = mu_a[mask] / vol_a[mask]
        
        if len(vol_a) == 0 or np.all(np.isnan(vol_a)):
            return None

        import plotly.graph_objects as go
        fig = go.Figure()
        
        # 绘制模拟散点
        fig.add_trace(go.Scatter(
            x=vol_a, y=mu_a,
            mode='markers',
            marker=dict(
                color=sharpe,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="夏普比率"),
                size=5,
                opacity=0.3
            ),
            name="模拟组合",
            hovertemplate="波动率: %{x:.2%}<br>预期收益: %{y:.2%}<br>夏普比率: %{marker.color:.2f}"
        ))
        
        # 找到特殊点
        if not np.all(np.isnan(sharpe)):
            idx_max_sharpe = np.nanargmax(sharpe)
            fig.add_trace(go.Scatter(
                x=[vol_a[idx_max_sharpe]], y=[mu_a[idx_max_sharpe]],
                mode='markers',
                marker=dict(color='red', size=15, symbol='star'),
                name="最大夏普点 (最优解)",
                hovertemplate="<b>最优建议点</b><br>波动率: %{x:.2%}<br>收益: %{y:.2%}"
            ))
            
        if not np.all(np.isnan(vol_a)):
            idx_min_vol = np.nanargmin(vol_a)
            fig.add_trace(go.Scatter(
                x=[vol_a[idx_min_vol]], y=[mu_a[idx_min_vol]],
                mode='markers',
                marker=dict(color='blue', size=12, symbol='circle'),
                name="最小波动点 (保守解)"
            ))

        fig.update_layout(
            xaxis_title="年化波动率 (风险血压)",
            yaxis_title="年化收益率 (营养摄入)",
            template="plotly_white",
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        return fig
    except Exception as e:
        st.error(f"有效边界计算失败: {e}")
        return None

def _rebalance_weights(changed_sym: str, new_val: float, current_weights: Dict[str, float], active_symbols: List[str]):
    """
    当一个资产权重改变时，按比例调整其他资产权重，使总和保持为 1.0。
    """
    if len(active_symbols) <= 1:
        return {active_symbols[0]: 1.0} if active_symbols else {}
    
    new_weights = current_weights.copy()
    new_weights[changed_sym] = new_val
    
    other_symbols = [s for s in active_symbols if s != changed_sym]
    other_sum = sum(current_weights.get(s, 0.0) for s in other_symbols)
    
    target_other_sum = 1.0 - new_val
    
    if other_sum > 0:
        for s in other_symbols:
            new_weights[s] = (current_weights.get(s, 0.0) / other_sum) * target_other_sum
    else:
        # 如果原来其他权重都是0，则平分剩余权重
        for s in other_symbols:
            new_weights[s] = target_other_sum / len(other_symbols)
            
    return new_weights


def main():
    # Step 1: 核心框架搭建 - 配置三栏布局
    st.set_page_config(page_title="金融智能投资组合优化系统", layout="wide")
    
    # 自定义 CSS 实现资产泡泡样式及健康卡片动画
    st.markdown("""
        <style>
        .asset-bubble {
            display: inline-block;
            padding: 5px 12px;
            margin: 4px;
            border-radius: 20px;
            cursor: pointer;
            font-size: 0.85em;
            transition: all 0.3s;
            border: 1px solid #ddd;
        }
        .asset-active {
            background-color: #2e7d32;
            color: white;
            border-color: #1b5e20;
        }
        .asset-inactive {
            background-color: #f5f5f5;
            color: #9e9e9e;
            border-color: #e0e0e0;
        }
        /* 血压计高压闪烁动画 */
        @keyframes blink {
            0% { opacity: 1; border-color: red; }
            50% { opacity: 0.5; border-color: darkred; }
            100% { opacity: 1; border-color: red; }
        }
        .high-vol-blink {
            animation: blink 1s infinite;
            border-width: 15px !important;
        }
        /* 状态灯样式 */
        .status-light {
            height: 12px;
            width: 12px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 5px;
        }
        .light-green { background-color: #00ff00; }
        .light-yellow { background-color: #ffff00; }
        .light-red { background-color: #ff0000; }
        </style>
    """, unsafe_allow_html=True)

    # 初始化 Session State
    if 'active_assets' not in st.session_state:
        st.session_state['active_assets'] = []
    if 'status_code' not in st.session_state:
        st.session_state['status_code'] = 0 # 0: 绿, 1: 黄, 2: 红
    if 'user_weights' not in st.session_state:
        st.session_state['user_weights'] = {}

    data = _load_data()
    models = _load_models()
    
    # 侧边栏保持原有功能，但增加 Step 1 要求的参数引擎
    st.sidebar.title("🛠️ 参数引擎 (Control Suite)")
    
    # 1. 资产泡泡组件 (Asset Bubbles)
    st.sidebar.subheader("资产选择 (Asset Bubbles)")
    close_all = data["close_universe"]
    all_symbols = list(close_all.columns)
    
    if not st.session_state.get('active_assets'):
        st.session_state['active_assets'] = all_symbols

    # 渲染资产泡泡
    cols_bubbles = st.sidebar.columns(3) # 改为 3 列以适应侧边栏宽度
    for i, sym in enumerate(all_symbols):
        is_active = sym in st.session_state['active_assets']
        btn_type = "primary" if is_active else "secondary"
        if cols_bubbles[i % 3].button(f"{sym}", key=f"bubble_{sym}", type=btn_type, use_container_width=True):
            if is_active:
                if len(st.session_state['active_assets']) > 1: # 至少保留一个资产
                    st.session_state['active_assets'].remove(sym)
                else:
                    st.toast("⚠️ 至少需选择一个资产")
            else:
                st.session_state['active_assets'].append(sym)
            st.rerun()

    # 2. 动作按钮
    st.sidebar.markdown("---")
    col_add, col_del = st.sidebar.columns(2)
    with col_add:
        if st.button("➕ 新增", type="secondary", use_container_width=True):
            st.toast("正在集成异步下载功能 (Step 2)...")
    with col_del:
        if st.button("➖ 删除", type="primary", use_container_width=True):
            st.toast("激活删除模式：点击上方泡泡即可移除")

    # 3. 偏好参数
    st.sidebar.markdown("---")
    risk_val = st.sidebar.select_slider("风险偏好 (1-5)", options=[1, 2, 3, 4, 5], value=3)
    roll_window = st.sidebar.number_input("滚动窗口 (天)", min_value=5, max_value=252, value=30)
    sim_count = st.sidebar.slider("模拟次数", min_value=1000, max_value=100000, value=50000, step=1000)

    # 4. 初始化/更新权重逻辑
    risk_map = {1: "低", 2: "低", 3: "中", 4: "高", 5: "高"}
    risk_str = risk_map[risk_val]
    
    # 如果风险偏好改变，重置权重
    if 'last_risk_val' not in st.session_state:
        st.session_state['last_risk_val'] = risk_val
    
    if st.session_state['last_risk_val'] != risk_val:
        st.session_state['user_weights'] = _recommend_weights(models, "universe", risk_str)
        st.session_state['last_risk_val'] = risk_val

    if not st.session_state.get('user_weights'):
        st.session_state['user_weights'] = _recommend_weights(models, "universe", risk_str)

    # Calculate metrics for the selected assets
    selected = st.session_state['active_assets']
    close_sel = _subset_close(data["close_universe"], selected)
    ret_sel = daily_returns(close_sel).dropna(how="all")
    metrics_df = annual_metrics(ret_sel)
    corr_df = corr_matrix(ret_sel)
    
    # 权重联动回调函数
    def on_weight_change(sym=None):
        # 如果是通过 data_editor 改变的，sym 为 None
        if sym is None:
            # 获取 data_editor 的更新
            if "weight_editor" in st.session_state:
                edited_df = st.session_state["weight_editor"]["edited_rows"]
                # 处理数据编辑器的更新
                pass # 这种方式比较复杂，我们直接在主逻辑中处理数据编辑器
        else:
            # 兼容旧逻辑
            side_key = f"side_w_{sym}"
            mid_key = f"mid_w_{sym}"
            old_val = st.session_state['user_weights'].get(sym, 0.0)
            new_val_side = st.session_state.get(side_key, old_val)
            new_val_mid = st.session_state.get(mid_key, old_val)
            new_val = new_val_side if new_val_side != old_val else new_val_mid
            st.session_state['user_weights'] = _rebalance_weights(sym, new_val, st.session_state['user_weights'], selected)

    # 计算组合层面指标 (Portfolio Metrics)
    user_w_dict = st.session_state['user_weights']
    active_w = {s: user_w_dict.get(s, 0.0) for s in selected}
    w_sum = sum(active_w.values())
    if w_sum > 0:
        for s in selected:
            st.session_state['user_weights'][s] = active_w[s] / w_sum
        w_arr = np.array([st.session_state['user_weights'][s] for s in selected])
        from portfolio import _portfolio_stats
        mu_p_ann, vol_p_ann, sharpe_p = _portfolio_stats(ret_sel.mean().values, ret_sel.cov().values, w_arr)
    else:
        for s in selected:
            st.session_state['user_weights'][s] = 1.0 / len(selected)
        mu_p_ann, vol_p_ann, sharpe_p = 0.0, 0.0, 0.0

    # 左栏资产配置面板 (改为饼图 + 数据编辑器)
    st.sidebar.subheader("资产配比 (Weights)")
    weight_df = pd.DataFrame([
        {"资产": s, "权重 (%)": float(st.session_state['user_weights'].get(s, 0.0)) * 100} 
        for s in selected
    ])
    
    # 侧边栏饼图展示
    fig_side_pie = px.pie(weight_df, values="权重 (%)", names="资产", hole=0.4)
    fig_side_pie.update_layout(showlegend=False, height=250, margin=dict(l=0, r=0, t=0, b=0))
    st.sidebar.plotly_chart(fig_side_pie, use_container_width=True)

    # 侧边栏数据编辑器
    edited_weight_df = st.sidebar.data_editor(
        weight_df,
        column_config={
            "权重 (%)": st.column_config.NumberColumn(
                "权重 (%)",
                help="调整资产权重 (0-100)",
                min_value=0,
                max_value=100,
                step=1,
                format="%d%%"
            ),
            "资产": st.column_config.TextColumn("资产", disabled=True)
        },
        hide_index=True,
        use_container_width=True,
        key="weight_editor_sidebar"
    )

    # 如果数据被编辑，更新权重
    if not weight_df.equals(edited_weight_df):
        # 找出哪个被改了
        for i, row in edited_weight_df.iterrows():
            sym = row["资产"]
            new_val = row["权重 (%)"] / 100.0
            if new_val != weight_df.iloc[i]["权重 (%)"] / 100.0:
                st.session_state['user_weights'] = _rebalance_weights(sym, new_val, st.session_state['user_weights'], selected)
                st.rerun()

    # 5. 系统状态逻辑 (Status Code Logic)
    # 根据模型信号一致性更新状态灯
    def update_status_code():
        lgb_data = models.get("universe", {}).get("lightgbm", {})
        if not lgb_data or not lgb_data.get("available", False):
            return 1 # 数据缺失，黄灯警告
        
        cls_res = lgb_data.get("classification", {})
        if not cls_res:
            return 1
            
        # 简单逻辑：根据资产的平均预测准确率来决定系统灯
        # 如果选中资产的平均准确度低于 50%，设为黄灯；低于 40%，设为红灯
        if not selected:
            return 0
            
        accs = []
        for sym in selected:
            res = cls_res.get(sym, {})
            # 如果模型没有该资产的预测，默认给 0.5 (中立)
            accs.append(res.get("accuracy", 0.5))
        
        avg_acc = sum(accs) / len(accs)
        
        if avg_acc < 0.4:
            return 2 # 红灯：模型不可靠
        elif avg_acc < 0.55:
            return 1 # 黄灯：建议谨慎
        return 0 # 绿灯：系统运行稳健

    st.session_state['status_code'] = update_status_code()

    # 核心布局：参数(侧边)、沙盘(中)、评估(右)
    col_mid, col_right = st.columns([2, 1])

    with col_mid:
        st.title("🏗️ 决策沙盘 (Decision Sandbox)")
        
        # 状态层（Overlay）
        status_colors = {0: "#00FF0033", 1: "#FFFF0033", 2: "#FF000033"}
        status_labels = {0: "🟢 系统运行稳健", 1: "🟡 建议谨慎操作", 2: "🔴 逻辑严重紊乱"}
        
        st.markdown(f"""
            <div style="background-color: {status_colors[st.session_state['status_code']]}; padding: 15px; border-radius: 10px; border: 2px solid #ddd; margin-bottom: 20px;">
                <h3 style="margin:0;">{status_labels[st.session_state['status_code']]}</h3>
                <p style="margin:5px 0 0 0;">{ "当前模型信号一致，建议参考最优解。" if st.session_state['status_code'] == 0 else ("市场波动剧烈，建议降低总仓位至 30% 以下。" if st.session_state['status_code'] == 1 else "模型间存在严重信号冲突，请谨慎参考以下预测数据。") }</p>
            </div>
        """, unsafe_allow_html=True)

        # 移除决策锁定逻辑
        # if st.session_state['status_code'] == 2:
        #     st.stop()
        
        # 影子博弈布局
        sandbox_col1, sandbox_col2 = st.columns([1, 1.5])
        
        with sandbox_col1:
            st.subheader("资产配置与预测")
            # 每一行显示一个资产：状态灯、精简预测条
            for sym in selected:
                with st.container():
                    # 状态灯与精简预测条
                    mu_sim = ret_sel[sym].mean() if sym in ret_sel.columns else 0.0
                    sigma_sim = ret_sel[sym].std() if sym in ret_sel.columns else 0.01
                    prob_text = get_prob_summary(mu_sim, sigma_sim)
                    
                    # 获取该资产的模型准确率
                    lgb_res = models.get("universe", {}).get("lightgbm", {}).get("classification", {}).get(sym, {})
                    acc = lgb_res.get("accuracy", 1.0)
                    
                    if acc < 0.4:
                        light_class = "light-red"
                    elif acc < 0.6:
                        light_class = "light-yellow"
                    else:
                        light_class = "light-green"
                    
                    st.markdown(f"""
                        <div style="display: flex; align-items: center; margin-bottom: 15px;">
                            <div style="width: 80px; font-weight: bold;">{sym}</div>
                            <span class="status-light {light_class}"></span>
                            <div style="background-color: #f0f2f6; padding: 5px 10px; border-radius: 5px; font-family: monospace; font-size: 0.85em; flex-grow: 1;">
                                {prob_text}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

        with sandbox_col2:
            st.subheader("📊 影子博弈 (Efficient Frontier)")
            fig_frontier = _plot_efficient_frontier(ret_sel, sim_count=sim_count)
            if fig_frontier:
                # 引力感点位
                user_w_arr = np.array([st.session_state['user_weights'].get(s, 0.0) for s in selected])
                if np.sum(user_w_arr) > 0:
                    user_w_arr = user_w_arr / np.sum(user_w_arr)
                    from portfolio import _portfolio_stats
                    
                    # 关键修复：_portfolio_stats 返回的已经是年化指标，无需再次乘以 252
                    mu_user, vol_user, _ = _portfolio_stats(ret_sel.mean().values, ret_sel.cov().values, user_w_arr)
                    fig_frontier.add_trace(go.Scatter(
                        x=[vol_user], y=[mu_user],
                        mode='markers+text', marker=dict(color='orange', size=12, symbol='x'),
                        name="当前组合", text=["你的位置"], textposition="top center"
                    ))
                st.plotly_chart(fig_frontier, use_container_width=True)
                st.info("💡 **引力感提示**：红星代表系统建议的最优解。若您的组合偏离过远，建议向红星靠拢。")

    with col_right:
        st.title("📋 健康卡片 (Health Card)")
        
        # 1. 血压计 (Volatility)
        st.subheader("💓 血压计 (Volatility)")
        vol = vol_p_ann # 使用组合年化波动率
        vol_color = "red" if vol > 0.3 else ("orange" if vol > 0.15 else "green")
        blink_class = "high-vol-blink" if vol > 0.3 else ""
        st.markdown(f"""
            <div class="{blink_class}" style="padding: 20px; border-radius: 50%; width: 140px; height: 140px; border: 10px solid {vol_color}; display: flex; align-items: center; justify-content: center; margin: auto;">
                <div style="text-align: center;">
                    <div style="font-size: 1.8em; font-weight: bold;">{vol:.1%}</div>
                    <div style="font-size: 0.7em; color: gray;">组合年化波动</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # 血压计医嘱
        st.markdown(f"""
            <div style="background-color: #fff4f4; padding: 10px; border-radius: 8px; margin-top: 10px; font-size: 0.9em;">
                <strong>当前状态：</strong>{ "你的组合目前像是在暴风雨中的小船，起伏剧烈。" if vol > 0.3 else "目前组合生命体征平稳，风险控制良好。" }<br>
                <strong>核心病因：</strong>{ "资产配置过于集中或选择了高波动标的，缺乏避震缓冲。" if vol > 0.3 else "当前的权重分配有效抵消了单一风险。" }<br>
                <strong>行动处方：</strong>{ "建议增加低相关性资产（如黄金）的占比，降低组合总血压。" if vol > 0.3 else "继续保持当前的平衡态势。" }
            </div>
        """, unsafe_allow_html=True)
        
        # 2. 免疫热图 (Correlation)
        st.subheader("🛡️ 免疫热图 (Correlation)")
        # 仅显示权重 > 0 的资产，实现动态同步
        active_symbols_with_weight = [s for s in selected if user_w_dict.get(s, 0.0) > 0]
        if active_symbols_with_weight:
            filtered_corr = corr_df.loc[active_symbols_with_weight, active_symbols_with_weight]
            fig_corr = _heatmap_fig(filtered_corr, "")
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # 免疫热图医嘱
            avg_corr = filtered_corr.mean().mean()
        else:
            st.info("请先分配资产权重以查看免疫系统表现。")
            avg_corr = 0
        st.markdown(f"""
            <div style="background-color: #f0fdf4; padding: 10px; border-radius: 8px; margin-top: 10px; font-size: 0.9em;">
                <strong>当前状态：</strong>{ "资产间高度同步，一旦感冒就会集体发烧。" if avg_corr > 0.5 else "各器官免疫系统独立，具备良好的抗风险屏障。" }<br>
                <strong>核心病因：</strong>{ "所选标的行业过于集中，缺乏分散化。" if avg_corr > 0.5 else "资产种类丰富，互不干扰。" }<br>
                <strong>行动处方：</strong>{ "引入与当前组合负相关的资产，增强系统免疫力。" if avg_corr > 0.5 else "目前的隔离措施很到位。" }
            </div>
        """, unsafe_allow_html=True)
        
        # 3. 代谢曲线 (Sharpe)
        st.subheader("🔄 代谢曲线 (Sharpe)")
        # 这里展示组合的 Sharpe 和 收益率
        st.metric("组合年化收益 (营养摄入)", f"{mu_p_ann:.1%}")
        st.metric("组合夏普比率 (代谢效率)", f"{sharpe_p:.2f}")
        
        # 代谢曲线医嘱
        avg_sharpe = sharpe_p
        st.markdown(f"""
            <div style="background-color: #f0f7ff; padding: 10px; border-radius: 8px; margin-top: 10px; font-size: 0.9em;">
                <strong>当前状态：</strong>{ "目前的营养转化效率极低，每一份风险并没有换回相应的回报。" if avg_sharpe < 0.5 else "身体吸收能力很强，投入的每一份风险都在转化为实打实的收益。" }<br>
                <strong>核心病因：</strong>{ "风险收益比失衡，代谢效率低下。" if avg_sharpe < 0.5 else "当前的资产配比极具性价比。" }<br>
                <strong>行动处方：</strong>{ "参考决策沙盘中的'红星'点位调整配比，提升新陈代谢效率。" if avg_sharpe < 0.5 else "保持当前的锻炼强度。" }
            </div>
        """, unsafe_allow_html=True)

    # 兼容性收纳原有逻辑
    with st.expander("🔍 原始数据看板 (Legacy Console)"):
        st.subheader("基础信息")
        st.write(data.get("meta", {}))
        st.subheader("价格数据（清洗后）")
        st.dataframe(close_sel.tail(20))
        st.subheader("核心统计指标（年化）")
        st.dataframe(metrics_df)

    return # End of main

if __name__ == "__main__":
    main()
