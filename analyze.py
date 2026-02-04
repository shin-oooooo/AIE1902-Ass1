import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from ass1_core import (
    annual_metrics,
    corr_matrix,
    daily_returns,
    gaussian_kde_1d,
    load_bundle,
    normalize_prices,
    rolling_volatility,
)


def _ensure_plotly():
    import plotly.express as px
    import plotly.graph_objects as go

    return px, go


def _default_paths() -> Tuple[str, str]:
    out_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(out_dir, "data.json")
    return out_dir, json_path


def _write_df(df: pd.DataFrame, path: str):
    df.to_csv(path, encoding="utf-8-sig")


def _save_fig(fig, path: str):
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)


def _kde_figure(returns: pd.DataFrame, title: str):
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


def _normalized_price_figure(close: pd.DataFrame, title: str):
    px, _ = _ensure_plotly()
    norm = normalize_prices(close)
    fig = px.line(norm, x=norm.index, y=norm.columns, title=title)
    fig.update_layout(xaxis_title="date", yaxis_title="normalized close")
    return fig


def _rolling_vol_figure(returns: pd.DataFrame, title: str, window: int = 30):
    px, _ = _ensure_plotly()
    rv = rolling_volatility(returns, window=window)
    fig = px.line(rv, x=rv.index, y=rv.columns, title=title)
    fig.update_layout(xaxis_title="date", yaxis_title=f"{window}D rolling volatility (annualized)")
    return fig


def _corr_heatmap(corr: pd.DataFrame, title: str):
    px, _ = _ensure_plotly()
    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        text_auto=True,
        aspect="auto",
        title=title,
    )
    fig.update_layout(coloraxis_colorbar_title="corr")
    return fig


def analyze_one(out_dir: str, name: str, close: pd.DataFrame):
    close = close.dropna(how="all")
    returns = daily_returns(close).dropna(how="all")

    metrics = annual_metrics(returns)
    corr = corr_matrix(returns)

    _write_df(metrics, os.path.join(out_dir, f"{name}_metrics.csv"))
    _write_df(corr, os.path.join(out_dir, f"{name}_corr.csv"))

    price_fig = _normalized_price_figure(close, f"{name.upper()} normalized prices")
    kde_fig = _kde_figure(returns, f"{name.upper()} returns KDE")
    vol_fig = _rolling_vol_figure(returns, f"{name.upper()} 30D rolling volatility")
    corr_fig = _corr_heatmap(corr, f"{name.upper()} correlation heatmap")

    _save_fig(price_fig, os.path.join(out_dir, f"{name}_prices.html"))
    _save_fig(kde_fig, os.path.join(out_dir, f"{name}_returns_kde.html"))
    _save_fig(vol_fig, os.path.join(out_dir, f"{name}_rolling_vol.html"))
    _save_fig(corr_fig, os.path.join(out_dir, f"{name}_corr_heatmap.html"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["assets", "stocks", "universe", "all"], default="all")
    args = parser.parse_args()

    out_dir, json_path = _default_paths()
    bundle = load_bundle(json_path)

    if args.dataset in ["assets", "all"]:
        analyze_one(out_dir, "assets", bundle.close_assets)
    if args.dataset in ["stocks", "all"]:
        analyze_one(out_dir, "stocks", bundle.close_stocks)
    if args.dataset in ["universe", "all"]:
        analyze_one(out_dir, "universe", bundle.close_universe)


if __name__ == "__main__":
    main()
