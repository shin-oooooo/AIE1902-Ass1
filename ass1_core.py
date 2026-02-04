import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd


Section = Literal["assets", "stocks", "universe"]


@dataclass(frozen=True)
class DataBundle:
    meta: Dict[str, Any]
    close_assets: pd.DataFrame
    close_stocks: pd.DataFrame
    close_universe: pd.DataFrame


def load_bundle(json_path: str) -> DataBundle:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    close_assets = _records_to_wide_close(data.get("assets", {}))
    close_stocks = _records_to_wide_close(data.get("stocks", {}))
    meta = data.get("meta", {})
    universe_symbols = meta.get("universe", []) or [
        "NVDA",
        "MSFT",
        "TSMC",
        "GOOGL",
        "AMZN",
        "AAPL",
        "ASML",
        "META",
        "AVGO",
        "ORCL",
        "SPY",
        "AU0",
    ]

    close_universe = close_stocks.copy()
    for sym in universe_symbols:
        if sym in close_universe.columns:
            continue
        if sym in close_assets.columns:
            close_universe[sym] = close_assets[sym]
    close_universe = close_universe.loc[:, [c for c in universe_symbols if c in close_universe.columns]].copy()

    if "universe" not in meta:
        meta = dict(meta)
        meta["universe"] = universe_symbols
    return DataBundle(meta=meta, close_assets=close_assets, close_stocks=close_stocks, close_universe=close_universe)


def _records_to_wide_close(records_map: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    frames = []
    for sym, rows in records_map.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        if "date" not in df.columns or "close" not in df.columns:
            continue
        df = df[["date", "close"]].copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["date", "close"])
        df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        df = df.set_index("date").rename(columns={"close": sym})
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1).sort_index()
    return out


def slice_symbols(close: pd.DataFrame, symbols: Iterable[str]) -> pd.DataFrame:
    cols = [s for s in symbols if s in close.columns]
    return close.loc[:, cols].copy()


def daily_returns(close: pd.DataFrame) -> pd.DataFrame:
    if close is None or close.empty:
        return pd.DataFrame()
    out = close.pct_change()
    out = out.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    return out


def annual_metrics(returns: pd.DataFrame, trading_days: int = 252) -> pd.DataFrame:
    if returns is None or returns.empty:
        return pd.DataFrame()
    mu_d = returns.mean(skipna=True)
    vol_d = returns.std(skipna=True, ddof=1)
    mu_a = mu_d * trading_days
    vol_a = vol_d * float(np.sqrt(trading_days))
    sharpe = mu_a / vol_a.replace(0.0, np.nan)
    out = pd.DataFrame(
        {
            "ann_return_mean": mu_a,
            "ann_volatility": vol_a,
            "sharpe_rf0": sharpe,
        }
    )
    out.index.name = "symbol"
    return out.sort_values("sharpe_rf0", ascending=False)


def corr_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    if returns is None or returns.empty:
        return pd.DataFrame()
    return returns.corr()


def normalize_prices(close: pd.DataFrame) -> pd.DataFrame:
    if close is None or close.empty:
        return pd.DataFrame()
    first = close.ffill().bfill().iloc[0]
    first = first.replace(0.0, np.nan)
    return close / first


def rolling_volatility(returns: pd.DataFrame, window: int = 30, trading_days: int = 252) -> pd.DataFrame:
    if returns is None or returns.empty:
        return pd.DataFrame()
    rv = returns.rolling(window=window).std(ddof=1) * float(np.sqrt(trading_days))
    return rv.dropna(how="all")


def gaussian_kde_1d(x: np.ndarray, grid: np.ndarray, bandwidth: Optional[float] = None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.zeros_like(grid, dtype=float)
    if bandwidth is None:
        std = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
        if std == 0.0 or not np.isfinite(std):
            std = float(np.abs(x).mean()) if np.isfinite(np.abs(x).mean()) else 1.0
        bandwidth = 1.06 * std * (x.size ** (-1.0 / 5.0))
        if bandwidth <= 0.0 or not np.isfinite(bandwidth):
            bandwidth = 1e-3
    u = (grid[:, None] - x[None, :]) / bandwidth
    dens = np.exp(-0.5 * u * u).sum(axis=1) / (x.size * bandwidth * np.sqrt(2.0 * np.pi))
    return dens


def iso_now() -> str:
    return datetime.utcnow().isoformat() + "Z"
