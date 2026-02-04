import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from ass1_core import DataBundle, annual_metrics, daily_returns, load_bundle


Objective = Literal["max_sharpe", "min_vol"]


@dataclass(frozen=True)
class PortfolioResult:
    weights: Dict[str, float]
    objective: Objective
    ann_return_mean: float
    ann_volatility: float
    sharpe_rf0: float


def _portfolio_stats(mu_daily: np.ndarray, cov_daily: np.ndarray, w: np.ndarray, trading_days: int = 252) -> Tuple[float, float, float]:
    mu_a = float(w @ mu_daily) * trading_days
    vol_a = float(np.sqrt(max(w @ cov_daily @ w, 0.0))) * float(np.sqrt(trading_days))
    sharpe = mu_a / vol_a if vol_a > 0 else float("nan")
    return mu_a, vol_a, sharpe


def optimize_monte_carlo(
    returns: pd.DataFrame,
    mu_daily: Optional[np.ndarray] = None,
    n_samples: int = 50000,
    seed: int = 42,
    trading_days: int = 252,
) -> Dict[Objective, PortfolioResult]:
    rets = returns.dropna(how="any")
    if rets.empty:
        raise RuntimeError("returns 为空，无法优化")

    cols = list(rets.columns)
    mu_d = rets.mean().to_numpy(dtype=float) if mu_daily is None else np.asarray(mu_daily, dtype=float)
    cov_d = rets.cov().to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    w_all = rng.dirichlet(alpha=np.ones(len(cols)), size=n_samples)
    mu_a = (w_all @ mu_d) * trading_days
    var_a = np.einsum("ij,jk,ik->i", w_all, cov_d, w_all) * trading_days
    vol_a = np.sqrt(np.maximum(var_a, 0.0))
    sharpe = np.divide(mu_a, vol_a, out=np.full_like(mu_a, np.nan), where=vol_a > 0)

    idx_sharpe = int(np.nanargmax(sharpe))
    idx_vol = int(np.nanargmin(vol_a))

    def to_result(idx: int, obj: Objective) -> PortfolioResult:
        w = w_all[idx]
        mu, vol, sh = _portfolio_stats(mu_d, cov_d, w, trading_days=trading_days)
        return PortfolioResult(weights={c: float(w[i]) for i, c in enumerate(cols)}, objective=obj, ann_return_mean=mu, ann_volatility=vol, sharpe_rf0=sh)

    return {"max_sharpe": to_result(idx_sharpe, "max_sharpe"), "min_vol": to_result(idx_vol, "min_vol")}


def naive_forecast(returns: pd.DataFrame, lookback: int = 30, horizon: int = 7) -> pd.DataFrame:
    rets = returns.dropna(how="all").copy()
    if rets.empty:
        return pd.DataFrame()
    tail = rets.tail(lookback)
    mu = tail.mean(skipna=True)
    out = pd.DataFrame({"pred_daily_return": mu})
    out["pred_7d_cum_return"] = (1.0 + out["pred_daily_return"]) ** horizon - 1.0
    out.index.name = "symbol"
    return out.sort_values("pred_7d_cum_return", ascending=False)


def _to_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
    return out.dropna(axis=0, how="all").sort_index()


def _slice_dates(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    dfx = _to_dt_index(df)
    m = (dfx.index >= pd.to_datetime(start)) & (dfx.index <= pd.to_datetime(end))
    return dfx.loc[m].copy()


def _make_features(ret_s: pd.Series) -> pd.DataFrame:
    s = ret_s.astype(float).replace([np.inf, -np.inf], np.nan)
    df = pd.DataFrame({"ret": s})
    for i in range(1, 6):
        df[f"lag_{i}"] = df["ret"].shift(i)
    for w in [5, 10, 20]:
        df[f"roll_mean_{w}"] = df["ret"].rolling(w).mean()
        df[f"roll_std_{w}"] = df["ret"].rolling(w).std(ddof=1)
        df[f"mom_{w}"] = (1.0 + df["ret"]).rolling(w).apply(lambda x: float(np.prod(x) - 1.0), raw=False)
    df["y"] = df["ret"].shift(-1)
    df = df.drop(columns=["ret"]).dropna()
    return df


def _try_import_lightgbm():
    try:
        import lightgbm as lgb
    except Exception as e:
        return None, str(e)
    return lgb, ""


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "direction_acc": float("nan")}
    yt = y_true[mask]
    yp = y_pred[mask]
    mae = float(np.mean(np.abs(yp - yt)))
    rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
    direction_acc = float(np.mean(np.sign(yp) == np.sign(yt)))
    return {"mae": mae, "rmse": rmse, "direction_acc": direction_acc}


def lightgbm_train_eval(
    returns: pd.DataFrame,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    seed: int = 42,
) -> Dict[str, Any]:
    lgb, err = _try_import_lightgbm()
    if lgb is None:
        return {"available": False, "error": err}

    rets = _to_dt_index(returns).dropna(how="any")
    if rets.empty:
        return {"available": True, "models": {}, "metrics": {}, "test_pred": {}, "error": "returns为空"}

    train_mask = (rets.index >= pd.to_datetime(train_start)) & (rets.index <= pd.to_datetime(train_end))
    test_mask = (rets.index >= pd.to_datetime(test_start)) & (rets.index <= pd.to_datetime(test_end))
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        return {"available": True, "models": {}, "metrics": {}, "test_pred": {}, "error": "训练集或测试集为空"}

    models: Dict[str, Any] = {}
    metrics: Dict[str, Any] = {}
    test_pred: Dict[str, Any] = {}

    for sym in rets.columns:
        feat = _make_features(rets[sym])
        feat = feat.loc[feat.index.intersection(rets.index)].copy()
        train_feat = feat.loc[(feat.index >= pd.to_datetime(train_start)) & (feat.index <= pd.to_datetime(train_end))]
        test_feat = feat.loc[(feat.index >= pd.to_datetime(test_start)) & (feat.index <= pd.to_datetime(test_end))]
        if train_feat.empty or test_feat.empty:
            continue
        X_train = train_feat.drop(columns=["y"])
        y_train = train_feat["y"].to_numpy(dtype=float)
        X_test = test_feat.drop(columns=["y"])
        y_test = test_feat["y"].to_numpy(dtype=float)

        model = lgb.LGBMRegressor(
            n_estimators=800,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        models[sym] = {"feature_names": list(X_train.columns)}
        metrics[sym] = _regression_metrics(y_test, y_pred)
        test_pred[sym] = {
            "dates": [d.strftime("%Y-%m-%d") for d in X_test.index.to_pydatetime()],
            "y_true": [float(x) for x in y_test],
            "y_pred": [float(x) for x in y_pred],
        }

    return {"available": True, "models": models, "metrics": metrics, "test_pred": test_pred, "train": {"start": train_start, "end": train_end}, "test": {"start": test_start, "end": test_end}}


def lightgbm_forecast_horizon(
    returns: pd.DataFrame,
    lightgbm_state: Dict[str, Any],
    horizon: int = 7,
) -> Dict[str, Any]:
    lgb, err = _try_import_lightgbm()
    if lgb is None:
        return {"available": False, "error": err}
    if not lightgbm_state.get("available", False):
        return {"available": False, "error": lightgbm_state.get("error", "")}

    rets = _to_dt_index(returns).dropna(how="any")
    if rets.empty:
        return {"available": True, "forecast": {}, "error": "returns为空"}

    out: Dict[str, Any] = {"available": True, "forecast": {}, "horizon": horizon}
    for sym in rets.columns:
        feat_full = _make_features(rets[sym])
        if feat_full.empty:
            continue
        last_row = feat_full.drop(columns=["y"]).iloc[-1].copy()
        hist = rets[sym].copy()

        model = None
        try:
            train_start = lightgbm_state.get("train", {}).get("start", "")
            train_end = lightgbm_state.get("train", {}).get("end", "")
            train_mask = (rets.index >= pd.to_datetime(train_start)) & (rets.index <= pd.to_datetime(train_end))
            feat = _make_features(rets[sym])
            train_feat = feat.loc[feat.index.intersection(rets.index)]
            train_feat = train_feat.loc[(train_feat.index >= pd.to_datetime(train_start)) & (train_feat.index <= pd.to_datetime(train_end))]
            if train_feat.empty:
                continue
            X_train = train_feat.drop(columns=["y"])
            y_train = train_feat["y"].to_numpy(dtype=float)
            model = lgb.LGBMRegressor(
                n_estimators=800,
                learning_rate=0.03,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
            )
            model.fit(X_train, y_train)
        except Exception:
            model = None

        if model is None:
            continue

        preds: List[float] = []
        for _ in range(horizon):
            x = pd.DataFrame([last_row.to_dict()])
            y_hat = float(model.predict(x)[0])
            preds.append(y_hat)
            hist = pd.concat([hist, pd.Series([y_hat], index=[hist.index[-1] + pd.Timedelta(days=1)])])
            feat_next = _make_features(hist)
            if feat_next.empty:
                break
            last_row = feat_next.drop(columns=["y"]).iloc[-1].copy()

        out["forecast"][sym] = {"pred_daily_return": preds, "pred_7d_cum_return": float(np.prod([1.0 + p for p in preds]) - 1.0) if preds else float("nan")}
    return out


def _blend_weights(w_low: Dict[str, float], w_high: Dict[str, float], alpha: float) -> Dict[str, float]:
    keys = sorted(set(w_low) | set(w_high))
    w = {k: (1.0 - alpha) * float(w_low.get(k, 0.0)) + alpha * float(w_high.get(k, 0.0)) for k in keys}
    s = sum(w.values())
    if s <= 0:
        return {}
    return {k: v / s for k, v in w.items()}


def _subset_renorm(weights: Dict[str, float], symbols: list) -> Dict[str, float]:
    sset = set(symbols)
    w = {k: float(v) for k, v in weights.items() if k in sset}
    s = sum(w.values())
    if s <= 0:
        return {k: 1.0 / len(symbols) for k in symbols} if symbols else {}
    return {k: v / s for k, v in w.items()}


def _weights_arrays(weights: Dict[str, float], symbols: Optional[list] = None) -> Dict[str, Any]:
    syms = symbols or list(weights.keys())
    return {"symbols": syms, "weights": [float(weights.get(s, 0.0)) for s in syms]}


def _mu_from_naive(naive_df: pd.DataFrame, symbols: list) -> np.ndarray:
    if naive_df is None or naive_df.empty:
        return np.zeros(len(symbols), dtype=float)
    s = naive_df["pred_daily_return"] if "pred_daily_return" in naive_df.columns else pd.Series(dtype=float)
    if naive_df.index.name != "symbol" and "symbol" in naive_df.columns:
        naive_df = naive_df.set_index("symbol")
        s = naive_df["pred_daily_return"]
    return np.asarray([float(s.get(sym, 0.0)) for sym in symbols], dtype=float)


def _mu_from_lightgbm_forecast(ret_df: pd.DataFrame, lgb_state: Dict[str, Any], symbols: list, horizon: int = 7) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    forecast = lightgbm_forecast_horizon(ret_df, lgb_state, horizon=horizon)
    if not forecast.get("available", False):
        return None, forecast
    mu = []
    for sym in symbols:
        preds = forecast.get("forecast", {}).get(sym, {}).get("pred_daily_return", [])
        if preds:
            mu.append(float(np.mean(np.asarray(preds, dtype=float))))
        else:
            mu.append(0.0)
    return np.asarray(mu, dtype=float), forecast


def run_models(json_path: str, out_dir: Optional[str] = None) -> Dict[str, Any]:
    bundle = load_bundle(json_path)
    out_dir = out_dir or os.path.dirname(os.path.abspath(json_path))

    close_assets = bundle.close_assets
    close_stocks = bundle.close_stocks
    close_universe = bundle.close_universe

    ret_assets = daily_returns(close_assets).dropna(how="all")
    ret_stocks = daily_returns(close_stocks).dropna(how="all")
    ret_universe = daily_returns(close_universe).dropna(how="all")

    meta = bundle.meta or {}
    train_start = meta.get("train_start", "2024-11-01")
    train_end = meta.get("train_end", "2025-11-30")
    test_start = meta.get("test_start", "2025-12-01")
    test_end = meta.get("test_end", "2026-01-31")

    assets_naive = naive_forecast(_slice_dates(ret_assets, train_start, train_end))
    stocks_naive = naive_forecast(_slice_dates(ret_stocks, train_start, train_end))
    universe_naive = naive_forecast(_slice_dates(ret_universe, train_start, train_end))

    lgb_assets = lightgbm_train_eval(ret_assets, train_start, train_end, test_start, test_end, seed=7)
    lgb_stocks = lightgbm_train_eval(ret_stocks, train_start, train_end, test_start, test_end, seed=11)
    lgb_universe = lightgbm_train_eval(ret_universe, train_start, train_end, test_start, test_end, seed=19)

    ret_assets_train = _slice_dates(ret_assets, train_start, train_end)
    ret_stocks_train = _slice_dates(ret_stocks, train_start, train_end)
    ret_universe_train = _slice_dates(ret_universe, train_start, train_end)

    assets_syms = list(ret_assets_train.columns)
    stocks_syms = list(ret_stocks_train.columns)
    universe_syms = list(ret_universe_train.columns)

    mu_assets, fc_assets = _mu_from_lightgbm_forecast(ret_assets, lgb_assets, assets_syms)
    mu_stocks, fc_stocks = _mu_from_lightgbm_forecast(ret_stocks, lgb_stocks, stocks_syms)
    mu_universe, fc_universe = _mu_from_lightgbm_forecast(ret_universe, lgb_universe, universe_syms)

    if mu_assets is None:
        mu_assets = _mu_from_naive(assets_naive, assets_syms)
        mu_assets_source = "naive"
    else:
        mu_assets_source = "lightgbm"
    if mu_stocks is None:
        mu_stocks = _mu_from_naive(stocks_naive, stocks_syms)
        mu_stocks_source = "naive"
    else:
        mu_stocks_source = "lightgbm"
    if mu_universe is None:
        mu_universe = _mu_from_naive(universe_naive, universe_syms)
        mu_universe_source = "naive"
    else:
        mu_universe_source = "lightgbm"

    assets_opt = optimize_monte_carlo(ret_assets_train, mu_daily=mu_assets, n_samples=50000, seed=7)
    stocks_opt = optimize_monte_carlo(ret_stocks_train, mu_daily=mu_stocks, n_samples=150000, seed=11)
    universe_opt = optimize_monte_carlo(ret_universe_train, mu_daily=mu_universe, n_samples=200000, seed=19)

    universe_w_low = universe_opt["min_vol"].weights
    universe_w_high = universe_opt["max_sharpe"].weights
    universe_final = {
        "低": universe_w_low,
        "中": _blend_weights(universe_w_low, universe_w_high, alpha=0.5),
        "高": universe_w_high,
    }

    universe_meta_arrays = {
        "all": _weights_arrays(universe_final["中"], symbols=universe_syms),
        "assets": _weights_arrays(_subset_renorm(universe_final["中"], ["SPY", "AU0"]), symbols=["SPY", "AU0"]),
        "stocks": _weights_arrays(_subset_renorm(universe_final["中"], [s for s in universe_syms if s not in ["SPY", "AU0"]]), symbols=[s for s in universe_syms if s not in ["SPY", "AU0"]]),
    }

    stocks_derived = {k: _subset_renorm(v, [s for s in universe_syms if s not in ["SPY", "AU0"]]) for k, v in universe_final.items()}
    assets_derived = {k: _subset_renorm(v, ["SPY", "AU0"]) for k, v in universe_final.items()}

    payload: Dict[str, Any] = {
        "meta": {
            "source_data": os.path.basename(json_path),
            "generated_at": bundle.meta.get("generated_at", ""),
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
        },
        "assets": {
            "naive": assets_naive.reset_index().to_dict(orient="records"),
            "opt": {k: vars(v) for k, v in assets_opt.items()},
            "lightgbm": lgb_assets,
            "lightgbm_forecast": fc_assets,
            "mu_source": mu_assets_source,
        },
        "stocks": {
            "naive": stocks_naive.reset_index().to_dict(orient="records"),
            "opt": {k: vars(v) for k, v in stocks_opt.items()},
            "lightgbm": lgb_stocks,
            "lightgbm_forecast": fc_stocks,
            "mu_source": mu_stocks_source,
            "derived_from_universe": stocks_derived,
        },
        "universe": {
            "naive": universe_naive.reset_index().to_dict(orient="records"),
            "opt": {k: vars(v) for k, v in universe_opt.items()},
            "lightgbm": lgb_universe,
            "lightgbm_forecast": fc_universe,
            "mu_source": mu_universe_source,
            "final_weights": universe_final,
            "weight_arrays": universe_meta_arrays,
            "derived_assets": assets_derived,
        },
    }

    with open(os.path.join(out_dir, "models.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    assets_naive.to_csv(os.path.join(out_dir, "assets_naive.csv"), encoding="utf-8-sig")
    stocks_naive.to_csv(os.path.join(out_dir, "stocks_naive.csv"), encoding="utf-8-sig")
    universe_naive.to_csv(os.path.join(out_dir, "universe_naive.csv"), encoding="utf-8-sig")

    return payload
