import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Ensure ass1_core can be imported
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from ass1_core import load_bundle, daily_returns

def calculate_backtest_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "cum_diff": float("nan")}
    yt = y_true[mask]
    yp = y_pred[mask]
    mae = float(np.mean(np.abs(yp - yt)))
    rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
    
    # Calculate cumulative return difference
    cum_true = np.prod(1 + yt) - 1.0
    cum_pred = np.prod(1 + yp) - 1.0
    cum_diff = float(abs(cum_pred - cum_true))
    
    return {"mae": mae, "rmse": rmse, "cum_diff": cum_diff}

def analyze_backtest(
    json_path: str, models_path: str, output_path: str = "backtest_results.json"
):
    bundle = load_bundle(json_path)
    
    # Load models.json to get predictions
    with open(models_path, "r", encoding="utf-8") as f:
        models_data = json.load(f)
    
    # Define backtest period (the week after test_end)
    # The models were trained up to test_end (2025-01-31) and forecasted 7 days ahead
    # So we need real data from 2025-02-01 to 2025-02-07 (or 08)
    backtest_start = "2025-02-01"
    backtest_end = "2025-02-08"
    
    # Process each scope: assets, stocks, universe
    results = {}
    
    for scope in ["assets", "stocks", "universe"]:
        scope_data = getattr(bundle, f"close_{scope}")
        if scope_data is None or scope_data.empty:
            continue
            
        ret_df = daily_returns(scope_data)
        
        # Extract real returns for the backtest period
        real_returns = ret_df.loc[
            (ret_df.index >= pd.to_datetime(backtest_start)) & 
            (ret_df.index <= pd.to_datetime(backtest_end))
        ]
        
        if real_returns.empty:
            print(f"Warning: No real data found for {scope} in backtest period {backtest_start} to {backtest_end}")
            continue
            
        # Limit to 7 days as the forecast horizon is 7
        real_returns = real_returns.head(7)
        
        scope_results = {
            "dates": [d.strftime("%Y-%m-%d") for d in real_returns.index],
            "metrics": {},
            "comparison": {},
            "portfolio": {}  # 新增：组合层面回测
        }

        # Get Naive predictions
        naive_preds = models_data.get(scope, {}).get("naive", [])
        # Convert list of dicts to dict of preds
        naive_dict = {item["symbol"]: item["pred_daily_return"] for item in naive_preds}

        # Get LightGBM predictions
        lgb_forecast = models_data.get(scope, {}).get("lightgbm_forecast", {}).get("forecast", {})

        # 获取优化权重 (从 portfolio_optimization 结果)
        opt_weights = models_data.get(scope, {}).get("opt", {})
        
        # Compare for each symbol
        for sym in real_returns.columns:
            real_vals = real_returns[sym].values
            
            # Naive: constant prediction for all days
            naive_val = naive_dict.get(sym, 0.0)
            naive_seq = np.full(len(real_vals), naive_val)
            
            # LightGBM: sequence prediction
            lgb_seq = lgb_forecast.get(sym, {}).get("pred_daily_return", [])
            # Truncate or pad lgb_seq to match real_vals length
            if len(lgb_seq) > len(real_vals):
                lgb_seq = lgb_seq[:len(real_vals)]
            elif len(lgb_seq) < len(real_vals):
                lgb_seq = lgb_seq + [0.0] * (len(real_vals) - len(lgb_seq))
            lgb_seq = np.array(lgb_seq)
            
            # Calculate metrics
            naive_metrics = calculate_backtest_metrics(real_vals, naive_seq)
            lgb_metrics = calculate_backtest_metrics(real_vals, lgb_seq)
            
            scope_results["comparison"][sym] = {
                "real": [float(x) for x in real_vals],
                "naive_pred": [float(x) for x in naive_seq],
                "lgb_pred": [float(x) for x in lgb_seq],
                "naive_metrics": naive_metrics,
                "lgb_metrics": lgb_metrics
            }

        # 计算组合层面回测 (按优化权重加权)
        if opt_weights:
            for risk_profile in ["min_vol", "max_sharpe"]:
                weights_dict = opt_weights.get(risk_profile, {}).get("weights", {})
                if not weights_dict:
                    continue

                # 确保权重和为1
                total_w = sum(weights_dict.values())
                if total_w > 0:
                    weights_dict = {k: v/total_w for k, v in weights_dict.items()}

                # 计算加权组合收益率
                portfolio_real = []
                portfolio_naive = []
                portfolio_lgb = []

                for i in range(len(real_returns)):
                    # 当日各资产收益率加权
                    real_daily = 0.0
                    naive_daily = 0.0
                    lgb_daily = 0.0

                    for sym, w in weights_dict.items():
                        if sym in real_returns.columns:
                            real_daily += w * real_returns.iloc[i][sym]
                            comp = scope_results["comparison"].get(sym, {})
                            naive_pred = comp.get("naive_pred", [])
                            lgb_pred = comp.get("lgb_pred", [])
                            if i < len(naive_pred):
                                naive_daily += w * naive_pred[i]
                                lgb_daily += w * lgb_pred[i]

                    portfolio_real.append(real_daily)
                    portfolio_naive.append(naive_daily)
                    portfolio_lgb.append(lgb_daily)

                # 计算组合层面指标
                portfolio_metrics = {
                    "naive": calculate_backtest_metrics(
                        np.array(portfolio_real),
                        np.array(portfolio_naive)
                    ),
                    "lgb": calculate_backtest_metrics(
                        np.array(portfolio_real),
                        np.array(portfolio_lgb)
                    )
                }

                scope_results["portfolio"][risk_profile] = {
                    "weights": weights_dict,
                    "real": [float(x) for x in portfolio_real],
                    "naive_pred": [float(x) for x in portfolio_naive],
                    "lgb_pred": [float(x) for x in portfolio_lgb],
                    "metrics": portfolio_metrics
                }

        results[scope] = scope_results
        
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"Backtest analysis saved to {output_path}")
    return results

if __name__ == "__main__":
    analyze_backtest("data.json", "models.json")
