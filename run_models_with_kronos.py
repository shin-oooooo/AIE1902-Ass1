"""
运行所有模型（Naive + LightGBM + Kronos）并生成组合优化结果
"""
import argparse
import json
import os
import sys
from typing import Any, Dict

import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from portfolio import run_models, _portfolio_stats, optimize_monte_carlo, daily_returns
from kronos_predictor import kronos_forecast, KRONOS_AVAILABLE


def run_all_models(json_path: str, out_dir: str = None):
    """
    运行所有模型：Naive、LightGBM、Kronos
    """
    print("="*60)
    print("开始运行全部模型 (Naive + LightGBM + Kronos)")
    print("="*60)

    # 1. 运行原有模型（Naive + LightGBM）
    print("\n[1/3] 运行原有模型 (Naive + LightGBM)...")
    try:
        payload = run_models(json_path, out_dir=out_dir)
        print("✓ Naive + LightGBM 完成")
    except Exception as e:
        print(f"✗ Naive + LightGBM 失败: {e}")
        payload = {}

    # 2. 运行 Kronos 模型
    print("\n[2/3] 运行 Kronos 深度学习模型...")
    if not KRONOS_AVAILABLE:
        print("✗ Kronos 不可用，跳过")
        kronos_payload = {}
    else:
        try:
            from ass1_core import load_bundle, DataBundle

            bundle = load_bundle(json_path)

            kronos_payload = {}
            for dataset_name, close_df in [
                ("assets", bundle.close_assets),
                ("stocks", bundle.close_stocks),
                ("universe", bundle.close_universe)
            ]:
                print(f"  - 处理 {dataset_name} ({len(close_df.columns)} 个标的)...")

                # Run Kronos forecast
                forecast = kronos_forecast(
                    close_df,
                    symbols=list(close_df.columns),
                    lookback=60,
                    pred_len=7,
                    model_name="kronos-small"
                )

                # Run portfolio optimization using Kronos predictions
                rets = daily_returns(close_df)

                if not forecast["regression"].empty:
                    # Use predicted returns as expected returns for optimization
                    pred_returns = forecast["regression"]["pred_daily_return"].reindex(close_df.columns).fillna(rets.mean())

                    # Run Monte Carlo optimization
                    opt_results = optimize_monte_carlo(
                        returns=rets,
                        mu_daily=pred_returns.values,
                        n_samples=50000,
                        seed=42
                    )

                    kronos_payload[dataset_name] = {
                        "forecast": forecast,
                        "opt": {
                            "max_sharpe": {
                                "weights": opt_results["max_sharpe"].weights,
                                "sharpe": opt_results["max_sharpe"].sharpe_rf0,
                                "ann_return": opt_results["max_sharpe"].ann_return_mean,
                                "ann_vol": opt_results["max_sharpe"].ann_volatility
                            },
                            "min_vol": {
                                "weights": opt_results["min_vol"].weights,
                                "sharpe": opt_results["min_vol"].sharpe_rf0,
                                "ann_return": opt_results["min_vol"].ann_return_mean,
                                "ann_vol": opt_results["min_vol"].ann_volatility
                            }
                        }
                    }

                    # Save weight pie chart
                    _save_weight_pies(out_dir or os.path.dirname(json_path), f"kronos_{dataset_name}", kronos_payload[dataset_name]["opt"])

                else:
                    print(f"    警告: {dataset_name} 预测结果为空")

            print("✓ Kronos 完成")

        except Exception as e:
            import traceback
            print(f"✗ Kronos 失败: {e}")
            traceback.print_exc()
            kronos_payload = {}

    # 3. 合并结果
    print("\n[3/3] 合并结果...")

    # Update payload with Kronos results
    if kronos_payload:
        for dataset in ["assets", "stocks", "universe"]:
            if dataset in kronos_payload:
                if dataset not in payload:
                    payload[dataset] = {}
                payload[dataset]["kronos"] = kronos_payload[dataset]

    # Save to models.json
    if out_dir:
        models_path = os.path.join(out_dir, "models.json")
    else:
        models_path = os.path.join(os.path.dirname(json_path), "models.json")

    with open(models_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"✓ 结果已保存至: {models_path}")
    print("\n" + "="*60)
    print("全部模型运行完成!")
    print("="*60)

    return payload


def _save_weight_pies(out_dir: str, label: str, opt_block: Dict[str, Any]):
    """Save weight pie charts"""
    import plotly.express as px

    for obj in ["max_sharpe", "min_vol"]:
        if obj not in opt_block:
            continue
        w = opt_block[obj]["weights"]
        df = pd.DataFrame({"symbol": list(w.keys()), "weight": list(w.values())}).sort_values("weight", ascending=False)
        fig = px.pie(df, names="symbol", values="weight", title=f"{label.upper()} weights - {obj}")
        fig.write_html(os.path.join(out_dir, f"{label}_weights_{obj}.html"), include_plotlyjs="cdn", full_html=True)


def main():
    parser = argparse.ArgumentParser(description="Run all models including Kronos")
    parser.add_argument("--json", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json"))
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.json))
    run_all_models(args.json, out_dir=out_dir)


if __name__ == "__main__":
    main()
