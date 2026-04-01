"""
Kronos 股价预测模块 - 为 Ass1 项目提供深度学习预测能力
"""
import json
import os
import sys
import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Ensure the project root is in path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# Check for local weights - only use local if both the directory exists AND it contains model folders
_WEIGHTS_DIR = os.path.join(_THIS_DIR, "kronos_weights")
_USE_LOCAL = os.path.exists(_WEIGHTS_DIR) and (
    os.path.exists(os.path.join(_WEIGHTS_DIR, "kronos-small")) or 
    os.path.exists(os.path.join(_WEIGHTS_DIR, "kronos-mini"))
)

# Import ass1_core functions at module level to avoid circular imports
try:
    from ass1_core import load_bundle as _load_bundle, daily_returns as _daily_returns
    _ASS1_CORE_AVAILABLE = True
except ImportError:
    _ASS1_CORE_AVAILABLE = False
    _load_bundle = None
    _daily_returns = None

try:
    from kronos_model import Kronos, KronosTokenizer, KronosPredictor
    KRONOS_AVAILABLE = True
except ImportError as e:
    KRONOS_AVAILABLE = False
    print(f"Kronos not available: {e}")


def prepare_ohlcv_from_close(
    close_df: pd.DataFrame,
    volume_df: pd.DataFrame = None,
    symbol: str = None
) -> pd.DataFrame:
    """
    从收盘价生成模拟的 OHLCV 数据（Kronos 需要）

    Args:
        close_df: 收盘价 DataFrame，index 为日期，columns 为标的
        volume_df: 可选的成交量 DataFrame
        symbol: 要处理的标的代码

    Returns:
        OHLCV DataFrame
    """
    if symbol and symbol in close_df.columns:
        close = close_df[symbol].copy()
    else:
        close = close_df.iloc[:, 0].copy()  # 默认取第一列

    # 生成模拟的 OHLC（基于收盘价）
    ohlcv_data = []
    for i, (date, c) in enumerate(close.items()):
        # 模拟开盘价 = 前一日收盘价或当日收盘价
        if i > 0:
            o = close.iloc[i-1]
        else:
            o = c

        # 模拟波动：基于收盘价生成合理的高低价
        daily_range = abs(c - o) * 2 + c * 0.01  # 日波动范围
        h = max(o, c) + daily_range * 0.3
        l = min(o, c) - daily_range * 0.3

        # 确保合理性
        h = max(h, o, c)
        l = min(l, o, c)

        # 成交量（模拟或从外部传入）
        if volume_df is not None and symbol in volume_df.columns:
            vol = volume_df.loc[date, symbol] if date in volume_df.index else np.random.randint(10000, 100000)
        else:
            vol = np.random.randint(10000, 100000)

        ohlcv_data.append({
            'timestamps': pd.to_datetime(date),
            'open': round(float(o), 2),
            'high': round(float(h), 2),
            'low': round(float(l), 2),
            'close': round(float(c), 2),
            'volume': int(vol)
        })

    return pd.DataFrame(ohlcv_data)


def kronos_forecast(
    close_df: pd.DataFrame,
    symbols: List[str] = None,
    lookback: int = 60,
    pred_len: int = 7,
    model_name: str = "kronos-small",
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    使用 Kronos 模型预测未来收益

    Args:
        close_df: 收盘价 DataFrame
        symbols: 要预测的标的列表，None 则预测所有
        lookback: 历史数据长度
        pred_len: 预测未来天数
        model_name: kronos-mini, kronos-small, kronos-base
        device: cpu 或 cuda

    Returns:
        {
            "regression": DataFrame with predicted returns,
            "classification": Dict with class predictions,
            "model_info": str
        }
    """
    if not KRONOS_AVAILABLE:
        return {
            "regression": pd.DataFrame(),
            "classification": {},
            "model_info": "Kronos not available"
        }

    # Model mapping - use local paths if available
    if _USE_LOCAL:
        model_configs = {
            'kronos-mini': {
                'model_path': os.path.join(_WEIGHTS_DIR, 'kronos-mini'),
                'tokenizer_path': os.path.join(_WEIGHTS_DIR, 'tokenizer-2k'),
                'context_length': 2048
            },
            'kronos-small': {
                'model_path': os.path.join(_WEIGHTS_DIR, 'kronos-small'),
                'tokenizer_path': os.path.join(_WEIGHTS_DIR, 'tokenizer-base'),
                'context_length': 512
            },
            'kronos-base': {
                'model_path': 'NeoQuasar/Kronos-base',  # base not cached locally
                'tokenizer_path': os.path.join(_WEIGHTS_DIR, 'tokenizer-base'),
                'context_length': 512
            }
        }
    else:
        model_configs = {
            'kronos-mini': {
                'model_id': 'NeoQuasar/Kronos-mini',
                'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-2k',
                'context_length': 2048
            },
            'kronos-small': {
                'model_id': 'NeoQuasar/Kronos-small',
                'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
                'context_length': 512
            },
            'kronos-base': {
                'model_id': 'NeoQuasar/Kronos-base',
                'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
                'context_length': 512
            }
        }

    config = model_configs.get(model_name, model_configs['kronos-small'])

    try:
        # Load model from local or HuggingFace
        if _USE_LOCAL:
            print(f"Loading Kronos model from local: {config['model_path']}")
            tokenizer = KronosTokenizer.from_pretrained(config['tokenizer_path'])
            model = Kronos.from_pretrained(config['model_path'])
        else:
            print(f"Downloading Kronos model from HuggingFace...")
            tokenizer = KronosTokenizer.from_pretrained(config['tokenizer_id'])
            model = Kronos.from_pretrained(config['model_id'])
        predictor = KronosPredictor(model, tokenizer, device=device, max_context=config['context_length'])
    except Exception as e:
        print(f"Failed to load Kronos model: {e}")
        return {
            "regression": pd.DataFrame(),
            "classification": {},
            "model_info": f"Model load failed: {e}"
        }

    if symbols is None:
        symbols = list(close_df.columns)

    reg_results = []
    cls_results = {}

    for symbol in symbols:
        if symbol not in close_df.columns:
            continue

        try:
            # Prepare OHLCV data
            ohlcv_df = prepare_ohlcv_from_close(close_df, symbol=symbol)

            if len(ohlcv_df) < lookback:
                print(f"Insufficient data for {symbol}: {len(ohlcv_df)} < {lookback}")
                continue

            # Use last lookback points for prediction
            hist_df = ohlcv_df.tail(lookback).reset_index(drop=True)

            # Prepare timestamps for prediction (next pred_len days)
            last_date = hist_df['timestamps'].iloc[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=pred_len, freq='B')  # Business days

            # Run prediction
            x_df = hist_df[['open', 'high', 'low', 'close', 'volume']]
            x_timestamp = hist_df['timestamps']
            y_timestamp = pd.Series(future_dates, name='timestamps')

            pred_df = predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                T=1.0,
                top_p=0.9,
                sample_count=1,
                verbose=False
            )

            # Calculate predicted return
            last_close = hist_df['close'].iloc[-1]
            pred_last_close = pred_df['close'].iloc[-1]
            pred_daily_return = (pred_last_close / last_close) ** (1/pred_len) - 1
            pred_cum_return = pred_last_close / last_close - 1

            reg_results.append({
                "symbol": symbol,
                "pred_daily_return": float(pred_daily_return),
                "pred_7d_cum_return": float(pred_cum_return),
                "last_close": float(last_close),
                "pred_future_close": float(pred_last_close)
            })

            # Classification based on predicted direction
            if pred_cum_return > 0.02:
                pred_class = 2  # Up
                probs = [0.1, 0.2, 0.7]
            elif pred_cum_return < -0.02:
                pred_class = 0  # Down
                probs = [0.7, 0.2, 0.1]
            else:
                pred_class = 1  # Flat
                probs = [0.2, 0.6, 0.2]

            cls_results[symbol] = {
                "pred_class": pred_class,
                "pred_probs": probs,
                "description": f"Kronos {model_name} prediction",
                "pred_cum_return": float(pred_cum_return)
            }

        except Exception as e:
            print(f"Prediction failed for {symbol}: {e}")
            continue

    reg_df = pd.DataFrame(reg_results)
    if not reg_df.empty:
        reg_df = reg_df.set_index("symbol").sort_values("pred_7d_cum_return", ascending=False)

    return {
        "regression": reg_df,
        "classification": cls_results,
        "model_info": f"Kronos {model_name} (lookback={lookback}, pred_len={pred_len})"
    }


def run_kronos_optimization(
    data_json_path: str,
    dataset: str = "universe",
    lookback: int = 60,
    pred_len: int = 7,
    model_name: str = "kronos-small",
    out_dir: str = None
) -> Dict[str, Any]:
    """
    运行 Kronos 预测并返回与现有模型兼容的格式

    类似于 portfolio.run_models 的接口
    """
    if not _ASS1_CORE_AVAILABLE or _load_bundle is None or _daily_returns is None:
        return {
            "regression": {},
            "classification": {},
            "model_info": "ass1_core not available",
            "metrics": {}
        }

    bundle = _load_bundle(data_json_path)

    if dataset == "assets":
        close = bundle.close_assets
    elif dataset == "stocks":
        close = bundle.close_stocks
    else:
        close = bundle.close_universe

    # Run Kronos forecast
    forecast = kronos_forecast(
        close,
        symbols=list(close.columns),
        lookback=lookback,
        pred_len=pred_len,
        model_name=model_name
    )

    # Calculate returns for optimization (using predicted returns as expected returns)
    rets = _daily_returns(close)

    # Prepare output compatible with existing format
    result = {
        "regression": forecast["regression"].to_dict() if not forecast["regression"].empty else {},
        "classification": forecast["classification"],
        "model_info": forecast["model_info"],
        "metrics": {
            "mse": None,  # Kronos doesn't provide MSE directly
            "mae": None,
            "accuracy": None
        },
        "note": "Kronos is a foundation model for time series prediction"
    }

    # Save to file if out_dir provided
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"kronos_{dataset}_forecast.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    return result


if __name__ == "__main__":
    # Test
    test_data_path = "/Users/dyl/Downloads/AIE1902-Ass1-main/data.json"
    if os.path.exists(test_data_path):
        result = run_kronos_optimization(test_data_path, dataset="stocks")
        print("Kronos forecast completed!")
        print(f"Symbols predicted: {list(result['classification'].keys())}")
        print(f"Model: {result['model_info']}")
    else:
        print("Test data not found")
