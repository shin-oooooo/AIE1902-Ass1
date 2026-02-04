import json
import os
import argparse
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FetchConfig:
    start: str = "2024-11-01"
    end: str = "2026-01-31"
    zscore_threshold: float = 3.0


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _to_yyyymmdd(date_str: str) -> str:
    return date_str.replace("-", "")


_REQUESTS_HTTP_PATCHED = False
_SYMBOL_ALIASES = {"TSMC": "TSM"}


def _enable_http_for_requests(logs: List[Dict[str, Any]]):
    global _REQUESTS_HTTP_PATCHED
    if _REQUESTS_HTTP_PATCHED:
        return
    import requests

    original = requests.sessions.Session.request

    def patched(self, method, url, *args, **kwargs):
        if isinstance(url, str) and url.startswith("https://"):
            url = "http://" + url[len("https://") :]
        return original(self, method, url, *args, **kwargs)

    requests.sessions.Session.request = patched
    _REQUESTS_HTTP_PATCHED = True
    logs.append({"time": _now_iso(), "level": "INFO", "code": "REQUESTS_HTTP_PATCHED", "message": "force https->http"})


def _normalize_us_hist_symbol(symbol: str) -> str:
    s = symbol.strip()
    if "." in s:
        return s
    return f"105.{s}"


def _standardize_date_close(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "close"])
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]

    date_col = None
    for c in ["date", "日期", "time", "datetime", "timestamp"]:
        if c.lower() in out.columns:
            date_col = c.lower()
            break
    close_col = None
    for c in ["close", "收盘价", "收盘", "close_price", "closing"]:
        if c.lower() in out.columns:
            close_col = c.lower()
            break

    if date_col is None or close_col is None:
        raise RuntimeError(f"无法识别 date/close 列: columns={list(out.columns)}")

    out = out[[date_col, close_col]].rename(columns={date_col: "date", close_col: "close"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date.astype("string")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["date", "close"])
    out["date"] = out["date"].astype(str)
    out = out.sort_values("date").reset_index(drop=True)
    return out


def _filter_date_range(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "close"])
    m = (df["date"] >= start) & (df["date"] <= end)
    out = df.loc[m].copy()
    out = out.sort_values("date").reset_index(drop=True)
    return out


def _dedup_by_date(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "close"]), 0
    before = len(df)
    out = df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return out, before - len(out)


def _zscore_filter_on_returns(df: pd.DataFrame, threshold: float) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    if df is None or df.empty or len(df) < 5:
        return df, []
    out = df.copy()
    out["ret"] = out["close"].pct_change()
    rets = out["ret"].to_numpy(dtype=float)
    mask = np.isfinite(rets)
    if mask.sum() < 5:
        out = out.drop(columns=["ret"])
        return out, []
    mu = float(np.nanmean(rets[mask]))
    sigma = float(np.nanstd(rets[mask]))
    if sigma == 0.0 or not np.isfinite(sigma):
        out = out.drop(columns=["ret"])
        return out, []
    z = (rets - mu) / sigma
    outliers: List[Dict[str, Any]] = []
    keep = np.ones(len(out), dtype=bool)
    for i in range(len(out)):
        if not np.isfinite(z[i]):
            continue
        if abs(float(z[i])) > threshold:
            keep[i] = False
            outliers.append(
                {"date": str(out["date"].iloc[i]), "close": float(out["close"].iloc[i]), "ret": float(rets[i]), "z": float(z[i])}
            )
    out = out.loc[keep].drop(columns=["ret"]).reset_index(drop=True)
    return out, outliers


def fetch_us_daily_qfq(symbol: str, start: str, end: str, logs: List[Dict[str, Any]]) -> pd.DataFrame:
    _enable_http_for_requests(logs)
    try:
        import akshare as ak
    except Exception as e:
        logs.append({"time": _now_iso(), "level": "ERROR", "code": "AK_IMPORT_FAIL", "message": str(e), "context": {"symbol": symbol}})
        raise

    df = None
    try:
        df = ak.stock_us_daily(symbol=symbol, adjust="qfq")
        logs.append({"time": _now_iso(), "level": "INFO", "code": "AK_US_DAILY_OK", "message": "stock_us_daily", "context": {"symbol": symbol}})
    except Exception as e:
        logs.append({"time": _now_iso(), "level": "WARN", "code": "AK_US_DAILY_FAIL", "message": str(e), "context": {"symbol": symbol}})

    if df is None or getattr(df, "empty", True):
        try:
            df = ak.stock_us_hist(
                symbol=_normalize_us_hist_symbol(symbol),
                period="daily",
                start_date=_to_yyyymmdd(start),
                end_date=_to_yyyymmdd(end),
                adjust="qfq",
            )
            logs.append({"time": _now_iso(), "level": "INFO", "code": "AK_US_HIST_OK", "message": "stock_us_hist", "context": {"symbol": symbol}})
        except Exception as e:
            logs.append({"time": _now_iso(), "level": "ERROR", "code": "AK_US_HIST_FAIL", "message": str(e), "context": {"symbol": symbol}})
            raise

    return _standardize_date_close(df)


def fetch_futures_au0_daily(symbol: str, start: str, end: str, logs: List[Dict[str, Any]]) -> pd.DataFrame:
    _enable_http_for_requests(logs)
    try:
        import akshare as ak
    except Exception as e:
        logs.append({"time": _now_iso(), "level": "ERROR", "code": "AK_IMPORT_FAIL", "message": str(e), "context": {"symbol": symbol}})
        raise

    y1 = _to_yyyymmdd(start)
    y2 = _to_yyyymmdd(end)

    if not hasattr(ak, "futures_main_sina"):
        raise RuntimeError("当前 AkShare 版本缺少 futures_main_sina，无法拉取 AU0")

    last_err: Optional[Exception] = None
    for sym_try in [symbol, symbol.upper(), symbol.lower()]:
        try:
            df = ak.futures_main_sina(symbol=sym_try, start_date=y1, end_date=y2)
            logs.append(
                {"time": _now_iso(), "level": "INFO", "code": "AK_FUT_MAIN_SINA_OK", "message": "futures_main_sina", "context": {"symbol": sym_try}}
            )
            return _standardize_date_close(df)
        except Exception as e:
            last_err = e
            logs.append(
                {"time": _now_iso(), "level": "WARN", "code": "AK_FUT_MAIN_SINA_FAIL", "message": str(e), "context": {"symbol": sym_try}}
            )
    raise RuntimeError(str(last_err) if last_err is not None else "futures_main_sina failed")


def _summary_stats(symbol: str, df_raw: pd.DataFrame, df_clean: pd.DataFrame, outliers: List[Dict[str, Any]], dup_removed: int) -> Dict[str, Any]:
    s: Dict[str, Any] = {
        "symbol": symbol,
        "n_raw": int(len(df_raw)) if df_raw is not None else 0,
        "n_clean": int(len(df_clean)) if df_clean is not None else 0,
        "duplicate_date_removed": int(dup_removed),
        "zscore_outlier_removed_count": int(len(outliers)),
    }
    if df_clean is None or df_clean.empty:
        return s
    closes = df_clean["close"].astype(float)
    s.update(
        {
            "date_min": str(df_clean["date"].iloc[0]),
            "date_max": str(df_clean["date"].iloc[-1]),
            "close_mean": float(closes.mean()),
            "close_min": float(closes.min()),
            "close_max": float(closes.max()),
            "first_close": float(closes.iloc[0]),
            "last_close": float(closes.iloc[-1]),
        }
    )
    return s


def _monthly_means(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    tmp = df.copy()
    tmp["ym"] = tmp["date"].str.slice(0, 7)
    g = tmp.groupby("ym", as_index=False).agg(mean_close=("close", "mean"), n=("close", "count"))
    out = []
    for _, r in g.iterrows():
        out.append({"month": str(r["ym"]), "mean_close": float(r["mean_close"]), "n": int(r["n"])})
    out = sorted(out, key=lambda x: x["month"])
    return out


def _default_symbols() -> Tuple[List[str], List[str]]:
    assets = ["SPY", "AU0"]
    stocks = ["NVDA", "MSFT", "TSMC", "GOOGL", "AMZN", "AAPL", "ASML", "META", "AVGO", "ORCL"]
    return assets, stocks


def _universe_symbols() -> List[str]:
    return ["NVDA", "MSFT", "TSMC", "GOOGL", "AMZN", "AAPL", "ASML", "META", "AVGO", "ORCL", "SPY", "AU0"]


def _train_test_meta(cfg: FetchConfig) -> Dict[str, str]:
    train_start = "2024-11-01"
    train_end = "2025-11-30"
    test_start = "2025-12-01"
    test_end = "2026-01-31"
    return {
        "train_start": max(cfg.start, train_start),
        "train_end": min(cfg.end, train_end),
        "test_start": max(cfg.start, test_start),
        "test_end": min(cfg.end, test_end),
    }


def _new_payload(cfg: FetchConfig) -> Dict[str, Any]:
    assets, stocks = _default_symbols()
    universe = _universe_symbols()
    split = _train_test_meta(cfg)
    return {
        "meta": {
            "source": "akshare",
            "generated_at": _now_iso(),
            "start": cfg.start,
            "end": cfg.end,
            "assets": assets,
            "stocks": stocks,
            "universe": universe,
            **split,
            "zscore_threshold": cfg.zscore_threshold,
        },
        "assets": {},
        "stocks": {},
        "summary": {"assets": {}, "stocks": {}},
        "outliers": {},
        "monthly_means": {},
        "logs": [],
    }


def _load_payload(path: str, cfg: FetchConfig) -> Dict[str, Any]:
    if not os.path.exists(path):
        return _new_payload(cfg)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "meta" not in data:
        return _new_payload(cfg)
    assets, stocks = _default_symbols()
    universe = _universe_symbols()
    split = _train_test_meta(cfg)
    data["meta"]["generated_at"] = _now_iso()
    data["meta"]["start"] = cfg.start
    data["meta"]["end"] = cfg.end
    data["meta"]["zscore_threshold"] = cfg.zscore_threshold
    data["meta"]["assets"] = assets
    data["meta"]["stocks"] = stocks
    data["meta"]["universe"] = universe
    data["meta"].update(split)
    for k in ["assets", "stocks", "summary", "outliers", "monthly_means", "logs"]:
        if k not in data:
            data[k] = {} if k != "logs" else []
    if "assets" not in data["summary"]:
        data["summary"]["assets"] = {}
    if "stocks" not in data["summary"]:
        data["summary"]["stocks"] = {}
    return data


def _write_outputs(out_dir: str, payload: Dict[str, Any]):
    json_path = os.path.join(out_dir, "data.json")
    txt_path = os.path.join(out_dir, "read.txt")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    universe = payload.get("meta", {}).get("universe", None) or _universe_symbols()
    lines: List[str] = []
    for sym in universe:
        lines.append(sym)
        for row in payload.get("monthly_means", {}).get(sym, []):
            lines.append(f"  {row['month']}: mean_close={row['mean_close']:.6f}, n={row['n']}")
        lines.append("")
    read_txt = "\n".join(lines).rstrip() + "\n"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(read_txt)


def download_one(cfg: FetchConfig, symbol: str, kind: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    logs: List[Dict[str, Any]] = []
    sym = symbol.strip().upper()
    fetch_sym = _SYMBOL_ALIASES.get(sym, sym)
    if fetch_sym != sym:
        logs.append({"time": _now_iso(), "level": "INFO", "code": "SYMBOL_ALIAS", "message": "alias", "context": {"symbol": sym, "fetch_symbol": fetch_sym}})
    if sym == "AU0":
        df0 = fetch_futures_au0_daily(sym, cfg.start, cfg.end, logs)
    else:
        df0 = fetch_us_daily_qfq(fetch_sym, cfg.start, cfg.end, logs)
    df0 = _filter_date_range(df0, cfg.start, cfg.end)
    df1, dup_removed = _dedup_by_date(df0)
    df2, outliers = _zscore_filter_on_returns(df1, cfg.zscore_threshold)

    if kind == "asset":
        payload["assets"][sym] = df2.to_dict(orient="records")
        payload["summary"]["assets"][sym] = _summary_stats(sym, df0, df2, outliers, dup_removed)
    else:
        payload["stocks"][sym] = df2.to_dict(orient="records")
        payload["summary"]["stocks"][sym] = _summary_stats(sym, df0, df2, outliers, dup_removed)
    payload["outliers"][sym] = outliers
    payload["monthly_means"][sym] = _monthly_means(df2)

    payload["logs"].extend(logs)
    payload["logs"].append(
        {
            "time": _now_iso(),
            "level": "INFO",
            "code": "SYMBOL_DONE",
            "message": "processed",
            "context": {"symbol": sym, "kind": kind, "n_raw": int(len(df0)), "n_clean": int(len(df2)), "dup_removed": int(dup_removed), "outliers": int(len(outliers))},
        }
    )
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default=FetchConfig.start)
    parser.add_argument("--end", type=str, default=FetchConfig.end)
    parser.add_argument("--z", type=float, default=FetchConfig.zscore_threshold)
    parser.add_argument("--symbol", type=str, default="")
    parser.add_argument("--kind", type=str, choices=["asset", "stock"], default="")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    cfg = FetchConfig(start=args.start, end=args.end, zscore_threshold=float(args.z))
    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "data.json")

    if args.reset:
        payload = _new_payload(cfg)
    else:
        payload = _load_payload(json_path, cfg)

    assets, stocks = _default_symbols()
    sym = args.symbol.strip().upper()
    kind = args.kind
    if not sym:
        raise RuntimeError("必须提供 --symbol")
    if not kind:
        kind = "asset" if sym in set(assets) else "stock"

    payload = download_one(cfg, sym, kind, payload)
    _write_outputs(out_dir, payload)


if __name__ == "__main__":
    main()
