# src/processing/feature_builder.py
from __future__ import annotations

import argparse
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import math

from config.settings import get_settings
from src.utils.helpers import get_logger

log = get_logger(__name__)
S = get_settings()

DATASET_VERSION = "v1"
FEATURE_VERSION = "v1"

def _raw_ohlcv_path(exchange: str, symbol: str, date_str: str) -> Path:
    return S.persist.dir_raw / exchange / symbol / date_str / "ohlcv.jsonl.gz"

def _features_path(exchange: str, symbol: str, date_str: str) -> Path:
    out_dir = S.persist.dir_ml / exchange / symbol / date_str
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"features_{FEATURE_VERSION}.parquet"

def _read_jsonl_gz(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def _to_df(rows: List[Dict[str, Any]]):
    import pandas as pd  # heavy only offline
    if not rows:
        return pd.DataFrame(columns=["t","o","h","l","c","v"])
    df = pd.DataFrame.from_records(rows, columns=["t","o","h","l","c","v"])
    df = df.sort_values("t").reset_index(drop=True)
    return df

def _compute_features(df):
    import pandas as pd
    eps = 1e-12

    # 1m returns (close/close)
    ret_1m = df["c"].pct_change(1)
    # 3m/5m returns
    ret_3m = df["c"].pct_change(3)
    ret_5m = df["c"].pct_change(5)

    # Rolling volatility of 1m returns (15m window)
    vol_15m = ret_1m.rolling(15, min_periods=15).std()

    # SMA gap (5 vs 20), normalized by SMA20
    sma5 = df["c"].rolling(5, min_periods=5).mean()
    sma20 = df["c"].rolling(20, min_periods=20).mean()
    sma_gap = (sma5 - sma20) / (sma20.abs() + eps)

    # Candle body ratio for current bar
    body = (df["c"] - df["o"]).abs()
    range_ = (df["h"] - df["l"]).abs()
    body_ratio = body / (range_ + eps)

    # Time of day (UTC)
    dt = (df["t"] // 1000).apply(datetime.utcfromtimestamp)
    hour = dt.dt.hour + dt.dt.minute / 60.0
    # cyclical encoding (24h)
    x_hour_sin = (2 * math.pi * hour / 24.0).apply(math.sin)
    x_hour_cos = (2 * math.pi * hour / 24.0).apply(math.cos)

    # Build features frame (aligned by t)
    feats = pd.DataFrame({
        "t": df["t"].astype("int64"),
        "x_ret_1m": ret_1m,
        "x_ret_3m": ret_3m,
        "x_ret_5m": ret_5m,
        "x_vol_15m": vol_15m,
        "x_sma_gap_5_20": sma_gap,
        "x_body_ratio": body_ratio,
        "x_hour_sin": x_hour_sin,
        "x_hour_cos": x_hour_cos,
        "dataset_version": DATASET_VERSION,
        "feature_version": FEATURE_VERSION,
    })
    # Drop rows that don't have enough lookback (NaNs)
    feats = feats.dropna().reset_index(drop=True)
    return feats

def _write_parquet(path: Path, df) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd
        import pyarrow as pa  # noqa
        import pyarrow.parquet as pq  # noqa
        df.to_parquet(path, index=False)
        log.info(f"Wrote features: {path}")
    except Exception as e:
        # Fallback to JSONL.gz if parquet unavailable
        alt = path.with_suffix(".jsonl.gz")
        with gzip.open(alt, "wb") as f:
            for rec in df.to_dict(orient="records"):
                f.write((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))
        log.warning(f"Parquet unavailable ({e}); wrote JSONL.gz: {alt}")

def build_for_partition(exchange: str, symbol: str, date_str: str) -> int:
    rows = _read_jsonl_gz(_raw_ohlcv_path(exchange, symbol, date_str))
    if not rows:
        log.info(f"No OHLCV for {exchange}/{symbol} on {date_str}; skipping.")
        return 0
    df = _to_df(rows)
    feats = _compute_features(df)
    if feats.empty:
        log.info(f"No feature rows (insufficient history) for {exchange}/{symbol} on {date_str}")
        return 0
    _write_parquet(_features_path(exchange, symbol, date_str), feats)
    return len(feats)

def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def main_cli():
    ap = argparse.ArgumentParser(description="Build v1 features per partition (exchange/symbol/date).")
    ap.add_argument("--exchange", default=S.api.exchange)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--date", default=_today_utc())
    args = ap.parse_args()

    n = build_for_partition(args.exchange, args.symbol, args.date)
    print(f"Built {n} feature rows for {args.exchange}/{args.symbol} on {args.date}")

if __name__ == "__main__":
    main_cli()
