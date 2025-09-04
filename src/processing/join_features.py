# src/processing/join_features.py
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List

import pandas as pd

from config.settings import get_settings
from src.utils.helpers import get_logger

log = get_logger(__name__)
S = get_settings()

def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def _date_range_utc(start: str, end: str) -> List[str]:
    d0 = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    d1 = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    out = []
    cur = d0
    while cur <= d1:
        out.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return out

def _feat_path(exchange: str, symbol: str, date_str: str) -> Path:
    return S.persist.dir_ml / exchange / symbol / date_str / "features_v1.parquet"

def main_cli():
    ap = argparse.ArgumentParser(description="Join dataset_v1 with features_v1 into a train-ready table.")
    ap.add_argument("--dataset", default="data/processed/train/dataset_v1.parquet")
    ap.add_argument("--exchange", default=S.api.exchange)
    ap.add_argument("--symbols", required=True, help="Comma list (e.g., BTC-USD,ETH-USD)")
    ap.add_argument("--start", default=_today_utc())
    ap.add_argument("--end", default=_today_utc())
    ap.add_argument("--out", default="data/processed/train/dataset_v1_ml.parquet")
    args = ap.parse_args()

    df = pd.read_parquet(args.dataset)
    syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
    dates = _date_range_utc(args.start, args.end)

    # Load & concat features across requested partitions
    feat_frames = []
    for sym in syms:
        for day in dates:
            p = _feat_path(args.exchange, sym, day)
            if p.exists():
                f = pd.read_parquet(p)
                f["symbol"] = sym  # ensure symbol present
                feat_frames.append(f)

    if not feat_frames:
        print("No features found; did you run feature_builder?")
        return

    feats = pd.concat(feat_frames, ignore_index=True)

    # Join on (symbol, t)
    merged = df.merge(feats, on=["symbol", "t"], how="left", suffixes=("", "_feat"))

    # Keep only rows with features present (dropna on one core feature)
    merged = merged.dropna(subset=["x_ret_1m"]).reset_index(drop=True)

    # Write output
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.out, index=False)
    print(f"Wrote: {args.out} (rows={len(merged)}, cols={len(merged.columns)})")

if __name__ == "__main__":
    main_cli()
