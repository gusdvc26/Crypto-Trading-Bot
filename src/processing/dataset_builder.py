# src/processing/dataset_builder.py
from __future__ import annotations

import argparse
import gzip
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

ROOT_DEFAULT = "data/processed/coinbase"

HORIZONS_MIN = (1, 5, 15)  # must match labeler
BOOL_COLS_BASE = ("mask", "tp_first", "sl_first")
FLOAT_COLS_BASE = ("dir_ret",)

def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

@dataclass
class Args:
    root: str
    start: str
    end: str
    symbols: Optional[str]
    out: str

def parse_args() -> Args:
    p = argparse.ArgumentParser(
        description="Build a labeled training dataset by merging decisions and labels."
    )
    p.add_argument("--root", default=ROOT_DEFAULT, help="root folder of processed exchange data")
    p.add_argument("--start", required=True, help="start date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="end date YYYY-MM-DD (inclusive)")
    p.add_argument("--symbols", default="ALL", help="comma-separated list or 'ALL'")
    p.add_argument("--out", required=True, help="output parquet path (without .parquet extension or with; both ok)")
    a = p.parse_args()

    out = a.out
    if out.endswith(".parquet"):
        out = out[:-8]

    return Args(root=a.root, start=a.start, end=a.end, symbols=a.symbols, out=out)

def _date_range(start: str, end: str) -> List[str]:
    s = datetime.strptime(start, "%Y-%m-%d").date()
    e = datetime.strptime(end, "%Y-%m-%d").date()
    cur = s
    out: List[str] = []
    while cur <= e:
        out.append(cur.isoformat())
        cur += timedelta(days=1)
    return out

def _list_symbols(root: Path) -> List[str]:
    return sorted([p.name for p in root.iterdir() if p.is_dir()])

def _load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    recs: List[dict] = []
    if path.suffix == ".gz" or path.name.endswith(".jsonl.gz"):
        opener = gzip.open
        mode = "rt"
    else:
        opener = open
        mode = "r"
    with opener(path, mode, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except Exception:
                # ignore malformed line
                continue
    return recs

def _normalize_sv(sv: str, keep_full: bool) -> str:
    if keep_full:
        return sv
    # collapse to family prefix, e.g. "v1"
    return sv.split("|", 1)[0] if sv else "v1"

def _ensure_bool(series: pd.Series) -> pd.Series:
    # Convert a variety of possible encodings to bool
    if series.dtype == "bool":
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(bool)
    # strings like "True"/"False" or "true"/"false" or "0"/"1"
    return series.fillna("False").astype(str).str.lower().isin(("true", "1", "t", "yes", "y"))

def _merge_labels(
    decisions: pd.DataFrame,
    labels: pd.DataFrame,
    horizon_min: int,
    time_tolerance_ms: int,
) -> pd.DataFrame:
    labels = labels.copy()
    labels["t"] = pd.to_datetime(labels["t"], utc=False)
    # Drop duplicate timestamps in labels (keep last)
    labels = labels.sort_values("t").drop_duplicates(subset=["t"], keep="last")

    # Rename horizon-specific columns
    ren = {}
    for b in BOOL_COLS_BASE:
        if b in labels.columns:
            ren[b] = f"{b}_{horizon_min}m"
    for b in FLOAT_COLS_BASE:
        if b in labels.columns:
            ren[b] = f"{b}_{horizon_min}m"
    labels = labels.rename(columns=ren)

    # Keep only columns we need
    keep_cols = ["t"] + list(ren.values())
    labels = labels[keep_cols]

    dec = decisions.sort_values("t").copy()

    if time_tolerance_ms > 0:
        # nearest-merge with tolerance
        merged = pd.merge_asof(
            dec,
            labels.sort_values("t"),
            on="t",
            direction="nearest",
            tolerance=pd.to_timedelta(time_tolerance_ms, unit="ms"),
        )
    else:
        merged = dec.merge(labels, on="t", how="left", validate="one_to_one")

    # Coerce types
    for b in BOOL_COLS_BASE:
        col = f"{b}_{horizon_min}m"
        if col in merged.columns:
            merged[col] = _ensure_bool(merged[col])
        else:
            merged[col] = False

    for b in FLOAT_COLS_BASE:
        col = f"{b}_{horizon_min}m"
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
        else:
            merged[col] = pd.NA

    return merged

def main():
    args = parse_args()
    root = Path(args.root)
    dates = _date_range(args.start, args.end)

    keep_full_sv = _env_bool("DATASET_KEEP_FULL_SV", False)
    log_info = _env_bool("DATASET_LOG_LEVEL", False) or os.getenv("DATASET_LOG_LEVEL", "").upper() == "INFO"
    time_tol_ms = _env_int("DATASET_TIME_TOL_MS", 0)

    if args.symbols.upper() == "ALL":
        symbols = _list_symbols(root)
    else:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    frames: List[pd.DataFrame] = []

    for sym in symbols:
        for d in dates:
            base = root / sym / d
            dec_path = base / "decisions.jsonl.gz"
            if not dec_path.exists():
                continue

            # Load decisions
            dec_recs = _load_jsonl(dec_path)
            if not dec_recs:
                continue
            dec = pd.DataFrame(dec_recs)

            # Sanity: required cols
            if "t" not in dec.columns:
                continue

            dec["t"] = pd.to_datetime(dec["t"], utc=False)
            # Respect strategy_version policy
            sv_col = dec.get("strategy_version")
            if sv_col is not None:
                dec["strategy_version"] = dec["strategy_version"].astype(str).map(lambda s: _normalize_sv(s, keep_full_sv))
            else:
                dec["strategy_version"] = "v1"

            dec["symbol"] = sym

            # Merge labels for each horizon
            merged = dec
            for h in HORIZONS_MIN:
                lab_path = base / f"labels_{h}m.jsonl.gz"
                lab_recs = _load_jsonl(lab_path)
                if not lab_recs:
                    # add empty columns for this horizon
                    for b in BOOL_COLS_BASE:
                        merged[f"{b}_{h}m"] = False
                    for b in FLOAT_COLS_BASE:
                        merged[f"{b}_{h}m"] = pd.NA
                    if log_info:
                        print(f"[INFO] {sym} {d}: NO labels_{h}m.jsonl.gz")
                    continue

                labs = pd.DataFrame(lab_recs)
                if "t" not in labs.columns:
                    # corrupt label file; make empty columns
                    for b in BOOL_COLS_BASE:
                        merged[f"{b}_{h}m"] = False
                    for b in FLOAT_COLS_BASE:
                        merged[f"{b}_{h}m"] = pd.NA
                    if log_info:
                        print(f"[WARN] {sym} {d}: labels_{h}m.jsonl.gz missing 't' column")
                    continue

                merged = _merge_labels(merged, labs, horizon_min=h, time_tolerance_ms=time_tol_ms)

            # Minimal cleanup
            merged = merged.sort_values("t").reset_index(drop=True)
            frames.append(merged)

            if log_info:
                m5 = int(merged["mask_5m"].sum()) if "mask_5m" in merged.columns else 0
                print(f"[INFO] {sym} {d}: rows={len(merged)}  mask_5m_true={m5}")

    if not frames:
        print("No data to write.")
        return

    df = pd.concat(frames, ignore_index=True)

    # Output
    out_path = Path(args.out).with_suffix(".parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    # Report a quick summary in logs
    by_sv = df["strategy_version"].value_counts().head(10)
    m5_true = int(df.get("mask_5m", pd.Series([False] * len(df))).astype(bool).sum())
    print(f"Wrote Parquet: {out_path.as_posix()}")
    print(f"Built dataset: {out_path.as_posix()} (rows={len(df)})")
    print("strategy_version sample:\n", by_sv.to_string())
    print("mask_5m True rows:", m5_true)

if __name__ == "__main__":
    main()
