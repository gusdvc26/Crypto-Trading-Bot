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
    # Preserve strategy_version exactly by default; keep_full flag is respected as a no-op.
    return sv

def _ensure_bool(series: pd.Series) -> pd.Series:
    # Convert a variety of possible encodings to bool
    if series.dtype == "bool":
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(bool)
    # strings like "True"/"False" or "true"/"false" or "0"/"1"
    return series.fillna("False").astype(str).str.lower().isin(("true", "1", "t", "yes", "y"))

def _merge_labels_asof(
    decisions: pd.DataFrame,
    labels: pd.DataFrame,
    tol_ms: int,
) -> pd.DataFrame:
    """As-of tolerance join decisions to labels per symbol using UTC ms columns.

    Expects decisions to have `ts_ms` and labels to have `decision_ts_ms` (both ms ints).
    Falls back to using `t` if ms columns are missing.
    """
    dec = decisions.copy()
    lab = labels.copy()

    # Determine ms columns with safe fallbacks
    if "ts_ms" not in dec.columns and "t" in dec.columns:
        dec["ts_ms"] = pd.to_numeric(dec["t"], errors="coerce")
    if "decision_ts_ms" not in lab.columns and "t" in lab.columns:
        lab["decision_ts_ms"] = pd.to_numeric(lab["t"], errors="coerce")

    # Parse as UTC datetimes
    dec["ts_ms_dt"] = pd.to_datetime(dec["ts_ms"], unit="ms", utc=True)
    lab["decision_ts_ms_dt"] = pd.to_datetime(lab["decision_ts_ms"], unit="ms", utc=True)

    # Ensure symbol present on both sides
    if "symbol" not in dec.columns and "symbol" in lab.columns:
        # broadcast from labels if needed (rare)
        if lab["symbol"].nunique() == 1:
            dec["symbol"] = lab["symbol"].iloc[0]
    if "symbol" not in lab.columns and "symbol" in dec.columns:
        lab["symbol"] = dec["symbol"]

    # Merge as-of with tolerance by symbol
    merged = pd.merge_asof(
        dec.sort_values(["symbol", "ts_ms_dt"]).reset_index(drop=True),
        lab.sort_values(["symbol", "decision_ts_ms_dt"]).reset_index(drop=True),
        by="symbol",
        left_on="ts_ms_dt",
        right_on="decision_ts_ms_dt",
        direction="nearest",
        tolerance=pd.Timedelta(milliseconds=int(tol_ms)),
    )

    return merged

def main():
    args = parse_args()
    root = Path(args.root)
    dates = _date_range(args.start, args.end)

    keep_full_sv = _env_bool("DATASET_KEEP_FULL_SV", False)
    log_info = _env_bool("DATASET_LOG_LEVEL", False) or os.getenv("DATASET_LOG_LEVEL", "").upper() == "INFO"
    time_tol_ms = _env_int("DATASET_TIME_TOL_MS", 2000)

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

            # Ensure strategy_version present and preserved as-is
            if "strategy_version" not in dec.columns:
                dec["strategy_version"] = "v1"
            else:
                dec["strategy_version"] = dec["strategy_version"].astype(str)

            dec["symbol"] = sym

            # Load labels (preferred single-file) and merge with tolerance
            lab_single = base / "labels.jsonl.gz"
            if lab_single.exists():
                lab_recs = _load_jsonl(lab_single)
                labs = pd.DataFrame(lab_recs) if lab_recs else pd.DataFrame()
                if not labs.empty:
                    merged = _merge_labels_asof(dec, labs, tol_ms=time_tol_ms)
                else:
                    merged = dec.copy()
            else:
                # Fallback: horizon-specific files; maintain prior behavior with tolerance on 't'
                merged = dec.copy()
                if "t" in merged.columns:
                    merged["t"] = pd.to_datetime(merged["t"], unit=None, utc=False, errors="coerce")
                for h in HORIZONS_MIN:
                    lab_path = base / f"labels_{h}m.jsonl.gz"
                    lab_recs = _load_jsonl(lab_path)
                    if not lab_recs:
                        for b in BOOL_COLS_BASE:
                            merged[f"{b}_{h}m"] = False
                        for b in FLOAT_COLS_BASE:
                            merged[f"{b}_{h}m"] = pd.NA
                        if log_info:
                            print(f"[INFO] {sym} {d}: NO labels_{h}m.jsonl.gz")
                        continue
                    labs = pd.DataFrame(lab_recs)
                    if "t" not in labs.columns:
                        for b in BOOL_COLS_BASE:
                            merged[f"{b}_{h}m"] = False
                        for b in FLOAT_COLS_BASE:
                            merged[f"{b}_{h}m"] = pd.NA
                        if log_info:
                            print(f"[WARN] {sym} {d}: labels_{h}m.jsonl.gz missing 't' column")
                        continue
                    labs = labs.copy()
                    labs["t"] = pd.to_datetime(labs["t"], utc=False, errors="coerce")
                    # Rename horizon-specific columns
                    ren = {}
                    for b in BOOL_COLS_BASE:
                        if b in labs.columns:
                            ren[b] = f"{b}_{h}m"
                    for b in FLOAT_COLS_BASE:
                        if b in labs.columns:
                            ren[b] = f"{b}_{h}m"
                    labs = labs.rename(columns=ren)
                    keep_cols = ["t"] + list(ren.values())
                    labs = labs[keep_cols]
                    merged = pd.merge_asof(
                        merged.sort_values("t"),
                        labs.sort_values("t"),
                        on="t",
                        direction="nearest",
                        tolerance=pd.to_timedelta(time_tol_ms, unit="ms"),
                    )

                # Type coercion for fallback path
                for h in HORIZONS_MIN:
                    for b in BOOL_COLS_BASE:
                        col = f"{b}_{h}m"
                        if col in merged.columns:
                            merged[col] = _ensure_bool(merged[col])
                        else:
                            merged[col] = False
                    for b in FLOAT_COLS_BASE:
                        col = f"{b}_{h}m"
                        if col in merged.columns:
                            merged[col] = pd.to_numeric(merged[col], errors="coerce")
                        else:
                            merged[col] = pd.NA

            # Minimal cleanup: sort by decision time if available
            sort_col = "ts_ms" if "ts_ms" in merged.columns else ("t" if "t" in merged.columns else None)
            if sort_col is not None:
                merged = merged.sort_values(sort_col).reset_index(drop=True)
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
    # QA and optional gate
    mask_cols = [c for c in ["mask_1m", "mask_5m", "mask_15m"] if c in df.columns]
    qa = {f"{c}_true_pct": float(df[c].mean()) if c in df.columns else None for c in mask_cols}
    print("DATASET_QA", qa)
    import sys as _sys
    if os.getenv('ENFORCE_QA','0')=='1' and 'mask_5m' in df.columns and (df['mask_5m'].mean() < 0.30):
        print("QA_GATE_FAIL: mask_5m < 30%"); _sys.exit(2)

    m5_true = int(df.get("mask_5m", pd.Series([False] * len(df))).astype(bool).sum())
    print(f"Wrote Parquet: {out_path.as_posix()}")
    print(f"Built dataset: {out_path.as_posix()} (rows={len(df)})")
    print("strategy_version sample:\n", by_sv.to_string())
    print("mask_5m True rows:", m5_true)

if __name__ == "__main__":
    main()
