# src/processing/dataset_builder.py
from __future__ import annotations

import argparse
import gzip
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

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


# ------------------- NEW: robust timestamp sanitizer -------------------
def _coerce_ts_ms(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    Ensure df has integer millisecond column `col_name`, drop nulls, and add a
    UTC datetime companion column `<col_name>_dt`. Also drops rows with null symbol.
    """
    df = df.copy()

    # Drop rows with missing symbol (prevents group-by/asof issues)
    if "symbol" in df.columns:
        df = df.dropna(subset=["symbol"])

    # If the join column is missing, try fallbacks (older formats)
    if col_name not in df.columns:
        if "ts_ms" in df.columns:
            df[col_name] = df["ts_ms"]
        elif "ts" in df.columns:
            def to_utc_ms(x):
                if isinstance(x, (int, float)):
                    return int(x)
                # ISO strings, allow trailing Z
                return int(
                    datetime.fromisoformat(str(x).replace("Z", "+00:00"))
                    .replace(tzinfo=timezone.utc)
                    .timestamp()
                    * 1000
                )
            df[col_name] = df["ts"].map(to_utc_ms)
        elif "t" in df.columns:
            # common case where 't' is epoch ms already or ISO string
            def t_to_ms(x):
                # try numeric first
                x_num = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
                if pd.notna(x_num):
                    return int(x_num)
                # else parse as datetime
                return int(pd.to_datetime(x, utc=True, errors="coerce").timestamp() * 1000)
            df[col_name] = df["t"].map(t_to_ms)
        else:
            raise ValueError(f"Required column {col_name} missing and no fallback present.")

    # Coerce to integer ms and drop nulls
    df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
    null_pct = float(df[col_name].isna().mean())
    if null_pct > 0:
        print(f"WARN: dropping {null_pct:.2%} rows with null {col_name}")
        df = df.dropna(subset=[col_name])

    df[col_name] = df[col_name].astype("int64")
    df[col_name + "_dt"] = pd.to_datetime(df[col_name], unit="ms", utc=True)
    return df
# ----------------------------------------------------------------------


def _merge_labels_asof(
    decisions: pd.DataFrame,
    labels: pd.DataFrame,
    tol_ms: int,
) -> pd.DataFrame:
    """
    As-of tolerance join decisions to labels per symbol using UTC ms columns.
    Expects decisions to have a usable 'ts_ms' and labels a 'decision_ts_ms' (both ms).
    Falls back to other columns via _coerce_ts_ms.
    """
    dec = decisions.copy()
    labs = labels.copy()

    # Ensure the canonical join keys exist; coerce + drop nulls; add *_dt
    dec = _coerce_ts_ms(dec, "ts_ms")
    labs = _coerce_ts_ms(labs, "decision_ts_ms")

    # Ensure symbol present on both sides (final guard)
    if "symbol" not in dec.columns and "symbol" in labs.columns and labs["symbol"].nunique() == 1:
        dec["symbol"] = labs["symbol"].iloc[0]
    if "symbol" not in labs.columns and "symbol" in dec.columns and dec["symbol"].nunique() == 1:
        labs["symbol"] = dec["symbol"].iloc[0]

    # Sort by ['symbol', on] for merge_asof with by=
    dec = dec.sort_values(["symbol", "ts_ms_dt"]).reset_index(drop=True)
    labs = labs.sort_values(["symbol", "decision_ts_ms_dt"]).reset_index(drop=True)

    merged = pd.merge_asof(
        dec,
        labs,
        by="symbol",
        left_on="ts_ms_dt",
        right_on="decision_ts_ms_dt",
        direction="nearest",
        tolerance=pd.Timedelta(milliseconds=int(tol_ms)),
    )

    # Ensure mask columns are real bools
    for _h in (1, 5, 15):
        _c = f"mask_{_h}m"
        if _c in merged.columns:
            merged[_c] = _ensure_bool(merged[_c])

    # --- reconcile duplicated columns after merge (e.g., strategy_version_x/_y) ---
    cand = [c for c in ["strategy_version_x", "strategy_version", "strategy_version_y", "strategy"] if c in merged.columns]
    if cand:
        merged["strategy_version"] = (
            merged[cand]
            .astype("string")
            .bfill(axis=1)
            .iloc[:, 0]
            .fillna("v1")
            .astype(str)
        )
        drop_cols = [c for c in ["strategy_version_x", "strategy_version_y"] if c in merged.columns]
        merged = merged.drop(columns=drop_cols, errors="ignore")

    # (Optional) action coalesce
    cand_act = [c for c in ["action_x", "action", "action_y"] if c in merged.columns]
    if cand_act:
        merged["action"] = (
            merged[cand_act]
            .astype("string")
            .bfill(axis=1)
            .iloc[:, 0]
            .fillna("HOLD")
            .astype(str)
        )
        merged = merged.drop(columns=[c for c in ["action_x", "action_y"] if c in merged.columns], errors="ignore")

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
            if keep_full_sv:
                dec["strategy_version"] = dec["strategy_version"].map(lambda s: _normalize_sv(s, keep_full=True))

            dec["symbol"] = sym

            # Load labels (preferred single merged labels file) and merge with tolerance
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
                    merged["t"] = pd.to_datetime(merged["t"], utc=False, errors="coerce")
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
    sv_col = "strategy_version" if "strategy_version" in df.columns else ("strategy" if "strategy" in df.columns else None)
    if sv_col is None:
        by_sv = pd.Series(dtype=int)
    else:
        by_sv = df[sv_col].astype(str).value_counts().head(10)

    # ===== Dataset QA (keep this) =====
    mask_cols = [c for c in ["mask_1m", "mask_5m", "mask_15m"] if c in df.columns]
    qa = {f"{c}_true_pct": float(df[c].mean()) if c in df.columns else None for c in mask_cols}
    qa["rows"] = int(len(df))
    print("DATASET_QA", qa)

    # Coverage by symbol for 5m (top/bottom few to avoid spam)
    if "mask_5m" in df.columns:
        # mask_5m is boolean → mean is coverage fraction
        cov_by_sym = (
            df.groupby("symbol")["mask_5m"]
              .mean()
              .astype(float)
              .sort_values(ascending=False)
        )
        print("MASK5M_COVERAGE_BY_SYMBOL_TOP", cov_by_sym.head(15).round(3).to_dict())
        print("MASK5M_COVERAGE_BY_SYMBOL_BOTTOM", cov_by_sym.tail(15).round(3).to_dict())

    import sys as _sys
    if os.getenv("ENFORCE_QA", "0") == "1" and "mask_5m" in df.columns and (df["mask_5m"].mean() < 0.30):
        print("QA_GATE_FAIL: mask_5m < 30%")
        _sys.exit(2)
    # ==================================

    m5_true = int(df.get("mask_5m", pd.Series([False] * len(df))).astype(bool).sum())
    print(f"Wrote Parquet: {out_path.as_posix()}")
    print(f"Built dataset: {out_path.as_posix()} (rows={len(df)})")
    print("strategy_version sample:\n", by_sv.to_string())
    print("mask_5m True rows:", m5_true)


if __name__ == "__main__":
    main()
