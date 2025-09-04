from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd

from config.settings import get_settings


S = get_settings()


def _choose_ms_col(df: pd.DataFrame) -> Optional[str]:
    """Pick a millisecond timestamp column for day partitioning.

    Preference order: ts_ms, t. Returns None if neither exists.
    """
    if "ts_ms" in df.columns:
        return "ts_ms"
    if "t" in df.columns:
        return "t"
    return None


def _to_date_series(df: pd.DataFrame, ms_col: str) -> pd.Series:
    """Convert a millisecond epoch column to YYYY-MM-DD (UTC) strings."""
    return pd.to_datetime(pd.to_numeric(df[ms_col], errors="coerce"), unit="ms", utc=True).dt.strftime("%Y-%m-%d")


def _manifest_for(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute manifest payload for a partitioned day DataFrame."""
    row_count = int(len(df))
    sv_col = "strategy_version" if "strategy_version" in df.columns else ("strategy" if "strategy" in df.columns else None)
    if sv_col is not None:
        strategy_versions = sorted(map(str, pd.unique(df[sv_col].astype(str))))
    else:
        strategy_versions = []
    mask_true = float(df["mask_5m"].mean()) if "mask_5m" in df.columns else float("nan")
    bad_rows = float(df["mask_5m"].isna().mean()) if "mask_5m" in df.columns else float("nan")
    build_ts_utc = datetime.now(timezone.utc).isoformat()
    return {
        "row_count": row_count,
        "strategy_versions": strategy_versions,
        "mask_5m_true_pct": mask_true,
        "bad_rows_pct": bad_rows,
        "build_ts_utc": build_ts_utc,
    }


def materialize_from_dataset(data_path: str, out_root: Optional[str] = None) -> int:
    """Materialize a gold daily partition and manifest from a merged dataset parquet.

    - Reads dataset parquet at `data_path`.
    - Splits rows by UTC day based on `ts_ms` or `t` (epoch ms).
    - For each day, writes a single file `part-00000.parquet` under
      `<out_root>/YYYY/MM/DD/` and a `manifest.json` with summary metrics.

    Returns total number of rows materialized across all days.
    """
    out_base = Path(out_root) if out_root else (S.data_dir / "gold" / "merged")
    df = pd.read_parquet(data_path)
    ms_col = _choose_ms_col(df)
    if ms_col is None:
        raise KeyError("Dataset is missing a millisecond timestamp column ('ts_ms' or 't') for partitioning")

    dates = _to_date_series(df, ms_col)
    total = 0
    for day, g in df.groupby(dates):
        y, m, d = day.split("-")
        out_dir = out_base / y / m / d
        out_dir.mkdir(parents=True, exist_ok=True)
        # Write a single part file for this day
        part_path = out_dir / "part-00000.parquet"
        g.to_parquet(part_path, index=False)
        # Write manifest
        manifest = _manifest_for(g)
        (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        total += len(g)
    return int(total)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Materialize daily gold parquet and manifest from merged dataset parquet.")
    ap.add_argument("--data", required=True, help="Path to merged dataset parquet file")
    ap.add_argument("--out-root", default=str((S.data_dir / "gold" / "merged").as_posix()), help="Output root for gold partitions")
    return ap.parse_args()


def main_cli() -> None:
    args = _parse_args()
    n = materialize_from_dataset(args.data, out_root=args.out_root)
    print(f"Materialized gold rows: {n}")


if __name__ == "__main__":
    main_cli()

